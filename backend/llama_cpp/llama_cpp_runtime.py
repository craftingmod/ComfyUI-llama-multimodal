from __future__ import annotations

import asyncio
import atexit
import hashlib
import ipaddress
import json
import logging
import os
import platform
import shutil
import socket
import stat
import subprocess
import sys
import tarfile
import tempfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from threading import Lock, Thread
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urljoin, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener, urlopen

from aiohttp import web

from ..llm_model_paths import (
    get_default_llm_model_directory,
    get_llm_model_directories,
    resolve_llm_model_directory,
)
from ..scoped_process import ScopedProcess
from .llama_cpp_release_b11429 import (
    _LLAMA_RELEASE,
    _LLAMA_RELEASE_URL,
    _RELEASE_ASSETS,
)

RUNTIME_ROUTE = "/ollama_image_list/llama_cpp/runtime"
RUNTIME_RESTART_ROUTE = f"{RUNTIME_ROUTE}/restart"
RUNTIME_DOWNLOAD_ROUTE = f"{RUNTIME_ROUTE}/download"
_DEFAULT_PORT = 18582
_MIN_PORT = 1024
_MAX_PORT = 65535
_CONFIG_NAME = "settings.json"
_DEFAULT_CTX_SIZE = 16384
_MIN_CTX_SIZE = 512
_MAX_CTX_SIZE = 1048576
_STARTUP_TIMEOUT_SECONDS = 30
_HEALTH_REQUEST_TIMEOUT_SECONDS = 0.5
_HEALTH_POLL_INTERVAL_SECONDS = 0.2
_SUPERVISOR_SHUTDOWN_TIMEOUT_SECONDS = 8
_VERSION_PROBE_TIMEOUT = 3
_VERSION_DISPLAY_LIMIT = 256
_DOWNLOAD_CHUNK_SIZE = 1024 * 1024
_DOWNLOAD_TIMEOUT_SECONDS = 30
_INSTALL_MARKER = "install.json"
_ALLOWED_REDIRECT_HOSTS = {
    "github.com",
    "release-assets.githubusercontent.com",
    "objects.githubusercontent.com",
}
_logger = logging.getLogger(__name__)
_lock = Lock()
_restart_lock = Lock()
_download_lock = Lock()
_auto_start = False
_ctx_size = _DEFAULT_CTX_SIZE
_port = _DEFAULT_PORT
_model_dir: str | None = None
_active_port: int | None = None
_state = "stopped"
_start_attempted = False
_process: ScopedProcess | None = None
_last_error: str | None = None
_routes_registered = False
_version_probe_executable: str | None = None
_llama_version: str | None = None
_download_state = "idle"
_download_target: str | None = None
_download_bytes_received = 0
_download_bytes_total: int | None = None
_download_error: str | None = None
_download_thread: Thread | None = None
_CONFIG_DIRECTORY_ERROR = (
    "This ComfyUI version does not provide the protected system user directory "
    "API; internal llama.cpp settings are unavailable."
)


class _RestartDisabled(RuntimeError):
    pass


def _start_supervisor(command: list[str], environment: dict[str, str]) -> ScopedProcess:
    supervisor_path = Path(__file__).with_name("llama_cpp_supervisor.py")
    return ScopedProcess.start(
        [sys.executable, str(supervisor_path), "--", *command],
        stdin=subprocess.PIPE,
        close_fds=True,
        env=environment,
    )


@dataclass
class OwnedLlamaServer:
    url: str
    _process: ScopedProcess

    def close(self) -> None:
        self._process.close(graceful_timeout=_SUPERVISOR_SHUTDOWN_TIMEOUT_SECONDS)


def _ephemeral_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as port_probe:
        port_probe.bind(("127.0.0.1", 0))
        return int(port_probe.getsockname()[1])


def _wait_for_owned_server_ready(process: ScopedProcess, port: int) -> None:
    deadline = time.monotonic() + _STARTUP_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"llama supervisor exited before the server became ready "
                f"(code {process.returncode})."
            )
        if _server_ready(port):
            time.sleep(_HEALTH_POLL_INTERVAL_SECONDS)
            if (
                process.poll() is None
                and _server_ready(port)
                and process.poll() is None
            ):
                return
        time.sleep(_HEALTH_POLL_INTERVAL_SECONDS)
    raise RuntimeError(
        f"llama server did not become ready on port {port} within "
        f"{_STARTUP_TIMEOUT_SECONDS} seconds."
    )


def start_owned_llama_server(
    executable: str,
    server_args: list[str],
    *,
    internal: bool,
    api_key: str | None = None,
) -> OwnedLlamaServer:
    """Start a local server; server_args are options after the ``server`` command."""
    port = _ephemeral_loopback_port()
    command = [
        executable,
        "server",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--cors-origins",
        "localhost",
        "--models-max",
        "1",
        "--parallel",
        "1",
        *server_args,
    ]
    environment = _llama_child_environment(Path(executable), internal=internal)
    if api_key is not None:
        environment["LLAMA_API_KEY"] = api_key
    process = _start_supervisor(command, environment)
    handle = OwnedLlamaServer(f"http://127.0.0.1:{port}", process)
    try:
        _wait_for_owned_server_ready(process, port)
    except BaseException:
        try:
            handle.close()
        except (OSError, RuntimeError):
            _logger.exception("Could not clean up failed owned llama server startup.")
        raise
    return handle


def _config_path() -> Path:
    import folder_paths

    get_system_user_directory = getattr(folder_paths, "get_system_user_directory", None)
    if callable(get_system_user_directory):
        return Path(get_system_user_directory("llama_cpp")) / _CONFIG_NAME
    raise RuntimeError(_CONFIG_DIRECTORY_ERROR)


def _select_release_asset(
    platform_name: str, machine: str, backend: str
) -> dict[str, Any]:
    os_name = {
        "win32": "windows",
        "darwin": "macos",
        "linux": "linux",
    }.get(platform_name)
    if os_name is None:
        raise ValueError(f"llama.cpp downloads are unsupported on {platform_name}.")

    architecture = machine.casefold().replace("_", "").replace("-", "")
    arch = {
        "amd64": "x64",
        "x8664": "x64",
        "x64": "x64",
        "aarch64": "arm64",
        "arm64": "arm64",
    }.get(architecture)
    if arch is None:
        raise ValueError(f"llama.cpp downloads are unsupported on {machine}.")

    if os_name == "macos":
        backend = "cpu"
    if backend not in {"cpu", "cuda", "rocm"}:
        raise ValueError(f"Unsupported llama.cpp backend: {backend}.")
    key = (os_name, arch, backend)
    assets = _RELEASE_ASSETS.get(key)
    if assets is None:
        description = f"{os_name} {arch} {backend}"
        raise ValueError(
            f"No {_LLAMA_RELEASE} llama.cpp build is available for {description}."
        )

    backend_label = {
        "cpu": "CPU",
        "cuda": "CUDA 13.4",
        "rocm": "ROCm 10.0",
    }[backend]
    os_label = {"windows": "Windows", "macos": "macOS", "linux": "Linux"}[os_name]
    label = (
        f"{os_label} {arch}"
        if os_name == "macos"
        else f"{os_label} {arch} {backend_label}"
    )
    return {
        "os": os_name,
        "arch": arch,
        "backend": backend,
        "directory": f"{os_name}-{arch}-{backend}",
        "label": label,
        "assets": assets,
    }


def _current_release_selection() -> dict[str, Any]:
    if sys.platform not in {"win32", "darwin", "linux"}:
        return _select_release_asset(sys.platform, platform.machine(), "cpu")
    if sys.platform == "darwin":
        return _select_release_asset(sys.platform, platform.machine(), "cpu")
    try:
        import torch

        torch_version = torch.version
        backend = (
            "rocm" if torch_version.hip else "cuda" if torch_version.cuda else "cpu"
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not determine the ComfyUI PyTorch accelerator: {exc}"
        ) from exc
    return _select_release_asset(sys.platform, platform.machine(), backend)


def _artifacts_directory() -> Path:
    return _config_path().parent / "artifacts"


def _installation_directory(
    selection: dict[str, Any], release: str = _LLAMA_RELEASE
) -> Path:
    return _artifacts_directory() / release / selection["directory"]


def _installed_llama_executable(
    selection: dict[str, Any], release: str = _LLAMA_RELEASE
) -> str | None:
    try:
        install_dir = _installation_directory(selection, release)
        marker_path = install_dir / _INSTALL_MARKER
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        executable_name = marker["executable"]
        if (
            marker.get("release") != release
            or marker.get("target") != selection["directory"]
            or not isinstance(executable_name, str)
        ):
            return None
        relative_path = PurePosixPath(executable_name.replace("\\", "/"))
        if (
            relative_path.is_absolute()
            or not relative_path.parts
            or any(part in {".", ".."} or ":" in part for part in relative_path.parts)
        ):
            return None
        root = install_dir.resolve()
        executable = install_dir.joinpath(*relative_path.parts).resolve()
        if root not in executable.parents or not executable.is_file():
            return None
        if os.name != "nt" and not os.access(executable, os.X_OK):
            return None
        return str(executable)
    except (KeyError, OSError, RuntimeError, TypeError, ValueError):
        return None


def _older_installed_llama_executable(
    selection: dict[str, Any],
) -> tuple[str, str] | None:
    try:
        releases = tuple(_artifacts_directory().iterdir())
    except OSError:
        return None
    current_build = int(_LLAMA_RELEASE[1:])
    older_releases = sorted(
        (
            path
            for path in releases
            if path.is_dir()
            and not path.is_symlink()
            and path.name.startswith("b")
            and path.name[1:].isdigit()
            and int(path.name[1:]) < current_build
        ),
        key=lambda path: int(path.name[1:]),
        reverse=True,
    )
    for release_dir in older_releases:
        if (release_dir / selection["directory"]).is_symlink():
            continue
        executable = _installed_llama_executable(selection, release_dir.name)
        if executable:
            return release_dir.name, executable
    return None


def _path_llama_executable() -> str | None:
    executable = shutil.which("llama")
    return str(Path(executable).resolve()) if executable else None


def _resolve_llama_executable(
    selection: dict[str, Any] | None = None,
) -> tuple[str | None, str | None]:
    executable = _path_llama_executable()
    if executable:
        return executable, "path"
    if selection is None:
        try:
            selection = _current_release_selection()
        except (RuntimeError, ValueError):
            return None, None
    executable = _installed_llama_executable(selection)
    if executable:
        return executable, "internal"
    older_install = _older_installed_llama_executable(selection)
    return (older_install[1], "internal") if older_install else (None, None)


def _valid_ctx_size(value: Any) -> bool:
    return type(value) is int and _MIN_CTX_SIZE <= value <= _MAX_CTX_SIZE


def _valid_port(value: Any) -> bool:
    return type(value) is int and _MIN_PORT <= value <= _MAX_PORT


def _model_directory_options() -> tuple[list[str], str]:
    import folder_paths

    default_model_dir = get_default_llm_model_directory(folder_paths)
    model_dirs = get_llm_model_directories(folder_paths)
    default_identity = os.path.normcase(default_model_dir)
    ordered_model_dirs = [
        default_model_dir,
        *(path for path in model_dirs if os.path.normcase(path) != default_identity),
    ]
    return ordered_model_dirs, default_model_dir


def _load_runtime_settings(
    model_dirs: list[str], default_model_dir: str
) -> tuple[bool, int, int, str]:
    global _last_error
    try:
        data = json.loads(_config_path().read_text(encoding="utf-8"))
    except FileNotFoundError:
        return False, _DEFAULT_CTX_SIZE, _DEFAULT_PORT, default_model_dir
    except RuntimeError as exc:
        _last_error = str(exc)
        _logger.error("Internal llama.cpp configuration is unavailable: %s", exc)
        return False, _DEFAULT_CTX_SIZE, _DEFAULT_PORT, default_model_dir
    except (OSError, json.JSONDecodeError) as exc:
        _last_error = f"Could not read internal llama.cpp settings: {exc}"
        _logger.warning("Could not read internal llama.cpp settings: %s", exc)
        return False, _DEFAULT_CTX_SIZE, _DEFAULT_PORT, default_model_dir
    if not isinstance(data, dict):
        _logger.warning("Internal llama.cpp settings must be a JSON object.")
        return False, _DEFAULT_CTX_SIZE, _DEFAULT_PORT, default_model_dir
    ctx_size = data.get("ctx_size", _DEFAULT_CTX_SIZE)
    if not _valid_ctx_size(ctx_size):
        _logger.warning("Ignoring invalid context size in internal llama.cpp settings.")
        ctx_size = _DEFAULT_CTX_SIZE
    port = data.get("port", _DEFAULT_PORT)
    if not _valid_port(port):
        _logger.warning("Ignoring invalid port in internal llama.cpp settings.")
        port = _DEFAULT_PORT
    model_dir = data.get("model_dir", default_model_dir)
    resolved_model_dir = (
        resolve_llm_model_directory(model_dir, model_dirs)
        if isinstance(model_dir, str)
        else None
    )
    if resolved_model_dir is None:
        _logger.warning("Ignoring unavailable LLM model directory in settings.")
        resolved_model_dir = default_model_dir
    return data.get("auto_start") is True, ctx_size, port, resolved_model_dir


def _save_runtime_settings(
    auto_start: bool, ctx_size: int, port: int, model_dir: str
) -> None:
    path = _config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "auto_start": auto_start,
                "ctx_size": ctx_size,
                "port": port,
                "model_dir": model_dir,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _llama_executable() -> str | None:
    return _resolve_llama_executable()[0]


def _probe_llama_version(executable: str | None) -> str | None:
    global _version_probe_executable, _llama_version
    if not executable:
        return None
    if executable == _version_probe_executable:
        return _llama_version

    _version_probe_executable = executable
    _llama_version = None
    try:
        result = subprocess.run(
            [executable, "--version"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=_VERSION_PROBE_TIMEOUT,
            check=True,
            shell=False,
        )
        lines = (result.stdout + "\n" + result.stderr).splitlines()
        version_line = next(
            (
                line.strip()
                for line in lines
                if line.strip().lower().startswith("version:")
            ),
            next((line.strip() for line in lines if line.strip()), None),
        )
        _llama_version = version_line[:_VERSION_DISPLAY_LIMIT] if version_line else None
    except (OSError, subprocess.SubprocessError) as exc:
        _logger.warning("Could not probe llama version: %s", exc)
    return _llama_version


def _validate_download_url(url: str) -> None:
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or parsed.hostname not in _ALLOWED_REDIRECT_HOSTS
        or parsed.port not in (None, 443)
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise RuntimeError("Rejected an unsafe llama.cpp download redirect.")


class _GitHubRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        redirected_url = urljoin(req.full_url, newurl)
        try:
            _validate_download_url(redirected_url)
        except RuntimeError as exc:
            raise HTTPError(req.full_url, code, str(exc), headers, fp) from exc
        return super().redirect_request(req, fp, code, msg, headers, redirected_url)


def _download_asset(
    filename: str,
    expected_sha256: str,
    destination: Path,
) -> None:
    global _download_bytes_total, _download_bytes_received
    url = _LLAMA_RELEASE_URL.format(filename)
    _validate_download_url(url)
    request = Request(url, headers={"User-Agent": "ComfyUI-Ollama-Multimodal"})
    opener = build_opener(_GitHubRedirectHandler())
    digest = hashlib.sha256()
    with opener.open(request, timeout=_DOWNLOAD_TIMEOUT_SECONDS) as response:
        _validate_download_url(response.geturl())
        content_length = response.headers.get("Content-Length")
        try:
            expected_size = int(content_length) if content_length is not None else None
        except ValueError:
            expected_size = None
        with _download_lock:
            if expected_size is None:
                _download_bytes_total = None
            elif _download_bytes_total is not None:
                _download_bytes_total += expected_size
        with destination.open("xb") as output:
            while chunk := response.read(_DOWNLOAD_CHUNK_SIZE):
                output.write(chunk)
                digest.update(chunk)
                with _download_lock:
                    _download_bytes_received += len(chunk)
    actual_sha256 = digest.hexdigest()
    if actual_sha256 != expected_sha256:
        destination.unlink(missing_ok=True)
        raise RuntimeError(
            f"SHA-256 verification failed for {filename}: "
            f"expected {expected_sha256}, received {actual_sha256}."
        )


def _safe_archive_path(root: Path, member_name: str) -> Path:
    name = member_name.replace("\\", "/")
    relative_path = PurePosixPath(name)
    if name in {"", "."}:
        return root
    if relative_path.is_absolute() or any(
        part in {"..", "."} or ":" in part for part in relative_path.parts
    ):
        raise RuntimeError(f"Archive contains an unsafe path: {member_name!r}.")
    resolved_root = root.resolve()
    destination = root.joinpath(*relative_path.parts).resolve()
    if resolved_root not in destination.parents:
        raise RuntimeError(f"Archive path escapes its destination: {member_name!r}.")
    return destination


def _extract_archive(archive_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path) as archive:
            for info in archive.infolist():
                target = _safe_archive_path(destination, info.filename)
                mode = (info.external_attr >> 16) & 0xFFFF
                file_type = stat.S_IFMT(mode)
                if file_type == stat.S_IFLNK or file_type not in {
                    0,
                    stat.S_IFREG,
                    stat.S_IFDIR,
                }:
                    raise RuntimeError(
                        f"Archive contains a link or special file: {info.filename!r}."
                    )
                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(info) as source, target.open("xb") as output:
                        shutil.copyfileobj(source, output)
                    if mode & 0o111:
                        target.chmod(mode & 0o777)
        return

    try:
        archive = tarfile.open(archive_path, mode="r:*")
    except (tarfile.TarError, OSError) as exc:
        raise RuntimeError(f"Could not read downloaded archive: {exc}") from exc
    with archive:
        for member in archive.getmembers():
            target = _safe_archive_path(destination, member.name)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                raise RuntimeError(
                    f"Archive contains a link or special file: {member.name!r}."
                )
            target.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise RuntimeError(f"Could not read archive member: {member.name!r}.")
            with source, target.open("xb") as output:
                shutil.copyfileobj(source, output)
            if member.mode & 0o111:
                target.chmod(member.mode & 0o777)


def _archive_content_root(extracted: Path) -> Path:
    entries = list(extracted.iterdir())
    if len(entries) == 1 and entries[0].is_dir():
        return entries[0]
    return extracted


def _find_llama_executable(root: Path, os_name: str) -> Path:
    expected_name = "llama.exe" if os_name == "windows" else "llama"
    matches = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.name.casefold() == expected_name
    )
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one {expected_name} in the "
            f"{_LLAMA_RELEASE} archive; found {len(matches)}."
        )
    return matches[0]


def _is_runtime_library(path: Path) -> bool:
    name = path.name.casefold()
    return name.endswith((".dll", ".dylib")) or ".so" in name


def _copy_runtime_files(source_root: Path, destination: Path, executable: Path) -> Path:
    executable_relative = executable.relative_to(source_root)
    runtime_files = [
        path
        for path in source_root.rglob("*")
        if path.is_file() and (path == executable or _is_runtime_library(path))
    ]
    for source in runtime_files:
        relative_path = source.relative_to(source_root)
        target = destination / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise RuntimeError(f"Multiple archives contain {relative_path}.")
        shutil.copyfile(source, target)
    return destination / executable_relative


def _llama_child_environment(executable: Path, *, internal: bool) -> dict[str, str]:
    environment = os.environ.copy()
    if internal and sys.platform == "linux":
        library_path = str(executable.parent)
        existing_library_path = environment.get("LD_LIBRARY_PATH")
        environment["LD_LIBRARY_PATH"] = os.pathsep.join(
            path for path in (library_path, existing_library_path) if path
        )
    return environment


def _probe_downloaded_executable(
    executable: Path, selection: dict[str, Any]
) -> str | None:
    result = subprocess.run(
        [str(executable), "--version"],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=_VERSION_PROBE_TIMEOUT,
        check=True,
        shell=False,
        env=_llama_child_environment(
            executable,
            internal=selection["os"] == "linux",
        ),
    )
    lines = (result.stdout + "\n" + result.stderr).splitlines()
    version_line = next(
        (line.strip() for line in lines if line.strip().lower().startswith("version:")),
        next((line.strip() for line in lines if line.strip()), None),
    )
    return version_line[:_VERSION_DISPLAY_LIMIT] if version_line else None


def _set_download_state(
    state: str,
    *,
    target: str | None = None,
    error: str | None = None,
) -> None:
    global _download_state, _download_target, _download_error
    with _download_lock:
        _download_state = state
        if target is not None:
            _download_target = target
        _download_error = error


def _download_and_install(selection: dict[str, Any]) -> None:
    global _version_probe_executable, _llama_version, _download_thread
    try:
        is_update = _older_installed_llama_executable(selection) is not None
        artifacts_dir = _artifacts_directory()
        version_dir = artifacts_dir / _LLAMA_RELEASE
        version_dir.mkdir(parents=True, exist_ok=True)
        install_dir = _installation_directory(selection)
        if install_dir.exists():
            if _installed_llama_executable(selection):
                _set_download_state("installed", target=selection["label"])
                return
            raise RuntimeError(
                f"An incomplete installation already exists at {install_dir}; "
                "it was left untouched."
            )

        with tempfile.TemporaryDirectory(
            prefix=".llama-download-", dir=version_dir
        ) as temporary_directory:
            temporary_root = Path(temporary_directory)
            downloaded_archives: list[tuple[Path, str]] = []
            for filename, expected_sha256 in selection["assets"]:
                archive_path = temporary_root / filename
                _download_asset(filename, expected_sha256, archive_path)
                downloaded_archives.append((archive_path, filename))

            _set_download_state("installing", target=selection["label"])
            install_stage = temporary_root / "install"
            install_stage.mkdir()
            primary_archive, _primary_filename = downloaded_archives[0]
            primary_extract = temporary_root / "primary"
            _extract_archive(primary_archive, primary_extract)
            primary_root = _archive_content_root(primary_extract)
            source_executable = _find_llama_executable(primary_root, selection["os"])
            executable = _copy_runtime_files(
                primary_root, install_stage, source_executable
            )

            for archive_path, _filename in downloaded_archives[1:]:
                runtime_extract = temporary_root / "runtime"
                _extract_archive(archive_path, runtime_extract)
                runtime_root = _archive_content_root(runtime_extract)
                for source in runtime_root.rglob("*"):
                    if not source.is_file() or not _is_runtime_library(source):
                        continue
                    relative_path = source.relative_to(runtime_root)
                    target = install_stage / relative_path
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if target.exists():
                        raise RuntimeError(
                            f"Multiple archives contain runtime library {relative_path}."
                        )
                    shutil.copyfile(source, target)

            if os.name != "nt":
                executable.chmod(
                    executable.stat().st_mode
                    | stat.S_IXUSR
                    | stat.S_IXGRP
                    | stat.S_IXOTH
                )
            version = _probe_downloaded_executable(executable, selection)
            marker = {
                "release": _LLAMA_RELEASE,
                "target": selection["directory"],
                "executable": executable.relative_to(install_stage).as_posix(),
                "version": version,
                "assets": [filename for _path, filename in downloaded_archives],
            }
            (install_stage / _INSTALL_MARKER).write_text(
                json.dumps(marker, indent=2) + "\n",
                encoding="utf-8",
            )
            os.replace(install_stage, install_dir)

        installed_executable = str(install_dir / marker["executable"])
        with _lock:
            _version_probe_executable = installed_executable
            _llama_version = version
            restart_after_install = _auto_start and (
                is_update or _last_error == "'llama' executable was not found in PATH."
            )
        if restart_after_install:
            try:
                restart_runtime(wait=True)
            except _RestartDisabled:
                pass
        _set_download_state("installed", target=selection["label"])
    except Exception as exc:
        _logger.exception("Could not download or install llama.cpp.")
        _set_download_state("error", target=selection["label"], error=str(exc))
    finally:
        with _download_lock:
            _download_thread = None


def _start_runtime_download() -> str:
    global _download_state, _download_target, _download_bytes_received
    global _download_bytes_total, _download_error, _download_thread
    with _download_lock:
        if _download_state in {"downloading", "installing"}:
            raise RuntimeError("A llama.cpp download is already in progress.")
        if _path_llama_executable():
            raise RuntimeError("A llama executable is already available in PATH.")
        selection = _current_release_selection()
        if _installed_llama_executable(selection):
            _download_state = "installed"
            _download_target = selection["label"]
            _download_error = None
            return "installed"
        install_dir = _installation_directory(selection)
        if install_dir.exists():
            raise RuntimeError(
                f"An incomplete installation already exists at {install_dir}; "
                "it was left untouched."
            )
        _download_state = "downloading"
        _download_target = selection["label"]
        _download_bytes_received = 0
        _download_bytes_total = 0
        _download_error = None
        _download_thread = Thread(
            target=_download_and_install,
            args=(selection,),
            name="llama-cpp-download",
            daemon=True,
        )
        _download_thread.start()
        return "started"


def _observe_process_exit_locked() -> None:
    global _process, _active_port, _state, _start_attempted, _last_error
    process = _process
    if process is None:
        return
    exit_code = process.poll()
    if exit_code is None:
        return
    try:
        process.close()
    except (OSError, subprocess.TimeoutExpired):
        _logger.exception("Could not release the exited llama supervisor scope")
        return
    _process = None
    _active_port = None
    _start_attempted = True
    if _auto_start:
        _state = "failed"
        _last_error = f"llama supervisor exited unexpectedly with code {exit_code}."
    else:
        _state = "stopped"
        _last_error = None


def _stop_locked(*, clear_error: bool = True) -> bool:
    global _process, _active_port, _state, _start_attempted, _last_error
    process = _process
    _state = "stopping" if process is not None else "stopped"
    if process is None:
        _active_port = None
        _start_attempted = False
        if clear_error:
            _last_error = None
        return True

    try:
        process.close(graceful_timeout=_SUPERVISOR_SHUTDOWN_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        _state = "failed"
        _last_error = "llama supervisor did not stop before the shutdown timeout."
        _logger.error("%s", _last_error)
        return False
    except OSError as exc:
        _state = "failed"
        _last_error = f"Could not wait for llama supervisor to stop: {exc}"
        _logger.error("%s", _last_error, exc_info=True)
        return False

    _process = None
    _active_port = None
    _state = "stopped"
    _start_attempted = False
    if clear_error:
        _last_error = None
    return True


def _server_ready(port: int) -> bool:
    request = Request(f"http://127.0.0.1:{port}/health", method="GET")
    try:
        with urlopen(request, timeout=_HEALTH_REQUEST_TIMEOUT_SECONDS) as response:
            if response.status != 200:
                return False
            health = json.loads(response.read())
    except (OSError, ValueError):
        return False
    return isinstance(health, dict) and health.get("status") == "ok"


def _wait_for_server_ready(process: ScopedProcess, port: int) -> bool:
    deadline = time.monotonic() + _STARTUP_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if process.poll() is not None:
            _observe_process_exit_locked()
            return False
        if _server_ready(port):
            return True
        time.sleep(_HEALTH_POLL_INTERVAL_SECONDS)
    return False


def _start_locked(*, repair_model_dir: bool = True) -> None:
    global _process, _active_port, _state, _start_attempted, _last_error, _model_dir
    if _process is not None:
        if _process.poll() is None:
            return
        _observe_process_exit_locked()
    if _start_attempted:
        return
    _start_attempted = True
    _state = "starting"
    _active_port = None
    _last_error = None

    executable = _llama_executable()
    executable_source = "path" if _path_llama_executable() == executable else "internal"
    if executable is None:
        _state = "failed"
        _last_error = "'llama' executable was not found in PATH."
        _logger.warning("Internal llama.cpp auto-start failed: %s", _last_error)
        return

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as port_probe:
            port_probe.bind(("127.0.0.1", _port))
        model_dirs, default_model_dir = _model_directory_options()
        models_dir_path = (
            resolve_llm_model_directory(_model_dir, model_dirs) if _model_dir else None
        )
        if models_dir_path is None:
            if not repair_model_dir:
                raise RuntimeError(
                    f"Configured LLM models directory is unavailable: {_model_dir}"
                )
            _model_dir = default_model_dir
            _save_runtime_settings(_auto_start, _ctx_size, _port, _model_dir)
            models_dir_path = default_model_dir
        models_dir = Path(models_dir_path)
        models_dir.mkdir(parents=True, exist_ok=True)
        command = [
            executable,
            "server",
            "--host",
            "127.0.0.1",
            "--port",
            str(_port),
            "--models-dir",
            str(models_dir),
            "--models-max",
            "1",
            "--parallel",
            "1",
            "--ctx-size",
            str(_ctx_size),
            "--cors-origins",
            "localhost",
            "--log-verbosity",
            "3",
        ]
        child_environment = _llama_child_environment(
            Path(executable), internal=executable_source == "internal"
        )
        process = _start_supervisor(command, child_environment)
        _process = process
        if not _wait_for_server_ready(process, _port):
            startup_error = _last_error or (
                f"llama server did not become ready on port {_port} within "
                f"{_STARTUP_TIMEOUT_SECONDS} seconds."
            )
            stopped = _stop_locked(clear_error=False)
            _state = "failed"
            _start_attempted = True
            if stopped:
                _last_error = startup_error
            elif _last_error and _last_error != startup_error:
                _last_error = f"{startup_error} {_last_error}"
            _logger.error("Internal llama.cpp auto-start failed: %s", _last_error)
            return
        _active_port = _port
        _state = "running"
        _last_error = None
    except (OSError, RuntimeError) as exc:
        start_error = f"Could not start llama server on port {_port}: {exc}"
        if _process is not None:
            _stop_locked(clear_error=False)
        _active_port = None
        _state = "failed"
        _start_attempted = True
        _last_error = start_error
        _logger.error("%s", _last_error, exc_info=True)


def initialize_llama_cpp_runtime() -> None:
    global _auto_start, _ctx_size, _port, _model_dir
    with _lock:
        model_dirs, default_model_dir = _model_directory_options()
        _auto_start, _ctx_size, _port, _model_dir = _load_runtime_settings(
            model_dirs, default_model_dir
        )
        if _auto_start:
            _start_locked()


def _runtime_status_locked() -> dict[str, Any]:
    _observe_process_exit_locked()
    path_executable = _path_llama_executable()
    try:
        selection = _current_release_selection()
        _artifacts_directory()
        download_support_error = None
    except (RuntimeError, ValueError) as exc:
        selection = None
        download_support_error = str(exc)
    update_available = False
    current_internal_executable: str | None = None
    if path_executable is not None:
        executable = path_executable
        llama_source = "path"
    elif selection is not None:
        current_internal_executable = _installed_llama_executable(selection)
        older_install = (
            _older_installed_llama_executable(selection)
            if current_internal_executable is None
            else None
        )
        executable = current_internal_executable or (
            older_install[1] if older_install else None
        )
        update_available = older_install is not None
        llama_source = "internal" if executable is not None else None
    else:
        executable = None
        llama_source = None
    model_dirs, default_model_dir = _model_directory_options()
    model_dir = (
        resolve_llm_model_directory(_model_dir, model_dirs) if _model_dir else None
    )
    if model_dir is None:
        model_dir = default_model_dir
    llama_version = _probe_llama_version(executable)
    version = llama_version
    if version and version.lower().startswith("version:"):
        version = version.partition(":")[2].strip()
        version = version.split(maxsplit=1)[0] if version else None
    running = _state == "running"
    with _download_lock:
        download_state = _download_state
        download_error = _download_error
        if current_internal_executable is not None and download_state not in {
            "downloading",
            "installing",
        }:
            download_state = "installed"
            download_error = None
        download_target = _download_target or (
            selection["label"] if selection else None
        )
        download_bytes_received = _download_bytes_received
        download_bytes_total = _download_bytes_total
    return {
        "auto_start": _auto_start,
        "ctx_size": _ctx_size,
        "port": _port,
        "active_port": _active_port,
        "model_dir": model_dir,
        "model_dirs": model_dirs,
        "default_model_dir": default_model_dir,
        "state": _state,
        "llama_available": executable is not None,
        "llama_executable": executable,
        "llama_source": llama_source,
        "llama_path_executable": path_executable,
        "llama_version": llama_version,
        "update_available": update_available,
        "download_supported": selection is not None,
        "download_support_error": download_support_error,
        "download_state": download_state,
        "download_target": download_target,
        "download_bytes_received": download_bytes_received,
        "download_bytes_total": download_bytes_total,
        "download_error": download_error,
        "running": running,
        "base_url": (
            f"http://127.0.0.1:{_active_port}"
            if running and _active_port is not None
            else None
        ),
        "version": version,
        "error": _last_error,
    }


def get_runtime_status() -> dict[str, Any]:
    with _lock:
        return _runtime_status_locked()


def update_runtime_settings(
    auto_start: bool | None = None,
    ctx_size: int | None = None,
    port: int | None = None,
    model_dir: str | None = None,
) -> dict[str, Any]:
    global _auto_start, _ctx_size, _port, _model_dir
    with _lock:
        previous_auto_start = _auto_start
        previous_launch_config = (_ctx_size, _port, _model_dir)
        next_auto_start = _auto_start if auto_start is None else auto_start
        next_ctx_size = _ctx_size if ctx_size is None else ctx_size
        next_port = _port if port is None else port
        next_model_dir = _model_dir
        if model_dir is not None:
            model_dirs, _default_model_dir = _model_directory_options()
            next_model_dir = resolve_llm_model_directory(model_dir, model_dirs)
            if next_model_dir is None:
                raise ValueError(
                    "model_dir must be one of the available LLM directories."
                )
        if next_model_dir is None:
            _model_dirs, next_model_dir = _model_directory_options()
        assert next_model_dir is not None
        _save_runtime_settings(
            next_auto_start, next_ctx_size, next_port, next_model_dir
        )
        _auto_start, _ctx_size, _port, _model_dir = (
            next_auto_start,
            next_ctx_size,
            next_port,
            next_model_dir,
        )
        launch_config_changed = previous_launch_config != (
            next_ctx_size,
            next_port,
            next_model_dir,
        )
        if not next_auto_start:
            _stop_locked()
        elif not previous_auto_start or launch_config_changed:
            if not previous_auto_start or _stop_locked():
                _start_locked()
        return _runtime_status_locked()


def restart_runtime(*, wait: bool = False) -> dict[str, Any]:
    if not _restart_lock.acquire(blocking=wait):
        with _lock:
            _observe_process_exit_locked()
            if not _auto_start:
                raise _RestartDisabled(
                    "Enable the internal llama.cpp runtime before restarting it."
                )
            return _runtime_status_locked()

    try:
        with _lock:
            _observe_process_exit_locked()
            if not _auto_start:
                raise _RestartDisabled(
                    "Enable the internal llama.cpp runtime before restarting it."
                )
            if not _stop_locked():
                return _runtime_status_locked()
            _start_locked(repair_model_dir=False)
            return _runtime_status_locked()
    finally:
        _restart_lock.release()


def _is_local_request(request: Any) -> bool:
    if any(
        name.lower() == "forwarded"
        or name.lower().startswith("x-forwarded-")
        or name.lower() in {"x-real-ip", "x-client-ip"}
        for name in request.headers
    ):
        return False
    remote = request.remote
    if not remote:
        return False
    try:
        address = ipaddress.ip_address(remote)
    except ValueError:
        return False
    mapped_address = getattr(address, "ipv4_mapped", None)
    if not (address.is_loopback or (mapped_address and mapped_address.is_loopback)):
        return False
    host = urlsplit(f"//{request.host}").hostname
    if not host:
        return False
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


async def runtime_status_endpoint(request: Any):
    if not _is_local_request(request):
        return web.json_response(
            {"error": "Runtime status is only available locally."}, status=403
        )
    return web.json_response(await asyncio.to_thread(get_runtime_status))


async def update_runtime_endpoint(request: Any):
    if not _is_local_request(request):
        return web.json_response(
            {"error": "This setting can only be changed locally."}, status=403
        )
    try:
        data = await request.json()
    except Exception:
        return web.json_response(
            {"error": "Request body must be valid JSON."}, status=400
        )
    if (
        not isinstance(data, dict)
        or not data
        or data.keys() - {"auto_start", "ctx_size", "port", "model_dir"}
        or ("auto_start" in data and not isinstance(data["auto_start"], bool))
        or ("ctx_size" in data and not _valid_ctx_size(data["ctx_size"]))
        or ("port" in data and not _valid_port(data["port"]))
        or ("model_dir" in data and not isinstance(data["model_dir"], str))
    ):
        return web.json_response(
            {
                "error": (
                    "Provide auto_start as a boolean, ctx_size as an integer "
                    f"from {_MIN_CTX_SIZE} to {_MAX_CTX_SIZE}, port as an "
                    f"integer from {_MIN_PORT} to {_MAX_PORT}, and/or model_dir "
                    "as an available LLM directory."
                )
            },
            status=400,
        )
    try:
        status = await asyncio.to_thread(
            update_runtime_settings,
            auto_start=data.get("auto_start"),
            ctx_size=data.get("ctx_size"),
            port=data.get("port"),
            model_dir=data.get("model_dir"),
        )
        return web.json_response(status)
    except ValueError as exc:
        return web.json_response({"error": str(exc)}, status=400)
    except (OSError, RuntimeError) as exc:
        _logger.exception("Could not save internal llama.cpp settings.")
        return web.json_response({"error": str(exc)}, status=500)


async def restart_runtime_endpoint(request: Any):
    if not _is_local_request(request):
        return web.json_response(
            {"error": "This setting can only be changed locally."}, status=403
        )
    try:
        status = await asyncio.to_thread(restart_runtime)
        return web.json_response(status)
    except _RestartDisabled as exc:
        return web.json_response({"error": str(exc)}, status=409)
    except (OSError, RuntimeError) as exc:
        _logger.exception("Could not restart the internal llama.cpp runtime.")
        return web.json_response({"error": str(exc)}, status=500)


async def download_runtime_endpoint(request: Any):
    if not _is_local_request(request):
        return web.json_response(
            {"error": "This setting can only be changed locally."}, status=403
        )
    try:
        result = await asyncio.to_thread(_start_runtime_download)
        status = await asyncio.to_thread(get_runtime_status)
        return web.json_response(status, status=202 if result == "started" else 200)
    except (RuntimeError, ValueError) as exc:
        return web.json_response({"error": str(exc)}, status=409)
    except OSError as exc:
        _logger.exception("Could not start the llama.cpp download.")
        return web.json_response({"error": str(exc)}, status=500)


def register_runtime_routes() -> None:
    global _routes_registered
    if _routes_registered:
        return
    from server import PromptServer

    PromptServer.instance.routes.get(RUNTIME_ROUTE)(runtime_status_endpoint)
    PromptServer.instance.routes.post(RUNTIME_ROUTE)(update_runtime_endpoint)
    PromptServer.instance.routes.post(RUNTIME_RESTART_ROUTE)(restart_runtime_endpoint)
    PromptServer.instance.routes.post(RUNTIME_DOWNLOAD_ROUTE)(download_runtime_endpoint)
    _routes_registered = True


def _stop_at_exit() -> None:
    with _lock:
        _stop_locked()


atexit.register(_stop_at_exit)


__all__ = [
    "RUNTIME_ROUTE",
    "RUNTIME_RESTART_ROUTE",
    "RUNTIME_DOWNLOAD_ROUTE",
    "OwnedLlamaServer",
    "download_runtime_endpoint",
    "get_runtime_status",
    "initialize_llama_cpp_runtime",
    "register_runtime_routes",
    "restart_runtime",
    "restart_runtime_endpoint",
    "runtime_status_endpoint",
    "start_owned_llama_server",
    "update_runtime_endpoint",
    "update_runtime_settings",
]
