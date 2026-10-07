import asyncio
import io
import json
import subprocess
import zipfile

import pytest

import backend.llama_cpp.llama_cpp_runtime as runtime


class FakeProcess:
    stdin = None

    def __init__(self, *, fail_wait=False):
        self.fail_wait = fail_wait
        self.waited = False

    def poll(self):
        return None

    def wait(self, timeout):
        if self.fail_wait:
            raise subprocess.TimeoutExpired("llama-supervisor", timeout)
        self.waited = True
        return 0

    def close(self, graceful_timeout=8):
        self.wait(timeout=graceful_timeout)


class FreePortProbe:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def bind(self, _address):
        return None


def set_status_dependencies(monkeypatch, model_dir):
    monkeypatch.setattr(
        runtime, "_model_directory_options", lambda: ([model_dir], model_dir)
    )
    monkeypatch.setattr(runtime, "_llama_executable", lambda: "llama")
    monkeypatch.setattr(runtime, "_probe_llama_version", lambda _path: "test")


def test_restart_retries_failed_daemon_with_saved_settings(monkeypatch):
    model_dir = "C:/models/LLM"
    set_status_dependencies(monkeypatch, model_dir)
    old_process = FakeProcess()
    monkeypatch.setattr(runtime, "_auto_start", True)
    monkeypatch.setattr(runtime, "_ctx_size", 8192)
    monkeypatch.setattr(runtime, "_port", 18600)
    monkeypatch.setattr(runtime, "_model_dir", model_dir)
    monkeypatch.setattr(runtime, "_process", old_process)
    monkeypatch.setattr(runtime, "_active_port", None)
    monkeypatch.setattr(runtime, "_state", "failed")
    monkeypatch.setattr(runtime, "_start_attempted", True)
    monkeypatch.setattr(runtime, "_last_error", "previous startup failure")
    monkeypatch.setattr(
        runtime,
        "_save_runtime_settings",
        lambda *_args: pytest.fail("restart must not save settings"),
    )
    events = []

    def start(*, repair_model_dir):
        assert runtime._start_attempted is False
        assert repair_model_dir is False
        events.append(
            (
                "start",
                runtime._ctx_size,
                runtime._port,
                runtime._model_dir,
                repair_model_dir,
            )
        )
        monkeypatch.setattr(runtime, "_process", FakeProcess())
        monkeypatch.setattr(runtime, "_active_port", runtime._port)
        monkeypatch.setattr(runtime, "_state", "running")
        monkeypatch.setattr(runtime, "_last_error", None)
        monkeypatch.setattr(runtime, "_start_attempted", True)

    real_stop = runtime._stop_locked

    def stop():
        events.append(("stop",))
        return real_stop()

    monkeypatch.setattr(runtime, "_stop_locked", stop)
    monkeypatch.setattr(runtime, "_start_locked", start)

    status = runtime.restart_runtime()

    assert old_process.waited
    assert events == [
        ("stop",),
        ("start", 8192, 18600, model_dir, False),
    ]
    assert (
        runtime._auto_start,
        runtime._ctx_size,
        runtime._port,
        runtime._model_dir,
    ) == (
        True,
        8192,
        18600,
        model_dir,
    )
    assert status["state"] == "running"
    assert status["active_port"] == 18600


def test_restart_does_not_repair_or_save_an_unavailable_model_directory(monkeypatch):
    missing_model_dir = "C:/models/removed"
    default_model_dir = "C:/models/LLM"
    monkeypatch.setattr(
        runtime,
        "_model_directory_options",
        lambda: ([default_model_dir], default_model_dir),
    )
    monkeypatch.setattr(runtime, "_llama_executable", lambda: "llama")
    monkeypatch.setattr(runtime, "_probe_llama_version", lambda _path: "test")
    monkeypatch.setattr(runtime, "_auto_start", True)
    monkeypatch.setattr(runtime, "_ctx_size", 8192)
    monkeypatch.setattr(runtime, "_port", 18600)
    monkeypatch.setattr(runtime, "_model_dir", missing_model_dir)
    monkeypatch.setattr(runtime, "_process", None)
    monkeypatch.setattr(runtime, "_start_attempted", True)
    monkeypatch.setattr(runtime, "_state", "failed")
    monkeypatch.setattr(
        runtime,
        "_save_runtime_settings",
        lambda *_args: pytest.fail("restart must not save settings"),
    )

    monkeypatch.setattr(runtime.socket, "socket", lambda *_args: FreePortProbe())

    status = runtime.restart_runtime()

    assert status["state"] == "failed"
    assert "directory is unavailable" in status["error"]
    assert runtime._model_dir == missing_model_dir


def test_restart_reports_running_only_after_health_check(monkeypatch, tmp_path):
    model_dir = str(tmp_path)
    set_status_dependencies(monkeypatch, model_dir)
    monkeypatch.setattr(runtime, "_auto_start", True)
    monkeypatch.setattr(runtime, "_ctx_size", 8192)
    monkeypatch.setattr(runtime, "_port", 18600)
    monkeypatch.setattr(runtime, "_model_dir", model_dir)
    monkeypatch.setattr(runtime, "_process", None)
    monkeypatch.setattr(runtime, "_start_attempted", True)
    monkeypatch.setattr(runtime, "_state", "failed")
    monkeypatch.setattr(runtime.socket, "socket", lambda *_args: FreePortProbe())
    monkeypatch.setattr(
        runtime, "_save_runtime_settings", lambda *_args: pytest.fail("must not save")
    )
    commands = []

    def start_supervisor(command, **_kwargs):
        commands.append(command)
        return FakeProcess()

    monkeypatch.setattr(runtime.subprocess, "Popen", start_supervisor)
    health_checks = []
    monkeypatch.setattr(
        runtime,
        "_server_ready",
        lambda port: health_checks.append(port) or port == 18600,
    )

    status = runtime.restart_runtime()

    assert health_checks == [18600]
    assert "--ctx-size" in commands[0]
    assert commands[0][commands[0].index("--ctx-size") + 1] == "8192"
    assert status["state"] == "running"
    assert status["active_port"] == 18600


def test_restart_reports_failed_when_health_check_never_succeeds(monkeypatch, tmp_path):
    model_dir = str(tmp_path)
    set_status_dependencies(monkeypatch, model_dir)
    monkeypatch.setattr(runtime, "_auto_start", True)
    monkeypatch.setattr(runtime, "_port", 18600)
    monkeypatch.setattr(runtime, "_model_dir", model_dir)
    monkeypatch.setattr(runtime, "_process", None)
    monkeypatch.setattr(runtime, "_start_attempted", True)
    monkeypatch.setattr(runtime, "_state", "failed")
    monkeypatch.setattr(runtime.socket, "socket", lambda *_args: FreePortProbe())
    monkeypatch.setattr(runtime, "_STARTUP_TIMEOUT_SECONDS", 0.02)
    monkeypatch.setattr(runtime, "_HEALTH_POLL_INTERVAL_SECONDS", 0.005)
    monkeypatch.setattr(runtime, "_server_ready", lambda _port: False)
    monkeypatch.setattr(
        runtime, "_save_runtime_settings", lambda *_args: pytest.fail("must not save")
    )
    supervisor = FakeProcess()
    monkeypatch.setattr(
        runtime.subprocess, "Popen", lambda _command, **_kwargs: supervisor
    )

    status = runtime.restart_runtime()

    assert supervisor.waited
    assert status["state"] == "failed"
    assert status["active_port"] is None
    assert "did not become ready" in status["error"]


def test_restart_does_not_start_if_supervisor_will_not_stop(monkeypatch):
    model_dir = "C:/models/LLM"
    set_status_dependencies(monkeypatch, model_dir)
    monkeypatch.setattr(runtime, "_auto_start", True)
    monkeypatch.setattr(runtime, "_ctx_size", 4096)
    monkeypatch.setattr(runtime, "_port", 18582)
    monkeypatch.setattr(runtime, "_model_dir", model_dir)
    monkeypatch.setattr(runtime, "_process", FakeProcess(fail_wait=True))
    monkeypatch.setattr(runtime, "_state", "running")
    monkeypatch.setattr(runtime, "_start_attempted", True)
    monkeypatch.setattr(runtime, "_SUPERVISOR_SHUTDOWN_TIMEOUT_SECONDS", 1)
    monkeypatch.setattr(
        runtime,
        "_start_locked",
        lambda **_kwargs: pytest.fail("must not start before the old supervisor stops"),
    )

    status = runtime.restart_runtime()

    assert status["state"] == "failed"
    assert "shutdown timeout" in status["error"]


class LocalRequest:
    headers = {}
    remote = "127.0.0.1"
    host = "127.0.0.1"


def test_restart_endpoint_requires_local_request_and_enabled_runtime(monkeypatch):
    monkeypatch.setattr(runtime, "_auto_start", False)

    remote_request = LocalRequest()
    remote_request.remote = "192.168.1.20"
    remote_response = asyncio.run(runtime.restart_runtime_endpoint(remote_request))
    assert remote_response.status == 403

    local_response = asyncio.run(runtime.restart_runtime_endpoint(LocalRequest()))
    assert local_response.status == 409
    assert "Enable" in json.loads(local_response.text)["error"]


def test_release_asset_selection_uses_fixed_platform_backend_mapping():
    windows_cuda = runtime._select_release_asset("win32", "AMD64", "cuda")
    assert windows_cuda["directory"] == "windows-x64-cuda"
    assert windows_cuda["assets"][0][0] == "llama-b11429-bin-win-cuda-13.4-x64.zip"
    assert windows_cuda["assets"][1][0] == "cudart-llama-bin-win-cuda-13.4-x64.zip"

    macos = runtime._select_release_asset("darwin", "arm64", "cuda")
    assert macos["assets"][0][0] == "llama-b11429-bin-macos-arm64.tar.gz"

    with pytest.raises(ValueError, match="No b11429 llama.cpp build"):
        runtime._select_release_asset("linux", "aarch64", "rocm")


def test_llama_executable_prefers_path_over_internal_install(monkeypatch):
    monkeypatch.setattr(runtime, "_path_llama_executable", lambda: "path/llama")
    monkeypatch.setattr(
        runtime,
        "_installed_llama_executable",
        lambda _selection: pytest.fail("PATH executable must win"),
    )

    assert runtime._llama_executable() == "path/llama"


def test_runtime_status_recognizes_persisted_install_after_restart(
    monkeypatch, tmp_path
):
    selection = runtime._select_release_asset("win32", "AMD64", "cpu")
    install_dir = (
        tmp_path / "artifacts" / runtime._LLAMA_RELEASE / selection["directory"]
    )
    install_dir.mkdir(parents=True)
    executable = install_dir / "llama.exe"
    executable.write_bytes(b"installed")
    executable.chmod(0o755)
    (install_dir / runtime._INSTALL_MARKER).write_text(
        json.dumps(
            {
                "release": runtime._LLAMA_RELEASE,
                "target": selection["directory"],
                "executable": "llama.exe",
            }
        ),
        encoding="utf-8",
    )
    model_dir = str(tmp_path / "models")
    monkeypatch.setattr(runtime, "_current_release_selection", lambda: selection)
    monkeypatch.setattr(runtime, "_artifacts_directory", lambda: tmp_path / "artifacts")
    monkeypatch.setattr(runtime, "_path_llama_executable", lambda: None)
    monkeypatch.setattr(
        runtime, "_model_directory_options", lambda: ([model_dir], model_dir)
    )
    monkeypatch.setattr(runtime, "_probe_llama_version", lambda _path: "b11429")
    monkeypatch.setattr(runtime, "_download_state", "error")
    monkeypatch.setattr(runtime, "_download_error", "stale download error")
    monkeypatch.setattr(runtime, "_process", None)

    status = runtime.get_runtime_status()

    assert status["llama_available"] is True
    assert status["llama_source"] == "internal"
    assert status["llama_executable"] == str(executable.resolve())
    assert status["download_state"] == "installed"
    assert status["download_error"] is None
    assert runtime._start_runtime_download() == "installed"
    assert runtime._download_state == "installed"


def test_download_asset_rejects_sha256_mismatch(monkeypatch, tmp_path):
    monkeypatch.setattr(runtime, "_download_bytes_received", 0)
    monkeypatch.setattr(runtime, "_download_bytes_total", 0)

    class FakeResponse:
        headers = {"Content-Length": "4"}

        def __init__(self):
            self._stream = io.BytesIO(b"fake")

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def geturl(self):
            return "https://github.com/ggml-org/llama.cpp/releases/download/b11429/test.zip"

        def read(self, size):
            return self._stream.read(size)

    class FakeOpener:
        def open(self, _request, timeout):
            assert timeout == runtime._DOWNLOAD_TIMEOUT_SECONDS
            return FakeResponse()

    monkeypatch.setattr(runtime, "build_opener", lambda *_args: FakeOpener())

    with pytest.raises(RuntimeError, match="SHA-256 verification failed"):
        runtime._download_asset("test.zip", "0" * 64, tmp_path / "test.zip")
    assert not (tmp_path / "test.zip").exists()


def test_archive_extraction_rejects_path_traversal(tmp_path):
    archive_path = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("../escape.txt", "unsafe")

    with pytest.raises(RuntimeError, match="unsafe path"):
        runtime._extract_archive(archive_path, tmp_path / "extracted")


def test_download_start_rejects_a_second_active_download(monkeypatch):
    monkeypatch.setattr(runtime, "_download_state", "downloading")
    monkeypatch.setattr(
        runtime,
        "_current_release_selection",
        lambda: pytest.fail("active download must be checked first"),
    )

    with pytest.raises(RuntimeError, match="already in progress"):
        runtime._start_runtime_download()


def test_download_endpoint_requires_local_request(monkeypatch):
    remote_request = LocalRequest()
    remote_request.remote = "192.168.1.20"

    response = asyncio.run(runtime.download_runtime_endpoint(remote_request))

    assert response.status == 403


def test_owned_server_startup_failure_closes_only_its_supervisor(monkeypatch):
    ephemeral_port = 49173
    server_args = ["--model", "C:/models/test.gguf", "--ctx-size", "4096"]
    daemon_globals = (
        runtime._auto_start,
        runtime._ctx_size,
        runtime._port,
        runtime._model_dir,
        runtime._active_port,
        runtime._state,
        runtime._start_attempted,
        runtime._process,
        runtime._last_error,
    )
    monkeypatch.setattr(runtime, "_ephemeral_loopback_port", lambda: ephemeral_port)
    monkeypatch.setattr(runtime, "_STARTUP_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(runtime, "_HEALTH_POLL_INTERVAL_SECONDS", 0.001)
    health_checks = []
    monkeypatch.setattr(
        runtime,
        "_server_ready",
        lambda port: health_checks.append(port) or False,
    )
    supervisor = FakeProcess()

    class FakePipe:
        closed = False

        def close(self):
            self.closed = True

    supervisor.stdin = FakePipe()
    launches = []

    def popen(command, **kwargs):
        launches.append((command, kwargs))
        return supervisor

    monkeypatch.setattr(runtime.subprocess, "Popen", popen)

    with pytest.raises(RuntimeError, match="did not become ready"):
        runtime.start_owned_llama_server(
            "C:/llama/llama.exe", server_args, internal=True
        )

    command, options = launches[0]
    assert command[0] == runtime.sys.executable
    assert command[1].endswith("llama_cpp_supervisor.py")
    assert command[2] == "--"
    assert command[3:7] == [
        "C:/llama/llama.exe",
        "server",
        "--host",
        "127.0.0.1",
    ]
    assert command[7:9] == ["--port", str(ephemeral_port)]
    assert command[9:] == [
        "--cors-origins",
        "localhost",
        "--models-max",
        "1",
        "--parallel",
        "1",
        *server_args,
    ]
    assert options["stdin"] == subprocess.PIPE
    assert options["close_fds"] is True
    assert health_checks and set(health_checks) == {ephemeral_port}
    assert supervisor.stdin.closed
    assert supervisor.waited
    assert daemon_globals == (
        runtime._auto_start,
        runtime._ctx_size,
        runtime._port,
        runtime._model_dir,
        runtime._active_port,
        runtime._state,
        runtime._start_attempted,
        runtime._process,
        runtime._last_error,
    )
