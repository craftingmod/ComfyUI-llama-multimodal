import io
import os
import subprocess
import sys
import time
from pathlib import Path

from backend.llama_cpp.llama_cpp_supervisor import _forward_server_stdout


def test_server_stdout_suppresses_bracketed_numeric_lines(monkeypatch):
    output = io.BytesIO()
    monkeypatch.setattr(sys, "stdout", io.TextIOWrapper(output))

    _forward_server_stdout(io.BytesIO(b"[ 12]\nvisible\n[\t3]\r\n[4] trailing\nlast"))

    assert output.getvalue() == b"visible\n[4] trailing\nlast"


def test_owner_pipe_eof_stops_the_owned_server(tmp_path):
    pid_path = tmp_path / "server.pid"
    child_code = (
        "import os, pathlib, time; "
        f"pathlib.Path({str(pid_path)!r}).write_text(str(os.getpid())); "
        "time.sleep(60)"
    )
    root = Path(__file__).resolve().parents[2]
    supervisor_path = root / "backend" / "llama_cpp" / "llama_cpp_supervisor.py"
    supervisor = subprocess.Popen(
        [
            sys.executable,
            str(supervisor_path),
            "--",
            sys.executable,
            "-c",
            child_code,
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        close_fds=True,
    )
    try:
        assert supervisor.stdin is not None
        deadline = time.monotonic() + 5
        server_pid = None
        while server_pid is None and supervisor.poll() is None:
            if time.monotonic() >= deadline:
                raise AssertionError("fake llama-server did not start")
            try:
                server_pid = int(pid_path.read_text(encoding="utf-8"))
            except (FileNotFoundError, ValueError):
                pass
            time.sleep(0.05)
        supervisor.stdin.close()
        assert supervisor.wait(timeout=10) == 0
        assert server_pid is not None
        try:
            os.kill(server_pid, 0)
        except OSError:
            pass
        else:
            raise AssertionError("llama-server child survived its owner")
    finally:
        if supervisor.poll() is None:
            if supervisor.stdin is not None and not supervisor.stdin.closed:
                supervisor.stdin.close()
            try:
                supervisor.wait(timeout=10)
            except subprocess.TimeoutExpired:
                supervisor.kill()
                supervisor.wait()
