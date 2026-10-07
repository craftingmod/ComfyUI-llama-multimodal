from __future__ import annotations

import logging
import os
import re
import signal
import subprocess
import sys
from threading import Event, Thread
from typing import BinaryIO

_SHUTDOWN_TIMEOUT_SECONDS = 5
_logger = logging.getLogger("llama-supervisor")
_SUPPRESSED_STDOUT_LINE = re.compile(rb"^\[\s*\d+\]\s*$")


def _request_stop(stop_requested: Event, _frame: object) -> None:
    stop_requested.set()


def _watch_owner(owner_pipe: BinaryIO, stop_requested: Event) -> None:
    try:
        owner_pipe.read()
    finally:
        stop_requested.set()


def _forward_server_stdout(server_stdout: BinaryIO) -> None:
    for line in server_stdout:
        if _SUPPRESSED_STDOUT_LINE.fullmatch(line.rstrip(b"\r\n")):
            continue
        sys.stdout.buffer.write(line)
        sys.stdout.buffer.flush()


def _stop_server(server: subprocess.Popen[bytes]) -> None:
    if server.poll() is not None:
        server.wait()
        return

    try:
        if os.name == "nt" and hasattr(signal, "CTRL_BREAK_EVENT"):
            server.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            server.terminate()
    except (OSError, ValueError):
        if server.poll() is None:
            try:
                server.terminate()
            except OSError:
                try:
                    server.kill()
                except OSError:
                    pass

    try:
        server.wait(timeout=_SHUTDOWN_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        _logger.warning("llama-server did not stop gracefully; forcing termination")
        try:
            server.kill()
        except OSError:
            pass
        server.wait()


def _supervise(command: list[str], owner_pipe: BinaryIO) -> int:
    stop_requested = Event()
    creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    try:
        server = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            close_fds=True,
            creationflags=creationflags,
        )
    except OSError:
        _logger.exception("Could not start llama-server")
        return 1

    _logger.info("llama-server started (pid %s)", server.pid)
    assert server.stdout is not None
    stdout_thread = Thread(
        target=_forward_server_stdout, args=(server.stdout,), daemon=True
    )
    stdout_thread.start()
    Thread(target=_watch_owner, args=(owner_pipe, stop_requested), daemon=True).start()
    handlers = {}
    stop_signals = [signal.SIGINT, signal.SIGTERM]
    if hasattr(signal, "SIGBREAK"):
        stop_signals.append(signal.SIGBREAK)
    for signum in stop_signals:
        handlers[signum] = signal.signal(
            signum, lambda _signum, frame: _request_stop(stop_requested, frame)
        )

    try:
        while True:
            exit_code = server.poll()
            if exit_code is not None:
                _logger.info("llama-server exited with code %s", exit_code)
                return exit_code
            if stop_requested.wait(0.1):
                _logger.info("shutdown requested; stopping llama-server")
                return 0
    finally:
        _stop_server(server)
        stdout_thread.join()
        server.stdout.close()
        for signum, handler in handlers.items():
            signal.signal(signum, handler)
        _logger.info("supervisor exited")


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="[llama-supervisor] %(message)s",
        stream=sys.stderr,
    )
    args = sys.argv[1:] if argv is None else argv
    if len(args) < 3 or args[0] != "--":
        _logger.error("expected '--' followed by the llama-server command")
        return 2
    return _supervise(args[1:], sys.stdin.buffer)


if __name__ == "__main__":
    raise SystemExit(main())
