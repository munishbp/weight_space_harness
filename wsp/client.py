"""Send teacher code to the kernel and print the output.

Use:  uv run python -m wsp.client path/to/code.py
      echo "print(model)" | uv run python -m wsp.client -
"""

import json
import socket
import sys
from pathlib import Path

from wsp.kernel import SOCKET_PATH


def run(source: str) -> dict:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        try:
            sock.connect(str(SOCKET_PATH))
        except (FileNotFoundError, ConnectionRefusedError) as error:
            raise SystemExit(
                f"The kernel is not reachable at {SOCKET_PATH} ({error}). "
                "No code ran. Start the kernel with `uv run python -m wsp.kernel`."
            ) from error
        sock.sendall(source.encode("utf-8"))
        sock.shutdown(socket.SHUT_WR)
        chunks = []
        while chunk := sock.recv(1 << 16):
            chunks.append(chunk)
    raw = b"".join(chunks)
    if not raw:
        # The kernel died during the request, for example from an out-of-memory kill.
        raise SystemExit(
            "The kernel closed the connection with no reply. The code may have run in part. "
            "Check the kernel log, restart the kernel, and apply the saved edit again."
        )
    return json.loads(raw)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Use: python -m wsp.client <file.py | ->")
    source = sys.stdin.read() if sys.argv[1] == "-" else Path(sys.argv[1]).read_text()
    reply = run(source)
    print(reply["output"], end="")
    print(f"\n[{'ok' if reply['ok'] else 'ERROR'} in {reply['seconds']} s]")
    sys.exit(0 if reply["ok"] else 1)


if __name__ == "__main__":
    main()
