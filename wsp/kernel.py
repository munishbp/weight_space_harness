"""Persistent kernel. It loads the student model one time and runs teacher code against it.

The teacher sends Python source over a Unix socket. The kernel runs the source in one
namespace that persists between requests, so a tensor that the teacher computes stays in memory.

Start:  uv run python -m wsp.kernel --model Qwen/Qwen3.5-4B --run runs/<name>
"""

import argparse
import contextlib
import gc
import io
import json
import os
import socketserver
import time
import traceback
from pathlib import Path

import torch

from wsp import tools

SOCKET_PATH = Path(os.environ.get("WSP_SOCKET", "/tmp/wsp-kernel.sock"))
# Stop the kernel with `kill $(cat /tmp/wsp-kernel.pid)`.
PID_PATH = SOCKET_PATH.with_suffix(".pid")


class Handler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        source = self.rfile.read().decode("utf-8")
        # The run directory keeps every piece of teacher code and its output, in order.
        # This is the record of how the teacher reached the edit.
        index = self.server.next_index
        self.server.next_index += 1
        code_dir = tools.RUN_DIR / "code"
        (code_dir / f"{index:03d}.py").write_text(source)
        started = time.time()
        out = io.StringIO()
        ok = True
        # A defect in teacher code is an expected failure. The kernel returns the traceback
        # as text and stays alive, because a restart costs a full model load.
        try:
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
                exec(compile(source, "<teacher>", "exec"), self.server.namespace)
        except BaseException:  # noqa: BLE001 - SystemExit from teacher code must not stop the kernel
            ok = False
            out.write(traceback.format_exc())
        if not ok:
            # A failed run leaves large tensors alive in the traceback frames. The frames are
            # free only after the except block ends, so the cleanup must happen here.
            gc.collect()
            torch.cuda.empty_cache()
        seconds = round(time.time() - started, 2)
        (code_dir / f"{index:03d}.out").write_text(out.getvalue())
        tools.log("python", index=index, ok=ok, seconds=seconds)
        reply = {"ok": ok, "seconds": seconds, "output": out.getvalue()}
        self.wfile.write(json.dumps(reply).encode("utf-8"))


class Server(socketserver.UnixStreamServer):
    namespace: dict
    next_index: int


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", default=None, help="Commit hash of the model repository.")
    parser.add_argument("--run", required=True, help="Directory for the ledger and the edit files.")
    args = parser.parse_args()

    # The method forbids gradient descent. The kernel disables autograd for the whole process.
    torch.set_grad_enabled(False)

    tools.start_run(Path(args.run))
    started = time.time()
    tools.load_model(args.model, args.revision)
    tools.log("load_model", model=args.model, seconds=round(time.time() - started, 2))

    if SOCKET_PATH.exists():
        SOCKET_PATH.unlink()
    server = Server(str(SOCKET_PATH), Handler)
    # The socket runs arbitrary code. Only the owner may connect.
    SOCKET_PATH.chmod(0o600)
    code_dir = tools.RUN_DIR / "code"
    code_dir.mkdir(exist_ok=True)
    # A restarted kernel continues the numbering of the same run.
    server.next_index = len(list(code_dir.glob("*.py"))) + 1
    server.namespace = {"torch": torch, "tools": tools, "model": tools.MODEL, "processor": tools.PROCESSOR}
    PID_PATH.write_text(str(os.getpid()))
    print(f"kernel ready on {SOCKET_PATH}", flush=True)
    try:
        server.serve_forever()
    finally:
        SOCKET_PATH.unlink(missing_ok=True)
        PID_PATH.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
