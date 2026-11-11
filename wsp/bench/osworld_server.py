"""The OSWorld side of the bridge: one process that holds the virtual machine and answers requests.

The OSWorld environment package (`desktop-env` on PyPI) pins old versions of torch and transformers,
so it lives in its own environment (`.venv-osworld`). The harness talks to this process through a
Unix socket. Run it with the Python of that environment, from the directory that holds the VM data:

    .venv-osworld/bin/python wsp/bench/osworld_server.py --socket /tmp/wsp-osworld.sock

Add `--user-network` on a host that has no `ip_tables` kernel module (the VM boots, but no port reaches it).

This file imports only the standard library and `desktop_env`. It must not import `wsp`.

Protocol: one JSON object per line in each direction.

    {"op": "reset", "task": {...}}       -> {"instruction": "...", "screenshot": "<base64 PNG>"}
    {"op": "step", "code": "...", "pause": 2}  -> {"screenshot": "<base64 PNG>", "done": false}
    {"op": "evaluate"}                   -> {"score": 1.0}
    {"op": "close"}                      -> {"closed": true}

A request that fails gives {"error": "...", "detail": "..."} and the process stays up. The caller
decides what to do: a `reset` is safe to send again, a `step` is not.
"""

import argparse
import base64
import json
import os
import socket
import traceback


def handle(env, request: dict) -> dict:
    """Answer one request with the environment. `env` has the methods of `desktop_env.DesktopEnv`."""
    op = request.get("op")
    if op == "reset":
        obs = env.reset(task_config=request["task"])
        return {"instruction": obs["instruction"], "screenshot": base64.b64encode(obs["screenshot"]).decode()}
    if op == "step":
        obs, _, done, _ = env.step(request["code"], pause=request.get("pause", 2))
        return {"screenshot": base64.b64encode(obs["screenshot"]).decode(), "done": bool(done)}
    if op == "evaluate":
        return {"score": float(env.evaluate())}
    raise ValueError(f"Unknown op {op!r}. Use reset, step, evaluate, or close.")


def serve(env, path: str) -> None:
    """Accept one client at a time until a client sends `close`."""
    if os.path.exists(path):
        os.remove(path)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
        server.bind(path)
        server.listen(1)
        print(f"osworld server ready on {path}", flush=True)
        while True:
            connection, _ = server.accept()
            with connection, connection.makefile("rwb") as stream:
                for line in stream:
                    request = json.loads(line)
                    if request.get("op") == "close":
                        stream.write(b'{"closed": true}\n')
                        stream.flush()
                        return
                    try:
                        reply = handle(env, request)
                    except Exception as error:  # The caller needs the cause, and the VM must stay up for the next task.
                        reply = {"error": f"{type(error).__name__}: {error}", "detail": traceback.format_exc()[-2000:]}
                    stream.write(json.dumps(reply).encode() + b"\n")
                    stream.flush()


def use_user_network() -> None:
    """Make the Docker provider start the VM with user-mode networking and forward the three service ports.

    The default network of the VM container needs the `ip_tables` kernel module on the host. On a host
    without it the VM boots and no port reaches it. The package has this setup for another case
    (a Docker without Linux capabilities), so this function only turns it on.
    """
    from desktop_env.providers.docker.provider import DockerProvider

    start = DockerProvider.start_emulator

    def start_with_user_network(self, *args, **kwargs):
        self.environment = {**self.environment, "NETWORK": "user", "USER_PORTS": "5000,9222,8080"}
        return start(self, *args, **kwargs)

    DockerProvider.start_emulator = start_with_user_network


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", default="/tmp/wsp-osworld.sock")
    parser.add_argument("--provider", default="docker")
    parser.add_argument("--user-network", action="store_true", help="For a host without the ip_tables kernel module.")
    args = parser.parse_args()
    from desktop_env.desktop_env import DesktopEnv

    if args.user_network:
        use_user_network()

    # No accessibility tree: the student sees the screenshot only, as in the other benchmarks of the harness.
    env = DesktopEnv(provider_name=args.provider, action_space="pyautogui", headless=True, require_a11y_tree=False, os_type="Ubuntu")
    try:
        serve(env, args.socket)
    finally:
        env.close()
        if os.path.exists(args.socket):
            os.remove(args.socket)


if __name__ == "__main__":
    main()
