"""A stand-in for the Docker client, for the sandbox tests: ``python fake_docker.py STATE_DIR <docker args>``.

It understands exactly the calls ``synthic.sandbox`` makes. The "container" is
the host: a command runs with the engine's host shell in the host folder its working
directory is mounted from, and a container path outside every mount lives
under ``STATE_DIR/fs``. Every call is appended to ``STATE_DIR/calls.jsonl``
with the values of the variables it was asked to pass, so a test can check
what reached the container and that no value appeared on a command line.
"""

from __future__ import annotations

import json
import os
import posixpath
import subprocess
import sys
from pathlib import Path

from synthic.proc import shell_argv
from synthic.sandbox import _ALIVE, _KILL, _READ, _WRAPPER, _WRITE
from synthic.types import EnvironmentSettings

STATE = Path(sys.argv[1])
ARGS = sys.argv[2:]


def state() -> dict:
    path = STATE / "state.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def save(data: dict) -> None:
    (STATE / "state.json").write_text(json.dumps(data), encoding="utf-8")


def log(entry: dict) -> None:
    """One unbuffered write per call: several fakes run at once, and a buffered writer splits a long line
    into pieces that another process can write between."""
    with (STATE / "calls.jsonl").open("ab", buffering=0) as sink:
        sink.write((json.dumps(entry) + "\n").encode())


def host(path: str) -> Path:
    for target, source, _ in sorted(state().get("mounts", []), key=lambda m: len(m[0]), reverse=True):
        if path == target or path.startswith(target.rstrip("/") + "/"):
            return Path(source) / path[len(target) :].lstrip("/")
    return STATE / "fs" / path.lstrip("/")


def run() -> int:
    mounts, names, network, name = [], [], "default", ""
    for index, arg in enumerate(ARGS):
        following = ARGS[index + 1] if index + 1 < len(ARGS) else ""
        if arg == "--mount":
            fields = dict(f.strip('"').split("=", 1) for f in following.split(",") if "=" in f)
            mounts.append([fields["target"], fields["source"], "readonly" in following.split(",")])
        elif arg == "-e":
            names.append(following)
        elif arg == "--network":
            network = following
        elif arg == "--name":
            name = following
    if (STATE / "fail_run").exists():
        print("Unable to find image 'missing:latest' locally", flush=True)
        return 125
    save({"running": True, "mounts": mounts, "network": network, "name": name})
    log({"cmd": "run", "argv": ARGS, "env": {n: os.environ.get(n) for n in names}})
    sys.stdin.buffer.read()
    data = state()
    data["running"] = False
    save(data)
    return 0


def execute() -> int:
    rest = ARGS[1:]
    workdir, names, data = "/", [], None
    while rest and rest[0].startswith("-"):
        flag = rest.pop(0)
        if flag == "-w":
            workdir = rest.pop(0)
        elif flag == "-e":
            names.append(rest.pop(0))
        elif flag == "-i":
            data = True
    _name, _sh, _c, script, _zero, *params = rest
    log({"cmd": "exec", "script": script, "params": params, "workdir": workdir, "env": {n: os.environ.get(n) for n in names}})
    if script == _WRAPPER:
        command = params[-1]
        return subprocess.run([*shell_argv(EnvironmentSettings()), command], cwd=host(workdir)).returncode
    if script in (_KILL, _ALIVE):
        return 0 if script == _KILL else 1
    target = host(posixpath.normpath(params[0]))
    if script == _READ:
        if not target.exists():
            return 3
        if target.is_dir():
            return 4
        sys.stdout.buffer.write(target.read_bytes())
        return 0
    if script == _WRITE:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(sys.stdin.buffer.read() if data else b"")
        return 0
    if script == '[ -e "$1" ]':
        return 0 if target.exists() else 1
    if script == '[ -d "$1" ]':
        return 0 if target.is_dir() else 1
    return 2


def main() -> int:
    match ARGS[0]:
        case "run":
            return run()
        case "inspect":
            current = state()
            print("true" if current.get("running") and not current.get("removed") else "false")
            return 0
        case "exec":
            return execute()
        case "ps":
            log({"cmd": "ps", "argv": ARGS})
            others = STATE / "others.txt"
            if others.is_file():
                sys.stdout.write(others.read_text(encoding="utf-8"))
            return 0
        case "rm":
            log({"cmd": "rm", "argv": ARGS})
            current = state()
            if current and ARGS[-1] == current.get("name"):
                current["removed"] = True
                save(current)
            return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
