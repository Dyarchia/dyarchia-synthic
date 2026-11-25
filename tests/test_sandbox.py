"""The Docker sandbox: policy, path mapping, and whole runs against ``fake_docker.py``.

The last tests talk to a real Docker daemon and are skipped when none answers
or when the small public image they use is not already present.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from test_loop import done, go, make_policy, project, records, results

from synthic.journal import beat
from synthic.policy import PolicyError
from synthic.sandbox import _KILL, _WRAPPER, DockerSandbox
from synthic.testing import reply, tool_use
from synthic.tools.jobs import JobManager, wait_job
from synthic.types import DockerSettings, EnvironmentSettings

FAKE = Path(__file__).with_name("fake_docker.py")
IMAGE = "alpine:3.20"


def docker_task(state: Path, **docker: Any) -> dict[str, Any]:
    state.mkdir(parents=True, exist_ok=True)
    settings = {"command": [sys.executable, str(FAKE), str(state)], "image": IMAGE, **docker}
    return {"environment": {"mode": "docker", "docker": settings}, "memory": {"enabled": False}}


def calls(state: Path) -> list[dict[str, Any]]:
    path = state / "calls.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.is_file() else []


def test_defaults_keep_the_host_and_docker_needs_an_image(tmp_path: Path) -> None:
    policy = make_policy(tmp_path).policy
    assert policy.environment.mode == "in_place" and policy.environment.docker == DockerSettings()
    with pytest.raises(PolicyError) as error:
        make_policy(tmp_path, {"environment": {"mode": "docker"}})
    assert any("environment.docker.image" in p for p in error.value.problems)


def test_the_engine_key_never_reaches_the_container(tmp_path: Path) -> None:
    task = docker_task(tmp_path / "state", pass_env=["anthropic_api_key"])
    task["environment"]["env"] = {"ANTHROPIC_API_KEY": "x"}
    with pytest.raises(PolicyError) as error:
        make_policy(tmp_path, task)
    assert sum("ANTHROPIC_API_KEY" in p for p in error.value.problems) == 2
    with pytest.raises(PolicyError) as error:
        make_policy(tmp_path, docker_task(tmp_path / "state", pass_env=["synthic_anthropic_api_key"]))
    assert sum("SYNTHIC_ANTHROPIC_API_KEY" in p for p in error.value.problems) == 1
    with pytest.raises(PolicyError):
        make_policy(tmp_path, docker_task(tmp_path / "state", workdir="relative"))


def test_paths_map_to_mounts_and_everything_else_stays_inside(tmp_path: Path) -> None:
    root, ref = project(tmp_path), tmp_path / "ref"
    ref.mkdir()
    settings = EnvironmentSettings(mode="docker", docker=DockerSettings(image=IMAGE, mounts={"/ref": str(ref)}))
    sandbox = DockerSandbox(settings, root, tmp_path, "run")
    inside = sandbox.locate("src/a.py", root)
    assert inside.shown == "/workspace/src/a.py" and inside.host == (root / "src" / "a.py").resolve()
    assert sandbox.locate("/workspace/b.txt", root).host == (root / "b.txt").resolve()
    escaped = sandbox.locate("../etc/passwd", root)
    assert escaped.shown == "/etc/passwd" and escaped.host is None
    assert sandbox.locate("/tmp/x", root).host is None
    read_only = sandbox.locate("/ref/data.csv", root)
    assert read_only.readonly and read_only.host == (ref / "data.csv").resolve()
    with pytest.raises(PermissionError):
        asyncio.run(sandbox.write(read_only, b"x"))
    assert sandbox.shown(root) == "/workspace" and sandbox.inner(tmp_path) is None


def test_a_symlink_out_of_the_mount_is_followed_inside_the_container(tmp_path: Path) -> None:
    root = project(tmp_path)
    outside = tmp_path / "secret.txt"
    outside.write_text("host secret", encoding="utf-8")
    try:
        (root / "link.txt").symlink_to(outside)
    except OSError:
        pytest.skip("symlinks need extra rights on this system")
    sandbox = DockerSandbox(EnvironmentSettings(mode="docker", docker=DockerSettings(image=IMAGE)), root, tmp_path, "r")
    assert sandbox.locate("link.txt", root).host is None
    (root / "plain.txt").write_text("kept", encoding="utf-8")
    listed = asyncio.run(sandbox.files(sandbox.locate(".", root), 1000))
    assert [name for name, _ in listed] == ["plain.txt"]


def test_a_run_works_inside_the_container(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    state = tmp_path / "state"
    monkeypatch.setenv("SYNTHIC_TEST_PASS", "passed-value")
    task = docker_task(state, pass_env=["SYNTHIC_TEST_PASS"], network=False, mounts={"/ref": str(tmp_path)})
    task["environment"]["env"] = {"SYNTHIC_TEST_SET": "set-value"}
    runtime, result, provider = go(
        tmp_path,
        [
            reply(tool_use("sh", {"command": 'echo "$SYNTHIC_JOB" > token.txt'})),
            reply(tool_use("edit", {"path": "notes.txt", "old": "", "new": "inside\n"})),
            reply(tool_use("edit", {"path": "/tmp/scratch.txt", "old": "", "new": "only in the container\n"})),
            reply(tool_use("edit", {"path": "/tmp/scratch.txt", "old": "only", "new": "still only"})),
            done(),
        ],
        task,
    )
    assert result.status == "delivered"
    root = project(tmp_path)
    assert (root / "notes.txt").read_text(encoding="utf-8") == "inside\n"
    assert (state / "fs" / "tmp" / "scratch.txt").read_text(encoding="utf-8") == "still only in the container\n"
    assert not (root / "tmp").exists()
    assert "Created /tmp/scratch.txt." in results(provider.requests[3])[0]["content"]
    log = calls(state)
    started = next(c for c in log if c["cmd"] == "run")
    assert started["env"] == {"SYNTHIC_TEST_SET": "set-value", "SYNTHIC_TEST_PASS": "passed-value"}
    argv = started["argv"]
    assert ["--network", "none"] == argv[argv.index("--network") : argv.index("--network") + 2]
    assert all("passed-value" not in arg and "set-value" not in arg for c in log for arg in c.get("argv", []))
    assert any(arg.startswith("type=bind,source=") and arg.endswith("target=/ref,readonly") for arg in argv)
    command = next(c for c in log if c["cmd"] == "exec" and c["script"] == _WRAPPER)
    assert command["workdir"] == "/workspace" and command["params"][-1] == 'echo "$SYNTHIC_JOB" > token.txt'
    token = command["env"]["SYNTHIC_JOB"]
    assert (root / "token.txt").read_text(encoding="utf-8").strip() == token
    assert log[-1]["cmd"] == "rm" and log[-1]["argv"][-1] == f"synthic-{runtime.run_id}"
    system = "".join(b["text"] for b in provider.requests[0].system)
    assert "Working folder: /workspace" in system and "Docker container (image alpine:3.20)" in system
    assert "Read-only folders: /ref." in system and str(root) not in system
    start = records(runtime)[0]["data"]
    assert start["cwd"] == str(root)


def test_a_killed_job_is_killed_inside_the_container(tmp_path: Path) -> None:
    state = tmp_path / "state"
    sleeper = f'"{Path(sys.executable).as_posix()}" -c "import time; time.sleep(60)"'
    _, result, provider = go(
        tmp_path,
        [
            reply(tool_use("sh", {"command": sleeper, "background": True})),
            reply(tool_use("job", {"action": "kill", "job": "j1"})),
            done(),
        ],
        docker_task(state),
    )
    assert result.status == "delivered"
    assert "Killed" in results(provider.requests[2])[0]["content"]
    """The fake may be killed before it logs its own start, so the token is checked by its shape."""
    kills = [c["params"] for c in calls(state) if c["cmd"] == "exec" and c["script"] == _KILL]
    assert len(kills) == 1 and kills[0][0].endswith("-j1")


def test_orphans_of_dead_runs_are_removed_and_live_ones_kept(tmp_path: Path) -> None:
    state = tmp_path / "state"
    dead, live = tmp_path / "dead-run", tmp_path / "live-run"
    dead.mkdir()
    live.mkdir()
    beat(live)
    state.mkdir()
    (state / "others.txt").write_text(f"synthic-old\t{dead}\nsynthic-busy\t{live}\n", encoding="utf-8")
    go(tmp_path, [reply(tool_use("sh", {"command": "echo ok"})), done()], docker_task(state))
    removed = [c["argv"][-1] for c in calls(state) if c["cmd"] == "rm"]
    assert "synthic-old" in removed and "synthic-busy" not in removed


def test_a_container_that_cannot_start_is_a_tool_error(tmp_path: Path) -> None:
    state = tmp_path / "state"
    state.mkdir()
    (state / "fail_run").write_text("", encoding="utf-8")
    _, result, provider = go(tmp_path, [reply(tool_use("sh", {"command": "echo ok"})), done()], docker_task(state))
    block = results(provider.requests[1])[0]
    assert block["is_error"] and "the container did not start" in block["content"]
    assert "Unable to find image" in block["content"] and result.status == "delivered"


def _real_docker() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        info = subprocess.run(["docker", "image", "inspect", IMAGE], capture_output=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return info.returncode == 0


real_docker = pytest.mark.skipif(not _real_docker(), reason=f"no Docker daemon, or {IMAGE} is not pulled")


def _exists(name: str) -> bool:
    out = subprocess.run(["docker", "ps", "-a", "-q", "--filter", f"name=^{name}$"], capture_output=True, text=True)
    return bool(out.stdout.strip())


@real_docker
def test_real_container_runs_kills_and_is_removed(tmp_path: Path) -> None:
    root = project(tmp_path)
    (root / "hello.txt").write_text("from the host\n", encoding="utf-8")
    settings = EnvironmentSettings(mode="docker", docker=DockerSettings(image=IMAGE, network=False))
    sandbox = DockerSandbox(settings, root, tmp_path, f"test-{tmp_path.name[-8:].lower()}")

    async def scenario() -> None:
        jobs = JobManager(tmp_path / "jobs", sandbox)
        try:
            job = await jobs.start("cat hello.txt; pwd; echo x > made.txt", root, {}, settings)
            assert await wait_job(job, 60, asyncio.Event()) == "done" and job.exit_code == 0
            assert job.take_new().split() == ["from", "the", "host", "/workspace"]
            assert (root / "made.txt").read_text(encoding="utf-8") == "x\n"
            sleeper = await jobs.start("setsid sleep 300 & sleep 301", root, {}, settings, background=True)
            await asyncio.sleep(1.0)
            await jobs.kill(sleeper.id)
            left = subprocess.run(["docker", "exec", sandbox.name, "ps"], capture_output=True, text=True).stdout
            assert "sleep 30" not in left
            location = sandbox.locate("/tmp/inner.txt", root)
            await sandbox.write(location, b"inner")
            assert await sandbox.read(location) == b"inner" and location.host is None
            blocked = await jobs.start("wget -q -T 3 -O- http://example.com >/dev/null 2>&1; echo $?", root, {}, settings)
            await wait_job(blocked, 60, asyncio.Event())
            assert blocked.take_new().strip() != "0"
        finally:
            await jobs.close()
            await sandbox.close()

    asyncio.run(scenario())
    assert not _exists(sandbox.name)


@real_docker
def test_real_run_with_a_scripted_model(tmp_path: Path) -> None:
    task = {"environment": {"mode": "docker", "docker": {"image": IMAGE}}, "memory": {"enabled": False}}
    runtime, result, provider = go(
        tmp_path,
        [
            reply(tool_use("sh", {"command": "uname -s > os.txt; ls /"})),
            reply(tool_use("view", {"path": "/etc/hostname"})),
            reply(tool_use("grep", {"pattern": "^root:", "path": "/etc", "glob": "*passwd"})),
            reply(tool_use("read", {"path": "os.txt"})),
            done(),
        ],
        {**task, "role": {"lead": {"tools": ["sh", "edit", "view", "grep", "read"]}}},
    )
    assert result.status == "delivered"
    assert (project(tmp_path) / "os.txt").read_text(encoding="utf-8").strip() == "Linux"
    assert "workspace" in results(provider.requests[1])[0]["content"]
    assert "/etc/hostname: not an image" in results(provider.requests[2])[0]["content"]
    assert results(provider.requests[3])[0]["content"].startswith("/etc/passwd:1: root:")
    assert results(provider.requests[4])[0]["content"] == "os.txt, lines 1-1 of 1:\n1: Linux"
    assert not _exists(f"synthic-{runtime.run_id}")


@real_docker
def test_real_container_dies_with_its_keeper(tmp_path: Path) -> None:
    settings = EnvironmentSettings(mode="docker", docker=DockerSettings(image=IMAGE))
    sandbox = DockerSandbox(settings, project(tmp_path), tmp_path, f"keeper-{tmp_path.name[-8:].lower()}")

    async def scenario() -> None:
        await sandbox.ensure()
        assert _exists(sandbox.name)
        keeper = sandbox._keeper
        assert keeper is not None
        await keeper.kill_tree()
        for _ in range(50):
            if not _exists(sandbox.name):
                break
            await asyncio.sleep(0.2)
        await sandbox.close()

    asyncio.run(scenario())
    assert not _exists(sandbox.name)
