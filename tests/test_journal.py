from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from synthic.journal import (
    ALIVE_FILE,
    FileJournal,
    Masker,
    agents_in,
    beat,
    ended,
    is_alive,
    list_runs,
    new_run,
    pending_intents,
    prune,
    read_records,
)
from synthic.types import JournalSettings, Record


def settings(tmp_path: Path, **kwargs: object) -> JournalSettings:
    return JournalSettings(dir=str(tmp_path / "runs"), **kwargs)


def test_new_run_creates_sortable_folder(tmp_path: Path) -> None:
    run = new_run(settings(tmp_path))
    assert run.is_dir()
    assert run.parent == tmp_path / "runs"
    assert len(run.name) == len("20261122-153012-ab12")
    assert new_run(settings(tmp_path), "custom").name == "custom"


def test_append_and_read_back(tmp_path: Path) -> None:
    journal = FileJournal(new_run(settings(tmp_path)), "root", Masker())
    first = journal.append("user", {"text": "héllo", "source": "human"})
    second = journal.append("done", {"status": "delivered", "summary": ""})
    assert (first.seq, second.seq) == (1, 2)
    assert journal.dir.name == "root"
    records = list(journal.records())
    assert [r.kind for r in records] == ["user", "done"]
    assert records[0].data["text"] == "héllo"
    assert records[0].agent_id == "root"
    assert records[0].ts > 0


def test_sequence_continues_after_reopening(tmp_path: Path) -> None:
    run = new_run(settings(tmp_path))
    journal = FileJournal(run, "a", Masker())
    for i in range(5):
        journal.append("user", {"text": str(i)})
    journal.close()
    reopened = FileJournal(run, "a", Masker())
    assert reopened.append("user", {"text": "next"}).seq == 6
    assert [r.seq for r in reopened.records()] == [1, 2, 3, 4, 5, 6]


def test_truncated_last_line_is_ignored_and_repaired(tmp_path: Path) -> None:
    run = new_run(settings(tmp_path))
    journal = FileJournal(run, "a", Masker())
    journal.append("user", {"text": "one"})
    journal.append("user", {"text": "two"})
    journal.close()
    with journal.path.open("ab") as handle:
        handle.write(b'{"seq": 3, "ts": 1.0, "agent_id": "a", "kind": "us')
    reopened = FileJournal(run, "a", Masker())
    assert [r.seq for r in reopened.records()] == [1, 2]
    assert reopened.append("user", {"text": "three"}).seq == 3
    assert [r.data["text"] for r in reopened.records()] == ["one", "two", "three"]


def test_truncated_single_line_file(tmp_path: Path) -> None:
    run = new_run(settings(tmp_path))
    agent_dir = run / "a"
    agent_dir.mkdir()
    (agent_dir / "journal.jsonl").write_bytes(b'{"seq": 1, "ts"')
    journal = FileJournal(run, "a", Masker())
    assert list(journal.records()) == []
    assert journal.append("user", {}).seq == 1


def test_long_lines_do_not_break_tail_reading(tmp_path: Path) -> None:
    run = new_run(settings(tmp_path))
    journal = FileJournal(run, "a", Masker())
    journal.append("user", {"text": "x" * 50_000})
    journal.append("user", {"text": "y" * 50_000})
    journal.close()
    assert FileJournal(run, "a", Masker()).append("user", {}).seq == 3


SECRETS = {
    "anthropic": "sk-ant-api03-AbCdEfGhIjKlMnOpQrStUvWxYz0123456789",
    "openai": "sk-proj-abcdefghijklmnopqrstuvwxyz1234",
    "aws": "AKIAIOSFODNN7EXAMPLE",
    "github": "ghp_" + "a1B2c3D4e5" * 4,
    "github_pat": "github_pat_11ABCDEFG0abcdefghijkl_mnopqrstuvwxyz",
    "bearer": "Bearer eyJhbGciOiJIUzI1NiJ9.payload.signature",
    "salesforce15": "00D5g000004ABCD!AQEAQAbCdEfGhIjKlMnOpQrStUvWxYz.1234",
    "salesforce18": "00D5g000004ABCDEAA!AQEAQAbCdEfGhIjKlMnOpQrStUvWxYz.1234",
    "sfdx": "force://PlatformCLI::5Aep861abcdefghijk@my-org.my.salesforce.com",
    "pem": "-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEA\nabcdef\n-----END RSA PRIVATE KEY-----",
}


@pytest.mark.parametrize("name", sorted(SECRETS))
def test_masker_masks_builtin_patterns(name: str) -> None:
    secret = SECRETS[name]
    masked = Masker().mask(f"before {secret} after")
    assert masked == "before [masked] after"


@pytest.mark.parametrize("key", ["password", "token", "secret", "access_token", "PASSWORD"])
def test_masker_masks_assignments_keeping_the_key(key: str) -> None:
    assert Masker().mask(f"{key}=hunter2hunter2hunter2 next") == f"{key}=[masked] next"
    assert Masker().mask(f'{key} = "hunter2hunter2hunter2"') == f'{key} = "[masked]"'


def test_masker_leaves_ordinary_text_alone() -> None:
    text = (
        "sha256 d41d8cd98f00b204e9800998ecf8427e da39a3ee5e6b4b0d3255bfef95601890afd80709\n"
        r"path C:\Users\me\project\src\main.py and /home/me/risk-assessment-framework-docs/readme.md"
        "\npassword=short and token: none; the bearer of news; 00D is a prefix; task-run-identifier-long-name"
    )
    assert Masker().mask(text) == text


def test_masker_extra_patterns_and_nested_objects() -> None:
    masker = Masker((r"internal-\d{4}",))
    data = {"a": ["x internal-1234", {"b": "ok"}], "n": 3, "t": ("internal-9999",)}
    assert masker.mask_obj(data) == {"a": ["x [masked]", {"b": "ok"}], "n": 3, "t": ["[masked]"]}


def test_masker_is_fast_on_large_strings() -> None:
    text = ("line of ordinary output with a path /usr/lib/x and value=42\n" * 20_000) + SECRETS["aws"]
    start = time.perf_counter()
    masked = Masker().mask(text)
    assert time.perf_counter() - start < 2.0
    assert masked.endswith("[masked]")


def test_journal_masks_before_writing(tmp_path: Path) -> None:
    journal = FileJournal(new_run(settings(tmp_path)), "a", Masker())
    record = journal.append("tool_result", {"content": f"key {SECRETS['aws']}", "nested": [SECRETS["github"]]})
    assert record.data == {"content": "key [masked]", "nested": ["[masked]"]}
    raw = journal.path.read_text(encoding="utf-8")
    assert SECRETS["aws"] not in raw
    assert SECRETS["github"] not in raw


def test_masker_leaves_thinking_signatures_and_image_data_alone() -> None:
    secret = SECRETS["openai"]
    content = [
        {"type": "thinking", "thinking": f"uses {secret}", "signature": secret},
        {"type": "redacted_thinking", "data": secret},
        {"type": "text", "text": f"key {secret}"},
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": secret}},
    ]
    masked = Masker().mask_obj(content)
    assert masked[:2] == content[:2]
    assert masked[2] == {"type": "text", "text": "key [masked]"}
    assert masked[3] == content[3]
    assert Masker().mask_obj(masked) == masked


def test_read_records_never_writes(tmp_path: Path) -> None:
    run = new_run(settings(tmp_path))
    journal = FileJournal(run, "a", Masker())
    journal.append("user", {"text": "one"})
    journal.close()
    with journal.path.open("ab") as handle:
        handle.write(b'{"seq": 2, "ts"')
    before = journal.path.read_bytes()
    assert [r.data["text"] for r in read_records(run, "a")] == ["one"]
    assert journal.path.read_bytes() == before
    assert list(read_records(run, "missing")) == []


def test_heartbeat(tmp_path: Path) -> None:
    run = new_run(settings(tmp_path))
    assert not is_alive(run)
    beat(run)
    assert is_alive(run) and (run / ALIVE_FILE).read_text(encoding="utf-8") == str(os.getpid())
    stale = time.time() - 60
    os.utime(run / ALIVE_FILE, (stale, stale))
    assert not is_alive(run) and is_alive(run, max_age=120)


def rec(seq: int, kind: str, call_id: str) -> Record:
    return Record(seq, 0.0, "a", kind, {"call_id": call_id})


def test_ended_is_the_last_done_not_followed_by_a_resume() -> None:
    first, second = rec(2, "done", ""), rec(5, "done", "")
    assert ended([rec(1, "start", "")]) is None
    assert ended([rec(1, "start", ""), first]) is first
    assert ended([first, rec(3, "resume", ""), rec(4, "user", "")]) is None
    assert ended([first, rec(3, "resume", ""), second, rec(6, "hook", "")]) is second


def test_pending_intents() -> None:
    records = [
        rec(1, "tool_intent", "c1"),
        rec(2, "tool_intent", "c2"),
        rec(3, "tool_result", "c1"),
        rec(4, "user", "c9"),
        rec(5, "tool_intent", "c3"),
    ]
    assert [r.data["call_id"] for r in pending_intents(records)] == ["c2", "c3"]
    assert pending_intents([]) == []


def make_run(cfg: JournalSettings, name: str, age_days: float) -> Path:
    run = new_run(cfg, name)
    journal = FileJournal(run, "root", Masker())
    journal.append("start", {})
    journal.close()
    stamp = time.time() - age_days * 86400
    os.utime(journal.path, (stamp, stamp))
    os.utime(run, (stamp, stamp))
    return run


def test_list_runs_agents_and_prune(tmp_path: Path) -> None:
    cfg = settings(tmp_path, retention_days=7)
    assert list_runs(cfg) == []
    old = make_run(cfg, "old", 30)
    mid = make_run(cfg, "mid", 3)
    new = make_run(cfg, "new", 0)
    FileJournal(new, "child", Masker()).close()
    assert [p.name for p in list_runs(cfg)] == ["new", "mid", "old"]
    assert agents_in(new) == ["child", "root"]
    assert prune(cfg) == 1
    assert not old.exists() and mid.exists() and new.exists()
    assert prune(cfg) == 0


def test_prune_is_a_no_op_without_retention(tmp_path: Path) -> None:
    cfg = settings(tmp_path)
    old = make_run(cfg, "old", 400)
    assert prune(cfg) == 0
    assert old.exists()
