# Synthic

Synthic is an agent harness: the engine and nothing else. It runs one or more model agents on a task,
with tools, a journal and a budget. Policy, instructions, capabilities and environment are separate
things that it loads from files or enforces, never things the agent decides. Once the rules are set,
a run goes on by itself, and the human can interrupt it at any moment.

## Install

```bash
py -m pip install -e .
```

Python 3.12 or newer. The only dependency is `httpx`. Set `ANTHROPIC_API_KEY` in the environment; it is
not passed to the agent's shell, hooks or MCP servers unless `[environment.env]` sets it.

## Smallest run

```bash
synthic run "Add a --verbose flag to cli.py" --cwd path/to/repo
```

`synthic run` never prompts and does not read stdin unless it is a terminal. The task can also come from
`--task-file FILE` or from stdin with `-`. Exit codes: 0 delivered, 1 failed, 2 usage or policy error,
3 budget, 130 stopped. `--quiet` prints only the final summary, `--json` prints one JSON object per
event, and the last one is the result.

Other commands: `runs`, `show [run-id] [--agent ID] [--full]`, `questions [run-id]`, `policy`, `prune`.
`synthic policy` prints every value in force and the level that set it, before you start anything.
`synthic runs` shows each run as `running`, `crashed` (no recorded end and no live process), or its
final status.

## Policy

```toml
initial_role = "build"
never_send = ["internal-\\d{4}"]

[role.build]
model = "sonnet"
effort = "high"
tools = ["sh", "edit", "spawn"]
can_spawn = ["explore"]

[budget]
dollars = 5.0
wall_clock_hours = 4

[human]
wait_seconds = 300

[provider]
retry_seconds = 1800

[permissions]
"sh: git push*" = "deny"
"edit: .github/*" = "ask"
```

Four levels, from most to least authority, over the packaged defaults:

```text
organization  the file named by the SYNTHIC_ORG_POLICY environment variable
user          ~/.synthic/policy.toml  (SYNTHIC_HOME replaces ~)
project       <repo>/.synthic/policy.toml
task          --policy FILE and --set KEY=VALUE (for example --set role.build.model=opus)
```

- A value comes from the lowest level that sets it, unless a higher level lists its key in `fixed`.
- Lists are replaced as a whole. Tables merge key by key.
- Numeric caps (budget, agents, `max_steps`) can only be tightened: the smallest value wins.
- A `deny` permission and `never_send` patterns always hold, from any level. Hooks accumulate.
- The project file is untrusted by default: only its prohibitions count. `--trust-project` accepts it.

Other pieces: `[provider]` sets `retry_seconds` (how long transient failures of one model call are
retried) and `timeout_seconds` (how long a stream may stay silent). `[models.<alias>].extra` and
`[role.<name>].extra` are free-form vendor request fields, the role's winning over the model's.
`agents.repeat_limit` bounds identical repeated tool calls. `never_send` and `journal.mask` hold regular
expressions (Python `re` syntax), not globs, and a pattern that does not compile is a policy error.
With `environment.mode = "copy"`, each child works in a copy, on the git branch
`synthic/<run_id>/<agent_id>`.

## Context and memory

- **Compaction** replaces the history with a summary the model writes from a template
  (`context.compaction_template`, yours to replace). It keeps the newest `compact_keep_turns` turns as
  they are, up to `compact_keep_chars`. It runs when the prompt reaches `compact_at` of the context
  window, and also between two tasks: a new text given to `resume` on a delivered run compacts the
  finished task first when its prompt had reached `compact_boundary_tokens` (0 turns this off).
- **Memory** is one folder per project under `memory.dir` (default `~/.synthic/memory`), never inside
  the repository. Its `MEMORY.md` is read once when an agent starts, cut at `memory.max_chars`, and the
  root agent, if it has `edit`, is told to keep it short and current before it delivers; spawned
  agents only read it. `memory.enabled = false`
  turns it off.

## Images and PDFs

```bash
synthic run "Make the page look like this" --attach mockup.png
synthic resume [run-id] "Now follow this spec too" --attach spec.pdf
synthic say [run-id] "This is the error" --attach screenshot.png
```

- `--attach` is repeatable. JPEG, PNG, GIF and WebP go as image blocks and PDFs as document blocks, before
  the text of the message, each after a one-line label with its path. The type comes from the file's bytes.
- A file the API would refuse is refused before anything starts: another type, an image over 10 MB of base64
  or 8000x8000 px, a PDF over 128 MB or 600 pages, an encrypted PDF. Above 20 images in one request the API
  applies a stricter size limit, which is not checked.
- The journal keeps each file once, by hash, under `<agent>/receipts/media/`, so a resume rebuilds the
  same bytes. Media counts toward the context size by its estimated tokens, not its base64 length.
- The `view` tool (grant it in `role.<name>.tools`; no default role has it) shows the model an image or a
  PDF from disk. A file the API would refuse is not sent: the result says why. There is no resizing.
- When the cache is cold, an agent that has `view` gets old attachments and old `view` results replaced by
  a one-line stub naming the file, like old tool outputs. Without `view` attachments are kept, since the
  model could not see them again. Compaction summarizes them with the rest.

## Web search and web fetch

```toml
[role.research]
model = "sonnet"
tools = ["web_search", "web_fetch"]

[web]
search_max_uses = 5
fetch_max_uses = 10
allowed_domains = ["docs.python.org"]

[permissions]
"web_fetch: pastebin.com" = "deny"
```

- These are server tools: the provider runs them inside the model call, and their calls and results come
  back in the reply, are journaled and replayed with it. No default role has them.
- `[web]` sets the tool versions (`search_tool`, `fetch_tool`, basic versions by default), `*_max_uses` per
  turn, `allowed_domains` or `blocked_domains`, `fetch_max_content_tokens`, `fetch_citations`, and
  `search_price` (dollars per 1,000 searches, 10 by default), which the ledger adds to the token cost.
- The limit: the engine sees a search or a fetch only after it happened, so no hook, `ask` rule or human
  can stop one call. Permissions apply when the request is built: a rule with pattern `*` decides whether
  the tool is offered (`ask` becomes `human.permission_fallback`), and a domain rule becomes a blocked or
  allowed domain. A deny pattern that is not a domain the API takes turns the tool off, with a warning.
- `never_send` does not mask what these tools return: the provider requires it back unchanged.
- After a paused turn (`pause_turn`), or a turn that called server and client tools together, the engine
  sends no new text until the provider has finished the call; messages that arrive meanwhile wait one step.

## Verification before delivery

```toml
[verify]
commands = ["python -m pytest -q", "git diff --check"]
review = "reviewer"
rounds = 2

[role.reviewer]
model = "sonnet"
tools = ["sh"]
```

- Off by default. It runs only when the root agent calls `done` as delivered: first the `before_done`
  hook, then the commands in the working folder (each must exit 0 within `verify.timeout_seconds`, 1800 by
  default, or it is killed and fails), then the review. Running children are cancelled first, as every
  `done` does. The root agent's system prompt names the checks, so it can run them itself first.
- The reviewer is a fresh agent (`v1`, `v2`...) with the role `review` names, as the policy defines it. It
  gets every message the human sent during the run, from the journal, with their attachments, the delivering
  agent's summary, and how to see the changes (`git status`, `git diff` and the commits since the run
  started). Its `done` is the verdict: `delivered` approves, `failed` rejects with the list of problems.
  Its cost is in the same ledger.
- A rejection is the error result of the agent's `done`, with the failing output or the reviewer's list,
  and the agent keeps working. After `rounds` rejections a delivery that still fails is accepted, and its
  summary starts with `NOT VERIFIED:` and the reason. A reviewer that ends without a verdict (budget,
  stopped, an error) does not use a round: the delivery is accepted as `NOT VERIFIED`.
- Child agents are not verified; their parent judges their work. Each check is a `verify` record in the
  journal, so a resumed run keeps the rejections it had.

## Interrupting a run

From the same terminal (when stdin is a terminal), type a line while the run goes:

```text
/stop                         stop the run
/redirect [@agent] <text>     abort the current work and continue with <text>
/answer <question-id> <text>  answer an open question
/questions                    list the open questions
/add [@agent] <text>          add a message for the agent; any other line does the same
```

Without `@agent` the root agent is addressed; `@a1` addresses child `a1`, and an unknown or ended
agent gives a one-line note. Ctrl+C once is a clean stop (every process the agent started is killed and
the summary is printed); a second Ctrl+C exits at once.

From another terminal, for a run started hours ago or by a scheduler:

```bash
synthic stop [run-id]
synthic say [--agent ID] [run-id] "use the staging org"
synthic redirect [--agent ID] [run-id] "forget the refactor, fix the failing test"
synthic answer [run-id] <question-id> "yes, go ahead"
```

Each command appends a JSON line to `control.jsonl` in the run folder. The running process polls that
file about twice a second. Without a run id, the most recently active run is used. A run keeps a file
named `alive` fresh while it runs; the four commands refuse with exit code 2 when the run is not alive,
and for `say` the message tells you to use `synthic resume` instead.

## Runs and resuming

Runs live in `journal.dir` (default `~/.synthic/runs`) as `<run-id>/<agent-id>/journal.jsonl`, one JSON
line per event, flushed as written.

```bash
synthic resume [run-id]
synthic resume [run-id] "now also update the docs"
```

`resume` continues a run that was stopped or cut by a crash, with the same options and interrupts as
`run`. A run that ended (delivered, failed or budget) is continued only when you give a text, which
enters as a new message from you; without one it just shows its recorded result. Giving the next task this way
keeps one session and lets the engine compact the finished task (see Context and memory). The run id may be
omitted (the most recent run is used); a word shaped like a run id that matches no run is an error, not
text. A run that is alive in another process is refused.

## Tools

```text
sh       run a shell command; a long one moves to the background
job      follow, wait for or kill a background command
edit     replace exact text in a file, or create a file
view     look at an image or a PDF
web_search, web_fetch   server tools, run by the provider (see Web search and web fetch)
spawn    start a child agent on a task
agent    list, wait for, message or cancel child agents
ask      ask the human, with the assumption to proceed on
receipt  read back the full text of a shortened tool result
done     end the run with a summary
skill    load a skill, when plugins provide any
mcp      call a tool of a configured MCP server
```

## Tests

```bash
py -m pip install pytest
py -m pytest -q
```

The tests run offline against a scripted model.
