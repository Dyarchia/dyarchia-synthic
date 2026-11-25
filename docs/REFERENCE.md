# Synthic reference

Everything the engine does and every policy key that controls it. The short version is in the
[README](../README.md).

## Smallest run

```bash
synthic run "Add a --verbose flag to cli.py" --cwd path/to/repo
```

`synthic run` never prompts and does not read stdin unless it is a terminal. The task can also come from
`--task-file FILE` or from stdin with `-`. Exit codes: 0 delivered or replied, 1 failed, 2 usage or policy error,
3 budget, 130 stopped. `--quiet` prints only the final summary, `--json` prints one JSON object per
event, and the last one is the result.

Other commands: `runs`, `show [run-id] [--agent ID] [--full]`, `questions [run-id]`, `policy`, `prune`,
`rewind [run-id] [--to STEP]` (see File checkpoints).
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
present = false
wait_seconds = 300

[provider]
retry_seconds = 1800

[permissions]
"sh: git push*" = "deny"
"edit: .github/*" = "ask"
```

`[human]` says what happens where a person would be needed. By default nobody is there: `ask` returns at once
(after `wait_seconds`, if set) with the agent's assumption, and a prose turn is nudged until the agent calls `done`.
With `present = true` (the terminal interface starts from it, under every policy file, so a file that sets it
to `false` wins) the root agent's prose turn ends as status `replied`, its text being the summary, and the run
waits for the next message (`resume` with a text, exit code 0). `ask` and a permission rule `ask` then wait
for the answer with no time limit, whatever `wait_seconds` says. Spawned agents are never waited on for prose.

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
`synthic/<run_id>/<agent_id>`. With `"docker"`, everything runs in a container (see Docker sandbox).

## Context and memory

- **Compaction** replaces the history with a summary the model writes from a template
  (`context.compaction_template`, yours to replace). It keeps the newest `compact_keep_turns` turns as
  they are, up to `compact_keep_chars`. It runs when the prompt reaches `compact_at` of the context
  window, or of the size where the model starts charging more (`long_prompt_threshold`, 100,000 tokens
  on Haiku) unless `compact_within_price_tier = false`, and also between two tasks: a new text given to `resume` on a delivered run compacts the
  finished task first when its prompt had reached `compact_boundary_tokens` (0 turns this off).
- **What the agent was asked is never summarized.** The task and every later message from the human
  (or from the parent, for a child) are carried over word for word ahead of the summary, up to
  `compact_pin_chars` (40000; the task always, then the newest that fit; 0 turns this off).
- **The summary is reread before it is used** (`compact_check`, on by default): one more call asks the
  agent to check its summary against the conversation it still has and to answer with the complete one.
- **Background summary** (`context.compact_background = true`, off by default): once the prompt reaches
  `compact_prepare_at` of the window (0.6), the summary is asked for in the background, right after a step, with
  that step's exact request plus the template, so it reads the prompt from the cache. It is asked again each time
  the prompt grows `compact_refresh_tokens` (50000) past the one it came from. When compaction is due, the ready
  summary replaces what it covers and every later message stays as it is, with no wait (one still in flight is
  awaited). A summary whose messages changed since (a clean, a compaction, a merged message) is dropped. Its
  calls are in the ledger and the journal (`purpose = "compaction_prepare"`), used or not; one in flight is
  cancelled when the agent ends, and one not yet used is lost if the process dies.
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

- **Reread** (`verify.reread`, on by default): the root agent's first `done` of a delivery is answered with
  what it was asked, word for word (the task and every later message), and it must go through it against
  its work before calling `done` again. It costs one step and is not a review by anyone else: it brings the
  agent's attention back to the request at the end of a long task. A new message from the human arms it again.
- The checks below are off by default. They run only when the root agent calls `done` as delivered: first the `before_done`
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

## Structured delivery

```toml
[role.triage.result_schema]
type = "object"
required = ["severity", "files"]
properties = { severity = { enum = ["low", "high"] }, files = { type = "array", items = { type = "string" } } }
```

- A role with `result_schema` must call `done` with a `result` that matches it; a mismatch is the error result of
  `done`, listing each problem, and the agent fixes it. A `failed` delivery needs no result. The schema goes into
  the `done` tool definition, so roles without one pay nothing. It is replaced as a whole across levels.
- `synthic run --result-schema FILE.json` sets it for the root's role. The result is `AgentResult.result`, the
  `result` field of the last `--json` line, a `result:` line of the text summary, in the `done` record of the
  journal, and in the notice a parent gets from a child.
- Checked keywords: `type`, `enum`, `const`, `properties`, `required`, `additionalProperties`, `items`,
  `minItems`, `maxItems`, `uniqueItems`, `minLength`, `maxLength`, `pattern`, `minimum`, `maximum`,
  `exclusiveMinimum`, `exclusiveMaximum`, `allOf`, `anyOf`, `oneOf`, `not`. Annotations (`title`, `description`,
  `default`, `examples`, `format`, `$schema`, `$id`, `$comment`) are ignored; any other keyword (`$ref`...) is a
  policy error.

## Docker sandbox

```toml
[environment]
mode = "docker"
env = { CI = "1" }

[environment.docker]
image = "python:3.12-slim"
network = false
pass_env = ["SF_ACCESS_TOKEN"]
mounts = { "/ref" = "C:/data/reference" }
```

- One container per run, shared by every agent of the run (they all work in the project folder, as with
  `in_place`). `sh`, background jobs, `[verify]` commands and the file access of `edit` and `view` happen inside
  it. Hooks, MCP servers and plugins still run on the host, and `plugins.path` folders and skill files are host
  paths the container cannot see.
- Mounts: the project read-write at `workdir` (`/workspace`), the memory folder at `/synthic/memory` when memory
  is on, and each of `mounts` read-only. Nothing else of the host is visible. The image needs `sh`.
- Environment: only `environment.env` and the host variables named in `pass_env`, given as `-e NAME` so no value
  is on a command line. `ANTHROPIC_API_KEY` is refused in this mode.
- Network: `network = true` (default) or `false` (`--network none`). There is no list of allowed domains: Docker
  cannot filter by domain without a proxy, which Synthic does not ship.
- Lifetime: started in the background when the run starts, removed with `docker rm -f` when it ends. If the
  engine dies, the container's main process sees its stdin close and `--rm` removes it; containers of runs that
  are no longer alive (label `synthic.run`) are also removed when a new docker-mode run starts.
- Every command carries `SYNTHIC_JOB=<token>`; a kill removes every process in the container that holds it,
  detached ones included. Output a leftover process writes more than about 2 s after its shell exited is lost.
- `user` sets `--user` (default: the host uid:gid on Linux and macOS), `options` adds `docker run` arguments,
  `shell` is the shell inside (`["sh", "-c"]`), and `command` the client (`["docker"]`).

## File checkpoints

```toml
[checkpoints]
edits = true
shell = true
```

```bash
synthic rewind [run-id] --to 3 --dry-run
synthic rewind [run-id] --to 3 [--agent a1]
```

- `edits`: before `edit` writes a file, its previous bytes are kept in the run folder (once, by hash), or the
  record says the file was new. `shell`: before a turn that runs `sh`, when the working folder is in a git work
  tree, a git tree of it (tracked and untracked files git does not ignore) is written with a private index: the
  repository's index, branches and stash are not touched, and its object store gains blobs that `git gc` later
  drops. The journal folder is never captured. Both are `checkpoint` records with the agent's step number, shown
  by `synthic show`. Off by default.
- `rewind` undoes, newest first, every checkpoint after step `STEP` (default 0: everything): edited files get
  their bytes back, created ones are deleted, and a git checkpoint puts the working folder back to its tree: the
  files that differ are written back byte for byte (unless `.gitattributes` converts them), and files git does
  not ignore that the tree does not hold are deleted. Steps are the root's (children's checkpoints
  count from the moment the root's next step began), or one agent's own with `--agent`. It refuses a live run.
- Not covered: what `sh` changes outside a git work tree or in ignored files, what a background job changes
  between two turns, `[verify]` commands, hooks, MCP servers, files that only existed in the Docker container, and
  anything remote (a Salesforce org, a pushed branch, a sent message). A resumed agent is not told about a rewind.

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
the summary is printed); a second Ctrl+C exits at once. In the terminal interface (`synthic` alone) the first
Ctrl+C clears the message box, closes a menu and stops a working run, and a second one within two seconds
leaves in every state; Ctrl+D with an empty box does the same, and with text it deletes a character.

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
`run`. A run that ended (delivered, replied, failed or budget) is continued only when you give a text, which
enters as a new message from you; without one it just shows its recorded result. Giving the next task this way
keeps one session and lets the engine compact the finished task (see Context and memory). The run id may be
omitted (the most recent run is used); a word shaped like a run id that matches no run is an error, not
text. A run that is alive in another process is refused.

A resumed conversation is byte for byte the one the process had, thinking blocks included, so its first
call reads the prompt cache instead of writing everything again (when it comes within `cache_ttl`). The
thinking is dropped only when the model of the role changed, when `context.resume_keep_thinking = false`,
or when the API rejects a block; the journal records each drop.

```bash
synthic show [run-id]              # the trace: one line per event, then cost and open questions
synthic show [run-id] --full       # texts and tool results whole
synthic show [run-id] --md > run.md   # the run as a Markdown document
```

## Tools

```text
sh       run a shell command; a long one moves to the background
job      follow, wait for or kill a background command
edit     replace exact text in a file, or create a file
view     look at an image or a PDF
read     a text file or a line range, with line numbers
grep     search files with a regular expression: path:line: text
todo     keep a short task list (pending, active, done, dropped)
web_search, web_fetch   server tools, run by the provider (see Web search and web fetch)
spawn    start a child agent on a task
agent    list, wait for, message or cancel child agents
ask      ask the human, with the assumption to proceed on
receipt  read back the full text of a shortened tool result
done     end the run with a summary
skill    load a skill, when plugins provide any
mcp      call a tool of a configured MCP server
```

`view`, `read`, `grep` and `todo` are granted like `sh` and `edit`, in `role.<name>.tools`; no default role has
them. `read` and `grep` give exact output with no shell noise, through the same output cap and receipts as any
tool, and inside the container in docker mode. `grep` takes `pattern` (Python `re`), `path`, `glob`, `ignore_case`
and `limit` (100 matches by default, at most 1000); inside a git work tree it searches only files git does not
ignore, and it skips binary files, files over 2 MB and lines past 300 characters. `todo`
takes `set` (the whole list), `update` (steps by number) and `add`. Its result holds only the counts and the
active step, and nothing reminds the agent of the list, so it costs only the calls themselves. Each change is a
`todo` record in the journal, printed live by `synthic run` and at the end of `synthic show`; a resumed agent
starts from the last one, and a compaction appends the list to its summary.
