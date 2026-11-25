"""The four views of a run. Each one draws the same ``Board`` and none of them talks to the engine.

``Column`` is one conversation. ``Panes`` puts the agents on the left, the conversation of one of them in
the centre and the full text of a step on the right. ``Fleet`` sets the agents side by side. ``Ledger`` is
the run as a table, one row per step. Only the view on screen is kept up to date: switching rebuilds it.
"""

from __future__ import annotations

import time
from typing import Any

from rich.console import Console, ConsoleOptions, Group, RenderableType, RenderResult
from rich.markdown import CodeBlock, Heading, Markdown
from rich.text import Text
from rich.theme import Theme
from textual.app import ComposeResult
from textual.containers import Horizontal, HorizontalScroll, Vertical, VerticalScroll
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Rule, Static

from .board import LIVE, ROOT, AgentView, Board, Step
from .theme import elapsed, money, number, seconds, state_mark

VIEWS = {"column": "Column", "panes": "Panes", "fleet": "Fleet", "ledger": "Ledger"}
SHOWN = 400
"""Steps a list keeps on screen; older ones stay in the journal and in ``synthic show``."""
BARS = " ▁▂▃▄▅▆▇█"


class Empty(Static):
    """What a part of a view says while there is nothing to draw in it: one quiet line."""

    def __init__(self, text: str, classes: str = "") -> None:
        super().__init__(text, classes=f"empty {classes}".strip())


class StepPicked(Message):
    def __init__(self, step: Step) -> None:
        super().__init__()
        self.step = step


class AgentPicked(Message):
    def __init__(self, agent_id: str) -> None:
        super().__init__()
        self.agent_id = agent_id


class FlatHeading(Heading):
    """A heading as a line of ink: no box, no centring."""

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        self.text.justify = "left"
        yield self.text


class FlatCode(CodeBlock):
    """A code block as it was written, on the sunken ground, with no syntax colours."""

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        yield Text(str(self.text).rstrip(), style=console.get_style("markdown.code_block"))


class Prose(Markdown):
    """What the model says, as Markdown drawn in ink alone: structure stays, colour and weight do not."""

    elements = {**Markdown.elements, "heading_open": FlatHeading, "fence": FlatCode, "code_block": FlatCode}

    def __init__(self, markup: str, tokens: dict[str, str]) -> None:
        super().__init__(markup)
        ink, muted, sunken = tokens["ink"], tokens["ink-muted"], tokens["ground-sunken"]
        plain = {f"markdown.{name}": ink for name in ("paragraph", "text", "strong", "em", "emph", "s", "item", "list")}
        heads = {f"markdown.h{level}": ink for level in range(1, 8)}
        self.theme = Theme({
            **plain, **heads, "markdown.h1.border": ink,
            "markdown.code": f"{ink} on {sunken}", "markdown.code_block": f"{ink} on {sunken}",
            "markdown.block_quote": muted, "markdown.item.bullet": muted, "markdown.item.number": muted,
            "markdown.hr": tokens["hairline"], "markdown.link": muted, "markdown.link_url": muted,
        })

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        with console.use_theme(self.theme):
            drawn = list(super().__rich_console__(console, options))
        yield from drawn


def cut(text: str, width: int) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= width else flat[: max(0, width - 1)] + "…"


def badge(state: str, tokens: dict[str, str], label: str = "") -> Text:
    """A state as the design system writes it: glyph and word in the colour of the state, on its tile."""
    glyph, colour = state_mark(state)
    tile = tokens.get(f"{colour}-tile")
    return Text(f"{glyph} {label or state}", style=f"{tokens[colour]} on {tile}" if tile else tokens[colour])


def name_of(board: Board, agent_id: str) -> str:
    """What an agent is called on screen: its role, or its id when two agents share the role."""
    view = board.agents.get(agent_id)
    if view is None or not view.role:
        return agent_id
    twins = [other for other in board.agents.values() if other.role == view.role]
    return view.role if len(twins) == 1 else f"{view.role} {agent_id}"


def brief(line: str) -> str:
    """A tool line with every path cut to its last part, for the places where a line has little room."""
    return " ".join("…/" + word.rsplit("/", 1)[1] if word.count("/") > 1 else word for word in line.split(" "))


def outcome(step: Step, tokens: dict[str, str]) -> Text:
    if step.result == "running":
        return Text("running", style=tokens["tide"])
    took = f" in {seconds(step.seconds)}" if step.seconds is not None else ""
    return Text(f"{step.result}{took}", style=tokens["down" if step.error else "ink-muted"])


def body(step: Step, tokens: dict[str, str], who: str = "", plain: bool = False) -> RenderableType:
    """One step as an entry of a conversation."""
    ink, muted = tokens["ink"], tokens["ink-muted"]
    lead = Text(f"{who}  ", style=muted) if who else Text()
    match step.kind:
        case "user":
            return Text.assemble(lead, ("> " + step.text, ink))
        case "text" if plain or who:
            return Text.assemble(lead, (step.text.strip(), ink))
        case "text":
            return Prose(step.text, tokens)
        case "tool":
            dot = tokens["tide"] if step.result == "running" else muted
            line = Text.assemble(lead, ("● ", dot), (brief(step.text) if plain else step.text, muted), "  ", outcome(step, tokens))
            return line
        case "question":
            open_ = step.result == "waiting"
            return Text.assemble(lead, ("◆ ", tokens["clay"] if open_ else muted), (step.text, ink if open_ else muted))
        case "end":
            glyph, colour = state_mark(step.result)
            mark = Text.assemble(lead, (f"{glyph} {step.result}", tokens[colour]))
            if not step.text.strip():
                return mark
            if plain or who:
                return Text.assemble(lead, (step.text.strip() + "\n", ink), mark)
            return Group(Prose(step.text, tokens), mark)
        case _:
            return Text.assemble(lead, ("! ", tokens["sun"]), (step.text, muted))


class Entry(Static):
    def __init__(self, step: Step, content: RenderableType, classes: str = "") -> None:
        super().__init__(content, classes=classes)
        self.step = step

    def on_click(self) -> None:
        self.post_message(StepPicked(self.step))


class Stream(VerticalScroll):
    """A conversation: one entry per step, the newest at the bottom."""

    can_focus = False

    def __init__(self, board: Board, agent_id: str | None = None, plain: bool = False, **more: Any) -> None:
        super().__init__(**more)
        self.board = board
        self.agent_id = agent_id
        self.plain = plain
        self.items: dict[int, Entry] = {}
        self.marked: int | None = None
        self.last: bool | None = None

    def mine(self, step: Step) -> bool:
        return self.agent_id is None or step.agent == self.agent_id

    def _entry(self, step: Step) -> Entry:
        dense = step.kind in ("tool", "question", "note")
        gap = " gap" if self.last is not None and not (dense and self.last) else ""
        self.last = dense
        entry = Entry(step, self._body(step), classes=f"entry {step.kind}{gap}")
        self.items[step.n] = entry
        return entry

    def _body(self, step: Step) -> RenderableType:
        several = self.agent_id is None and len(self.board.agents) > 1
        who = name_of(self.board, step.agent) if several and step.agent != ROOT else ""
        return body(step, self.app.tokens, who, self.plain)

    def rebuild(self, agent_id: str | None = None) -> None:
        self.agent_id = agent_id
        view = None if agent_id is None else self.board.agents.get(agent_id)
        steps = self.board.steps if agent_id is None else (view.steps if view else [])
        self.remove_children()
        self.items, self.last = {}, None
        self.mount(*[self._entry(step) for step in steps[-SHOWN:]])
        self.scroll_end(animate=False)

    def add(self, step: Step) -> None:
        if not self.mine(step):
            return
        if len(self.items) >= SHOWN:
            oldest = min(self.items)
            self.items.pop(oldest).remove()
        self.mount(self._entry(step))
        if self.app.follow:
            self.scroll_end(animate=False)

    def change(self, step: Step) -> None:
        entry = self.items.get(step.n)
        if entry is not None:
            entry.update(self._body(step))
            if self.app.follow:
                self.scroll_end(animate=False)

    def mark(self, step: Step | None) -> None:
        if self.marked in self.items:
            self.items[self.marked].remove_class("selected")
        self.marked = step.n if step is not None else None
        if self.marked in self.items:
            entry = self.items[self.marked]
            entry.add_class("selected")
            entry.scroll_visible(animate=False)


class Head(Widget):
    """A header row: a label on the left, a note on the right, both in muted ink."""

    def __init__(self, label: str, note: str = "", **more: Any) -> None:
        super().__init__(**more)
        self.label, self.note = label, note

    def set(self, label: str, note: str = "") -> None:
        self.label, self.note = label, note
        self.refresh()

    def render(self) -> Text:
        width, muted = self.size.width, self.app.tokens["ink-muted"]
        note = cut(self.note, max(0, width - len(self.label) - 2))
        return Text(self.label + " " * max(1, width - len(self.label) - len(note)) + note, style=muted, no_wrap=True)


class Detail(Vertical):
    """The full text of the selected step: a tool output, a diff, or what the model said."""

    def __init__(self, board: Board, meters: bool = False, **more: Any) -> None:
        super().__init__(**more)
        self.board = board
        self.meters = meters
        self.pending: Any = None

    def compose(self) -> ComposeResult:
        yield Head("DETAIL", classes="head")
        yield Rule(classes="rule")
        with VerticalScroll(classes="detail-body") as scroll:
            scroll.can_focus = False
            yield Static("", classes="detail-text")
        if self.meters:
            yield Meters(self.board)

    def show(self, step: Step | None) -> None:
        tokens = self.app.tokens
        head = self.query_one(Head)
        if step is None:
            head.set("DETAIL")
            self.query_one(".detail-text", Static).update(Text("Nothing selected", style=tokens["ink-faint"]))
            return
        head.set("DETAIL", f"{name_of(self.board, step.agent)} · step {step.n} · {time.strftime('%H:%M:%S', time.localtime(step.ts))}")
        self.draw(step, fetch=False)
        self.query_one(".detail-body", VerticalScroll).scroll_home(animate=False)
        if self.pending is not None:
            self.pending.stop()
        self.pending = None if self.board.known(step) else self.set_timer(0.3, lambda: self.draw(step, fetch=True))

    def draw(self, step: Step, fetch: bool) -> None:
        """Write what is known at once; the output of a call that just ended is read a moment later."""
        if fetch and step is not self.app.selected:
            return
        tokens = self.app.tokens
        tones = {"ink": tokens["ink"], "muted": tokens["ink-muted"], "up": tokens["up"], "down": tokens["down"]}
        text = Text()
        for tone, line in self.board.detail(step, fetch)[:2000]:
            text.append(line + "\n", style=tones[tone])
        self.query_one(".detail-text", Static).update(text)


class Meters(Widget):
    """Context of the agent in focus and budget of the run, as two small tracks."""

    def __init__(self, board: Board, **more: Any) -> None:
        super().__init__(**more)
        self.board = board

    def render(self) -> Text:
        app, width = self.app, self.size.width
        tokens = app.tokens
        text = Text(no_wrap=True)
        for label, used, limit, figure in app.gauges():
            share = min(1.0, used / limit) if limit else 0.0
            track = max(4, width - 9)
            on = round(share * track)
            percent = f"{share * 100:.1f}".replace(".", ",") + " %"
            text.append(label + " " * max(1, width - len(label) - len(figure)), style=tokens["ink-muted"])
            text.append(figure + "\n", style=tokens["ink"])
            text.append("▄" * on, style=tokens["sun"] if share >= 0.8 else tokens["ink-muted"])
            text.append("▁" * (track - on), style=tokens["line-control"])
            text.append(percent.rjust(width - track) + "\n\n", style=tokens["ink-muted"])
        return text


class AgentCard(Widget):
    """An agent in a list: name and spend, then state and model."""

    def __init__(self, board: Board, view: AgentView, **more: Any) -> None:
        super().__init__(**more)
        self.board, self.view = board, view

    def on_click(self) -> None:
        self.post_message(AgentPicked(self.view.id))

    def render(self) -> Text:
        tokens, width, view = self.app.tokens, self.size.width, self.view
        pad = "  " * min(view.depth, 3)
        name, cost = pad + cut(name_of(self.board, view.id), width - len(pad) - 9), money(view.cost)
        mark = badge(view.state, tokens)
        model = self.app.model_of(view.id)
        text = Text(no_wrap=True)
        text.append(name + " " * max(1, width - len(name) - len(cost)) + cost + "\n", style=tokens["ink"])
        text.append(pad)
        text.append(mark)
        text.append(" " * max(1, width - len(pad) - len(mark.plain) - len(model)) + model, style=tokens["ink-muted"])
        return text


class View(Widget):
    """What every view answers to. ``rebuild`` draws it from the board; the rest keep it up to date."""

    def __init__(self, board: Board, **more: Any) -> None:
        super().__init__(**more)
        self.board = board

    def rebuild(self) -> None:
        raise NotImplementedError

    def step_added(self, step: Step) -> None:
        raise NotImplementedError

    def step_changed(self, step: Step) -> None:
        raise NotImplementedError

    def agent_changed(self, view: AgentView) -> None:
        pass

    def selection_changed(self) -> None:
        pass

    def tick(self) -> None:
        pass

    def choices(self) -> list[Step]:
        """The steps the selection moves through in this view."""
        return []


class ColumnView(View):
    def compose(self) -> ComposeResult:
        yield Empty("No conversation yet")
        yield Stream(self.board, classes="measure")

    def sync(self) -> None:
        drawn = bool(self.board.steps)
        self.query_one(Stream).display = drawn
        self.query_one(Empty).display = not drawn

    def rebuild(self) -> None:
        self.query_one(Stream).rebuild(None)
        self.sync()

    def step_added(self, step: Step) -> None:
        self.query_one(Stream).add(step)
        self.sync()

    def step_changed(self, step: Step) -> None:
        self.query_one(Stream).change(step)


class PanesView(View):
    def compose(self) -> ComposeResult:
        with Horizontal():
            with Vertical(classes="side"):
                yield Head("AGENTS", classes="head")
                yield Rule(classes="rule")
                yield Empty("No agents yet", classes="no-agents")
                with VerticalScroll(classes="cards") as cards:
                    cards.can_focus = False
            with Vertical(classes="centre"):
                yield Head("", classes="head talk")
                yield Rule(classes="rule")
                yield Empty("No steps yet", classes="no-steps")
                yield Stream(self.board, plain=True)
            yield Detail(self.board, meters=True, classes="dock")

    def sync(self) -> None:
        agents, steps = bool(self.board.agents), bool(self.board.steps)
        self.query_one(".cards").display = agents
        self.query_one(".no-agents").display = not agents
        self.query_one(Stream).display = steps
        self.query_one(".no-steps").display = not steps

    def rebuild(self) -> None:
        cards = self.query_one(".cards", VerticalScroll)
        cards.remove_children()
        cards.mount(*[AgentCard(self.board, view, classes="card") for view in self.board.agents.values()])
        self.selection_changed(full=True)

    def selection_changed(self, full: bool = False) -> None:
        app, stream = self.app, self.query_one(Stream)
        focus = app.focus_agent
        for card in self.query(AgentCard):
            card.set_class(card.view.id == focus, "selected")
        view = self.board.agents.get(focus)
        self.query_one(".talk", Head).set(name_of(self.board, focus).upper() if view else "", view.task if view else "")
        if full or stream.agent_id != focus:
            stream.rebuild(focus)
        stream.mark(app.selected if app.selected is not None and app.selected.agent == focus else None)
        self.query_one(Detail).show(app.selected)
        self.query_one(Meters).refresh()
        self.sync()

    def step_added(self, step: Step) -> None:
        self.query_one(Stream).add(step)
        self.sync()

    def step_changed(self, step: Step) -> None:
        self.query_one(Stream).change(step)
        if step is self.app.selected:
            self.query_one(Detail).show(step)

    def agent_changed(self, view: AgentView) -> None:
        cards = {card.view.id: card for card in self.query(AgentCard)}
        if view.id in cards:
            cards[view.id].refresh()
        else:
            self.query_one(".cards", VerticalScroll).mount(AgentCard(self.board, view, classes="card"))
        self.sync()
        if view.id == self.app.focus_agent:
            self.query_one(".talk", Head).set(name_of(self.board, view.id).upper(), view.task)

    def tick(self) -> None:
        self.query_one(Meters).refresh()

    def choices(self) -> list[Step]:
        view = self.board.agents.get(self.app.focus_agent)
        return [step for step in view.steps if step.kind != "user"] if view else []


class Totals(Widget):
    """The run in one band: what it is, how long, how much, and the spend over time."""

    def __init__(self, board: Board, **more: Any) -> None:
        super().__init__(**more)
        self.board = board

    def render(self) -> Text:
        app, board, width = self.app, self.board, self.size.width
        tokens = app.tokens
        cap = app.loaded.policy.budget.dollars
        cells = [
            ("RUN", app.run_id or "no run", app.cwd.name),
            ("ELAPSED", elapsed(app.elapsed()), "from " + time.strftime("%H:%M", time.localtime(board.started)) if board.started else ""),
            ("SPENT", money(board.total), f"of {money(cap)}" if cap else "no cap"),
            ("STEPS", number(sum(view.calls for view in board.agents.values())), f"{len(board.agents)} agents"),
        ]
        widths = [max(len(part) for part in cell) + 3 for cell in cells]
        room = width - sum(widths) - 2
        rows = [Text(no_wrap=True), Text(no_wrap=True), Text(no_wrap=True)]
        for cell, size in zip(cells, widths):
            for row, part, tone in zip(rows, cell, ("ink-muted", "ink", "ink-muted")):
                row.append(part.ljust(size), style=tokens[tone])
        if room >= 16 and board.spend:
            low, high = self.chart(room)
            span = f"{time.strftime('%H:%M', time.localtime(board.started))} → {time.strftime('%H:%M')}"
            rows[0].append("  SPEND" + span.rjust(room - 5), style=tokens["ink-muted"])
            rows[1].append("  " + high, style=tokens["ink-muted"])
            rows[2].append("  " + low, style=tokens["ink-muted"])
        return Text("\n").join(rows)

    def chart(self, width: int) -> tuple[str, str]:
        """Spend so far as an area two rows tall, eight levels to a row."""
        board = self.board
        start, end = board.started, max(time.time(), board.spend[-1][0])
        peak = board.spend[-1][1] or 1.0
        low, high, at, spent = "", "", 0, 0.0
        for column in range(width):
            until = start + (end - start) * (column + 1) / width
            while at < len(board.spend) and board.spend[at][0] <= until:
                spent = board.spend[at][1]
                at += 1
            level = max(1, round(spent / peak * 16)) if spent else 0
            low += BARS[min(8, level)]
            high += BARS[max(0, level - 8)]
        return low, high


class FleetColumn(Vertical):
    def __init__(self, board: Board, view: AgentView, **more: Any) -> None:
        super().__init__(**more)
        self.board, self.view = board, view

    def compose(self) -> ComposeResult:
        yield FleetHead(self.board, self.view)
        yield Rule(classes="rule")
        yield Stream(self.board, self.view.id, plain=True)

    def on_click(self) -> None:
        self.post_message(AgentPicked(self.view.id))


class FleetHead(Widget):
    def __init__(self, board: Board, view: AgentView, **more: Any) -> None:
        super().__init__(**more)
        self.board, self.view = board, view

    def render(self) -> Text:
        app, view, width = self.app, self.view, self.size.width
        tokens = app.tokens
        tone = tokens["ink" if view.id == app.focus_agent else "ink-muted"]
        mark = badge(view.state, tokens)
        name = cut(name_of(self.board, view.id), width - len(mark.plain) - 1)
        model, cost = cut(app.model_of(view.id, effort=True), width - 10), money(view.cost)
        steps = f"{view.calls} step{'' if view.calls == 1 else 's'}"
        context = number(view.context)
        text = Text(no_wrap=True)
        text.append(name + " " * max(1, width - len(name) - len(mark.plain)), style=tone)
        text.append(mark)
        text.append("\n" + model + " " * max(1, width - len(model) - len(cost)), style=tokens["ink-muted"])
        text.append(cost, style=tone)
        text.append("\n" + steps + " " * max(1, width - len(steps) - len(context)) + context, style=tokens["ink-muted"])
        return text


class FleetView(View):
    """The root, every agent still working, and the last ones to end while there is room for them."""

    ROOM = 4

    def compose(self) -> ComposeResult:
        yield Totals(self.board, classes="totals")
        yield Empty("No agents yet")
        scroll = HorizontalScroll(classes="fleet")
        scroll.can_focus = False
        yield scroll

    def sync(self) -> None:
        agents = bool(self.board.agents)
        self.query_one(".fleet").display = agents
        self.query_one(Empty).display = not agents

    def wanted(self) -> list[str]:
        views = list(self.board.agents.values())
        live = [view.id for view in views if view.id == ROOT or view.state in LIVE]
        done = [view.id for view in reversed(views) if view.id not in live]
        keep = set(live) | set(done[: max(0, self.ROOM - len(live))])
        return [view.id for view in views if view.id in keep]

    def rebuild(self) -> None:
        fleet = self.query_one(".fleet", HorizontalScroll)
        fleet.remove_children()
        columns = [FleetColumn(self.board, self.board.agents[agent_id], classes="lane") for agent_id in self.wanted()]
        fleet.mount(*columns)
        self.shown = self.wanted()
        self.sync()
        self.call_after_refresh(self.fill)

    def lanes(self, agent_id: str | None = None) -> list[Stream]:
        """The conversations on screen. A lane mounted a moment ago has none yet: ``fill`` draws it whole."""
        columns = [column for column in self.query(FleetColumn) if agent_id in (None, column.view.id)]
        return [stream for column in columns for stream in column.query(Stream)]

    def fill(self) -> None:
        for stream in self.lanes():
            stream.rebuild(stream.agent_id)

    def step_added(self, step: Step) -> None:
        for stream in self.lanes(step.agent):
            stream.add(step)

    def step_changed(self, step: Step) -> None:
        for stream in self.lanes(step.agent):
            stream.change(step)

    def agent_changed(self, view: AgentView) -> None:
        if self.wanted() != getattr(self, "shown", []):
            self.rebuild()
            return
        for head in self.query(FleetHead):
            if head.view.id == view.id:
                head.refresh()

    def selection_changed(self) -> None:
        for head in self.query(FleetHead):
            head.refresh()

    def tick(self) -> None:
        self.query_one(Totals).refresh()


COLUMNS = (("TIME", 5, False), ("AGENT", 8, False), ("STEP", 4, True), ("ACTION", 0, False), ("RESULT", 7, False), ("TOOK", 6, True), ("COST", 7, True))
FIXED = sum(size for _, size, _ in COLUMNS) + 2 * (len(COLUMNS) - 1)


def ledger_cells(width: int, cells: list[tuple[str, str]]) -> Text:
    """One row of the ledger: every cell cut or padded to its column, figures to the right."""
    text = Text(no_wrap=True)
    for (_, size, right), (value, style) in zip(COLUMNS, cells):
        size = size or max(10, width - FIXED)
        value = cut(value, size)
        text.append(value.rjust(size) if right else value.ljust(size), style=style)
        text.append("  ")
    text.right_crop(2)
    return text


class LedgerHead(Widget):
    def render(self) -> Text:
        muted = self.app.tokens["ink-muted"]
        return ledger_cells(self.size.width, [(name, muted) for name, _, _ in COLUMNS])


class Row(Widget):
    def __init__(self, board: Board, step: Step, **more: Any) -> None:
        super().__init__(**more)
        self.board, self.step = board, step

    def on_click(self) -> None:
        self.post_message(StepPicked(self.step))

    def render(self) -> Text:
        tokens, step = self.app.tokens, self.step
        ink, muted = tokens["ink"], tokens["ink-muted"]
        first = step.text.strip().splitlines()[0] if step.text.strip() else ""
        first = brief(first) if step.kind == "tool" else first
        action = {"user": "> " + first, "question": "◆ " + first, "note": "! " + first}.get(step.kind, first)
        result, tone = step.result, muted
        if step.kind == "tool" and step.result == "running":
            tone = tokens["tide"]
        elif step.kind == "question" and step.result == "waiting":
            tone = tokens["clay"]
        elif step.kind == "end":
            tone = tokens[state_mark(step.result)[1]]
        elif step.error:
            tone = tokens["down"]
        return ledger_cells(self.size.width, [
            (time.strftime("%H:%M", time.localtime(step.ts)), muted),
            (name_of(self.board, step.agent), muted),
            (str(step.n), muted),
            (action, ink if step.kind in ("user", "text", "question", "end") else muted),
            (result, tone),
            (seconds(step.seconds) if step.seconds is not None else "", muted),
            (money(step.cost) if step.cost else "", ink),
        ])


class LedgerView(View):
    def compose(self) -> ComposeResult:
        with Horizontal():
            with Vertical(classes="table"):
                yield LedgerHead(classes="head")
                yield Rule(classes="rule")
                yield Empty("No steps yet")
                with VerticalScroll(classes="rows") as rows:
                    rows.can_focus = False
            yield Detail(self.board, classes="dock wide")

    def sync(self) -> None:
        drawn = bool(self.board.steps)
        self.query_one(".rows").display = drawn
        self.query_one(Empty).display = not drawn

    def rebuild(self) -> None:
        rows = self.query_one(".rows", VerticalScroll)
        rows.remove_children()
        self.rows = {step.n: Row(self.board, step, classes="row") for step in self.board.steps[-SHOWN:]}
        rows.mount(*self.rows.values())
        rows.scroll_end(animate=False)
        self.sync()
        self.selection_changed()

    def step_added(self, step: Step) -> None:
        rows = self.query_one(".rows", VerticalScroll)
        if len(self.rows) >= SHOWN:
            self.rows.pop(min(self.rows)).remove()
        self.rows[step.n] = Row(self.board, step, classes="row")
        rows.mount(self.rows[step.n])
        self.sync()
        if self.app.follow:
            rows.scroll_end(animate=False)

    def step_changed(self, step: Step) -> None:
        if step.n in self.rows:
            self.rows[step.n].refresh()
        if step is self.app.selected:
            self.query_one(Detail).show(step)

    def selection_changed(self) -> None:
        chosen = self.app.selected
        for number_, row in getattr(self, "rows", {}).items():
            row.set_class(chosen is not None and number_ == chosen.n, "selected")
        if chosen is not None and chosen.n in self.rows:
            self.rows[chosen.n].scroll_visible(animate=False)
        self.query_one(Detail).show(chosen)

    def choices(self) -> list[Step]:
        return self.board.steps
