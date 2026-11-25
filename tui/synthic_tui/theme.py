"""The look of Synthic in a terminal: the Space and Earth themes, and how figures are written.

The colours are the tokens of the Synthic design system. The interface has no accent colour: a colour of
the spectrum always means a state, and everything else is ink on one plane.
"""

from __future__ import annotations

from textual.app import App
from textual.theme import Theme

SPACE = {
    "ground": "#141413",
    "ground-raised": "#1a1a18",
    "ground-sunken": "#10100f",
    "ground-hover": "#1f1f1d",
    "ground-selected": "#24241f",
    "hairline": "#222220",
    "line-control": "#6a6964",
    "ink": "#f3f1ec",
    "ink-muted": "#a3a19a",
    "ink-faint": "#6e6c66",
    "focus": "#c9c6bf",
    "up": "#53c556",
    "down": "#e2735a",
    "moss": "#53c556",
    "moss-tile": "#1e301e",
    "clay": "#d97757",
    "clay-tile": "#34241e",
    "sky": "#4a90e8",
    "sky-tile": "#1d2835",
    "sun": "#e0a53a",
    "sun-tile": "#352b19",
    "nebula": "#b08ae6",
    "nebula-tile": "#2d2735",
    "tide": "#3fb8a9",
    "tide-tile": "#1b2e2b",
    "neutral-tile": "#262624",
}

EARTH = {
    "ground": "#e8e8e5",
    "ground-raised": "#f0f0ee",
    "ground-sunken": "#e0e0dc",
    "ground-hover": "#dfdfdb",
    "ground-selected": "#d8d8d4",
    "hairline": "#d6d6d2",
    "line-control": "#7d7d79",
    "ink": "#141414",
    "ink-muted": "#3d3d3b",
    "ink-faint": "#7a7a76",
    "focus": "#141414",
    "up": "#035814",
    "down": "#7e2401",
    "moss": "#035814",
    "moss-tile": "#b1cdb5",
    "clay": "#7e2401",
    "clay-tile": "#d9bdaf",
    "sky": "#024884",
    "sky-tile": "#b1c8d9",
    "sun": "#624001",
    "sun-tile": "#d0c6af",
    "nebula": "#5d02a5",
    "nebula-tile": "#cfb1e4",
    "tide": "#01544d",
    "tide-tile": "#b0ccc7",
    "neutral-tile": "#d7d7d4",
}

THEMES = {"synthic-space": SPACE, "synthic-earth": EARTH}

STATES = {
    "working": ("●", "tide"),
    "running": ("●", "tide"),
    "waiting": ("◆", "clay"),
    "queued": ("○", "sky"),
    "review": ("●", "nebula"),
    "delivered": ("✓", "moss"),
    "replied": ("○", "ink-muted"),
    "idle": ("○", "ink-muted"),
}
"""The glyph and the colour of each state; any other state is a failure and reads in ``down``."""


def state_mark(state: str) -> tuple[str, str]:
    return STATES.get(state, ("✗", "down"))


def _theme(name: str, tokens: dict[str, str], dark: bool) -> Theme:
    return Theme(
        name=name,
        primary=tokens["ink-muted"],
        secondary=tokens["ink-muted"],
        accent=tokens["focus"],
        success=tokens["moss"],
        warning=tokens["sun"],
        error=tokens["down"],
        foreground=tokens["ink"],
        background=tokens["ground"],
        surface=tokens["ground"],
        panel=tokens["ground-raised"],
        dark=dark,
        variables={
            **{key: value for key, value in tokens.items() if key != "ground"},
            "block-cursor-text-style": "none",
            "block-cursor-background": tokens["ground-selected"],
            "block-cursor-foreground": tokens["ink"],
            "block-cursor-blurred-text-style": "none",
            "block-cursor-blurred-background": tokens["ground-selected"],
            "block-cursor-blurred-foreground": tokens["ink"],
            "block-hover-background": tokens["ground-hover"],
            "input-selection-background": tokens["ground-selected"],
            "input-cursor-background": tokens["ink"],
            "input-cursor-foreground": tokens["ground"],
            "border": tokens["focus"],
            "border-blurred": tokens["hairline"],
            "scrollbar": tokens["hairline"],
            "scrollbar-hover": tokens["ink-faint"],
            "scrollbar-active": tokens["ink-muted"],
            "scrollbar-background": tokens["ground"],
            "scrollbar-corner-color": tokens["ground"],
            "link-color": tokens["ink"],
        },
    )


def register(app: App) -> None:
    """Call before the stylesheet is parsed, so its ``$moss``, ``$ink-muted``... variables exist."""
    app.register_theme(_theme("synthic-space", SPACE, True))
    app.register_theme(_theme("synthic-earth", EARTH, False))


"""Figures, written as the design system asks: es-ES separators, the unit after the number."""


def number(value: int) -> str:
    return f"{value:,}".replace(",", ".")


def money(value: float) -> str:
    digits = 2 if value == 0 or value >= 0.01 else 4
    return f"{value:,.{digits}f}".replace(",", " ").replace(".", ",").replace(" ", ".") + " $"


def seconds(value: float) -> str:
    return f"{value:.1f}".replace(".", ",") + "s"


def elapsed(value: float) -> str:
    whole = max(0, int(value))
    hours, rest = divmod(whole, 3600)
    return f"{hours}:{rest // 60:02d}:{rest % 60:02d}" if hours else f"{rest // 60}:{rest % 60:02d}"
