"""Built-in tools of the engine and their Anthropic definitions."""

from __future__ import annotations

from typing import Any

from ..types import EnvironmentSettings, Tool
from .edit import EditTool
from .files import GrepTool, ReadTool
from .jobs import JobManager, JobTool
from .sh import ShTool
from .view import ViewTool


def to_definition(tool: Tool) -> dict[str, Any]:
    """The Anthropic tool definition the model is shown."""
    return {"name": tool.name, "description": tool.description, "input_schema": tool.input_schema}


def builtin_tools(settings: EnvironmentSettings, jobs: JobManager) -> dict[str, Tool]:
    """``sh``, ``edit``, ``job``, ``view``, ``read`` and ``grep``, keyed by name; ``sh`` and ``job`` share ``jobs``."""
    tools: list[Tool] = [ShTool(settings, jobs), EditTool(), JobTool(jobs), ViewTool(), ReadTool(), GrepTool()]
    return {tool.name: tool for tool in tools}
