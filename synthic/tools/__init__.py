"""Built-in tools of the engine and their Anthropic definitions."""

from __future__ import annotations

from typing import Any

from ..types import EnvironmentSettings, Tool
from .edit import EditTool
from .jobs import JobManager, JobTool
from .sh import ShTool


def to_definition(tool: Tool) -> dict[str, Any]:
    """The Anthropic tool definition the model is shown."""
    return {"name": tool.name, "description": tool.description, "input_schema": tool.input_schema}


def builtin_tools(settings: EnvironmentSettings, jobs: JobManager) -> dict[str, Tool]:
    """``sh``, ``edit`` and ``job``, keyed by name; ``sh`` and ``job`` share ``jobs``."""
    tools: list[Tool] = [ShTool(settings, jobs), EditTool(), JobTool(jobs)]
    return {tool.name: tool for tool in tools}
