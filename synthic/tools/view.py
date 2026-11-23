"""The ``view`` tool: show the model an image or a PDF from disk, as it sees an attachment.

The file goes back inside the ``tool_result``, as an ``image`` or a
``document`` block, which the docs allow there:
https://platform.claude.com/docs/en/agents-and-tools/tool-use/handle-tool-calls
A file the API would refuse (type, size, dimensions, encryption; see
``media``) is not sent: the result says why, so the model can shrink or
convert it with its other tools. There is no resizing here, since the engine
uses the standard library only.
"""

from __future__ import annotations

import asyncio
from typing import Any

from ..media import MediaError, load
from ..types import DocumentData, ImageData, ToolContext, ToolOutput
from .edit import resolve


class ViewTool:
    name = "view"
    description = (
        "Look at an image (JPEG, PNG, GIF, WebP) or a PDF on disk: it is shown to you in the result. A file the "
        "model API cannot take (too large, other type, encrypted PDF) is refused with the reason. Relative paths "
        "are from the working folder."
    )
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    }

    def resource(self, tool_input: dict[str, Any], ctx: ToolContext) -> str | None:
        return None

    async def run(self, tool_input: dict[str, Any], ctx: ToolContext) -> ToolOutput:
        path = tool_input.get("path")
        if not isinstance(path, str) or not path:
            return ToolOutput("`path` must be a non-empty string.", is_error=True)
        target = resolve(path, ctx.cwd)
        try:
            attachment = await asyncio.to_thread(load, target)
        except MediaError as error:
            return ToolOutput(str(error), is_error=True)
        source = attachment.block["source"]
        meta = {"media": attachment.name}
        text = f"{attachment.name} ({attachment.note})"
        if attachment.block["type"] == "document":
            return ToolOutput(text, documents=[DocumentData(source["media_type"], source["data"])], meta=meta)
        return ToolOutput(text, images=[ImageData(source["media_type"], source["data"])], meta=meta)
