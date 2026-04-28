"""MCP (Model Context Protocol) helpers — minimal teaching wrappers.

This file gives a thin layer over the official `mcp` Python SDK so learners
can focus on concepts without wrestling with low-level transport / async.

MCP primitives covered:
- Tools: functions the model can call
- Resources: data the model can read (file / DB)
- Prompts: reusable prompt templates

If the `mcp` package isn't installed, fall back to a pure-Python EduMockServer
that simulates the protocol so learners can still run the notebook.
"""
from __future__ import annotations
import asyncio
import inspect
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

# ── Try real MCP SDK; fall back to mock if not installed ────────
try:
    from mcp.server.fastmcp import FastMCP
    MCP_AVAILABLE = True
except ImportError:
    FastMCP = None  # type: ignore
    MCP_AVAILABLE = False


# ============================================================
# Tool / Resource / Prompt — language-level dataclasses (transport-agnostic)
# ============================================================
@dataclass
class ToolDef:
    name: str
    description: str
    parameters: dict   # JSON schema
    func: Callable

    def call(self, **kwargs) -> Any:
        return self.func(**kwargs)


@dataclass
class ResourceDef:
    uri: str           # e.g. "file:///docs/readme.md"
    name: str
    mime_type: str
    reader: Callable   # () -> str

    def read(self) -> str:
        return self.reader()


@dataclass
class PromptDef:
    name: str
    description: str
    arguments: list[dict]   # [{"name": "topic", "description": "...", "required": True}]
    template: Callable      # (args dict) -> str

    def render(self, **kwargs) -> str:
        return self.template(**kwargs)


# ============================================================
# EduMCPServer — pure-Python teaching server (works without mcp SDK)
# ============================================================
class EduMCPServer:
    """Minimal MCP-style server. Mimics the real SDK's API for teaching."""

    def __init__(self, name: str = "edu-server"):
        self.name = name
        self._tools: dict[str, ToolDef] = {}
        self._resources: dict[str, ResourceDef] = {}
        self._prompts: dict[str, PromptDef] = {}
        self._auth_check: Optional[Callable[[str, str], bool]] = None  # (user_id, action) -> bool

    def add_tool(self, tool: ToolDef) -> None:
        self._tools[tool.name] = tool

    def add_resource(self, resource: ResourceDef) -> None:
        self._resources[resource.uri] = resource

    def add_prompt(self, prompt: PromptDef) -> None:
        self._prompts[prompt.name] = prompt

    def set_auth_check(self, fn: Callable[[str, str], bool]) -> None:
        """Optional: gate all tool calls by (user_id, action_name) → bool."""
        self._auth_check = fn

    def list_tools(self, user_id: Optional[str] = None) -> list[dict]:
        items = []
        for tool in self._tools.values():
            if self._auth_check and user_id and not self._auth_check(user_id, tool.name):
                continue
            items.append({"name": tool.name, "description": tool.description, "parameters": tool.parameters})
        return items

    def list_resources(self) -> list[dict]:
        return [{"uri": r.uri, "name": r.name, "mime_type": r.mime_type} for r in self._resources.values()]

    def list_prompts(self) -> list[dict]:
        return [{"name": p.name, "description": p.description, "arguments": p.arguments} for p in self._prompts.values()]

    def call_tool(self, name: str, arguments: dict, user_id: Optional[str] = None) -> Any:
        if name not in self._tools:
            raise ValueError(f"Unknown tool: {name}")
        if self._auth_check and user_id and not self._auth_check(user_id, name):
            raise PermissionError(f"User {user_id} not authorized for {name}")
        # Schema validation (light)
        tool = self._tools[name]
        required = tool.parameters.get("required", [])
        for r in required:
            if r not in arguments:
                raise ValueError(f"Missing required parameter: {r}")
        return tool.call(**arguments)

    def read_resource(self, uri: str) -> str:
        if uri not in self._resources:
            raise ValueError(f"Unknown resource: {uri}")
        return self._resources[uri].read()

    def get_prompt(self, name: str, **arguments) -> str:
        if name not in self._prompts:
            raise ValueError(f"Unknown prompt: {name}")
        return self._prompts[name].render(**arguments)


# ============================================================
# EduMCPClient — corresponds to EduMCPServer
# ============================================================
class EduMCPClient:
    """Client that talks to one or more EduMCPServer instances."""

    def __init__(self, user_id: Optional[str] = None):
        self.user_id = user_id
        self.servers: dict[str, EduMCPServer] = {}

    def connect(self, server: EduMCPServer) -> None:
        self.servers[server.name] = server

    def list_all_tools(self) -> list[dict]:
        out = []
        for srv_name, srv in self.servers.items():
            for t in srv.list_tools(user_id=self.user_id):
                out.append({**t, "server": srv_name})
        return out

    def call(self, server_name: str, tool_name: str, **arguments) -> Any:
        if server_name not in self.servers:
            raise ValueError(f"Not connected to server: {server_name}")
        return self.servers[server_name].call_tool(tool_name, arguments, user_id=self.user_id)

    def read(self, server_name: str, uri: str) -> str:
        return self.servers[server_name].read_resource(uri)


# ============================================================
# Convenience: turn a Python function into a ToolDef using introspection
# ============================================================
def tool_from_function(func: Callable, name: Optional[str] = None, description: Optional[str] = None) -> ToolDef:
    """Auto-generate a ToolDef from a regular Python function via signature inspection."""
    sig = inspect.signature(func)
    name = name or func.__name__
    description = description or (func.__doc__ or "").strip().split("\n")[0]
    properties = {}
    required = []
    for pname, param in sig.parameters.items():
        # Naive type → JSON schema
        ann = param.annotation
        if ann is int:
            ptype = "integer"
        elif ann is float:
            ptype = "number"
        elif ann is bool:
            ptype = "boolean"
        else:
            ptype = "string"
        properties[pname] = {"type": ptype}
        if param.default is inspect.Parameter.empty:
            required.append(pname)
    return ToolDef(
        name=name,
        description=description,
        parameters={"type": "object", "properties": properties, "required": required},
        func=func,
    )
