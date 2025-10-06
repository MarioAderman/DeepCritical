from __future__ import annotations


# Import the tool runners and utilities from utils
from ..utils.pydantic_ai_utils import (
    get_pydantic_ai_config as _get_cfg,
    build_builtin_tools as _build_builtin_tools,
    build_toolsets as _build_toolsets,
    build_agent as _build_agent,
    run_agent_sync as _run_sync,
)

# Registry overrides and additions
from .base import registry
from ..datatypes.pydantic_ai_tools import CodeExecBuiltinRunner, UrlContextBuiltinRunner

registry.register("pyd_code_exec", lambda: CodeExecBuiltinRunner())
registry.register("pyd_url_context", lambda: UrlContextBuiltinRunner())

# Export the functions for external use
__all__ = [
    "_build_builtin_tools",
    "_build_toolsets",
    "_build_agent",
    "_run_sync",
    "_get_cfg",
]
