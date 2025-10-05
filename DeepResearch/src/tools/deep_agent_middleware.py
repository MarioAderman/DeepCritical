"""
DeepAgent Middleware - Pydantic AI middleware for DeepAgent operations.

This module implements middleware components for planning, filesystem operations,
and subagent orchestration using Pydantic AI patterns that align with
DeepCritical's architecture.
"""

from __future__ import annotations


# Import existing DeepCritical types

# Import middleware types from datatypes module
from ..datatypes.middleware import (
    MiddlewareConfig,
    MiddlewareResult,
    BaseMiddleware,
    PlanningMiddleware,
    FilesystemMiddleware,
    SubAgentMiddleware,
    SummarizationMiddleware,
    PromptCachingMiddleware,
    MiddlewarePipeline,
    create_planning_middleware,
    create_filesystem_middleware,
    create_subagent_middleware,
    create_summarization_middleware,
    create_prompt_caching_middleware,
    create_default_middleware_pipeline,
)


# Export all middleware components
__all__ = [
    # Base classes
    "BaseMiddleware",
    "MiddlewarePipeline",
    # Middleware implementations
    "PlanningMiddleware",
    "FilesystemMiddleware",
    "SubAgentMiddleware",
    "SummarizationMiddleware",
    "PromptCachingMiddleware",
    # Configuration and results
    "MiddlewareConfig",
    "MiddlewareResult",
    # Factory functions
    "create_planning_middleware",
    "create_filesystem_middleware",
    "create_subagent_middleware",
    "create_summarization_middleware",
    "create_prompt_caching_middleware",
    "create_default_middleware_pipeline",
]
