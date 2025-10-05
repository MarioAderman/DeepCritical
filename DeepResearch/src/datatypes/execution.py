"""
Execution-related data types for DeepCritical's workflow orchestration.

This module defines data structures for workflow execution including
workflow steps, DAGs, execution contexts, and execution history.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, TYPE_CHECKING

if TYPE_CHECKING:
    from ..utils.execution_history import ExecutionHistory


@dataclass
class WorkflowStep:
    """A single step in a computational workflow."""

    tool: str
    parameters: Dict[str, Any]
    inputs: Dict[str, str]  # Maps input names to data sources
    outputs: Dict[str, str]  # Maps output names to data destinations
    success_criteria: Dict[str, Any]
    retry_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowDAG:
    """Directed Acyclic Graph representing a computational workflow."""

    steps: List[WorkflowStep]
    dependencies: Dict[str, List[str]]  # Maps step names to their dependencies
    execution_order: List[str]  # Topological sort of step names
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionContext:
    """Context for workflow execution."""

    workflow: WorkflowDAG
    history: "ExecutionHistory"
    data_bag: Dict[str, Any] = field(default_factory=dict)
    current_step: int = 0
    max_retries: int = 3
    manual_confirmation: bool = False
    adaptive_replanning: bool = True
