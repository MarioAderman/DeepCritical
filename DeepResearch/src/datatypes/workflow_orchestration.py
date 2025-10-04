"""
Workflow orchestration data types for DeepCritical's workflow-of-workflows architecture.

This module defines Pydantic models for orchestrating multiple specialized workflows
including RAG, bioinformatics, search, and multi-agent systems.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from pydantic import BaseModel, Field, validator
import uuid

if TYPE_CHECKING:
    pass


class WorkflowType(str, Enum):
    """Types of workflows that can be orchestrated."""

    PRIMARY_REACT = "primary_react"
    RAG_WORKFLOW = "rag_workflow"
    BIOINFORMATICS_WORKFLOW = "bioinformatics_workflow"
    SEARCH_WORKFLOW = "search_workflow"
    MULTI_AGENT_WORKFLOW = "multi_agent_workflow"
    HYPOTHESIS_GENERATION = "hypothesis_generation"
    HYPOTHESIS_TESTING = "hypothesis_testing"
    REASONING_WORKFLOW = "reasoning_workflow"
    CODE_EXECUTION_WORKFLOW = "code_execution_workflow"
    EVALUATION_WORKFLOW = "evaluation_workflow"
    NESTED_REACT_LOOP = "nested_react_loop"
    GROUP_CHAT_WORKFLOW = "group_chat_workflow"
    SEQUENTIAL_WORKFLOW = "sequential_workflow"
    SUBGRAPH_WORKFLOW = "subgraph_workflow"


class WorkflowStatus(str, Enum):
    """Status of workflow execution."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class AgentRole(str, Enum):
    """Roles for agents in multi-agent systems."""

    COORDINATOR = "coordinator"
    EXECUTOR = "executor"
    EVALUATOR = "evaluator"
    JUDGE = "judge"
    REVIEWER = "reviewer"
    LINTER = "linter"
    CODE_EXECUTOR = "code_executor"
    HYPOTHESIS_GENERATOR = "hypothesis_generator"
    HYPOTHESIS_TESTER = "hypothesis_tester"
    REASONING_AGENT = "reasoning_agent"
    SEARCH_AGENT = "search_agent"
    RAG_AGENT = "rag_agent"
    BIOINFORMATICS_AGENT = "bioinformatics_agent"
    ORCHESTRATOR_AGENT = "orchestrator_agent"
    SUBGRAPH_AGENT = "subgraph_agent"
    GROUP_CHAT_AGENT = "group_chat_agent"
    SEQUENTIAL_AGENT = "sequential_agent"


class DataLoaderType(str, Enum):
    """Types of data loaders for RAG workflows."""

    DOCUMENT_LOADER = "document_loader"
    WEB_SCRAPER = "web_scraper"
    DATABASE_LOADER = "database_loader"
    API_LOADER = "api_loader"
    FILE_LOADER = "file_loader"
    BIOINFORMATICS_LOADER = "bioinformatics_loader"
    SCIENTIFIC_PAPER_LOADER = "scientific_paper_loader"
    GENE_ONTOLOGY_LOADER = "gene_ontology_loader"
    PUBMED_LOADER = "pubmed_loader"
    GEO_LOADER = "geo_loader"


class WorkflowConfig(BaseModel):
    """Configuration for a specific workflow."""

    workflow_type: WorkflowType = Field(..., description="Type of workflow")
    name: str = Field(..., description="Workflow name")
    enabled: bool = Field(True, description="Whether workflow is enabled")
    priority: int = Field(0, description="Execution priority (higher = more priority)")
    max_retries: int = Field(3, description="Maximum retry attempts")
    timeout: Optional[float] = Field(None, description="Timeout in seconds")
    dependencies: List[str] = Field(
        default_factory=list, description="Dependent workflow names"
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict, description="Workflow-specific parameters"
    )
    output_format: str = Field("default", description="Expected output format")

    class Config:
        json_schema_extra = {
            "example": {
                "workflow_type": "rag_workflow",
                "name": "scientific_papers_rag",
                "enabled": True,
                "priority": 1,
                "max_retries": 3,
                "parameters": {
                    "collection_name": "scientific_papers",
                    "chunk_size": 1000,
                    "top_k": 5,
                },
            }
        }


class AgentConfig(BaseModel):
    """Configuration for an agent in multi-agent systems."""

    agent_id: str = Field(..., description="Unique agent identifier")
    role: AgentRole = Field(..., description="Agent role")
    model_name: str = Field("anthropic:claude-sonnet-4-0", description="Model to use")
    system_prompt: Optional[str] = Field(None, description="Custom system prompt")
    tools: List[str] = Field(default_factory=list, description="Available tools")
    max_iterations: int = Field(10, description="Maximum iterations")
    temperature: float = Field(0.7, description="Model temperature")
    enabled: bool = Field(True, description="Whether agent is enabled")

    class Config:
        json_schema_extra = {
            "example": {
                "agent_id": "hypothesis_generator_001",
                "role": "hypothesis_generator",
                "model_name": "anthropic:claude-sonnet-4-0",
                "tools": ["web_search", "rag_query", "reasoning"],
                "max_iterations": 5,
            }
        }


class DataLoaderConfig(BaseModel):
    """Configuration for data loaders in RAG workflows."""

    loader_type: DataLoaderType = Field(..., description="Type of data loader")
    name: str = Field(..., description="Loader name")
    enabled: bool = Field(True, description="Whether loader is enabled")
    parameters: Dict[str, Any] = Field(
        default_factory=dict, description="Loader parameters"
    )
    output_collection: str = Field(..., description="Output collection name")
    chunk_size: int = Field(1000, description="Chunk size for documents")
    chunk_overlap: int = Field(200, description="Chunk overlap")

    class Config:
        json_schema_extra = {
            "example": {
                "loader_type": "scientific_paper_loader",
                "name": "pubmed_loader",
                "parameters": {
                    "query": "machine learning",
                    "max_papers": 100,
                    "include_abstracts": True,
                },
                "output_collection": "scientific_papers",
            }
        }


class WorkflowExecution(BaseModel):
    """Execution context for a workflow."""

    execution_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()), description="Unique execution ID"
    )
    workflow_config: WorkflowConfig = Field(..., description="Workflow configuration")
    status: WorkflowStatus = Field(WorkflowStatus.PENDING, description="Current status")
    start_time: Optional[datetime] = Field(None, description="Start time")
    end_time: Optional[datetime] = Field(None, description="End time")
    input_data: Dict[str, Any] = Field(default_factory=dict, description="Input data")
    output_data: Dict[str, Any] = Field(default_factory=dict, description="Output data")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    retry_count: int = Field(0, description="Number of retries attempted")
    parent_execution_id: Optional[str] = Field(None, description="Parent execution ID")
    child_execution_ids: List[str] = Field(
        default_factory=list, description="Child execution IDs"
    )

    @property
    def duration(self) -> Optional[float]:
        """Get execution duration in seconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None

    @property
    def is_completed(self) -> bool:
        """Check if execution is completed."""
        return self.status == WorkflowStatus.COMPLETED

    @property
    def is_failed(self) -> bool:
        """Check if execution failed."""
        return self.status == WorkflowStatus.FAILED

    class Config:
        json_schema_extra = {
            "example": {
                "execution_id": "exec_123",
                "workflow_config": {},
                "status": "running",
                "input_data": {"query": "What is machine learning?"},
                "output_data": {},
            }
        }


class MultiAgentSystemConfig(BaseModel):
    """Configuration for multi-agent systems."""

    system_id: str = Field(..., description="System identifier")
    name: str = Field(..., description="System name")
    agents: List[AgentConfig] = Field(..., description="Agent configurations")
    coordination_strategy: str = Field(
        "sequential", description="Coordination strategy"
    )
    communication_protocol: str = Field("direct", description="Communication protocol")
    max_rounds: int = Field(10, description="Maximum coordination rounds")
    consensus_threshold: float = Field(0.8, description="Consensus threshold")
    enabled: bool = Field(True, description="Whether system is enabled")

    class Config:
        json_schema_extra = {
            "example": {
                "system_id": "hypothesis_system_001",
                "name": "Hypothesis Generation and Testing System",
                "agents": [],
                "coordination_strategy": "collaborative",
                "max_rounds": 5,
            }
        }


class JudgeConfig(BaseModel):
    """Configuration for LLM judges."""

    judge_id: str = Field(..., description="Judge identifier")
    name: str = Field(..., description="Judge name")
    model_name: str = Field("anthropic:claude-sonnet-4-0", description="Model to use")
    evaluation_criteria: List[str] = Field(..., description="Evaluation criteria")
    scoring_scale: str = Field("1-10", description="Scoring scale")
    enabled: bool = Field(True, description="Whether judge is enabled")

    class Config:
        json_schema_extra = {
            "example": {
                "judge_id": "quality_judge_001",
                "name": "Quality Assessment Judge",
                "evaluation_criteria": ["accuracy", "completeness", "clarity"],
                "scoring_scale": "1-10",
            }
        }


class WorkflowOrchestrationConfig(BaseModel):
    """Main configuration for workflow orchestration."""

    primary_workflow: WorkflowConfig = Field(
        ..., description="Primary REACT workflow config"
    )
    sub_workflows: List[WorkflowConfig] = Field(
        default_factory=list, description="Sub-workflow configs"
    )
    data_loaders: List[DataLoaderConfig] = Field(
        default_factory=list, description="Data loader configs"
    )
    multi_agent_systems: List[MultiAgentSystemConfig] = Field(
        default_factory=list, description="Multi-agent system configs"
    )
    judges: List[JudgeConfig] = Field(default_factory=list, description="Judge configs")
    execution_strategy: str = Field(
        "parallel", description="Execution strategy (parallel, sequential, hybrid)"
    )
    max_concurrent_workflows: int = Field(5, description="Maximum concurrent workflows")
    global_timeout: Optional[float] = Field(
        None, description="Global timeout in seconds"
    )
    enable_monitoring: bool = Field(True, description="Enable execution monitoring")
    enable_caching: bool = Field(True, description="Enable result caching")

    @validator("sub_workflows")
    def validate_sub_workflows(cls, v):
        """Validate sub-workflow configurations."""
        names = [w.name for w in v]
        if len(names) != len(set(names)):
            raise ValueError("Sub-workflow names must be unique")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "primary_workflow": {
                    "workflow_type": "primary_react",
                    "name": "main_research_workflow",
                    "enabled": True,
                },
                "sub_workflows": [],
                "data_loaders": [],
                "multi_agent_systems": [],
                "judges": [],
            }
        }


class WorkflowResult(BaseModel):
    """Result from workflow execution."""

    execution_id: str = Field(..., description="Execution ID")
    workflow_name: str = Field(..., description="Workflow name")
    status: WorkflowStatus = Field(..., description="Final status")
    output_data: Dict[str, Any] = Field(..., description="Output data")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Execution metadata"
    )
    quality_score: Optional[float] = Field(
        None, description="Quality score from judges"
    )
    execution_time: float = Field(..., description="Execution time in seconds")
    error_details: Optional[Dict[str, Any]] = Field(
        None, description="Error details if failed"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "execution_id": "exec_123",
                "workflow_name": "rag_workflow",
                "status": "completed",
                "output_data": {"answer": "Machine learning is..."},
                "quality_score": 8.5,
                "execution_time": 15.2,
            }
        }


class HypothesisDataset(BaseModel):
    """Dataset of hypotheses generated by workflows."""

    dataset_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()), description="Dataset ID"
    )
    name: str = Field(..., description="Dataset name")
    description: str = Field(..., description="Dataset description")
    hypotheses: List[Dict[str, Any]] = Field(..., description="Generated hypotheses")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Dataset metadata"
    )
    creation_date: datetime = Field(
        default_factory=datetime.now, description="Creation date"
    )
    source_workflows: List[str] = Field(
        default_factory=list, description="Source workflow names"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "dataset_id": "hyp_001",
                "name": "ML Research Hypotheses",
                "description": "Hypotheses about machine learning applications",
                "hypotheses": [
                    {
                        "hypothesis": "Deep learning improves protein structure prediction",
                        "confidence": 0.85,
                        "evidence": ["AlphaFold2 results", "ESMFold improvements"],
                    }
                ],
            }
        }


class HypothesisTestingEnvironment(BaseModel):
    """Environment for testing hypotheses."""

    environment_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()), description="Environment ID"
    )
    name: str = Field(..., description="Environment name")
    hypothesis: Dict[str, Any] = Field(..., description="Hypothesis to test")
    test_configuration: Dict[str, Any] = Field(..., description="Test configuration")
    expected_outcomes: List[str] = Field(..., description="Expected outcomes")
    success_criteria: Dict[str, Any] = Field(..., description="Success criteria")
    test_data: Dict[str, Any] = Field(default_factory=dict, description="Test data")
    results: Optional[Dict[str, Any]] = Field(None, description="Test results")
    status: WorkflowStatus = Field(WorkflowStatus.PENDING, description="Test status")

    class Config:
        json_schema_extra = {
            "example": {
                "environment_id": "test_001",
                "name": "Protein Structure Prediction Test",
                "hypothesis": {
                    "hypothesis": "Deep learning improves protein structure prediction",
                    "confidence": 0.85,
                },
                "test_configuration": {
                    "test_proteins": ["P04637", "P53"],
                    "metrics": ["RMSD", "GDT_TS"],
                },
            }
        }


class ReasoningResult(BaseModel):
    """Result from reasoning workflows."""

    reasoning_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()), description="Reasoning ID"
    )
    question: str = Field(..., description="Reasoning question")
    answer: str = Field(..., description="Reasoning answer")
    reasoning_chain: List[str] = Field(..., description="Reasoning steps")
    confidence: float = Field(..., description="Confidence score")
    supporting_evidence: List[Dict[str, Any]] = Field(
        ..., description="Supporting evidence"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Reasoning metadata"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "reasoning_id": "reason_001",
                "question": "Why does AlphaFold2 outperform traditional methods?",
                "answer": "AlphaFold2 uses deep learning to predict protein structures...",
                "reasoning_chain": [
                    "Analyze traditional methods limitations",
                    "Identify deep learning advantages",
                    "Compare performance metrics",
                ],
                "confidence": 0.92,
            }
        }


class WorkflowComposition(BaseModel):
    """Dynamic composition of workflows based on user input and config."""

    composition_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()), description="Composition ID"
    )
    user_input: str = Field(..., description="User input/query")
    selected_workflows: List[str] = Field(..., description="Selected workflow names")
    workflow_dependencies: Dict[str, List[str]] = Field(
        default_factory=dict, description="Workflow dependencies"
    )
    execution_order: List[str] = Field(..., description="Execution order")
    expected_outputs: Dict[str, str] = Field(
        default_factory=dict, description="Expected outputs by workflow"
    )
    composition_strategy: str = Field("adaptive", description="Composition strategy")

    class Config:
        json_schema_extra = {
            "example": {
                "composition_id": "comp_001",
                "user_input": "Analyze protein-protein interactions in cancer",
                "selected_workflows": [
                    "bioinformatics_workflow",
                    "rag_workflow",
                    "reasoning_workflow",
                ],
                "execution_order": [
                    "rag_workflow",
                    "bioinformatics_workflow",
                    "reasoning_workflow",
                ],
            }
        }


class OrchestrationState(BaseModel):
    """State of the workflow orchestration system."""

    state_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()), description="State ID"
    )
    active_executions: List[WorkflowExecution] = Field(
        default_factory=list, description="Active executions"
    )
    completed_executions: List[WorkflowResult] = Field(
        default_factory=list, description="Completed executions"
    )
    pending_workflows: List[WorkflowConfig] = Field(
        default_factory=list, description="Pending workflows"
    )
    current_composition: Optional[WorkflowComposition] = Field(
        None, description="Current composition"
    )
    system_metrics: Dict[str, Any] = Field(
        default_factory=dict, description="System metrics"
    )
    last_updated: datetime = Field(
        default_factory=datetime.now, description="Last update time"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "state_id": "state_001",
                "active_executions": [],
                "completed_executions": [],
                "system_metrics": {
                    "total_executions": 0,
                    "success_rate": 0.0,
                    "average_execution_time": 0.0,
                },
            }
        }


class MultiStateMachineMode(str, Enum):
    """Modes for multi-statemachine coordination."""

    GROUP_CHAT = "group_chat"
    SEQUENTIAL = "sequential"
    HIERARCHICAL = "hierarchical"
    PARALLEL = "parallel"
    PIPELINE = "pipeline"
    CONSENSUS = "consensus"


class SubgraphType(str, Enum):
    """Types of subgraphs that can be spawned."""

    RAG_SUBGRAPH = "rag_subgraph"
    SEARCH_SUBGRAPH = "search_subgraph"
    CODE_SUBGRAPH = "code_subgraph"
    BIOINFORMATICS_SUBGRAPH = "bioinformatics_subgraph"
    REASONING_SUBGRAPH = "reasoning_subgraph"
    EVALUATION_SUBGRAPH = "evaluation_subgraph"
    CUSTOM_SUBGRAPH = "custom_subgraph"


class LossFunctionType(str, Enum):
    """Types of loss functions for end conditions."""

    CONFIDENCE_THRESHOLD = "confidence_threshold"
    QUALITY_SCORE = "quality_score"
    CONSENSUS_LEVEL = "consensus_level"
    ITERATION_LIMIT = "iteration_limit"
    TIME_LIMIT = "time_limit"
    CUSTOM_LOSS = "custom_loss"


class BreakCondition(BaseModel):
    """Condition for breaking out of REACT loops."""

    condition_type: LossFunctionType = Field(..., description="Type of break condition")
    threshold: float = Field(..., description="Threshold value for the condition")
    operator: str = Field(">=", description="Comparison operator (>=, <=, ==, !=)")
    enabled: bool = Field(True, description="Whether this condition is enabled")
    custom_function: Optional[str] = Field(
        None, description="Custom function for custom_loss type"
    )


class NestedReactConfig(BaseModel):
    """Configuration for nested REACT loops."""

    loop_id: str = Field(..., description="Unique identifier for the nested loop")
    parent_loop_id: Optional[str] = Field(None, description="Parent loop ID if nested")
    max_iterations: int = Field(10, description="Maximum iterations for this loop")
    break_conditions: List[BreakCondition] = Field(
        default_factory=list, description="Break conditions"
    )
    state_machine_mode: MultiStateMachineMode = Field(
        MultiStateMachineMode.GROUP_CHAT, description="State machine mode"
    )
    subgraphs: List[SubgraphType] = Field(
        default_factory=list, description="Subgraphs to include"
    )
    agent_roles: List[AgentRole] = Field(
        default_factory=list, description="Agent roles for this loop"
    )
    tools: List[str] = Field(
        default_factory=list, description="Tools available to agents"
    )
    priority: int = Field(0, description="Execution priority")


class AgentOrchestratorConfig(BaseModel):
    """Configuration for agent-based orchestrators."""

    orchestrator_id: str = Field(..., description="Orchestrator identifier")
    agent_role: AgentRole = Field(
        AgentRole.ORCHESTRATOR_AGENT, description="Role of the orchestrator agent"
    )
    model_name: str = Field(
        "anthropic:claude-sonnet-4-0", description="Model for the orchestrator"
    )
    break_conditions: List[BreakCondition] = Field(
        default_factory=list, description="Break conditions"
    )
    max_nested_loops: int = Field(5, description="Maximum number of nested loops")
    coordination_strategy: str = Field(
        "collaborative", description="Coordination strategy"
    )
    can_spawn_subgraphs: bool = Field(
        True, description="Whether this orchestrator can spawn subgraphs"
    )
    can_spawn_agents: bool = Field(
        True, description="Whether this orchestrator can spawn agents"
    )


class SubgraphConfig(BaseModel):
    """Configuration for subgraphs."""

    subgraph_id: str = Field(..., description="Subgraph identifier")
    subgraph_type: SubgraphType = Field(..., description="Type of subgraph")
    state_machine_path: str = Field(
        ..., description="Path to state machine implementation"
    )
    entry_node: str = Field(..., description="Entry node for the subgraph")
    exit_node: str = Field(..., description="Exit node for the subgraph")
    parameters: Dict[str, Any] = Field(
        default_factory=dict, description="Subgraph parameters"
    )
    tools: List[str] = Field(
        default_factory=list, description="Tools available in subgraph"
    )
    max_execution_time: float = Field(
        300.0, description="Maximum execution time in seconds"
    )
    enabled: bool = Field(True, description="Whether this subgraph is enabled")


class AppMode(str, Enum):
    """Modes for app.py execution."""

    SINGLE_REACT = "single_react"
    MULTI_LEVEL_REACT = "multi_level_react"
    NESTED_ORCHESTRATION = "nested_orchestration"
    SUBGRAPH_COORDINATION = "subgraph_coordination"
    LOSS_DRIVEN = "loss_driven"
    CUSTOM_MODE = "custom_mode"


class AppConfiguration(BaseModel):
    """Main configuration for app.py modes."""

    mode: AppMode = Field(AppMode.SINGLE_REACT, description="Execution mode")
    primary_orchestrator: AgentOrchestratorConfig = Field(
        ..., description="Primary orchestrator config"
    )
    nested_react_configs: List[NestedReactConfig] = Field(
        default_factory=list, description="Nested REACT configurations"
    )
    subgraph_configs: List[SubgraphConfig] = Field(
        default_factory=list, description="Subgraph configurations"
    )
    loss_functions: List[BreakCondition] = Field(
        default_factory=list, description="Loss functions for end conditions"
    )
    global_break_conditions: List[BreakCondition] = Field(
        default_factory=list, description="Global break conditions"
    )
    execution_strategy: str = Field(
        "adaptive", description="Overall execution strategy"
    )
    max_total_iterations: int = Field(
        100, description="Maximum total iterations across all loops"
    )
    max_total_time: float = Field(
        3600.0, description="Maximum total execution time in seconds"
    )
