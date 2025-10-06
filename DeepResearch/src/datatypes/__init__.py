"""
Data types for DeepCritical research workflows.

This module provides Pydantic models and data structures for various
research workflows including bioinformatics and RAG operations.
"""

from .bioinformatics import (
    EvidenceCode,
    GOTerm,
    GOAnnotation,
    PubMedPaper,
    GEOPlatform,
    GEOSeries,
    GeneExpressionProfile,
    DrugTarget,
    PerturbationProfile,
    ProteinStructure,
    ProteinInteraction,
    FusedDataset,
    ReasoningTask,
    DataFusionRequest,
)

from .rag import (
    SearchType,
    EmbeddingModelType,
    LLMModelType,
    VectorStoreType,
    Document,
    EmbeddingsConfig,
    VLLMConfig,
    VectorStoreConfig,
    RAGQuery,
    RAGResponse,
    RAGConfig,
    IntegratedSearchRequest,
    IntegratedSearchResponse,
    Embeddings,
    VectorStore,
    LLMProvider,
    RAGSystem,
    RAGWorkflowState,
)

# from .vllm_agent import (
#     VLLMAgentDependencies,
#     VLLMAgentConfig,
# )

from .vllm_integration import (
    VLLMEmbeddings,
    VLLMLLMProvider,
    VLLMServerConfig,
    VLLMEmbeddingServerConfig,
    VLLMDeployment,
    VLLMRAGSystem,
)

from .analytics import (
    AnalyticsRequest,
    AnalyticsResponse,
    AnalyticsDataRequest,
    AnalyticsDataResponse,
)

from .search_agent import (
    SearchAgentConfig,
    SearchQuery,
    SearchResult,
    SearchAgentDependencies,
)

from .code_sandbox import (
    CodeSandboxRunner,
    CodeSandboxTool,
)

from .workflow_orchestration import (
    OrchestratorDependencies,
    NestedLoopRequest,
    SubgraphSpawnRequest,
    BreakConditionCheck,
    OrchestrationResult,
)
from .workflow_patterns import (
    InteractionPattern,
    MessageType,
    AgentInteractionMode,
    InteractionMessage,
    AgentInteractionState,
    WorkflowOrchestrator,
    InteractionConfig,
    AgentInteractionRequest,
    AgentInteractionResponse,
    create_interaction_state,
    create_workflow_orchestrator,
)

from .orchestrator import (
    Orchestrator,
)

from .planner import (
    Planner,
)

from .execution import (
    WorkflowStep,
    WorkflowDAG,
    ExecutionContext,
)

from .research import (
    ResearchOutcome,
    StepResult,
)

from .middleware import (
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

from .deep_agent_tools import (
    WriteTodosRequest,
    WriteTodosResponse,
    ListFilesResponse,
    ReadFileRequest,
    ReadFileResponse,
    WriteFileRequest,
    WriteFileResponse,
    EditFileRequest,
    EditFileResponse,
    TaskRequestModel,
    TaskResponse,
)

from .deepsearch import (
    EvaluationType,
    ActionType,
    SearchTimeFilter,
    MAX_URLS_PER_STEP,
    MAX_QUERIES_PER_STEP,
    MAX_REFLECT_PER_STEP,
    WebSearchRequest,
    URLVisitResult,
    ReflectionQuestion,
    PromptPair,
    DeepSearchSchemas,
)

from .docker_sandbox_datatypes import (
    DockerSandboxConfig,
    DockerExecutionRequest,
    DockerExecutionResult,
    DockerSandboxEnvironment,
    DockerSandboxPolicies,
    DockerSandboxContainerInfo,
    DockerSandboxMetrics,
    DockerSandboxRequest,
    DockerSandboxResponse,
)


from .tool_specs import (
    ToolSpec,
    ToolCategory,
    ToolInput,
    ToolOutput,
)

from .tools import (
    ToolMetadata,
    ExecutionResult,
    ToolRunner,
    MockToolRunner,
)

from .pydantic_ai_tools import (
    WebSearchBuiltinRunner,
    CodeExecBuiltinRunner,
    UrlContextBuiltinRunner,
)

from .agents import (
    AgentType,
    AgentStatus,
    AgentDependencies,
    AgentResult,
    ExecutionHistory,
)

from .multi_agent import (
    CoordinationStrategy,
    CommunicationProtocol,
    AgentState,
    CoordinationMessage,
    CoordinationRound,
    CoordinationResult,
    MultiAgentCoordinatorConfig,
    AgentRole,
)

from .agent_framework_types import (
    # Content types
    TextSpanRegion,
    CitationAnnotation,
    BaseContent,
    TextContent,
    TextReasoningContent,
    DataContent,
    UriContent,
    ErrorContent,
    FunctionCallContent,
    FunctionResultContent,
    UsageContent,
    HostedFileContent,
    HostedVectorStoreContent,
    FunctionApprovalRequestContent,
    FunctionApprovalResponseContent,
    Content,
    prepare_function_call_results,
    # Usage types
    UsageDetails,
    # Enum types
    Role,
    FinishReason,
    ToolMode,
    # Chat types
    ChatMessage,
    ChatResponseUpdate,
    ChatResponse,
    # Agent types
    AgentRunResponseUpdate,
    AgentRunResponse,
    # Options types
    ChatOptions,
)

__all__ = [
    # Tool specification types
    "ToolSpec",
    "ToolCategory",
    "ToolInput",
    "ToolOutput",
    "ToolMetadata",
    # Bioinformatics types
    "EvidenceCode",
    "GOTerm",
    "GOAnnotation",
    "PubMedPaper",
    "GEOPlatform",
    "GEOSeries",
    "GeneExpressionProfile",
    "DrugTarget",
    "PerturbationProfile",
    "ProteinStructure",
    "ProteinInteraction",
    "FusedDataset",
    "ReasoningTask",
    "DataFusionRequest",
    # RAG types
    "SearchType",
    "EmbeddingModelType",
    "LLMModelType",
    "VectorStoreType",
    "Document",
    "SearchResult",
    "EmbeddingsConfig",
    "VLLMConfig",
    "VectorStoreConfig",
    "RAGQuery",
    "RAGResponse",
    "RAGConfig",
    "IntegratedSearchRequest",
    "IntegratedSearchResponse",
    "Embeddings",
    "VectorStore",
    "LLMProvider",
    "RAGSystem",
    "RAGWorkflowState",
    # VLLM agent types
    # "VLLMAgentDependencies",
    # "VLLMAgentConfig",
    # VLLM integration types
    "VLLMEmbeddings",
    "VLLMLLMProvider",
    "VLLMServerConfig",
    "VLLMEmbeddingServerConfig",
    "VLLMDeployment",
    "VLLMRAGSystem",
    # Analytics types
    "AnalyticsRequest",
    "AnalyticsResponse",
    "AnalyticsDataRequest",
    "AnalyticsDataResponse",
    # Search agent types
    "SearchAgentConfig",
    "SearchQuery",
    "SearchResult",
    "SearchAgentDependencies",
    # Code sandbox types
    "CodeSandboxRunner",
    "CodeSandboxTool",
    # Workflow orchestration types
    "OrchestratorDependencies",
    "NestedLoopRequest",
    "SubgraphSpawnRequest",
    "BreakConditionCheck",
    "OrchestrationResult",
    # Workflow pattern types
    "InteractionPattern",
    "MessageType",
    "AgentInteractionMode",
    "InteractionMessage",
    "AgentInteractionState",
    "WorkflowOrchestrator",
    "InteractionConfig",
    "AgentInteractionRequest",
    "AgentInteractionResponse",
    "create_interaction_state",
    "create_workflow_orchestrator",
    "WorkflowStep",
    "WorkflowDAG",
    "ExecutionContext",
    "Orchestrator",
    "Planner",
    # Research types
    "ResearchOutcome",
    "StepResult",
    # Middleware types
    "MiddlewareConfig",
    "MiddlewareResult",
    "BaseMiddleware",
    "PlanningMiddleware",
    "FilesystemMiddleware",
    "SubAgentMiddleware",
    "SummarizationMiddleware",
    "PromptCachingMiddleware",
    "MiddlewarePipeline",
    "create_planning_middleware",
    "create_filesystem_middleware",
    "create_subagent_middleware",
    "create_summarization_middleware",
    "create_prompt_caching_middleware",
    "create_default_middleware_pipeline",
    # DeepAgent tools types
    "WriteTodosRequest",
    "WriteTodosResponse",
    "ListFilesResponse",
    "ReadFileRequest",
    "ReadFileResponse",
    "WriteFileRequest",
    "WriteFileResponse",
    "EditFileRequest",
    "EditFileResponse",
    "TaskRequestModel",
    "TaskResponse",
    # Deep search types
    "SearchTimeFilter",
    "MAX_URLS_PER_STEP",
    "MAX_QUERIES_PER_STEP",
    "MAX_REFLECT_PER_STEP",
    "EvaluationType",
    "ActionType",
    "SearchResult",
    "WebSearchRequest",
    "URLVisitResult",
    "ReflectionQuestion",
    "PromptPair",
    "DeepSearchSchemas",
    # Docker sandbox types
    "DockerSandboxConfig",
    "DockerExecutionRequest",
    "DockerExecutionResult",
    "DockerSandboxEnvironment",
    "DockerSandboxPolicies",
    "DockerSandboxContainerInfo",
    "DockerSandboxMetrics",
    "DockerSandboxRequest",
    "DockerSandboxResponse",
    # Pydantic AI tools types
    "WebSearchBuiltinRunner",
    "CodeExecBuiltinRunner",
    "UrlContextBuiltinRunner",
    # Core tool types
    "ExecutionResult",
    "ToolRunner",
    "MockToolRunner",
    # Agent types
    "AgentType",
    "AgentStatus",
    "AgentDependencies",
    "AgentResult",
    "ExecutionHistory",
    # Multi-agent types
    "CoordinationStrategy",
    "CommunicationProtocol",
    "AgentState",
    "CoordinationMessage",
    "CoordinationRound",
    "CoordinationResult",
    "MultiAgentCoordinatorConfig",
    "AgentRole",
    # Agent Framework types
    "TextSpanRegion",
    "CitationAnnotation",
    "BaseContent",
    "TextContent",
    "TextReasoningContent",
    "DataContent",
    "UriContent",
    "ErrorContent",
    "FunctionCallContent",
    "FunctionResultContent",
    "UsageContent",
    "HostedFileContent",
    "HostedVectorStoreContent",
    "FunctionApprovalRequestContent",
    "FunctionApprovalResponseContent",
    "Content",
    "prepare_function_call_results",
    "UsageDetails",
    "Role",
    "FinishReason",
    "ToolMode",
    "ChatMessage",
    "ChatResponseUpdate",
    "ChatResponse",
    "AgentRunResponseUpdate",
    "AgentRunResponse",
    "ChatOptions",
]
