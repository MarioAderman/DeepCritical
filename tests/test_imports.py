"""
Comprehensive import tests for DeepCritical src modules.

This module tests that all imports from the src directory work correctly,
including all submodules and their dependencies.
"""

import importlib
import pytest


class TestMainSrcImports:
    """Test imports for main src modules."""

    def test_agents_init_imports(self):
        """Test all imports from agents.__init__.py."""
        from DeepResearch.src.agents import (
            QueryParser,
            StructuredProblem,
            ScientificIntent,
            DataType,
            parse_query,
            PlanGenerator,
            WorkflowDAG,
            WorkflowStep,
            ToolSpec,
            ToolCategory,
            generate_plan,
            ToolExecutor,
            ExecutionContext,
            execute_workflow,
            Orchestrator,
            Planner,
            PydAIToolsetBuilder,
            ResearchAgent,
            ResearchOutcome,
            StepResult,
            run,
            ToolCaller,
        )
        # Verify they are all accessible
        assert QueryParser is not None
        assert StructuredProblem is not None
        assert ScientificIntent is not None
        assert DataType is not None
        assert parse_query is not None
        assert PlanGenerator is not None
        assert WorkflowDAG is not None
        assert WorkflowStep is not None
        assert ToolSpec is not None
        assert ToolCategory is not None
        assert generate_plan is not None
        assert ToolExecutor is not None
        assert ExecutionContext is not None
        assert execute_workflow is not None
        assert Orchestrator is not None
        assert Planner is not None
        assert PydAIToolsetBuilder is not None
        assert ResearchAgent is not None
        assert ResearchOutcome is not None
        assert StepResult is not None
        assert run is not None
        assert ToolCaller is not None

    def test_datatypes_init_imports(self):
        """Test all imports from datatypes.__init__.py."""
        from DeepResearch.src.datatypes import (
            # Bioinformatics types
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
            # RAG types
            SearchType,
            EmbeddingModelType,
            LLMModelType,
            VectorStoreType,
            Document,
            SearchResult,
            EmbeddingsConfig,
            VLLMConfig,
            VectorStoreConfig,
            RAGQuery,
            RAGResponse,
            RAGConfig,
            Embeddings,
            VectorStore,
            LLMProvider,
            RAGSystem,
            RAGWorkflowState,
            # VLLM integration types
            VLLMEmbeddings,
            VLLMLLMProvider,
            VLLMServerConfig,
            VLLMEmbeddingServerConfig,
            VLLMDeployment,
            VLLMRAGSystem,
        )
        # Verify they are all accessible
        assert EvidenceCode is not None
        assert GOTerm is not None
        assert GOAnnotation is not None
        assert PubMedPaper is not None
        assert GEOPlatform is not None
        assert GEOSeries is not None
        assert GeneExpressionProfile is not None
        assert DrugTarget is not None
        assert PerturbationProfile is not None
        assert ProteinStructure is not None
        assert ProteinInteraction is not None
        assert FusedDataset is not None
        assert ReasoningTask is not None
        assert DataFusionRequest is not None
        assert SearchType is not None
        assert EmbeddingModelType is not None
        assert LLMModelType is not None
        assert VectorStoreType is not None
        assert Document is not None
        assert SearchResult is not None
        assert EmbeddingsConfig is not None
        assert VLLMConfig is not None
        assert VectorStoreConfig is not None
        assert RAGQuery is not None
        assert RAGResponse is not None
        assert RAGConfig is not None
        assert Embeddings is not None
        assert VectorStore is not None
        assert LLMProvider is not None
        assert RAGSystem is not None
        assert RAGWorkflowState is not None
        assert VLLMEmbeddings is not None
        assert VLLMLLMProvider is not None
        assert VLLMServerConfig is not None
        assert VLLMEmbeddingServerConfig is not None
        assert VLLMDeployment is not None
        assert VLLMRAGSystem is not None

    def test_tools_init_imports(self):
        """Test all imports from tools.__init__.py."""
        from DeepResearch.src import tools
        # Test that the registry is accessible
        assert hasattr(tools, 'registry')
        assert tools.registry is not None

    def test_utils_init_imports(self):
        """Test all imports from utils.__init__.py."""
        from DeepResearch.src import utils
        # Test that utils module is accessible
        assert utils is not None

    def test_prompts_init_imports(self):
        """Test all imports from prompts.__init__.py."""
        from DeepResearch.src import prompts
        # Test that prompts module is accessible
        assert prompts is not None

    def test_statemachines_init_imports(self):
        """Test all imports from statemachines.__init__.py."""
        from DeepResearch.src import statemachines
        # Test that statemachines module is accessible
        assert statemachines is not None


class TestSubmoduleImports:
    """Test imports for individual submodules."""

    def test_agents_submodules(self):
        """Test that all agent submodules can be imported."""
        # Test individual agent modules
        from DeepResearch.src.agents import (
            prime_parser,
            prime_planner,
            prime_executor,
            orchestrator,
            planner,
            pyd_ai_toolsets,
            research_agent,
            tool_caller,
        )
        # Verify they are all accessible
        assert prime_parser is not None
        assert prime_planner is not None
        assert prime_executor is not None
        assert orchestrator is not None
        assert planner is not None
        assert pyd_ai_toolsets is not None
        assert research_agent is not None
        assert tool_caller is not None

    def test_datatypes_submodules(self):
        """Test that all datatype submodules can be imported."""
        from DeepResearch.src.datatypes import (
            bioinformatics,
            rag,
            vllm_integration,
            chunk_dataclass,
            document_dataclass,
            chroma_dataclass,
            postgres_dataclass,
            vllm_dataclass,
            markdown,
            deep_agent_state,
            deep_agent_types,
            workflow_orchestration,
        )
        # Verify they are all accessible
        assert bioinformatics is not None
        assert rag is not None
        assert vllm_integration is not None
        assert chunk_dataclass is not None
        assert document_dataclass is not None
        assert chroma_dataclass is not None
        assert postgres_dataclass is not None
        assert vllm_dataclass is not None
        assert markdown is not None
        assert deep_agent_state is not None
        assert deep_agent_types is not None
        assert workflow_orchestration is not None

    def test_tools_submodules(self):
        """Test that all tool submodules can be imported."""
        from DeepResearch.src.tools import (
            base,
            mock_tools,
            workflow_tools,
            pyd_ai_tools,
            code_sandbox,
            docker_sandbox,
            deepsearch_tools,
            deepsearch_workflow_tool,
            websearch_tools,
            analytics_tools,
            integrated_search_tools,
        )
        # Verify they are all accessible
        assert base is not None
        assert mock_tools is not None
        assert workflow_tools is not None
        assert pyd_ai_tools is not None
        assert code_sandbox is not None
        assert docker_sandbox is not None
        assert deepsearch_tools is not None
        assert deepsearch_workflow_tool is not None
        assert websearch_tools is not None
        assert analytics_tools is not None
        assert integrated_search_tools is not None

    def test_utils_submodules(self):
        """Test that all utils submodules can be imported."""
        from DeepResearch.src.utils import (
            config_loader,
            execution_history,
            execution_status,
            tool_registry,
            tool_specs,
            analytics,
            deepsearch_schemas,
            deepsearch_utils,
        )
        # Verify they are all accessible
        assert config_loader is not None
        assert execution_history is not None
        assert execution_status is not None
        assert tool_registry is not None
        assert tool_specs is not None
        assert analytics is not None
        assert deepsearch_schemas is not None
        assert deepsearch_utils is not None

    def test_prompts_submodules(self):
        """Test that all prompt submodules can be imported."""
        from DeepResearch.src.prompts import (
            agent,
            broken_ch_fixer,
            code_exec,
            code_sandbox,
            deep_agent_graph,
            deep_agent_prompts,
            error_analyzer,
            evaluator,
            finalizer,
            orchestrator,
            planner,
            query_rewriter,
            reducer,
            research_planner,
            serp_cluster,
        )
        # Verify they are all accessible
        assert agent is not None
        assert broken_ch_fixer is not None
        assert code_exec is not None
        assert code_sandbox is not None
        assert deep_agent_graph is not None
        assert deep_agent_prompts is not None
        assert error_analyzer is not None
        assert evaluator is not None
        assert finalizer is not None
        assert orchestrator is not None
        assert planner is not None
        assert query_rewriter is not None
        assert reducer is not None
        assert research_planner is not None
        assert serp_cluster is not None

    def test_statemachines_submodules(self):
        """Test that all statemachine submodules can be imported."""
        from DeepResearch.src.statemachines import (
            bioinformatics_workflow,
            deepsearch_workflow,
            rag_workflow,
            search_workflow,
        )
        # Verify they are all accessible
        assert bioinformatics_workflow is not None
        assert deepsearch_workflow is not None
        assert rag_workflow is not None
        assert search_workflow is not None


class TestDeepImportChains:
    """Test deep import chains and dependencies."""

    def test_agent_internal_imports(self):
        """Test that agents can import their internal dependencies."""
        # Test that prime_parser can import its dependencies
        from DeepResearch.src.agents.prime_parser import (
            QueryParser,
            StructuredProblem,
        )
        assert QueryParser is not None
        assert StructuredProblem is not None

    def test_datatype_internal_imports(self):
        """Test that datatypes can import their internal dependencies."""
        # Test that bioinformatics can import its dependencies
        from DeepResearch.src.datatypes.bioinformatics import (
            EvidenceCode,
            GOTerm,
        )
        assert EvidenceCode is not None
        assert GOTerm is not None

    def test_tool_internal_imports(self):
        """Test that tools can import their internal dependencies."""
        # Test that base tools can be imported
        from DeepResearch.src.tools.base import registry
        assert registry is not None

    def test_utils_internal_imports(self):
        """Test that utils can import their internal dependencies."""
        # Test that config_loader can be imported
        from DeepResearch.src.utils.config_loader import BioinformaticsConfigLoader
        assert BioinformaticsConfigLoader is not None

    def test_prompts_internal_imports(self):
        """Test that prompts can import their internal dependencies."""
        # Test that agent prompts can be imported
        from DeepResearch.src.prompts.agent import AgentPrompts
        assert AgentPrompts is not None


class TestCircularImportSafety:
    """Test for circular import issues."""

    def test_no_circular_imports_in_agents(self):
        """Test that importing agents doesn't cause circular imports."""
        # This test will fail if there are circular imports
        import DeepResearch.src.agents
        import DeepResearch.src.agents.prime_parser
        import DeepResearch.src.agents.prime_planner
        import DeepResearch.src.agents.prime_executor
        assert True  # If we get here, no circular imports

    def test_no_circular_imports_in_datatypes(self):
        """Test that importing datatypes doesn't cause circular imports."""
        # This test will fail if there are circular imports
        import DeepResearch.src.datatypes
        import DeepResearch.src.datatypes.bioinformatics
        import DeepResearch.src.datatypes.rag
        assert True  # If we get here, no circular imports

    def test_no_circular_imports_in_tools(self):
        """Test that importing tools doesn't cause circular imports."""
        # This test will fail if there are circular imports
        import DeepResearch.src.tools
        import DeepResearch.src.tools.base
        import DeepResearch.src.tools.mock_tools
        assert True  # If we get here, no circular imports

    def test_no_circular_imports_in_utils(self):
        """Test that importing utils doesn't cause circular imports."""
        # This test will fail if there are circular imports
        import DeepResearch.src.utils
        import DeepResearch.src.utils.config_loader
        import DeepResearch.src.utils.tool_registry
        assert True  # If we get here, no circular imports

    def test_no_circular_imports_in_prompts(self):
        """Test that importing prompts doesn't cause circular imports."""
        # This test will fail if there are circular imports
        import DeepResearch.src.prompts
        import DeepResearch.src.prompts.agent
        import DeepResearch.src.prompts.planner
        assert True  # If we get here, no circular imports

    def test_no_circular_imports_in_statemachines(self):
        """Test that importing statemachines doesn't cause circular imports."""
        # This test will fail if there are circular imports
        import DeepResearch.src.statemachines
        import DeepResearch.src.statemachines.bioinformatics_workflow
        import DeepResearch.src.statemachines.rag_workflow
        assert True  # If we get here, no circular imports
