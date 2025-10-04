"""
Import tests for DeepResearch statemachines modules.

This module tests that all imports from the statemachines subdirectory work correctly,
including all individual statemachine modules and their dependencies.
"""

import pytest


class TestStatemachinesModuleImports:
    """Test imports for individual statemachine modules."""

    def test_bioinformatics_workflow_imports(self):
        """Test all imports from bioinformatics_workflow module."""
        from DeepResearch.src.statemachines import bioinformatics_workflow

        from DeepResearch.src.statemachines.bioinformatics_workflow import (
            BioinformaticsState,
            DataFusionNode,
            ReasoningNode,
            QualityAssessmentNode,
            FinalAnswerNode,
        )

        # Verify they are all accessible and not None
        assert BioinformaticsState is not None
        assert DataFusionNode is not None
        assert ReasoningNode is not None
        assert QualityAssessmentNode is not None
        assert FinalAnswerNode is not None

    def test_deepsearch_workflow_imports(self):
        """Test all imports from deepsearch_workflow module."""
        from DeepResearch.src.statemachines import deepsearch_workflow

        from DeepResearch.src.statemachines.deepsearch_workflow import (
            DeepSearchState,
            QueryPlanningNode,
            SearchExecutionNode,
            ResultAggregationNode,
            FinalSynthesisNode,
        )

        # Verify they are all accessible and not None
        assert DeepSearchState is not None
        assert QueryPlanningNode is not None
        assert SearchExecutionNode is not None
        assert ResultAggregationNode is not None
        assert FinalSynthesisNode is not None

    def test_rag_workflow_imports(self):
        """Test all imports from rag_workflow module."""
        from DeepResearch.src.statemachines import rag_workflow

        from DeepResearch.src.statemachines.rag_workflow import (
            RAGState,
            DocumentRetrievalNode,
            QueryProcessingNode,
            AnswerGenerationNode,
            ResponseFormattingNode,
        )

        # Verify they are all accessible and not None
        assert RAGState is not None
        assert DocumentRetrievalNode is not None
        assert QueryProcessingNode is not None
        assert AnswerGenerationNode is not None
        assert ResponseFormattingNode is not None

    def test_search_workflow_imports(self):
        """Test all imports from search_workflow module."""
        from DeepResearch.src.statemachines import search_workflow

        from DeepResearch.src.statemachines.search_workflow import (
            SearchState,
            QueryReformulationNode,
            SearchExecutionNode,
            ResultFilteringNode,
            AnswerCompilationNode,
        )

        # Verify they are all accessible and not None
        assert SearchState is not None
        assert QueryReformulationNode is not None
        assert SearchExecutionNode is not None
        assert ResultFilteringNode is not None
        assert AnswerCompilationNode is not None


class TestStatemachinesCrossModuleImports:
    """Test cross-module imports and dependencies within statemachines."""

    def test_statemachines_internal_dependencies(self):
        """Test that statemachine modules can import from each other correctly."""
        # Test that modules can import shared patterns
        from DeepResearch.src.statemachines.bioinformatics_workflow import BioinformaticsState
        from DeepResearch.src.statemachines.rag_workflow import RAGState

        # This should work without circular imports
        assert BioinformaticsState is not None
        assert RAGState is not None

    def test_datatypes_integration_imports(self):
        """Test that statemachines can import from datatypes module."""
        # This tests the import chain: statemachines -> datatypes
        from DeepResearch.src.statemachines.bioinformatics_workflow import BioinformaticsState
        from DeepResearch.src.datatypes.bioinformatics import FusedDataset

        # If we get here without ImportError, the import chain works
        assert BioinformaticsState is not None
        assert FusedDataset is not None

    def test_agents_integration_imports(self):
        """Test that statemachines can import from agents module."""
        # This tests the import chain: statemachines -> agents
        from DeepResearch.src.statemachines.bioinformatics_workflow import DataFusionNode
        from DeepResearch.src.agents.bioinformatics_agents import BioinformaticsAgent

        # If we get here without ImportError, the import chain works
        assert DataFusionNode is not None
        assert BioinformaticsAgent is not None

    def test_pydantic_graph_imports(self):
        """Test that statemachines can import from pydantic_graph."""
        # Test that BaseNode and other pydantic_graph imports work
        from DeepResearch.src.statemachines.bioinformatics_workflow import BaseNode

        # If we get here without ImportError, the import chain works
        assert BaseNode is not None


class TestStatemachinesComplexImportChains:
    """Test complex import chains involving multiple modules."""

    def test_full_statemachines_initialization_chain(self):
        """Test the complete import chain for statemachines initialization."""
        try:
            from DeepResearch.src.statemachines.bioinformatics_workflow import (
                BioinformaticsState, DataFusionNode, ReasoningNode
            )
            from DeepResearch.src.statemachines.rag_workflow import (
                RAGState, DocumentRetrievalNode
            )
            from DeepResearch.src.statemachines.search_workflow import (
                SearchState, QueryReformulationNode
            )
            from DeepResearch.src.datatypes.bioinformatics import FusedDataset
            from DeepResearch.src.agents.bioinformatics_agents import BioinformaticsAgent

            # If all imports succeed, the chain is working
            assert BioinformaticsState is not None
            assert DataFusionNode is not None
            assert ReasoningNode is not None
            assert RAGState is not None
            assert DocumentRetrievalNode is not None
            assert SearchState is not None
            assert QueryReformulationNode is not None
            assert FusedDataset is not None
            assert BioinformaticsAgent is not None

        except ImportError as e:
            pytest.fail(f"Statemachines import chain failed: {e}")

    def test_workflow_execution_chain(self):
        """Test the complete import chain for workflow execution."""
        try:
            from DeepResearch.src.statemachines.bioinformatics_workflow import FinalAnswerNode
            from DeepResearch.src.statemachines.deepsearch_workflow import FinalSynthesisNode
            from DeepResearch.src.statemachines.rag_workflow import ResponseFormattingNode
            from DeepResearch.src.statemachines.search_workflow import AnswerCompilationNode

            # If all imports succeed, the chain is working
            assert FinalAnswerNode is not None
            assert FinalSynthesisNode is not None
            assert ResponseFormattingNode is not None
            assert AnswerCompilationNode is not None

        except ImportError as e:
            pytest.fail(f"Workflow execution import chain failed: {e}")


class TestStatemachinesImportErrorHandling:
    """Test import error handling for statemachines modules."""

    def test_missing_dependencies_handling(self):
        """Test that modules handle missing dependencies gracefully."""
        # Test that modules handle optional dependencies
        from DeepResearch.src.statemachines.bioinformatics_workflow import BaseNode

        # This should work even if pydantic_graph is not available in some contexts
        assert BaseNode is not None

    def test_circular_import_prevention(self):
        """Test that there are no circular imports in statemachines."""
        # This test will fail if there are circular imports
        import DeepResearch.src.statemachines.bioinformatics_workflow
        import DeepResearch.src.statemachines.rag_workflow
        import DeepResearch.src.statemachines.search_workflow

        # If we get here, no circular imports were detected
        assert True

    def test_state_class_instantiation(self):
        """Test that state classes can be instantiated."""
        from DeepResearch.src.statemachines.bioinformatics_workflow import BioinformaticsState

        # Test that we can create instances (basic functionality)
        try:
            state = BioinformaticsState(question="test question")
            assert state is not None
            assert state.question == "test question"
            assert state.go_annotations == []
            assert state.pubmed_papers == []
        except Exception as e:
            pytest.fail(f"State class instantiation failed: {e}")

    def test_node_class_instantiation(self):
        """Test that node classes can be instantiated."""
        from DeepResearch.src.statemachines.bioinformatics_workflow import DataFusionNode

        # Test that we can create instances (basic functionality)
        try:
            node = DataFusionNode()
            assert node is not None
        except Exception as e:
            pytest.fail(f"Node class instantiation failed: {e}")

    def test_pydantic_graph_compatibility(self):
        """Test that statemachines are compatible with pydantic_graph."""
        from DeepResearch.src.statemachines.bioinformatics_workflow import BaseNode

        # Test that BaseNode is properly imported from pydantic_graph
        assert BaseNode is not None

        # Test that common pydantic_graph attributes are available
        # (these might not exist if pydantic_graph is not installed)
        if hasattr(BaseNode, '__annotations__'):
            annotations = getattr(BaseNode, '__annotations__')
            assert isinstance(annotations, dict)
