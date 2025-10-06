"""
Tests for custom Pydantic AI model implementations.

This module tests the VLLMModel and OpenAICompatibleModel wrappers with both
config-based and direct parameter approaches.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from omegaconf import OmegaConf
from DeepResearch.src.models import VLLMModel, LlamaCppModel, OpenAICompatibleModel


class TestOpenAICompatibleModel:
    """Tests for OpenAICompatibleModel with direct parameters."""

    def test_from_vllm_direct_params(self):
        """Test from_vllm with direct parameters."""
        model = OpenAICompatibleModel.from_vllm(
            base_url="http://localhost:8000/v1",
            model_name="meta-llama/Llama-3-8B"
        )

        assert model.model_name == "meta-llama/Llama-3-8B"
        assert model.base_url == "http://localhost:8000/v1/"

    def test_from_vllm_with_config(self):
        """Test from_vllm with Hydra config."""
        config = OmegaConf.create({
            "provider": "vllm",
            "model_name": "meta-llama/Llama-3-8B",
            "base_url": "http://localhost:8000/v1",
            "generation": {
                "temperature": 0.8,
                "max_tokens": 256
            }
        })

        model = OpenAICompatibleModel.from_vllm(config=config)

        assert model.model_name == "meta-llama/Llama-3-8B"
        assert model.base_url == "http://localhost:8000/v1/"

    def test_from_vllm_with_custom_api_key(self):
        """Test that from_vllm accepts custom API key."""
        model = OpenAICompatibleModel.from_vllm(
            base_url="http://localhost:8000/v1",
            model_name="meta-llama/Llama-3-8B",
            api_key="custom-key"
        )

        # Model is created successfully with custom API key
        assert model.model_name == "meta-llama/Llama-3-8B"

    def test_from_llamacpp_creates_model(self):
        """Test that from_llamacpp factory method creates a model."""
        model = OpenAICompatibleModel.from_llamacpp(
            base_url="http://localhost:8080/v1",
            model_name="llama-3-8b.gguf"
        )

        assert model.model_name == "llama-3-8b.gguf"
        assert model.base_url == "http://localhost:8080/v1/"

    def test_from_llamacpp_default_model_name(self):
        """Test that from_llamacpp uses default model name."""
        model = OpenAICompatibleModel.from_llamacpp(
            base_url="http://localhost:8080/v1"
        )

        assert model.model_name == "llama"

    def test_from_llamacpp_with_config(self):
        """Test from_llamacpp with Hydra config."""
        config = OmegaConf.create({
            "provider": "llamacpp",
            "model_name": "llama-3-8b.gguf",
            "base_url": "http://localhost:8080/v1",
        })

        model = OpenAICompatibleModel.from_llamacpp(config=config)

        assert model.model_name == "llama-3-8b.gguf"
        assert model.base_url == "http://localhost:8080/v1/"

    def test_from_tgi_creates_model(self):
        """Test that from_tgi factory method creates a model."""
        model = OpenAICompatibleModel.from_tgi(
            base_url="http://localhost:3000/v1",
            model_name="bigscience/bloom"
        )

        assert model.model_name == "bigscience/bloom"
        assert model.base_url == "http://localhost:3000/v1/"

    def test_from_custom_creates_model(self):
        """Test that from_custom factory method creates a model."""
        model = OpenAICompatibleModel.from_custom(
            base_url="https://my-llm-server.com/v1",
            model_name="my-custom-model",
            api_key="my-secret-key"
        )

        assert model.model_name == "my-custom-model"
        assert model.base_url == "https://my-llm-server.com/v1/"

    def test_vllm_model_alias(self):
        """Test that VLLMModel is an alias for OpenAICompatibleModel."""
        assert VLLMModel is OpenAICompatibleModel

    def test_llamacpp_model_alias(self):
        """Test that LlamaCppModel is an alias for OpenAICompatibleModel."""
        assert LlamaCppModel is OpenAICompatibleModel


@pytest.mark.skip(reason="Custom VLLMModel implementation tests - requires additional setup")
class TestVLLMModelIntegration:
    """Integration tests for custom VLLMModel implementation."""

    def test_model_properties(self):
        """Test VLLMModel properties without running inference."""
        from DeepResearch.src.utils.vllm_client import VLLMClient

        # Create mock client
        mock_client = Mock(spec=VLLMClient)
        mock_client.base_url = "http://localhost:8000"

        # Import the actual VLLMModel from vllm_model.py
        from DeepResearch.src.models.vllm_model import VLLMModel as VLLMModelImpl

        model = VLLMModelImpl(
            client=mock_client,
            model_name="meta-llama/Llama-3-8B"
        )

        assert model.model_name == "meta-llama/Llama-3-8B"
        assert model.system == "vllm"
        assert model.base_url == "http://localhost:8000"


@pytest.mark.skip(reason="Integration tests require actual vLLM server and are slow")
class TestVLLMModelWithContainer:
    """Integration tests using TestContainers (optional, slow)."""

    @pytest.mark.asyncio
    async def test_vllm_model_request(self):
        """Test actual vLLM model request using TestContainers."""
        from DeepResearch.src.utils.vllm_client import VLLMClient
        from DeepResearch.src.models.vllm_model import VLLMModel as VLLMModelImpl
        from pydantic_ai.messages import ModelRequest, UserPromptPart
        from pydantic_ai.models.base import ModelRequestParameters

        # Start vLLM container
        from tests.testcontainers_vllm import VLLMPromptTester

        with VLLMPromptTester() as tester:
            # Create client
            client = VLLMClient(base_url=tester.container.get_connection_url())

            # Create model
            model = VLLMModelImpl(
                client=client,
                model_name=tester.model_name
            )

            # Make request
            messages = [
                ModelRequest(parts=[UserPromptPart(content="Say hello!")])
            ]

            response = await model.request(
                messages=messages,
                model_settings={"temperature": 0.7, "max_tokens": 50},
                model_request_parameters=ModelRequestParameters()
            )

            # Validate response
            assert response is not None
            assert len(response.parts) > 0
            assert response.model_name == tester.model_name
