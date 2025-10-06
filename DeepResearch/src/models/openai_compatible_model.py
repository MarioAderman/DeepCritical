"""
Pydantic AI model wrapper for OpenAI-compatible servers.

This module provides a generic OpenAICompatibleModel that can work with:
- vLLM (OpenAI-compatible API)
- llama.cpp server (OpenAI-compatible mode)
- Text Generation Inference (TGI)
- Any other server implementing the OpenAI Chat Completions API
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic_ai.models.openai import OpenAIChatModel, OpenAIProvider


@dataclass
class OpenAICompatibleModel(OpenAIChatModel):
    """Pydantic AI model for OpenAI-compatible servers.

    This is a thin wrapper around Pydantic AI's OpenAIChatModel that makes it
    easy to connect to local or custom OpenAI-compatible servers.

    Supports:
    - vLLM with OpenAI-compatible API
    - llama.cpp server in OpenAI mode
    - Text Generation Inference (TGI)
    - Any custom OpenAI-compatible endpoint

    Example:
        ```python
        from pydantic_ai import Agent
        from DeepResearch.src.models import OpenAICompatibleModel

        # Connect to vLLM server
        model = OpenAICompatibleModel.from_vllm(
            base_url="http://localhost:8000/v1",
            model_name="meta-llama/Llama-3-8B"
        )

        # Use with agent
        agent = Agent(model)
        result = agent.run_sync("Hello!")
        ```

    Example (llama.cpp):
        ```python
        # Connect to llama.cpp server
        model = OpenAICompatibleModel.from_llamacpp(
            base_url="http://localhost:8080/v1",
            model_name="llama-3-8b.gguf"
        )

        agent = Agent(model)
        result = agent.run_sync("Hello!")
        ```
    """

    @classmethod
    def from_vllm(
        cls, base_url: str, model_name: str, api_key: str = "EMPTY", **kwargs: Any
    ) -> OpenAICompatibleModel:
        """Create a model for a vLLM server.

        Args:
            base_url: The vLLM server URL (e.g., "http://localhost:8000/v1").
            model_name: The model name to use (e.g., "meta-llama/Llama-3-8B").
            api_key: API key if required (vLLM default is "EMPTY").
            **kwargs: Additional arguments passed to the model.

        Returns:
            Configured OpenAICompatibleModel instance.

        Example:
            ```python
            model = OpenAICompatibleModel.from_vllm(
                base_url="http://localhost:8000/v1",
                model_name="meta-llama/Llama-3-8B",
                api_key="my-secret-key"  # if auth is enabled
            )
            ```
        """
        provider = OpenAIProvider(
            base_url=base_url,
            api_key=api_key,
        )

        return cls(model_name=model_name, provider=provider, **kwargs)

    @classmethod
    def from_llamacpp(
        cls,
        base_url: str,
        model_name: str = "llama",
        api_key: str = "sk-no-key-required",
        **kwargs: Any,
    ) -> OpenAICompatibleModel:
        """Create a model for a llama.cpp server.

        Args:
            base_url: The llama.cpp server URL (e.g., "http://localhost:8080/v1").
            model_name: The model name (llama.cpp uses "llama" by default).
            api_key: API key (llama.cpp doesn't require one by default).
            **kwargs: Additional arguments passed to the model.

        Returns:
            Configured OpenAICompatibleModel instance.

        Example:
            ```python
            model = OpenAICompatibleModel.from_llamacpp(
                base_url="http://localhost:8080/v1",
                model_name="llama-3-8b.gguf"
            )
            ```
        """
        provider = OpenAIProvider(
            base_url=base_url,
            api_key=api_key,
        )

        return cls(model_name=model_name, provider=provider, **kwargs)

    @classmethod
    def from_tgi(
        cls, base_url: str, model_name: str, api_key: str | None = None, **kwargs: Any
    ) -> OpenAICompatibleModel:
        """Create a model for a Text Generation Inference (TGI) server.

        Args:
            base_url: The TGI server URL (e.g., "http://localhost:3000/v1").
            model_name: The model name.
            api_key: API key if required.
            **kwargs: Additional arguments passed to the model.

        Returns:
            Configured OpenAICompatibleModel instance.

        Example:
            ```python
            model = OpenAICompatibleModel.from_tgi(
                base_url="http://localhost:3000/v1",
                model_name="bigscience/bloom"
            )
            ```
        """
        provider = OpenAIProvider(
            base_url=base_url,
            api_key=api_key or "EMPTY",
        )

        return cls(model_name=model_name, provider=provider, **kwargs)

    @classmethod
    def from_custom(
        cls, base_url: str, model_name: str, api_key: str | None = None, **kwargs: Any
    ) -> OpenAICompatibleModel:
        """Create a model for any custom OpenAI-compatible server.

        Args:
            base_url: The server URL with /v1 path.
            model_name: The model name to use.
            api_key: API key if required.
            **kwargs: Additional arguments passed to the model.

        Returns:
            Configured OpenAICompatibleModel instance.

        Example:
            ```python
            model = OpenAICompatibleModel.from_custom(
                base_url="https://my-llm-server.com/v1",
                model_name="my-custom-model",
                api_key="my-secret-key"
            )
            ```
        """
        provider = OpenAIProvider(
            base_url=base_url,
            api_key=api_key or "EMPTY",
        )

        return cls(model_name=model_name, provider=provider, **kwargs)


# Convenience aliases
VLLMModel = OpenAICompatibleModel
"""Alias for OpenAICompatibleModel when using vLLM.

Example:
    ```python
    from DeepResearch.src.models import VLLMModel

    model = VLLMModel.from_vllm(
        base_url="http://localhost:8000/v1",
        model_name="meta-llama/Llama-3-8B"
    )
    ```
"""

LlamaCppModel = OpenAICompatibleModel
"""Alias for OpenAICompatibleModel when using llama.cpp.

Example:
    ```python
    from DeepResearch.src.models import LlamaCppModel

    model = LlamaCppModel.from_llamacpp(
        base_url="http://localhost:8080/v1",
        model_name="llama-3-8b.gguf"
    )
    ```
"""
