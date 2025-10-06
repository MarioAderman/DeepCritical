"""
Pydantic AI model wrapper for OpenAI-compatible servers.

This module provides a generic OpenAICompatibleModel that can work with:
- vLLM (OpenAI-compatible API)
- llama.cpp server (OpenAI-compatible mode)
- Text Generation Inference (TGI)
- Any other server implementing the OpenAI Chat Completions API

All configuration is managed through Hydra config files.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from omegaconf import DictConfig, OmegaConf
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider


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
    def from_config(
        cls,
        config: DictConfig | dict,
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs: Any,
    ) -> "OpenAICompatibleModel":
        """Create a model from Hydra configuration.

        Args:
            config: Hydra configuration (DictConfig) or dict with model settings.
            model_name: Override model name from config.
            base_url: Override base URL from config.
            api_key: Override API key from config.
            **kwargs: Additional arguments passed to the model.

        Returns:
            Configured OpenAICompatibleModel instance.

        Example:
            ```python
            from hydra import compose, initialize

            with initialize(config_path="../configs"):
                cfg = compose(config_name="config", overrides=["llm=vllm_local"])
                model = OpenAICompatibleModel.from_config(cfg.llm)
            ```
        """
        # Convert DictConfig to dict if needed
        if isinstance(config, DictConfig):
            config = OmegaConf.to_container(config, resolve=True)

        # Extract configuration with fallbacks
        final_model_name = (
            model_name
            or config.get("model_name")
            or config.get("model", {}).get("name", "gpt-3.5-turbo")
        )
        final_base_url = base_url or config.get("base_url") or os.getenv("LLM_BASE_URL")
        final_api_key = (
            api_key or config.get("api_key") or os.getenv("LLM_API_KEY", "EMPTY")
        )

        # Extract generation settings from config
        generation_config = config.get("generation", {})
        settings = kwargs.pop("settings", {})

        # Merge config-based settings with kwargs
        if generation_config:
            settings.update(
                {
                    k: v
                    for k, v in generation_config.items()
                    if k
                    in [
                        "temperature",
                        "max_tokens",
                        "top_p",
                        "frequency_penalty",
                        "presence_penalty",
                    ]
                }
            )

        if not final_base_url:
            raise ValueError(
                "base_url must be provided either in config, as argument, or via LLM_BASE_URL environment variable"
            )

        provider = OllamaProvider(
            base_url=final_base_url,
            api_key=final_api_key,
        )

        return cls(
            final_model_name, provider=provider, settings=settings or None, **kwargs
        )

    @classmethod
    def from_vllm(
        cls,
        config: Optional[DictConfig | dict] = None,
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs: Any,
    ) -> "OpenAICompatibleModel":
        """Create a model for a vLLM server.

        Args:
            config: Optional Hydra configuration with vLLM settings.
            model_name: Model name (overrides config if provided).
            base_url: vLLM server URL (overrides config if provided).
            api_key: API key (overrides config if provided).
            **kwargs: Additional arguments passed to the model.

        Returns:
            Configured OpenAICompatibleModel instance.

        Example:
            ```python
            # From config
            model = OpenAICompatibleModel.from_vllm(config=cfg.vllm)

            # Direct parameters (for testing/simple cases)
            model = OpenAICompatibleModel.from_vllm(
                base_url="http://localhost:8000/v1",
                model_name="meta-llama/Llama-3-8B"
            )
            ```
        """
        if config is not None:
            return cls.from_config(config, model_name, base_url, api_key, **kwargs)

        # Fallback for direct parameter usage
        if not base_url:
            raise ValueError("base_url is required when not using config")
        if not model_name:
            raise ValueError("model_name is required when not using config")

        provider = OllamaProvider(
            base_url=base_url,
            api_key=api_key or "EMPTY",
        )
        return cls(model_name, provider=provider, **kwargs)

    @classmethod
    def from_llamacpp(
        cls,
        config: Optional[DictConfig | dict] = None,
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs: Any,
    ) -> "OpenAICompatibleModel":
        """Create a model for a llama.cpp server.

        Args:
            config: Optional Hydra configuration with llama.cpp settings.
            model_name: Model name (overrides config if provided).
            base_url: llama.cpp server URL (overrides config if provided).
            api_key: API key (overrides config if provided).
            **kwargs: Additional arguments passed to the model.

        Returns:
            Configured OpenAICompatibleModel instance.

        Example:
            ```python
            # From config
            model = OpenAICompatibleModel.from_llamacpp(config=cfg.llamacpp)

            # Direct parameters
            model = OpenAICompatibleModel.from_llamacpp(
                base_url="http://localhost:8080/v1",
                model_name="llama-3-8b.gguf"
            )
            ```
        """
        if config is not None:
            # Use default llama model name if not specified
            if model_name is None and "model_name" not in config:
                model_name = "llama"
            return cls.from_config(config, model_name, base_url, api_key, **kwargs)

        # Fallback for direct parameter usage
        if not base_url:
            raise ValueError("base_url is required when not using config")

        provider = OllamaProvider(
            base_url=base_url,
            api_key=api_key or "sk-no-key-required",
        )
        return cls(model_name or "llama", provider=provider, **kwargs)

    @classmethod
    def from_tgi(
        cls,
        config: Optional[DictConfig | dict] = None,
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs: Any,
    ) -> "OpenAICompatibleModel":
        """Create a model for a Text Generation Inference (TGI) server.

        Args:
            config: Optional Hydra configuration with TGI settings.
            model_name: Model name (overrides config if provided).
            base_url: TGI server URL (overrides config if provided).
            api_key: API key (overrides config if provided).
            **kwargs: Additional arguments passed to the model.

        Returns:
            Configured OpenAICompatibleModel instance.

        Example:
            ```python
            # From config
            model = OpenAICompatibleModel.from_tgi(config=cfg.tgi)

            # Direct parameters
            model = OpenAICompatibleModel.from_tgi(
                base_url="http://localhost:3000/v1",
                model_name="bigscience/bloom"
            )
            ```
        """
        if config is not None:
            return cls.from_config(config, model_name, base_url, api_key, **kwargs)

        # Fallback for direct parameter usage
        if not base_url:
            raise ValueError("base_url is required when not using config")
        if not model_name:
            raise ValueError("model_name is required when not using config")

        provider = OllamaProvider(
            base_url=base_url,
            api_key=api_key or "EMPTY",
        )
        return cls(model_name, provider=provider, **kwargs)

    @classmethod
    def from_custom(
        cls,
        config: Optional[DictConfig | dict] = None,
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs: Any,
    ) -> "OpenAICompatibleModel":
        """Create a model for any custom OpenAI-compatible server.

        Args:
            config: Optional Hydra configuration with custom server settings.
            model_name: Model name (overrides config if provided).
            base_url: Server URL (overrides config if provided).
            api_key: API key (overrides config if provided).
            **kwargs: Additional arguments passed to the model.

        Returns:
            Configured OpenAICompatibleModel instance.

        Example:
            ```python
            # From config
            model = OpenAICompatibleModel.from_custom(config=cfg.custom_llm)

            # Direct parameters
            model = OpenAICompatibleModel.from_custom(
                base_url="https://my-llm-server.com/v1",
                model_name="my-custom-model",
                api_key="my-secret-key"
            )
            ```
        """
        if config is not None:
            return cls.from_config(config, model_name, base_url, api_key, **kwargs)

        # Fallback for direct parameter usage
        if not base_url:
            raise ValueError("base_url is required when not using config")
        if not model_name:
            raise ValueError("model_name is required when not using config")

        provider = OllamaProvider(
            base_url=base_url,
            api_key=api_key or "EMPTY",
        )
        return cls(model_name, provider=provider, **kwargs)


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
