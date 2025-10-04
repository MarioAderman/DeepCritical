"""
VLLM-powered Pydantic AI agent for DeepCritical.

This module provides a complete VLLM agent implementation that can be used
with Pydantic AI's CLI and agent system.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field

from ..vllm_client import VLLMClient
from ..datatypes.vllm_dataclass import (
    ChatCompletionRequest,
    CompletionRequest,
    EmbeddingRequest,
    VllmConfig,
    QuantizationMethod,
)


class VLLMAgentDependencies(BaseModel):
    """Dependencies for VLLM agent."""

    vllm_client: VLLMClient = Field(..., description="VLLM client instance")
    default_model: str = Field(
        "microsoft/DialoGPT-medium", description="Default model name"
    )
    embedding_model: Optional[str] = Field(None, description="Embedding model name")

    class Config:
        arbitrary_types_allowed = True


class VLLMAgentConfig(BaseModel):
    """Configuration for VLLM agent."""

    client_config: Dict[str, Any] = Field(
        default_factory=dict, description="VLLM client configuration"
    )
    default_model: str = Field("microsoft/DialoGPT-medium", description="Default model")
    embedding_model: Optional[str] = Field(None, description="Embedding model")
    system_prompt: str = Field(
        "You are a helpful AI assistant powered by VLLM. You can perform various tasks including text generation, conversation, and analysis.",
        description="System prompt for the agent",
    )
    max_tokens: int = Field(512, description="Maximum tokens for generation")
    temperature: float = Field(0.7, description="Sampling temperature")
    top_p: float = Field(0.9, description="Top-p sampling parameter")


class VLLMAgent:
    """VLLM-powered agent for Pydantic AI."""

    def __init__(self, config: VLLMAgentConfig):
        self.config = config
        self.client = VLLMClient(**config.client_config)
        self.dependencies = VLLMAgentDependencies(
            vllm_client=self.client,
            default_model=config.default_model,
            embedding_model=config.embedding_model,
        )

    async def initialize(self):
        """Initialize the VLLM agent."""
        # Test connection
        try:
            await self.client.health()
            print("✓ VLLM server connection established")
        except Exception as e:
            print(f"✗ Failed to connect to VLLM server: {e}")
            raise

    async def chat(
        self, messages: List[Dict[str, str]], model: Optional[str] = None, **kwargs
    ) -> str:
        """Chat with the VLLM model."""
        model = model or self.config.default_model

        request = ChatCompletionRequest(
            model=model,
            messages=messages,
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            temperature=kwargs.get("temperature", self.config.temperature),
            top_p=kwargs.get("top_p", self.config.top_p),
            **kwargs,
        )

        response = await self.client.chat_completions(request)
        return response.choices[0].message.content

    async def complete(self, prompt: str, model: Optional[str] = None, **kwargs) -> str:
        """Complete text with the VLLM model."""
        model = model or self.config.default_model

        request = CompletionRequest(
            model=model,
            prompt=prompt,
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            temperature=kwargs.get("temperature", self.config.temperature),
            top_p=kwargs.get("top_p", self.config.top_p),
            **kwargs,
        )

        response = await self.client.completions(request)
        return response.choices[0].text

    async def embed(
        self, texts: Union[str, List[str]], model: Optional[str] = None, **kwargs
    ) -> List[List[float]]:
        """Generate embeddings for texts."""
        if isinstance(texts, str):
            texts = [texts]

        embedding_model = (
            model or self.config.embedding_model or self.config.default_model
        )

        request = EmbeddingRequest(model=embedding_model, input=texts, **kwargs)

        response = await self.client.embeddings(request)
        return [item.embedding for item in response.data]

    async def chat_stream(
        self, messages: List[Dict[str, str]], model: Optional[str] = None, **kwargs
    ) -> str:
        """Stream chat completion."""
        model = model or self.config.default_model

        request = ChatCompletionRequest(
            model=model,
            messages=messages,
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            temperature=kwargs.get("temperature", self.config.temperature),
            top_p=kwargs.get("top_p", self.config.top_p),
            stream=True,
            **kwargs,
        )

        full_response = ""
        async for chunk in self.client.chat_completions_stream(request):
            full_response += chunk
            print(chunk, end="", flush=True)
        print()  # New line after streaming
        return full_response

    def to_pydantic_ai_agent(self):
        """Convert to Pydantic AI agent."""
        from pydantic_ai import Agent

        agent = Agent(
            "vllm-agent",
            deps_type=VLLMAgentDependencies,
            system_prompt=self.config.system_prompt,
        )

        # Chat completion tool
        @agent.tool
        async def chat_completion(
            ctx, messages: List[Dict[str, str]], model: Optional[str] = None, **kwargs
        ) -> str:
            """Chat with the VLLM model."""
            return (
                await ctx.deps.vllm_client.chat_completions(
                    ChatCompletionRequest(
                        model=model or ctx.deps.default_model,
                        messages=messages,
                        **kwargs,
                    )
                )
                .choices[0]
                .message.content
            )

        # Text completion tool
        @agent.tool
        async def text_completion(
            ctx, prompt: str, model: Optional[str] = None, **kwargs
        ) -> str:
            """Complete text with the VLLM model."""
            return (
                await ctx.deps.vllm_client.completions(
                    CompletionRequest(
                        model=model or ctx.deps.default_model, prompt=prompt, **kwargs
                    )
                )
                .choices[0]
                .text
            )

        # Embedding generation tool
        @agent.tool
        async def generate_embeddings(
            ctx, texts: Union[str, List[str]], model: Optional[str] = None, **kwargs
        ) -> List[List[float]]:
            """Generate embeddings using VLLM."""
            if isinstance(texts, str):
                texts = [texts]

            embedding_model = (
                model or ctx.deps.embedding_model or ctx.deps.default_model
            )

            return (
                await ctx.deps.vllm_client.embeddings(
                    EmbeddingRequest(model=embedding_model, input=texts, **kwargs)
                )
                .data[0]
                .embedding
                if len(texts) == 1
                else [
                    item.embedding
                    for item in await ctx.deps.vllm_client.embeddings(
                        EmbeddingRequest(model=embedding_model, input=texts, **kwargs)
                    ).data
                ]
            )

        # Model information tool
        @agent.tool
        async def get_model_info(ctx, model_name: str) -> Dict[str, Any]:
            """Get information about a specific model."""
            return await ctx.deps.vllm_client.get_model_info(model_name)

        # List models tool
        @agent.tool
        async def list_models(ctx) -> List[str]:
            """List available models."""
            response = await ctx.deps.vllm_client.models()
            return [model.id for model in response.data]

        # Tokenization tools
        @agent.tool
        async def tokenize(
            ctx, text: str, model: Optional[str] = None
        ) -> Dict[str, Any]:
            """Tokenize text."""
            return await ctx.deps.vllm_client.tokenize(
                text, model or ctx.deps.default_model
            )

        @agent.tool
        async def detokenize(
            ctx, token_ids: List[int], model: Optional[str] = None
        ) -> Dict[str, Any]:
            """Detokenize token IDs."""
            return await ctx.deps.vllm_client.detokenize(
                token_ids, model or ctx.deps.default_model
            )

        # Health check tool
        @agent.tool
        async def health_check(ctx) -> Dict[str, Any]:
            """Check server health."""
            return await ctx.deps.vllm_client.health()

        return agent


def create_vllm_agent(
    model_name: str = "microsoft/DialoGPT-medium",
    base_url: str = "http://localhost:8000",
    api_key: Optional[str] = None,
    embedding_model: Optional[str] = None,
    **kwargs,
) -> VLLMAgent:
    """Create a VLLM agent with default configuration."""

    config = VLLMAgentConfig(
        client_config={"base_url": base_url, "api_key": api_key, **kwargs},
        default_model=model_name,
        embedding_model=embedding_model,
    )

    return VLLMAgent(config)


def create_advanced_vllm_agent(
    model_name: str = "microsoft/DialoGPT-medium",
    base_url: str = "http://localhost:8000",
    quantization: Optional[QuantizationMethod] = None,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    **kwargs,
) -> VLLMAgent:
    """Create a VLLM agent with advanced configuration."""

    # Create VLLM configuration
    vllm_config = VllmConfig.from_config(
        model=model_name,
        quantization=quantization,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
    )

    config = VLLMAgentConfig(
        client_config={"base_url": base_url, "vllm_config": vllm_config, **kwargs},
        default_model=model_name,
    )

    return VLLMAgent(config)


# ============================================================================
# Example Usage
# ============================================================================


async def example_vllm_agent():
    """Example usage of VLLM agent."""
    print("Creating VLLM agent...")

    # Create agent
    agent = create_vllm_agent(
        model_name="microsoft/DialoGPT-medium",
        base_url="http://localhost:8000",
        temperature=0.8,
        max_tokens=100,
    )

    await agent.initialize()

    # Test chat
    print("\n--- Testing Chat ---")
    messages = [{"role": "user", "content": "Hello! How are you today?"}]
    response = await agent.chat(messages)
    print(f"Chat response: {response}")

    # Test completion
    print("\n--- Testing Completion ---")
    prompt = "The future of AI is"
    completion = await agent.complete(prompt)
    print(f"Completion: {completion}")

    # Test embeddings (if embedding model is available)
    if agent.config.embedding_model:
        print("\n--- Testing Embeddings ---")
        texts = ["Hello world", "AI is amazing"]
        embeddings = await agent.embed(texts)
        print(f"Generated {len(embeddings)} embeddings")
        print(f"First embedding dimension: {len(embeddings[0])}")

    print("\n✓ VLLM agent test completed!")


async def example_pydantic_ai_integration():
    """Example of using VLLM agent with Pydantic AI."""
    print("Creating VLLM agent for Pydantic AI...")

    # Create agent
    agent = create_vllm_agent(
        model_name="microsoft/DialoGPT-medium", base_url="http://localhost:8000"
    )

    await agent.initialize()

    # Convert to Pydantic AI agent
    pydantic_agent = agent.to_pydantic_ai_agent()

    print("\n--- Testing Pydantic AI Integration ---")

    # Test with dependencies
    result = await pydantic_agent.run(
        "Tell me about artificial intelligence", deps=agent.dependencies
    )

    print(f"Pydantic AI result: {result.data}")


if __name__ == "__main__":
    print("Running VLLM agent examples...")

    # Run basic example
    asyncio.run(example_vllm_agent())

    # Run Pydantic AI integration example
    asyncio.run(example_pydantic_ai_integration())

    print("All examples completed!")
