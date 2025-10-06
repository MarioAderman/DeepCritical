"""
Pydantic AI model wrapper for vLLM.

This module provides a VLLMModel class that implements the Pydantic AI Model interface,
allowing vLLM to be used as a model provider in Pydantic AI agents.
"""

from __future__ import annotations

import datetime
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import Model, StreamedResponse
from pydantic_ai.models.base import ModelRequestParameters, ModelSettings
from pydantic_ai.models.usage import Usage
from pydantic_ai.tools import ToolDefinition

from ..utils.vllm_client import VLLMClient, ChatCompletionRequest


@dataclass
class VLLMModel(Model):
    """Pydantic AI model that uses vLLM for inference.

    This model wraps the existing VLLMClient to make it compatible with
    Pydantic AI's Model interface.

    Example:
        ```python
        from pydantic_ai import Agent
        from DeepResearch.src.models import VLLMModel
        from DeepResearch.src.utils.vllm_client import VLLMClient

        # Create vLLM client
        client = VLLMClient(base_url="http://localhost:8000")

        # Create Pydantic AI model
        model = VLLMModel(client=client, model_name="meta-llama/Llama-3-8B")

        # Use with agent
        agent = Agent(model)
        result = agent.run_sync("Hello!")
        ```
    """

    client: VLLMClient = field(repr=False)
    """The vLLM client to use for inference."""

    model_name: str
    """The name of the vLLM model to use (e.g., 'meta-llama/Llama-3-8B')."""

    _system: str = field(default="vllm", repr=False)
    """The system identifier for OpenTelemetry."""

    def __post_init__(self):
        """Initialize the model after dataclass initialization."""
        super().__init__()

    @property
    def system(self) -> str:
        """The system / model provider identifier."""
        return self._system

    @property
    def base_url(self) -> str | None:
        """The base URL for the vLLM server."""
        return self.client.base_url

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        """Make a request to the vLLM model.

        Args:
            messages: The conversation history as Pydantic AI messages.
            model_settings: Optional model settings (temperature, max_tokens, etc.).
            model_request_parameters: Request parameters including tools and output configuration.

        Returns:
            A ModelResponse containing the model's reply.
        """
        # Convert Pydantic AI messages to vLLM chat format
        chat_messages = self._convert_messages(messages)

        # Build request parameters
        request_params = self._build_request_params(
            chat_messages, model_settings, model_request_parameters
        )

        # Make request to vLLM
        chat_request = ChatCompletionRequest(
            model=self.model_name, messages=chat_messages, **request_params
        )

        response = await self.client.chat_completions(chat_request)

        # Convert vLLM response to Pydantic AI format
        return self._convert_response(response)

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: Any | None = None,
    ) -> AsyncIterator[StreamedResponse]:
        """Make a streaming request to the vLLM model.

        Args:
            messages: The conversation history.
            model_settings: Optional model settings.
            model_request_parameters: Request parameters.
            run_context: Optional run context (unused by vLLM).

        Yields:
            A StreamedResponse for iterating over response chunks.
        """
        # Convert messages
        chat_messages = self._convert_messages(messages)

        # Build request parameters
        request_params = self._build_request_params(
            chat_messages, model_settings, model_request_parameters
        )

        # Create streaming request
        chat_request = ChatCompletionRequest(
            model=self.model_name, messages=chat_messages, stream=True, **request_params
        )

        # Yield streamed response
        yield VLLMStreamedResponse(
            client=self.client,
            request=chat_request,
            model_name=self.model_name,
            model_request_parameters=model_request_parameters,
        )

    def _convert_messages(self, messages: list[ModelMessage]) -> list[dict[str, str]]:
        """Convert Pydantic AI messages to vLLM chat format.

        Args:
            messages: Pydantic AI messages.

        Returns:
            List of chat messages in vLLM format.
        """
        chat_messages = []

        for msg in messages:
            if isinstance(msg, ModelRequest):
                # User message
                for part in msg.parts:
                    if isinstance(part, UserPromptPart):
                        chat_messages.append({"role": "user", "content": part.content})
                    elif isinstance(part, ToolReturnPart):
                        # Tool results are handled as system messages
                        chat_messages.append(
                            {
                                "role": "system",
                                "content": f"Tool {part.tool_name} returned: {part.content}",
                            }
                        )
            elif isinstance(msg, ModelResponse):
                # Assistant message
                content_parts = []
                for part in msg.parts:
                    if isinstance(part, TextPart):
                        content_parts.append(part.content)
                    elif isinstance(part, ToolCallPart):
                        # Represent tool calls in content
                        content_parts.append(
                            f"[Calling tool {part.tool_name} with args {part.args}]"
                        )

                if content_parts:
                    chat_messages.append(
                        {"role": "assistant", "content": " ".join(content_parts)}
                    )

        return chat_messages

    def _build_request_params(
        self,
        chat_messages: list[dict[str, str]],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> dict[str, Any]:
        """Build vLLM request parameters from Pydantic AI settings.

        Args:
            chat_messages: The chat messages.
            model_settings: Optional model settings.
            model_request_parameters: Request parameters.

        Returns:
            Dictionary of vLLM request parameters.
        """
        params: dict[str, Any] = {}

        # Apply model settings
        if model_settings:
            if "temperature" in model_settings:
                params["temperature"] = model_settings["temperature"]
            if "max_tokens" in model_settings:
                params["max_tokens"] = model_settings["max_tokens"]
            if "top_p" in model_settings:
                params["top_p"] = model_settings["top_p"]

        # Note: vLLM doesn't natively support function calling like OpenAI,
        # so we handle tools by adding them to the system prompt
        if model_request_parameters.function_tools:
            tools_description = self._format_tools_for_prompt(
                model_request_parameters.function_tools
            )
            # Prepend tools description to first message
            if chat_messages and chat_messages[0]["role"] == "user":
                chat_messages[0]["content"] = (
                    f"{tools_description}\n\n{chat_messages[0]['content']}"
                )

        return params

    def _format_tools_for_prompt(self, tools: list[ToolDefinition]) -> str:
        """Format tools as a prompt for the model.

        Since vLLM doesn't support native function calling, we describe
        tools in the prompt and ask the model to output tool calls as JSON.

        Args:
            tools: List of available tools.

        Returns:
            Formatted tools description.
        """
        if not tools:
            return ""

        tools_text = "Available tools:\n"
        for tool in tools:
            tools_text += f"\n- {tool.name}: {tool.description}\n"
            tools_text += f"  Parameters: {tool.parameters_json_schema}\n"

        tools_text += "\nTo use a tool, respond with JSON: "
        tools_text += '{"tool": "tool_name", "args": {...}}'

        return tools_text

    def _convert_response(self, response: Any) -> ModelResponse:
        """Convert vLLM response to Pydantic AI ModelResponse.

        Args:
            response: vLLM ChatCompletionResponse.

        Returns:
            Pydantic AI ModelResponse.
        """
        # Extract text from response
        content = response.choices[0].message.content

        # Build response parts
        parts = [TextPart(content=content)]

        # Build usage stats
        usage = Usage(
            request_tokens=response.usage.prompt_tokens,
            response_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens,
        )

        return ModelResponse(
            parts=parts,
            model_name=self.model_name,
            timestamp=datetime.datetime.now(datetime.timezone.utc),
            usage=usage,
        )


@dataclass
class VLLMStreamedResponse(StreamedResponse):
    """Streamed response from vLLM."""

    client: VLLMClient
    request: ChatCompletionRequest
    model_name: str
    model_request_parameters: ModelRequestParameters

    _timestamp: datetime.datetime = field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc)
    )

    @property
    def timestamp(self) -> datetime.datetime:
        """Get the timestamp of the response."""
        return self._timestamp

    async def _get_event_iterator(self):
        """Get an async iterator of stream events.

        Yields:
            ModelResponseStreamEvent instances.
        """
        # Import here to avoid circular dependencies
        from pydantic_ai.messages import ModelResponseStreamEvent, TextPartDelta

        full_content = ""

        async for chunk in self.client.chat_completions_stream(self.request):
            full_content += chunk

            # Yield text delta event
            yield ModelResponseStreamEvent(
                part=TextPartDelta(content=chunk),
                timestamp=datetime.datetime.now(datetime.timezone.utc),
            )

        # Update usage after stream completes
        # Note: vLLM doesn't provide token counts in stream, so we estimate
        self._usage = Usage(
            request_tokens=len(full_content) // 4,  # Rough estimate
            response_tokens=len(full_content) // 4,
            total_tokens=len(full_content) // 2,
        )
