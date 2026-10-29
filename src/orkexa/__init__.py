"""Orkexa: one agent runtime for Anthropic Claude, OpenAI, Google Gemini and
OpenAI-compatible endpoints.

Quick start::

    from orkexa import Agent, Tool

    @Tool
    def add(a: int, b: int) -> int:
        \"\"\"Add two integers.

        Args:
            a: First summand.
            b: Second summand.

        Returns:
            The sum of a and b.
        \"\"\"
        return a + b

    agent = Agent(model="openai:gpt-5.5", tools=[add])   # or "anthropic:...", "gemini:..."
    print(agent.ask("What is 17 + 25?"))

``ClaudeAgent`` / ``AsyncClaudeAgent`` keep the ``claude-agent`` 0.8.0 API.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from .agent import Agent, RefusalMode, ToolCall
from .async_agent import AsyncAgent
from .content import ContentBlock, Prompt, ToolContent, document, image, uploaded_file
from .core.capabilities import ModelCapabilities, Support
from .core.conversion import HistoryConversionWarning
from .core.types import (
    Effort,
    Message,
    ModelRequest,
    ModelResponse,
    ReasoningDelta,
    ResponseCompleted,
    StreamEvent,
    TextDelta,
    ToolCallDelta,
    ToolCallStarted,
    ToolChoice,
    TurnCompleted,
    TurnFailed,
    UsageReported,
)
from .errors import (
    AuthenticationError,
    ClaudeAgentError,
    ConcurrentTurnError,
    ConfigurationError,
    EvaluationFailedError,
    HistoryConversionError,
    IncompleteResponseError,
    InvalidRequestError,
    MaxToolRoundsError,
    MemoryToolError,
    MissingDependencyError,
    OrkexaError,
    ProviderConnectionError,
    ProviderError,
    ProviderServerError,
    RateLimitError,
    RefusalError,
    StructuredOutputError,
    ToolDefinitionError,
    ToolExecutionError,
    ToolInputError,
    ToolResultError,
    ToolSchemaError,
    UnexpectedResponseError,
    UnknownCapabilityError,
    UnsupportedCapabilityError,
)
from .evaluation import (
    Criterion,
    CriterionResult,
    Evaluation,
    EvaluationCase,
    Evaluator,
    Refinement,
    refine,
)
from .files import FileStore, LocalFileStore, SQLiteFileStore, StoredFile
from .memory import EphemeralMemory, FileMemory, MemoryBackend, SQLiteMemory
from .pricing import MODEL_PRICING, ModelPricing, register_pricing, resolve_pricing
from .providers import (
    AnthropicOptions,
    CompatibleOptions,
    GeminiOptions,
    OpenAIOptions,
    Provider,
    ProviderConfig,
    register_provider,
)
from .retrieval import Chunk, Citation, Hit, Index, SQLiteIndex, citations, split_text
from .tools import Tool
from .usage import UsageRecord, UsageSummary, UsageTracker

try:
    __version__ = version("orkexa")
except PackageNotFoundError:  # pragma: no cover - running from a source checkout
    __version__ = "0.0.0"

__all__ = [
    "MODEL_PRICING",
    "Agent",
    "AnthropicOptions",
    "AsyncAgent",
    "AuthenticationError",
    "Chunk",
    "Citation",
    "ClaudeAgentError",
    "CompatibleOptions",
    "ConcurrentTurnError",
    "ConfigurationError",
    "ContentBlock",
    "Criterion",
    "CriterionResult",
    "Effort",
    "EphemeralMemory",
    "Evaluation",
    "EvaluationCase",
    "EvaluationFailedError",
    "Evaluator",
    "FileMemory",
    "FileStore",
    "GeminiOptions",
    "HistoryConversionError",
    "HistoryConversionWarning",
    "Hit",
    "IncompleteResponseError",
    "Index",
    "InvalidRequestError",
    "LocalFileStore",
    "MaxToolRoundsError",
    "MemoryBackend",
    "MemoryToolError",
    "Message",
    "MissingDependencyError",
    "ModelCapabilities",
    "ModelPricing",
    "ModelRequest",
    "ModelResponse",
    "OpenAIOptions",
    "OrkexaError",
    "Prompt",
    "Provider",
    "ProviderConfig",
    "ProviderConnectionError",
    "ProviderError",
    "ProviderServerError",
    "RateLimitError",
    "ReasoningDelta",
    "Refinement",
    "RefusalError",
    "RefusalMode",
    "ResponseCompleted",
    "SQLiteFileStore",
    "SQLiteIndex",
    "SQLiteMemory",
    "StoredFile",
    "StreamEvent",
    "StructuredOutputError",
    "Support",
    "TextDelta",
    "Tool",
    "ToolCall",
    "ToolCallDelta",
    "ToolCallStarted",
    "ToolChoice",
    "ToolContent",
    "ToolDefinitionError",
    "ToolExecutionError",
    "ToolInputError",
    "ToolResultError",
    "ToolSchemaError",
    "TurnCompleted",
    "TurnFailed",
    "UnexpectedResponseError",
    "UnknownCapabilityError",
    "UnsupportedCapabilityError",
    "UsageRecord",
    "UsageReported",
    "UsageSummary",
    "UsageTracker",
    "__version__",
    "citations",
    "document",
    "image",
    "refine",
    "register_pricing",
    "register_provider",
    "resolve_pricing",
    "split_text",
    "uploaded_file",
]
