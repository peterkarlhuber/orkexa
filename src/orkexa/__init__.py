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

``ClaudeAgent`` / ``AsyncClaudeAgent`` are Anthropic-native agents with the Messages API
history format and SDK objects.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from .agent import Agent, RefusalMode, ToolCall
from .async_agent import AsyncAgent
from .claude import AsyncClaudeAgent, ClaudeAgent, Fallbacks, ThinkingDisplay, ThinkingMode
from .content import ContentBlock, Prompt, ToolContent, document, image, uploaded_file
from .core.capabilities import ModelCapabilities, ModelLimits, Support
from .core.checkpoint import CheckpointClaims, TurnCheckpoint, WaitingCall
from .core.conversion import HistoryConversionWarning
from .core.reasoning import ReasoningStream, ReasoningText
from .core.types import (
    Effort,
    LimitHandled,
    Message,
    ModelRequest,
    ModelResponse,
    ReasoningDelta,
    ReasoningKind,
    ReasoningStatus,
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
    CheckpointError,
    ConcurrentTurnError,
    ConfigurationError,
    DataResidencyError,
    EvaluationFailedError,
    HistoryConversionError,
    IncompleteResponseError,
    InvalidRequestError,
    InvalidRequestReason,
    MaxToolCallsError,
    MaxToolRoundsError,
    MemoryToolError,
    MissingCredentialsError,
    MissingDependencyError,
    OrkexaError,
    ProviderConnectionError,
    ProviderError,
    ProviderServerError,
    RateLimitError,
    RefusalError,
    RequestSuspended,
    StructuredOutputError,
    ToolCallSuspended,
    ToolDefinitionError,
    ToolExecutionError,
    ToolInputError,
    ToolResultError,
    ToolSchemaError,
    TurnSuspended,
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
    tool_results_of_last_turn,
)
from .files import FileStore, LocalFileStore, StoredFile
from .memory import EphemeralMemory, FileMemory, MemoryBackend
from .pricing import MODEL_PRICING, ModelPricing, register_pricing, resolve_pricing
from .providers import (
    AnthropicOptions,
    BedrockOptions,
    CompatibleOptions,
    GeminiOptions,
    OpenAIOptions,
    Provider,
    ProviderConfig,
    register_provider,
)
from .retrieval import Chunk, Citation, Hit, Index, citations, retrieval_sources, split_text
from .tools import Tool
from .usage import Reservation, UsageRecord, UsageSummary, UsageTracker

# isort: split
# The SQLite backends import the contract modules above (retrieval, files, memory), so they come last.
from .backends.sqlite import SQLiteFileStore, SQLiteIndex, SQLiteMemory

try:
    __version__ = version("orkexa")
except PackageNotFoundError:  # pragma: no cover - running from a source checkout
    __version__ = "0.0.0"

__all__ = [
    "MODEL_PRICING",
    "Agent",
    "AnthropicOptions",
    "AsyncAgent",
    "AsyncClaudeAgent",
    "AuthenticationError",
    "BedrockOptions",
    "CheckpointClaims",
    "CheckpointError",
    "Chunk",
    "Citation",
    "ClaudeAgent",
    "CompatibleOptions",
    "ConcurrentTurnError",
    "ConfigurationError",
    "ContentBlock",
    "Criterion",
    "CriterionResult",
    "DataResidencyError",
    "Effort",
    "EphemeralMemory",
    "Evaluation",
    "EvaluationCase",
    "EvaluationFailedError",
    "Evaluator",
    "Fallbacks",
    "FileMemory",
    "FileStore",
    "GeminiOptions",
    "HistoryConversionError",
    "HistoryConversionWarning",
    "Hit",
    "IncompleteResponseError",
    "Index",
    "InvalidRequestError",
    "InvalidRequestReason",
    "LimitHandled",
    "LocalFileStore",
    "MaxToolCallsError",
    "MaxToolRoundsError",
    "MemoryBackend",
    "MemoryToolError",
    "Message",
    "MissingCredentialsError",
    "MissingDependencyError",
    "ModelCapabilities",
    "ModelLimits",
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
    "ReasoningKind",
    "ReasoningStatus",
    "ReasoningStream",
    "ReasoningText",
    "Refinement",
    "RefusalError",
    "RefusalMode",
    "RequestSuspended",
    "Reservation",
    "ResponseCompleted",
    "SQLiteFileStore",
    "SQLiteIndex",
    "SQLiteMemory",
    "StoredFile",
    "StreamEvent",
    "StructuredOutputError",
    "Support",
    "TextDelta",
    "ThinkingDisplay",
    "ThinkingMode",
    "Tool",
    "ToolCall",
    "ToolCallDelta",
    "ToolCallStarted",
    "ToolCallSuspended",
    "ToolChoice",
    "ToolContent",
    "ToolDefinitionError",
    "ToolExecutionError",
    "ToolInputError",
    "ToolResultError",
    "ToolSchemaError",
    "TurnCheckpoint",
    "TurnCompleted",
    "TurnFailed",
    "TurnSuspended",
    "UnexpectedResponseError",
    "UnknownCapabilityError",
    "UnsupportedCapabilityError",
    "UsageRecord",
    "UsageReported",
    "UsageSummary",
    "UsageTracker",
    "WaitingCall",
    "__version__",
    "citations",
    "document",
    "image",
    "refine",
    "register_pricing",
    "register_provider",
    "resolve_pricing",
    "retrieval_sources",
    "split_text",
    "tool_results_of_last_turn",
    "uploaded_file",
]
