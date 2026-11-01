"""Scripted SDK clients for every provider adapter, used by the shared contract tests.

Each harness builds *real* SDK response objects (``anthropic.types.Message``,
``openai.types.responses.Response``, ``google.genai.types.GenerateContentResponse``,
``openai.types.chat.ChatCompletion``, a real ``boto3`` ``bedrock-runtime`` client behind
``botocore.stub.Stubber``) and records the native keyword arguments the adapter sent, so
the same scenario can be checked against every adapter without network access.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from copy import deepcopy
import inspect
import json
from types import SimpleNamespace
from typing import Any

import anthropic
from anthropic.resources.beta.messages import Messages as _BetaMessagesSDK
from anthropic.resources.messages import Messages as _MessagesSDK
from anthropic.types import Message as AnthropicMessage
import boto3
from botocore.exceptions import ClientError
from botocore.stub import Stubber
from botocore.validate import validate_parameters
from google.genai import errors as genai_errors
from google.genai import types as genai_types
from google.genai.models import Models as _GeminiSDK
import httpx2
import openai
from openai.resources.chat.completions import Completions as _ChatSDK
from openai.resources.responses import Responses as _ResponsesSDK
from openai.resources.responses.input_tokens import InputTokens as _InputTokensSDK
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.responses import Response

from orkexa import Agent, AsyncAgent, CompatibleOptions

SECRET = "sk-contract-test-secret-key-123456"

# Every scripted response reports: 100 prompt tokens of which 20 cached, 50 output
# tokens of which 10 reasoning. Normalized: input 80, cache read 20, output 50, thinking 10.
EXPECTED_USAGE = {
    "input_tokens": 80,
    "cache_read_input_tokens": 20,
    "output_tokens": 50,
    "thinking_tokens": 10,
}


class Script:
    """Pops scripted responses and records every call's keyword arguments."""

    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def next(self, kwargs: dict[str, Any], method: Any = None) -> Any:
        if method is not None:
            accepts(method, kwargs)
        self.calls.append(
            deepcopy({k: v for k, v in kwargs.items() if k not in ("text_format", "output_format")})
        )
        if not self.responses:
            raise AssertionError("no scripted response left")
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


def accepts(method: Any, kwargs: dict[str, Any]) -> None:
    """Fail like the real SDK method would on keyword arguments it does not take."""
    inspect.signature(method).bind(None, **kwargs)


class Harness:
    """What a contract test needs from one adapter."""

    name: str
    provider: str
    async_client: bool = False
    reports_thinking_tokens: bool = True
    """Whether the API reports reasoning tokens separately (Bedrock Converse does not)."""
    reports_model_version: bool = True
    """Whether responses name the (dated) model version that answered (Bedrock Converse does not)."""

    def __init__(self) -> None:
        self.script = Script([])

    # scripted responses
    def text(self, text: str) -> Any: ...
    def tool_calls(self, calls: list[tuple[str, str, dict[str, Any]]]) -> Any: ...
    def refusal(self) -> Any: ...
    def truncated(self, text: str) -> Any: ...
    def status_error(self, status: int) -> BaseException: ...

    # inspection of sent requests
    def tool_results(self, call: dict[str, Any]) -> list[tuple[str, str, bool]]: ...
    def native_replayed(self, call: dict[str, Any]) -> bool:
        """Whether the provider metadata of the tool-call response was sent back unchanged."""
        return True

    def user_texts(self, call: dict[str, Any]) -> list[str]: ...

    # agents
    def client(self, asynchronous: bool) -> Any: ...

    def agent_kwargs(self) -> dict[str, Any]:
        return {}

    def agent(self, responses: list[Any], **kwargs: Any) -> Agent[Any]:
        self.script = Script(responses)
        return Agent(provider=self.provider, client=self.client(False), **{**self.agent_kwargs(), **kwargs})

    def async_agent(self, responses: list[Any], **kwargs: Any) -> AsyncAgent[Any]:
        self.script = Script(responses)
        return AsyncAgent(
            provider=self.provider, client=self.client(True), **{**self.agent_kwargs(), **kwargs}
        )


def _response(status: int) -> httpx2.Response:
    return httpx2.Response(status, request=httpx2.Request("POST", "https://api.example.com/v1"))


# ---- Anthropic ------------------------------------------------------------------------------


class _AnthropicStream:
    def __init__(self, message: Any) -> None:
        self.message = message

    def _events(self) -> list[Any]:
        events: list[Any] = []
        for index, block in enumerate(self.message.content):
            events.append(SimpleNamespace(type="content_block_start", index=index, content_block=block))
            if block.type == "text":
                events.append(SimpleNamespace(type="text", text=block.text))
            elif block.type == "thinking":
                events.append(SimpleNamespace(type="thinking", thinking=block.thinking))
            elif block.type == "tool_use":
                events.append(SimpleNamespace(type="input_json", partial_json=json.dumps(block.input)))
        return events

    def __enter__(self) -> _AnthropicStream:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def __iter__(self) -> Iterator[Any]:
        return iter(self._events())

    def get_final_message(self) -> Any:
        return self.message

    async def __aenter__(self) -> _AnthropicStream:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def __aiter__(self) -> AsyncIterator[Any]:
        for event in self._events():
            yield event

    async def aget_final_message(self) -> Any:
        return self.message


class AnthropicHarness(Harness):
    name = "anthropic"
    provider = "anthropic"
    model = "claude-opus-5-5"

    def agent_kwargs(self) -> dict[str, Any]:
        return {"model": self.model}

    def _message(self, content: list[dict[str, Any]], stop_reason: str) -> Any:
        return AnthropicMessage.model_validate(
            {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": self.model,
                "content": content,
                "stop_reason": stop_reason,
                "stop_sequence": None,
                "usage": {
                    "input_tokens": 80,
                    "cache_read_input_tokens": 20,
                    "cache_creation_input_tokens": 0,
                    "output_tokens": 50,
                    "output_tokens_details": {"thinking_tokens": 10},
                },
            }
        )

    def text(self, text: str) -> Any:
        return self._message([{"type": "text", "text": text}], "end_turn")

    def tool_calls(self, calls: list[tuple[str, str, dict[str, Any]]]) -> Any:
        blocks: list[dict[str, Any]] = [{"type": "thinking", "thinking": "plan", "signature": "sig-abc"}]
        blocks += [{"type": "tool_use", "id": i, "name": n, "input": a} for i, n, a in calls]
        return self._message(blocks, "tool_use")

    def refusal(self) -> Any:
        return self._message([{"type": "text", "text": "I can't help with that."}], "refusal")

    def truncated(self, text: str) -> Any:
        return self._message([{"type": "text", "text": text}], "max_tokens")

    def status_error(self, status: int) -> BaseException:
        cls = anthropic.RateLimitError if status == 429 else anthropic.APIStatusError
        return cls(f"rate limited for key {SECRET}", response=_response(status), body=None)

    def tool_results(self, call: dict[str, Any]) -> list[tuple[str, str, bool]]:
        last = call["messages"][-1]["content"]
        return [(b["tool_use_id"], _flatten(b.get("content")), bool(b.get("is_error"))) for b in last]

    def native_replayed(self, call: dict[str, Any]) -> bool:
        blocks = [
            b
            for m in call["messages"]
            if m["role"] == "assistant"
            for b in m["content"]
            if isinstance(b, dict)
        ]
        return {"type": "thinking", "thinking": "plan", "signature": "sig-abc"} in blocks

    def user_texts(self, call: dict[str, Any]) -> list[str]:
        return [_flatten(m["content"]) for m in call["messages"] if m["role"] == "user"]

    def client(self, asynchronous: bool) -> Any:
        script = lambda: self.script  # noqa: E731

        def sdk(name: str, kwargs: dict[str, Any]) -> Any:
            return getattr(_BetaMessagesSDK if kwargs.get("betas") else _MessagesSDK, name)

        class Messages:
            def create(self, **kwargs: Any) -> Any:
                return script().next(kwargs, sdk("create", kwargs))

            def parse(self, **kwargs: Any) -> Any:
                message = script().next(kwargs, sdk("parse", kwargs))
                model = kwargs["output_format"]
                parsed = (
                    model.model_validate_json(message.content[0].text)
                    if message.stop_reason == "end_turn"
                    else None
                )
                return _Parsed(message, parsed)

            def stream(self, **kwargs: Any) -> Any:
                return _AnthropicStream(script().next(kwargs, sdk("stream", kwargs)))

            def count_tokens(self, **kwargs: Any) -> Any:
                accepts(sdk("count_tokens", kwargs), kwargs)
                script().calls.append(deepcopy({k: v for k, v in kwargs.items() if k != "output_format"}))
                return SimpleNamespace(input_tokens=321)

        class AsyncMessages(Messages):
            async def create(self, **kwargs: Any) -> Any:  # type: ignore[override]
                return super().create(**kwargs)

            async def parse(self, **kwargs: Any) -> Any:  # type: ignore[override]
                return super().parse(**kwargs)

            def stream(self, **kwargs: Any) -> Any:
                stream = super().stream(**kwargs)
                stream.get_final_message = stream.aget_final_message  # type: ignore[method-assign]
                return stream

            async def count_tokens(self, **kwargs: Any) -> Any:  # type: ignore[override]
                return super().count_tokens(**kwargs)

        messages = AsyncMessages() if asynchronous else Messages()
        return SimpleNamespace(messages=messages, beta=SimpleNamespace(messages=messages))


class _Parsed:
    def __init__(self, message: Any, parsed: Any) -> None:
        self._message = message
        self.parsed_output = parsed

    def __getattr__(self, name: str) -> Any:
        return getattr(self._message, name)


def _flatten(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(c.get("text", "")) for c in content if isinstance(c, dict))
    return "" if content is None else str(content)


# ---- OpenAI Responses -----------------------------------------------------------------------


class _ResponsesStream:
    def __init__(self, response: Response) -> None:
        self.response = response

    def _events(self) -> list[Any]:
        events: list[Any] = []
        for index, item in enumerate(self.response.output):
            if item.type == "message":
                for content in item.content:
                    if content.type == "output_text":
                        events.append(SimpleNamespace(type="response.output_text.delta", delta=content.text))
            elif item.type == "reasoning":
                for summary in item.summary:
                    events.append(
                        SimpleNamespace(type="response.reasoning_summary_text.delta", delta=summary.text)
                    )
            elif item.type == "function_call":
                events.append(
                    SimpleNamespace(type="response.output_item.added", item=item, output_index=index)
                )
                events.append(
                    SimpleNamespace(
                        type="response.function_call_arguments.delta",
                        item_id=item.id,
                        output_index=index,
                        delta=item.arguments,
                    )
                )
        if self.response.status in ("incomplete", "failed"):
            events.append(SimpleNamespace(type=f"response.{self.response.status}", response=self.response))
        return events

    def __enter__(self) -> _ResponsesStream:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def __iter__(self) -> Iterator[Any]:
        return iter(self._events())

    def get_final_response(self) -> Response:
        if self.response.status != "completed":  # like the SDK's ResponseStream
            raise RuntimeError("Didn't receive a `response.completed` event.")
        return self.response

    async def __aenter__(self) -> _ResponsesStream:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def __aiter__(self) -> AsyncIterator[Any]:
        for event in self._events():
            yield event


class OpenAIHarness(Harness):
    name = "openai"
    provider = "openai"
    model = "gpt-5.5"

    def agent_kwargs(self) -> dict[str, Any]:
        return {"model": self.model}

    def _response(
        self, output: list[dict[str, Any]], status: str = "completed", reason: str | None = None
    ) -> Response:
        return Response.model_validate(
            {
                "id": "resp_1",
                "created_at": 0,
                "model": self.model,
                "object": "response",
                "output": output,
                "parallel_tool_calls": True,
                "tool_choice": "auto",
                "tools": [],
                "status": status,
                "incomplete_details": {"reason": reason} if reason else None,
                "usage": {
                    "input_tokens": 100,
                    "input_tokens_details": {"cached_tokens": 20, "cache_write_tokens": 0},
                    "output_tokens": 50,
                    "output_tokens_details": {"reasoning_tokens": 10},
                    "total_tokens": 150,
                },
            }
        )

    @staticmethod
    def _message(text: str, kind: str = "output_text") -> dict[str, Any]:
        content = (
            {"type": kind, "text": text, "annotations": []}
            if kind == "output_text"
            else {"type": kind, "refusal": text}
        )
        return {
            "type": "message",
            "id": "msg_1",
            "role": "assistant",
            "status": "completed",
            "content": [content],
        }

    def text(self, text: str) -> Any:
        return self._response([self._message(text)])

    def tool_calls(self, calls: list[tuple[str, str, dict[str, Any]]]) -> Any:
        output: list[dict[str, Any]] = [
            {
                "type": "reasoning",
                "id": "rs_1",
                "summary": [{"type": "summary_text", "text": "plan"}],
                "encrypted_content": "ENCRYPTED",
            }
        ]
        output += [
            {
                "type": "function_call",
                "id": f"fc_{i}",
                "call_id": i,
                "name": n,
                "arguments": json.dumps(a),
                "status": "completed",
            }
            for i, n, a in calls
        ]
        return self._response(output)

    def refusal(self) -> Any:
        return self._response([self._message("I can't help with that.", "refusal")])

    def truncated(self, text: str) -> Any:
        return self._response([self._message(text)], status="incomplete", reason="max_output_tokens")

    def status_error(self, status: int) -> BaseException:
        cls = openai.RateLimitError if status == 429 else openai.APIStatusError
        return cls(f"rate limited for key {SECRET}", response=_response(status), body=None)

    def tool_results(self, call: dict[str, Any]) -> list[tuple[str, str, bool]]:
        items = [i for i in call["input"] if i.get("type") == "function_call_output"]
        last_round = _last_round(call["input"])
        return [
            (i["call_id"], str(i["output"]).removeprefix("Error: "), str(i["output"]).startswith("Error"))
            for i in items
            if id(i) in last_round
        ]

    def native_replayed(self, call: dict[str, Any]) -> bool:
        return any(
            i.get("type") == "reasoning" and i.get("encrypted_content") == "ENCRYPTED" for i in call["input"]
        )

    def user_texts(self, call: dict[str, Any]) -> list[str]:
        texts: list[str] = []
        for item in call["input"]:
            if item.get("role") == "user":
                content = item["content"]
                texts.append(
                    content if isinstance(content, str) else "\n".join(c.get("text", "") for c in content)
                )
        return texts

    def client(self, asynchronous: bool) -> Any:
        script = lambda: self.script  # noqa: E731

        def parse(**kwargs: Any) -> Any:
            response = script().next(kwargs, _ResponsesSDK.parse)
            model = kwargs["text_format"]
            if response.status == "completed" and response.output[0].type == "message":
                content = response.output[0].content[0]
                if content.type == "output_text":
                    object.__setattr__(response, "output_parsed", model.model_validate_json(content.text))
            return response

        def count(**kwargs: Any) -> Any:
            accepts(_InputTokensSDK.count, kwargs)
            script().calls.append(deepcopy(kwargs))
            return SimpleNamespace(input_tokens=321)

        if asynchronous:

            async def acreate(**kwargs: Any) -> Any:
                return script().next(kwargs, _ResponsesSDK.create)

            async def aparse(**kwargs: Any) -> Any:
                return parse(**kwargs)

            async def acount(**kwargs: Any) -> Any:
                return count(**kwargs)

            responses = SimpleNamespace(
                create=acreate,
                parse=aparse,
                stream=lambda **kw: _AsyncResponsesStream(script().next(kw, _ResponsesSDK.stream)),
                input_tokens=SimpleNamespace(count=acount),
            )
        else:
            responses = SimpleNamespace(
                create=lambda **kw: script().next(kw, _ResponsesSDK.create),
                parse=parse,
                stream=lambda **kw: _ResponsesStream(script().next(kw, _ResponsesSDK.stream)),
                input_tokens=SimpleNamespace(count=count),
            )
        return SimpleNamespace(responses=responses)


class _AsyncResponsesStream(_ResponsesStream):
    async def get_final_response(self) -> Response:  # type: ignore[override]
        return super().get_final_response()


def _last_round(items: list[dict[str, Any]]) -> set[int]:
    """Ids of the function_call_output items after the last function call batch."""
    result: set[int] = set()
    for item in reversed(items):
        if item.get("type") == "function_call_output":
            result.add(id(item))
        elif result:
            break
    return result


# ---- Gemini ---------------------------------------------------------------------------------


class GeminiHarness(Harness):
    name = "gemini"
    provider = "gemini"
    model = "gemini-3-pro"

    def agent_kwargs(self) -> dict[str, Any]:
        return {"model": self.model}

    def _response(self, parts: list[dict[str, Any]], finish: str = "STOP") -> Any:
        return genai_types.GenerateContentResponse.model_validate(
            {
                "candidates": [{"content": {"role": "model", "parts": parts}, "finish_reason": finish}],
                "usage_metadata": {
                    "prompt_token_count": 100,
                    "cached_content_token_count": 20,
                    "candidates_token_count": 40,
                    "thoughts_token_count": 10,
                },
                "model_version": self.model,
                "response_id": "gem-1",
            }
        )

    def text(self, text: str) -> Any:
        return self._response([{"text": text}])

    def tool_calls(self, calls: list[tuple[str, str, dict[str, Any]]]) -> Any:
        parts: list[dict[str, Any]] = [{"text": "plan", "thought": True}]
        for position, (call_id, name, args) in enumerate(calls):
            part: dict[str, Any] = {"function_call": {"id": call_id, "name": name, "args": args}}
            if position == 0:
                part["thought_signature"] = "c2lnbmF0dXJl"  # base64 of "signature"
            parts.append(part)
        return self._response(parts)

    def refusal(self) -> Any:
        return self._response([], finish="SAFETY")

    def truncated(self, text: str) -> Any:
        return self._response([{"text": text}], finish="MAX_TOKENS")

    def status_error(self, status: int) -> BaseException:
        return genai_errors.ClientError(
            status, {"error": {"message": f"quota for key {SECRET}", "status": "RESOURCE_EXHAUSTED"}}
        )

    def tool_results(self, call: dict[str, Any]) -> list[tuple[str, str, bool]]:
        last = call["contents"][-1]
        results = []
        for part in last.parts:
            response = part.function_response
            if response is not None:
                payload = response.response or {}
                results.append(
                    (
                        str(response.id),
                        str(payload.get("result", payload.get("error", ""))),
                        "error" in payload,
                    )
                )
        return results

    def native_replayed(self, call: dict[str, Any]) -> bool:
        for content in call["contents"]:
            for part in content.parts or []:
                if part.function_call is not None and part.thought_signature == b"signature":
                    return True
        return False

    def user_texts(self, call: dict[str, Any]) -> list[str]:
        return [
            "\n".join(p.text for p in c.parts if p.text is not None)
            for c in call["contents"]
            if c.role == "user" and any(p.text is not None for p in c.parts)
        ]

    def client(self, asynchronous: bool) -> Any:
        script = lambda: self.script  # noqa: E731

        def stream(**kwargs: Any) -> Iterator[Any]:
            response = script().next(kwargs, _GeminiSDK.generate_content_stream)
            parts = response.candidates[0].content.parts or []
            for index, part in enumerate(parts):
                last = index == len(parts) - 1
                yield genai_types.GenerateContentResponse(
                    candidates=[
                        genai_types.Candidate(
                            content=genai_types.Content(role="model", parts=[part]),
                            finish_reason=response.candidates[0].finish_reason if last else None,
                        )
                    ],
                    usage_metadata=response.usage_metadata if last else None,
                    model_version=response.model_version,
                )
            if not parts:
                yield response

        def count(**kwargs: Any) -> Any:
            accepts(_GeminiSDK.count_tokens, kwargs)
            script().calls.append(deepcopy(kwargs))
            return SimpleNamespace(total_tokens=321)

        models = SimpleNamespace(
            generate_content=lambda **kw: script().next(kw, _GeminiSDK.generate_content),
            generate_content_stream=stream,
            count_tokens=count,
        )

        async def agenerate(**kwargs: Any) -> Any:
            return script().next(kwargs, _GeminiSDK.generate_content)

        async def astream(**kwargs: Any) -> AsyncIterator[Any]:
            async def chunks() -> AsyncIterator[Any]:
                for chunk in stream(**kwargs):
                    yield chunk

            return chunks()

        async def acount(**kwargs: Any) -> Any:
            return count(**kwargs)

        aio = SimpleNamespace(
            models=SimpleNamespace(
                generate_content=agenerate, generate_content_stream=astream, count_tokens=acount
            )
        )
        return SimpleNamespace(models=models, aio=aio, vertexai=False)


# ---- Chat Completions (Ollama preset) ---------------------------------------------------------


class ChatHarness(Harness):
    name = "ollama-chat"
    provider = "ollama"
    model = "llama3.1:8b"

    def agent_kwargs(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "capabilities": {
                "tools": "supported",
                "forced_tool_choice": "supported",
                "parallel_tool_calls": "supported",
                "image_input": "supported",
                "token_counting": "unsupported",
            },
        }

    def _completion(self, message: dict[str, Any], finish: str) -> ChatCompletion:
        return ChatCompletion.model_validate(
            {
                "id": "chat-1",
                "object": "chat.completion",
                "created": 0,
                "model": self.model,
                "choices": [
                    {"index": 0, "message": {"role": "assistant", **message}, "finish_reason": finish}
                ],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                    "total_tokens": 150,
                    "prompt_tokens_details": {"cached_tokens": 20},
                    "completion_tokens_details": {"reasoning_tokens": 10},
                },
            }
        )

    def text(self, text: str) -> Any:
        return self._completion({"content": text}, "stop")

    def tool_calls(self, calls: list[tuple[str, str, dict[str, Any]]]) -> Any:
        return self._completion(
            {
                "content": None,
                "tool_calls": [
                    {"id": i, "type": "function", "function": {"name": n, "arguments": json.dumps(a)}}
                    for i, n, a in calls
                ],
            },
            "tool_calls",
        )

    def refusal(self) -> Any:
        return self._completion({"content": None, "refusal": "I can't help with that."}, "stop")

    def truncated(self, text: str) -> Any:
        return self._completion({"content": text}, "length")

    def status_error(self, status: int) -> BaseException:
        cls = openai.RateLimitError if status == 429 else openai.APIStatusError
        return cls(f"rate limited for key {SECRET}", response=_response(status), body=None)

    def tool_results(self, call: dict[str, Any]) -> list[tuple[str, str, bool]]:
        results: list[tuple[str, str, bool]] = []
        for message in reversed(call["messages"]):
            if message["role"] != "tool":
                break
            text = message["content"]
            results.insert(
                0, (message["tool_call_id"], text.removeprefix("Error: "), text.startswith("Error"))
            )
        return results

    def user_texts(self, call: dict[str, Any]) -> list[str]:
        return [
            m["content"]
            if isinstance(m["content"], str)
            else "\n".join(c.get("text", "") for c in m["content"])
            for m in call["messages"]
            if m["role"] == "user"
        ]

    def _chunks(self, completion: ChatCompletion) -> list[ChatCompletionChunk]:
        message = completion.choices[0].message
        chunks: list[dict[str, Any]] = []
        if message.content:
            middle = len(message.content) // 2 or 1
            for piece in (message.content[:middle], message.content[middle:]):
                if piece:
                    chunks.append({"content": piece})
        if message.refusal:
            chunks.append({"refusal": message.refusal})
        for index, call in enumerate(message.tool_calls or []):
            chunks.append(
                {
                    "tool_calls": [
                        {
                            "index": index,
                            "id": call.id,
                            "type": "function",
                            "function": {"name": call.function.name, "arguments": ""},
                        }
                    ]
                }
            )
            chunks.append(
                {"tool_calls": [{"index": index, "function": {"arguments": call.function.arguments}}]}
            )
        result = [
            ChatCompletionChunk.model_validate(
                {
                    "id": "chat-1",
                    "object": "chat.completion.chunk",
                    "created": 0,
                    "model": self.model,
                    "choices": [{"index": 0, "delta": delta}],
                }
            )
            for delta in chunks
        ]
        result.append(
            ChatCompletionChunk.model_validate(
                {
                    "id": "chat-1",
                    "object": "chat.completion.chunk",
                    "created": 0,
                    "model": self.model,
                    "choices": [
                        {"index": 0, "delta": {}, "finish_reason": completion.choices[0].finish_reason}
                    ],
                    "usage": completion.usage.model_dump() if completion.usage else None,
                }
            )
        )
        return result

    def client(self, asynchronous: bool) -> Any:
        script = lambda: self.script  # noqa: E731

        def create(**kwargs: Any) -> Any:
            completion = script().next(kwargs, _ChatSDK.create)
            return iter(self._chunks(completion)) if kwargs.get("stream") else completion

        async def acreate(**kwargs: Any) -> Any:
            result = create(**kwargs)
            if not kwargs.get("stream"):
                return result

            async def chunks() -> AsyncIterator[Any]:
                for chunk in result:
                    yield chunk

            return chunks()

        completions = SimpleNamespace(create=acreate if asynchronous else create)
        return SimpleNamespace(chat=SimpleNamespace(completions=completions))


class CompatibleResponsesHarness(OpenAIHarness):
    """A compatible endpoint that implements the Responses API (``api="responses"``)."""

    name = "compatible-responses"
    provider = "compatible"
    model = "my-model"

    def agent_kwargs(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "base_url": "http://localhost:9000/v1",
            "options": CompatibleOptions(api="responses"),
            "capabilities": {
                "tools": "supported",
                "forced_tool_choice": "supported",
                "parallel_tool_calls": "supported",
                "structured_output": "supported",
                "streaming": "supported",
                "strict_tool_schemas": "supported",
                "image_input": "supported",
                "system_messages": "supported",
            },
        }

    def native_replayed(self, call: dict[str, Any]) -> bool:
        return any(i.get("type") == "reasoning" for i in call["input"])


# ---- Amazon Bedrock (Converse) -----------------------------------------------------------------

#: Fake AWS credentials: requests are answered by a Stubber and never signed or sent.
FAKE_AWS = {
    "aws_access_key_id": "AKIAIOSFODNN7EXAMPLE",
    "aws_secret_access_key": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
}


def bedrock_client(service: str = "bedrock-runtime", region: str = "eu-central-1", **kwargs: Any) -> Any:
    """A real boto3 client with fake credentials (use it only behind a Stubber or a patch)."""
    return boto3.client(service, region_name=region, **FAKE_AWS, **kwargs)


def converse_response(
    content: list[dict[str, Any]],
    stop: str = "end_turn",
    *,
    usage: dict[str, Any] | None = None,
    request_id: str = "req-bedrock-1",
    **extra: Any,
) -> dict[str, Any]:
    """A Converse response as boto3 returns it."""
    return {
        "output": {"message": {"role": "assistant", "content": content}},
        "stopReason": stop,
        "usage": usage
        or {"inputTokens": 80, "outputTokens": 50, "totalTokens": 150, "cacheReadInputTokens": 20},
        "metrics": {"latencyMs": 12},
        "ResponseMetadata": {"RequestId": request_id, "HTTPStatusCode": 200},
        **extra,
    }


def converse_events(response: dict[str, Any]) -> list[dict[str, Any]]:
    """The ConverseStream events that carry the same answer as a Converse ``response``."""
    events: list[dict[str, Any]] = [{"messageStart": {"role": "assistant"}}]
    for index, block in enumerate(response["output"]["message"]["content"]):
        if "text" in block:
            text = block["text"]
            middle = len(text) // 2
            for piece in (text[:middle], text[middle:]):
                if piece:
                    events.append(
                        {"contentBlockDelta": {"contentBlockIndex": index, "delta": {"text": piece}}}
                    )
        elif "toolUse" in block:
            use = block["toolUse"]
            events.append(
                {
                    "contentBlockStart": {
                        "contentBlockIndex": index,
                        "start": {"toolUse": {"toolUseId": use["toolUseId"], "name": use["name"]}},
                    }
                }
            )
            arguments = json.dumps(use["input"])
            middle = len(arguments) // 2
            for piece in (arguments[:middle], arguments[middle:]):
                events.append(
                    {
                        "contentBlockDelta": {
                            "contentBlockIndex": index,
                            "delta": {"toolUse": {"input": piece}},
                        }
                    }
                )
        elif "reasoningContent" in block:
            reasoning = block["reasoningContent"]
            if "redactedContent" in reasoning:
                delta: dict[str, Any] = {"redactedContent": reasoning["redactedContent"]}
                events.append(
                    {"contentBlockDelta": {"contentBlockIndex": index, "delta": {"reasoningContent": delta}}}
                )
            else:
                text_block = reasoning["reasoningText"]
                for delta in ({"text": text_block["text"]}, {"signature": text_block.get("signature", "")}):
                    events.append(
                        {
                            "contentBlockDelta": {
                                "contentBlockIndex": index,
                                "delta": {"reasoningContent": delta},
                            }
                        }
                    )
        events.append({"contentBlockStop": {"contentBlockIndex": index}})
    events.append({"messageStop": {"stopReason": response["stopReason"]}})
    metadata: dict[str, Any] = {"usage": response["usage"], "metrics": response["metrics"]}
    if "serviceTier" in response:
        metadata["serviceTier"] = response["serviceTier"]
    events.append({"metadata": metadata})
    return events


class FakeEventStream:
    """Stands in for botocore's ``EventStream``: iterable, and records ``close()``."""

    def __init__(self, events: list[Any]) -> None:
        self.events = list(events)
        self.closed = False
        self.consumed = 0

    def __iter__(self) -> Iterator[Any]:
        for event in self.events:
            if self.closed:
                return
            self.consumed += 1
            if isinstance(event, BaseException):
                raise event
            yield event

    def close(self) -> None:
        self.closed = True


class BedrockHarness(Harness):
    """A real ``bedrock-runtime`` client: Converse answered by ``Stubber`` (which validates
    the request against the service model), ConverseStream by a patched method that
    validates the request the same way and returns a :class:`FakeEventStream`."""

    name = "bedrock"
    provider = "bedrock"
    model = "eu.anthropic.claude-sonnet-4-5-20250929-v1:0"
    reports_thinking_tokens = False
    reports_model_version = False

    def __init__(self) -> None:
        super().__init__()
        self.streams: list[FakeEventStream] = []

    def agent_kwargs(self) -> dict[str, Any]:
        return {"model": self.model}

    def text(self, text: str) -> Any:
        return converse_response([{"text": text}])

    def tool_calls(self, calls: list[tuple[str, str, dict[str, Any]]]) -> Any:
        blocks: list[dict[str, Any]] = [
            {"reasoningContent": {"reasoningText": {"text": "plan", "signature": "sig-abc"}}}
        ]
        blocks += [{"toolUse": {"toolUseId": i, "name": n, "input": a}} for i, n, a in calls]
        return converse_response(blocks, "tool_use")

    def refusal(self) -> Any:
        return converse_response(
            [{"text": "Sorry, the model cannot answer this question."}], "guardrail_intervened"
        )

    def truncated(self, text: str) -> Any:
        return converse_response([{"text": text}], "max_tokens")

    def status_error(self, status: int) -> BaseException:
        code = "ThrottlingException" if status == 429 else "InternalServerException"
        return ClientError(
            {
                "Error": {"Code": code, "Message": f"rate limited for key {SECRET}"},
                "ResponseMetadata": {"HTTPStatusCode": status, "RequestId": "req-err"},
            },
            "Converse",
        )

    def tool_results(self, call: dict[str, Any]) -> list[tuple[str, str, bool]]:
        results: list[tuple[str, str, bool]] = []
        for block in call["messages"][-1]["content"]:
            result = block.get("toolResult")
            if result is None:
                continue
            texts = [c["text"] for c in result["content"] if "text" in c]
            error = result.get("status") == "error" or (bool(texts) and texts[0] == "Error:")
            if texts and texts[0] == "Error:":
                texts = texts[1:]
            results.append((result["toolUseId"], "\n".join(texts), error))
        return results

    def native_replayed(self, call: dict[str, Any]) -> bool:
        blocks = [b for m in call["messages"] if m["role"] == "assistant" for b in m["content"]]
        return {"reasoningContent": {"reasoningText": {"text": "plan", "signature": "sig-abc"}}} in blocks

    def user_texts(self, call: dict[str, Any]) -> list[str]:
        texts: list[str] = []
        for message in call["messages"]:
            if message["role"] != "user":
                continue
            pieces = [b["text"] for b in message["content"] if "text" in b]
            if pieces:
                texts.append("\n".join(pieces))
        return texts

    def client(self, asynchronous: bool) -> Any:
        """The same synchronous boto3 client serves ``Agent`` and ``AsyncAgent``."""
        return self.scripted_client()

    def scripted_client(self, region: str = "eu-central-1", **kwargs: Any) -> Any:
        script = lambda: self.script  # noqa: E731
        client = bedrock_client(region=region, **kwargs)
        stubber = Stubber(client)
        stubber.activate()

        def provide(params: dict[str, Any], model: Any, **_: Any) -> None:
            if model.name != "Converse":
                return
            response = _next(script(), params)
            if isinstance(response, ClientError):
                error = response.response
                stubber.add_client_error(
                    "converse",
                    service_error_code=error["Error"]["Code"],
                    service_message=error["Error"]["Message"],
                    http_status_code=error.get("ResponseMetadata", {}).get("HTTPStatusCode", 400),
                )
            elif isinstance(response, BaseException):
                raise response
            else:
                stubber.add_response("converse", response)

        def converse_stream(**params: Any) -> Any:
            validate_parameters(
                params, client.meta.service_model.operation_model("ConverseStream").input_shape
            )
            response = _next(script(), params)
            if isinstance(response, BaseException):
                raise response
            stream = FakeEventStream(converse_events(response))
            self.streams.append(stream)
            return {"stream": stream, "ResponseMetadata": response["ResponseMetadata"]}

        client.meta.events.register("provide-client-params.bedrock-runtime.Converse", provide)
        client.converse_stream = converse_stream
        client.stubber = stubber
        return client


def _next(script: Script, params: dict[str, Any]) -> Any:
    script.calls.append(deepcopy(params))
    if not script.responses:
        raise AssertionError("no scripted response left")
    return script.responses.pop(0)


HARNESSES: list[type[Harness]] = [
    AnthropicHarness,
    OpenAIHarness,
    GeminiHarness,
    ChatHarness,
    CompatibleResponsesHarness,
    BedrockHarness,
]
