"""Scripted test doubles (MUP-ONA B1): no network, no SDK needed.

:class:`ScriptedClient` stands in for a provider SDK client (``anthropic.Anthropic``, ``openai.OpenAI``, ...):
every method path (``client.messages.create``, ``client.beta.messages.batches.retrieve``) answers from a
script and records the call::

    from types import SimpleNamespace
    from orkexa.testing.fakes import ScriptedClient

    client = ScriptedClient({"messages.batches.create": [SimpleNamespace(id="batch-1")]})
    agent = Agent(model="anthropic:claude-opus-5-5", client=client)
    ...
    (call,) = client.calls_to("messages.batches.create")
    call.kwargs["requests"]

For whole agent turns use the scripted provider :mod:`orkexa.providers.fake` (``FakeModel``) instead.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
import threading
from typing import Any

__all__ = ["ScriptedCall", "ScriptedClient"]


@dataclass(frozen=True)
class ScriptedCall:
    """One recorded call: the dotted method path and its arguments (deep copies, as they were at the call)."""

    path: str
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=dict)


class _Path:
    """An attribute path of a :class:`ScriptedClient`; calling it answers from the script."""

    def __init__(self, client: ScriptedClient, path: str) -> None:
        self._client = client
        self._path = path

    def __getattr__(self, name: str) -> _Path:
        if name.startswith("__"):
            raise AttributeError(name)
        return _Path(self._client, f"{self._path}.{name}")

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._client._answer(self._path, args, kwargs)

    def __repr__(self) -> str:
        return f"<ScriptedClient path {self._path!r}>"


class ScriptedClient:
    """A stand-in for an SDK client whose every method answers from a script and is recorded.

    Args:
        script: Either one sequence of answers shared by every method path, in call order, or a mapping from
            a dotted path (``"messages.create"``) to the answers of that path. An answer is returned as it is;
            an exception (instance or class) is raised; a callable is called with the :class:`ScriptedCall`
            and its result returned. A call with no answer left (or to a path the mapping does not name)
            raises ``AssertionError``.
        asynchronous: Calls return awaitables (for ``AsyncAnthropic``-style clients).
        attributes: Plain attributes the client also has (e.g. ``{"api_key": "test"}``).

    Thread-safe; ``calls`` lists every call in order.
    """

    def __init__(
        self,
        script: Sequence[Any] | Mapping[str, Sequence[Any]] = (),
        *,
        asynchronous: bool = False,
        attributes: Mapping[str, Any] | None = None,
    ) -> None:
        self._lock = threading.Lock()
        self._shared: list[Any] | None = None
        self._routes: dict[str, list[Any]] = {}
        if isinstance(script, Mapping):
            self._routes = {str(path): list(answers) for path, answers in script.items()}
        else:
            self._shared = list(script)
        self._asynchronous = asynchronous
        self.calls: list[ScriptedCall] = []
        for name, value in (attributes or {}).items():
            object.__setattr__(self, name, value)

    def __getattr__(self, name: str) -> _Path:
        if name.startswith("_"):
            raise AttributeError(name)
        return _Path(self, name)

    def calls_to(self, path: str) -> list[ScriptedCall]:
        """The recorded calls of one dotted path, in order."""
        with self._lock:
            return [call for call in self.calls if call.path == path]

    def remaining(self, path: str | None = None) -> int:
        """How many scripted answers are left (of ``path``, or of the shared script)."""
        with self._lock:
            if path is None:
                return len(self._shared) if self._shared is not None else sum(map(len, self._routes.values()))
            return len(self._routes.get(path, []))

    def _answer(self, path: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        call = ScriptedCall(path, deepcopy(args), deepcopy(kwargs))
        with self._lock:
            self.calls.append(call)
            queue = self._shared if self._shared is not None else self._routes.get(path)
            if not queue:
                raise AssertionError(f"ScriptedClient: no scripted answer left for {path}")
            answer = queue.pop(0)
        if self._asynchronous:

            async def awaitable() -> Any:
                return _resolve(answer, call)  # an exception is raised when awaited, as an SDK call would

            return awaitable()
        return _resolve(answer, call)


def _resolve(answer: Any, call: ScriptedCall) -> Any:
    if isinstance(answer, BaseException) or (isinstance(answer, type) and issubclass(answer, BaseException)):
        raise answer
    if callable(answer) and not isinstance(answer, type):
        function: Callable[[ScriptedCall], Any] = answer
        return function(call)
    return answer
