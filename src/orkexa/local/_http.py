"""A small HTTP/1.1 client for a local Ollama server (standard library only).

Orkexa's core has no HTTP dependency, so the Ollama management API, the embedding
adapter and capability discovery share this client. It never follows redirects (a
redirect is reported as an error), never sends credentials, refuses non-HTTP URLs and,
unless ``allow_remote=True``, any host that is not a loopback address. Response bodies
are size-limited.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
import contextlib
import http.client
import ipaddress
import json
import socket
import threading
from typing import Any
from urllib.parse import urlsplit
import weakref

from ..errors import ConfigurationError, OrkexaError

__all__ = ["DEFAULT_OLLAMA_URL", "OllamaError", "OllamaHTTP", "check_ollama_url", "stream_error_code"]

DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
_MAX_BODY = 64 * 1024 * 1024  # embeddings of large batches can be a few MB
_MAX_LINE = 1024 * 1024
#: The socket of each open connection (weakly, by connection), so a cancelled stream can shut it down.
socket_of: weakref.WeakKeyDictionary[http.client.HTTPConnection, socket.socket | None] = (
    weakref.WeakKeyDictionary()
)


class OllamaError(OrkexaError, RuntimeError):
    """An Ollama server could not be reached or refused a request.

    ``code`` is stable and machine-readable: ``unreachable``, ``timeout``, ``redirect``,
    ``http_error``, ``model_missing``, ``invalid_response``, ``input_too_long``,
    ``not_embedding_model``, ``dimension_mismatch``, ``cancelled``, ``too_large``,
    ``disk_full`` (no space left for a download), ``network`` (Ollama, or a download, could not
    reach the internet).
    """

    def __init__(self, code: str, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.status = status


#: Parts of Ollama's error messages when a download breaks: a full disk, or no route to the registry.
_DISK_FULL = ("no space left on device", "not enough space", "disk full", "insufficient space")
_NETWORK = (
    "dial tcp",
    "no such host",
    "connection refused",
    "connection reset",
    "i/o timeout",
    "tls handshake",
    "proxyconnect",
    "network is unreachable",
    "context deadline exceeded",
    "unexpected eof",
    "max retries exceeded",
    "server misbehaving",
    "temporary failure in name resolution",
)


def stream_error_code(message: str) -> str:
    """The code of an error line in a streamed answer (e.g. a download that broke in ``/api/pull``)."""
    text = message.lower()
    if any(part in text for part in _DISK_FULL):
        return "disk_full"
    if "not found" in text or "file does not exist" in text:
        return "model_missing"
    if any(part in text for part in _NETWORK):
        return "network"
    return "http_error"


def _is_loopback(host: str) -> bool:
    if host in ("localhost", "localhost.") or host.endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        return False


def check_ollama_url(url: str, *, allow_remote: bool = False) -> str:
    """The normalized base URL (``http://host:port``, no path) of an Ollama server.

    A trailing ``/v1`` (the OpenAI-compatible path) is removed. Only ``http`` and
    ``https`` are accepted, without credentials, query or fragment. Without
    ``allow_remote`` the host must be a loopback address: Ollama has no authentication,
    so a remote server is an explicit decision of the caller.
    """
    parts = urlsplit(url.strip())
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise ConfigurationError(f"Not an http(s) URL of an Ollama server: {url!r}")
    if parts.username or parts.password or parts.query or parts.fragment:
        raise ConfigurationError("An Ollama URL must not contain credentials, a query or a fragment")
    path = parts.path.rstrip("/")
    if path not in ("", "/v1"):
        raise ConfigurationError(f"An Ollama URL has no path (got {parts.path!r})")
    if not allow_remote and not _is_loopback(parts.hostname):
        raise ConfigurationError(
            f"{parts.hostname!r} is not a loopback address; Ollama has no authentication, "
            "pass allow_remote=True to use a server on another host"
        )
    port = parts.port or (443 if parts.scheme == "https" else 80)
    host = parts.hostname if ":" not in parts.hostname else f"[{parts.hostname}]"
    return f"{parts.scheme}://{host}:{port}"


class OllamaHTTP:
    """JSON requests to one Ollama server."""

    def __init__(
        self, base_url: str = DEFAULT_OLLAMA_URL, *, timeout: float = 30.0, allow_remote: bool = False
    ):
        self.base_url = check_ollama_url(base_url, allow_remote=allow_remote)
        parts = urlsplit(self.base_url)
        self._scheme = parts.scheme
        self._host = parts.hostname or "127.0.0.1"
        self._port = parts.port
        self.timeout = timeout

    def _connection(self, timeout: float | None) -> http.client.HTTPConnection:
        cls = http.client.HTTPSConnection if self._scheme == "https" else http.client.HTTPConnection
        return cls(self._host, self._port, timeout=timeout if timeout is not None else self.timeout)

    def _send(
        self,
        method: str,
        path: str,
        body: Any,
        timeout: float | None,
        headers: Mapping[str, str] | None = None,
    ) -> tuple[http.client.HTTPConnection, http.client.HTTPResponse]:
        connection = self._connection(timeout)
        if isinstance(body, (bytes, bytearray)) or hasattr(body, "read"):
            payload = body
            sent = {"Content-Type": "application/octet-stream", **(headers or {})}
        else:
            payload = json.dumps(body).encode() if body is not None else None
            sent = {"Content-Type": "application/json", **(headers or {})}
        try:
            connection.request(method, path, body=payload, headers=sent)
            # Kept: a response that closes the connection takes the socket over (see stream's cancel).
            socket_of[connection] = connection.sock
            response = connection.getresponse()
        except TimeoutError as exc:
            connection.close()
            raise OllamaError("timeout", f"Ollama at {self.base_url} did not answer in time") from exc
        except (OSError, http.client.HTTPException) as exc:
            connection.close()
            raise OllamaError("unreachable", f"Ollama is not reachable at {self.base_url}: {exc}") from exc
        if 300 <= response.status < 400:
            connection.close()
            raise OllamaError(
                "redirect", f"Ollama answered with a redirect ({response.status}); not followed"
            )
        return connection, response

    @staticmethod
    def _read(response: http.client.HTTPResponse) -> bytes:
        data = response.read(_MAX_BODY + 1)
        if len(data) > _MAX_BODY:
            raise OllamaError("too_large", "Ollama's answer is larger than allowed")
        return data

    def _error(self, response: http.client.HTTPResponse, data: bytes, model: str | None) -> OllamaError:
        try:
            message = str(json.loads(data).get("error") or "")
        except (ValueError, AttributeError):
            message = data[:300].decode("utf-8", "replace")
        status = response.status
        if status == 404 and ("not found" in message or model):
            return OllamaError("model_missing", message or f"Model {model!r} is not installed", status=status)
        if "input length exceeds" in message or "context length" in message:
            return OllamaError("input_too_long", message, status=status)
        if "does not support embeddings" in message or "does not support generate" in message:
            return OllamaError("not_embedding_model", message, status=status)
        code = stream_error_code(message)
        if code in ("disk_full", "network"):
            return OllamaError(code, f"Ollama answered {status}: {message}", status=status)
        return OllamaError("http_error", f"Ollama answered {status}: {message}", status=status)

    def request(
        self,
        method: str,
        path: str,
        body: Any = None,
        *,
        timeout: float | None = None,
        model: str | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> Any:
        """One request; returns the decoded JSON body (``None`` for an empty body)."""
        connection, response = self._send(method, path, body, timeout, headers)
        try:
            try:
                data = self._read(response)
            except TimeoutError as exc:
                raise OllamaError("timeout", f"Ollama at {self.base_url} did not answer in time") from exc
            except (OSError, http.client.HTTPException) as exc:
                raise OllamaError("unreachable", f"The connection to Ollama broke: {exc}") from exc
            if response.status >= 400:
                raise self._error(response, data, model)
            if not data.strip():
                return None
            try:
                return json.loads(data)
            except ValueError as exc:
                raise OllamaError("invalid_response", "Ollama's answer is not JSON") from exc
        finally:
            connection.close()

    def stream(
        self,
        method: str,
        path: str,
        body: Any = None,
        *,
        timeout: float | None = None,
        model: str | None = None,
        cancel: threading.Event | None = None,
    ) -> Iterator[dict[str, Any]]:
        """A request with a newline-delimited JSON answer (e.g. ``/api/pull``), line by line.

        ``cancel`` ends the answer at once, also while the server sends nothing (the connection is shut down,
        which tells Ollama that nobody waits any longer); the generator then raises ``cancelled``."""
        connection, response = self._send(method, path, body, timeout, None)
        finished = threading.Event()
        if cancel is not None:

            def watch() -> None:
                while not finished.is_set():
                    if cancel.wait(0.2):
                        sock = socket_of.get(connection)
                        if sock is not None and not finished.is_set():
                            with contextlib.suppress(OSError):
                                sock.shutdown(socket.SHUT_RDWR)
                        return

            threading.Thread(target=watch, name="orkexa-ollama-cancel", daemon=True).start()

        def cancelled() -> OllamaError:
            return OllamaError("cancelled", "The request was cancelled")

        try:
            if response.status >= 400:
                raise self._error(response, self._read(response), model)
            while True:
                try:
                    line = response.readline(_MAX_LINE)
                except TimeoutError as exc:
                    if cancel is not None and cancel.is_set():
                        raise cancelled() from None
                    raise OllamaError("timeout", "Ollama stopped answering") from exc
                except (OSError, http.client.HTTPException) as exc:
                    if cancel is not None and cancel.is_set():
                        raise cancelled() from None
                    raise OllamaError("unreachable", f"The connection to Ollama broke: {exc}") from exc
                if cancel is not None and cancel.is_set():
                    raise cancelled()
                if not line:
                    return
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except ValueError as exc:
                    raise OllamaError("invalid_response", "Ollama sent a line that is not JSON") from exc
                if isinstance(item, dict) and item.get("error"):
                    message = str(item["error"])
                    raise OllamaError(stream_error_code(message), message)
                yield item
        finally:
            finished.set()
            connection.close()


def port_is_free(host: str, port: int) -> bool:
    """Whether nothing listens on ``host:port`` (for choosing a managed server's port)."""
    with socket.socket(socket.AF_INET6 if ":" in host else socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.5)
        return probe.connect_ex((host, port)) != 0
