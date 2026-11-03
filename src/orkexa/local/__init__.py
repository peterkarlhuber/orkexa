"""Local models with Ollama: management API, managed server, runtime install, offline packages.

Nothing in this package runs on ``import orkexa`` or ``Agent(...)``: it installs,
starts or downloads only when you call it (or run ``python -m orkexa.local``).
Three steps are kept apart:

1. **Provide the runtime**: use an Ollama that already runs, or
   :func:`install_runtime` (from a local archive, or ``download=True``) plus
   :class:`ManagedOllama` (a child process on a free loopback port).
2. **Get models**: :meth:`OllamaClient.pull` (network), :meth:`OllamaClient.create_from_gguf`
   or :func:`import_package` (offline), :func:`build_model` (a pinned recipe such as
   :data:`MDENSEON`), or point ``models_dir`` at an existing folder.
3. **Run models**: ``Agent(model="ollama:qwen3.5:4b", base_url=client.openai_base_url)`` and
   :class:`~orkexa.embeddings.OllamaEmbeddings`; no network after steps 1 and 2.
"""

from ._http import DEFAULT_OLLAMA_URL, OllamaError, check_ollama_url
from .client import (
    EmbedResult,
    LocalModel,
    ModelDetails,
    OllamaClient,
    PullProgress,
    check_model_name,
    is_cloud_model,
    split_model_name,
)
from .convert import MDENSEON, BuildReport, GGUFRecipe, SourceFile, build_model
from .install import (
    OLLAMA_VERSION,
    RUNTIME_ARTIFACTS,
    artifact_for,
    install_runtime,
    installed_runtime,
    remove_runtime,
)
from .packages import PackageReport, StoredModel, export_package, import_package, scan_store, verify_package
from .server import DEFAULT_PORT, ManagedOllama, ServerState, detect_ollama, free_port

__all__ = [
    "DEFAULT_OLLAMA_URL",
    "DEFAULT_PORT",
    "MDENSEON",
    "OLLAMA_VERSION",
    "RUNTIME_ARTIFACTS",
    "BuildReport",
    "EmbedResult",
    "GGUFRecipe",
    "LocalModel",
    "ManagedOllama",
    "ModelDetails",
    "OllamaClient",
    "OllamaError",
    "PackageReport",
    "PullProgress",
    "ServerState",
    "SourceFile",
    "StoredModel",
    "artifact_for",
    "build_model",
    "check_model_name",
    "check_ollama_url",
    "detect_ollama",
    "export_package",
    "free_port",
    "import_package",
    "install_runtime",
    "installed_runtime",
    "is_cloud_model",
    "remove_runtime",
    "scan_store",
    "split_model_name",
    "verify_package",
]
