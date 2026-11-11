"""The public Orkexa API as of MUP-ONA B0, extended by B1 and B5 (T-API-1); the import contracts (T-PKG-0).

The snapshot is the reference for every later compatibility claim (backend move B2, PostgreSQL backend B5):
a name may be added (extend the snapshot in the same change), but removing or renaming one needs a minor
version and a CHANGELOG entry (see "Versioning and releases" in docs/development/releasing.md).
"""

from __future__ import annotations

import importlib
from pathlib import Path
import subprocess
import sys
import tomllib

import pytest

ROOT = Path(__file__).resolve().parent.parent

PUBLIC = {
    "orkexa": [
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
        "MODEL_PRICING",
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
    ],
    "orkexa.retrieval": [
        "Chunk",
        "Citation",
        "Embedder",
        "HYBRID_FUSIONS",
        "Hit",
        "HybridFusion",
        "Index",
        "LEXICAL_WEIGHT",
        "MAX_QUERY_CHARS",
        "MAX_QUERY_WORDS",
        "PdfLimits",
        "RETRIEVAL_ERROR_CODES",
        "RETRIEVAL_MODES",
        "ReindexableIndex",
        "RetrievalError",
        "RetrievalMode",
        "SEARCH_TOOL_NAME",
        "SQLiteIndex",
        "VectorError",
        "VectorStatus",
        "citations",
        "fold",
        "fuse_hybrid",
        "fuse_rrf",
        "lexical_evidence",
        "pdf_pages",
        "pdf_pages_isolated",
        "query_words",
        "retrieval_sources",
        "split_text",
        "tokenize",
        "word_weight",
    ],
    "orkexa.files": [
        "FILES_TOOL_NAME",
        "FileStore",
        "LocalFileStore",
        "SQLiteFileStore",
        "StoredFile",
    ],
    "orkexa.memory": [
        "EphemeralMemory",
        "FileMemory",
        "MEMORY_ROOT",
        "MEMORY_TOOL_NAME",
        "MemoryBackend",
        "SQLiteMemory",
    ],
    "orkexa.embeddings": [
        "CallableEmbeddings",
        "EmbeddingError",
        "EmbeddingIdentity",
        "EmbeddingModel",
        "EmbeddingUsage",
        "KNOWN_PREFIXES",
        "OllamaEmbeddings",
        "QWEN3_QUERY_INSTRUCTION",
        "validate_vector",
    ],
    "orkexa.backends.sqlite": ["SQLiteFileStore", "SQLiteIndex", "SQLiteMemory"],
    "orkexa.backends.postgres": [  # MUP-ONA B5
        "CATALOG",
        "Connection",
        "ConnectionSource",
        "Cursor",
        "MAX_SOURCES",
        "MISSING_PSYCOPG",
        "MODES",
        "PostgresFileStore",
        "PostgresIndex",
        "PostgresLayout",
        "PostgresMemory",
        "PostgresVectorStatus",
        "PsycopgSource",
        "SCHEMA_VERSION",
        "SchemaProblem",
        "SchemaReport",
        "Statement",
        "TransactionMode",
        "migrate",
        "render",
        "schema_steps",
        "search_words",
        "verify_schema",
    ],
    "orkexa.testing.conformance": [
        "ALL_FILE_STORE_CASES",
        "ALL_INDEX_CASES",
        "ALL_MEMORY_CASES",
        "CAPABILITIES",
        "Case",
        "CaseFailure",
        "Context",
        "DEFAULT_PARTITION",
        "FILE_STORE_CASES",
        "Factory",
        "INDEX_CASES",
        "KeywordEmbeddings",
        "MEMORY_CASES",
        "Outcome",
        "PARTITION_FILE_STORE_CASES",
        "PARTITION_INDEX_CASES",
        "PARTITION_MEMORY_CASES",
        "REINDEX_CASES",
        "Report",
        "Request",
        "SECURITY_NEEDS",
        "VECTOR_CASES",
        "expect",
        "expect_code",
        "expect_equal",
        "expect_error",
        "load_expected_skips",
        "run_cases",
    ],
    "orkexa.testing.fakes": [
        "ScriptedCall",
        "ScriptedClient",
    ],
    "orkexa.testing.sqlcheck": [  # MUP-ONA B5
        "ANY_REFUSED",
        "Finding",
        "KINDS",
        "MISSING_PGLAST",
        "PLACEHOLDERS",
        "check_statement",
        "to_postgres_params",
    ],
}
"""``__all__`` of every public module of the storage and retrieval layer, plus the package itself."""

DEEP_PATHS = {
    # Nodestro (nodestro-runtime) imports these names from these modules, even where the package
    # re-exports them.
    "orkexa.retrieval": [
        "HYBRID_FUSIONS",
        "HybridFusion",
        "LEXICAL_WEIGHT",
        "RetrievalError",
        "RetrievalMode",
        "VectorStatus",
        "_pdf_pages",
        "check_mode",
        "fuse_hybrid",
        "lexical_evidence",
        "word_weight",
        "query_words",
        "split_text",
        "pdf_pages",
        "pdf_pages_isolated",
        "PdfLimits",
        "MAX_QUERY_CHARS",
        "MAX_QUERY_WORDS",
    ],
    "orkexa.memory": ["MEMORY_ROOT", "MEMORY_TOOL_NAME", "MemoryBackend"],
    "orkexa.embeddings": ["CallableEmbeddings", "EmbeddingIdentity", "EmbeddingModel", "OllamaEmbeddings"],
    "orkexa.files": ["FileStore", "LocalFileStore", "SQLiteFileStore", "StoredFile"],
}

REEXPORTS = {
    # orkexa.<name> is the very object of its defining public module (``is``, not just equal).
    "orkexa.retrieval": [
        "Chunk",
        "Hit",
        "Index",
        "SQLiteIndex",
        "citations",
        "retrieval_sources",
        "split_text",
    ],
    "orkexa.files": ["FileStore", "LocalFileStore", "SQLiteFileStore", "StoredFile"],
    "orkexa.memory": ["EphemeralMemory", "FileMemory", "MemoryBackend", "SQLiteMemory"],
    "orkexa.backends.sqlite": ["SQLiteFileStore", "SQLiteIndex", "SQLiteMemory"],
}

SQLITE_CLASSES = {
    # MUP-ONA B2: class -> (contract module that defined it before B2, leaf module that defines it now).
    "SQLiteIndex": ("orkexa.retrieval", "orkexa.backends.sqlite.index"),
    "SQLiteFileStore": ("orkexa.files", "orkexa.backends.sqlite.files"),
    "SQLiteMemory": ("orkexa.memory", "orkexa.backends.sqlite.memory"),
}


# MUP-ONA T-API-1
@pytest.mark.parametrize("module", sorted(PUBLIC))
def test_the_public_names_are_unchanged(module: str) -> None:
    imported = importlib.import_module(module)
    assert sorted(imported.__all__) == PUBLIC[module]
    for name in PUBLIC[module]:
        assert hasattr(imported, name), f"{module}.{name}"


# MUP-ONA T-API-1
@pytest.mark.parametrize("module", sorted(DEEP_PATHS))
def test_the_deep_import_paths_resolve(module: str) -> None:
    imported = importlib.import_module(module)
    for name in DEEP_PATHS[module]:
        assert hasattr(imported, name), f"{module}.{name}"


# MUP-ONA T-API-1
@pytest.mark.parametrize("module", sorted(REEXPORTS))
def test_reexports_are_the_same_objects(module: str) -> None:
    import orkexa

    imported = importlib.import_module(module)
    for name in REEXPORTS[module]:
        assert name in orkexa.__all__
        assert getattr(orkexa, name) is getattr(imported, name), name


# MUP-ONA T-API-1
def test_every_package_name_from_a_storage_module_is_that_object() -> None:
    import orkexa

    for module in ("orkexa.retrieval", "orkexa.files", "orkexa.memory", "orkexa.embeddings"):
        imported = importlib.import_module(module)
        for name in set(orkexa.__all__) & set(imported.__all__):
            assert getattr(orkexa, name) is getattr(imported, name), f"{module}.{name}"
    from orkexa import retrieval

    assert retrieval._pdf_pages is retrieval.pdf_pages  # the kept private alias


# MUP-ONA T-API-1
@pytest.mark.parametrize("name", sorted(SQLITE_CLASSES))
def test_every_import_path_of_a_sqlite_backend_is_the_same_class(name: str) -> None:
    import orkexa
    import orkexa.backends.sqlite

    contract, leaf = SQLITE_CLASSES[name]
    cls = getattr(orkexa, name)
    assert cls is getattr(importlib.import_module(contract), name)
    assert cls is getattr(orkexa.backends.sqlite, name)
    assert cls is getattr(importlib.import_module(leaf), name)
    assert cls.__module__ == leaf  # the only visible change of B2
    assert cls.__qualname__ == name
    assert name in dir(importlib.import_module(contract))


def _contracts() -> list[dict[str, object]]:
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    contracts: list[dict[str, object]] = config["tool"]["importlinter"]["contracts"]
    return contracts


# MUP-ONA T-PKG-0
def test_the_import_contracts_cover_every_core_module() -> None:
    package = ROOT / "src" / "orkexa"
    top = {
        path.stem if path.is_file() else path.name
        for path in package.iterdir()
        if (path.suffix == ".py" and path.stem != "__init__") or (path / "__init__.py").is_file()
    }
    core = top - {"testing"}  # B2 adds orkexa.backends: then list it here and adjust the contracts
    expected = {"orkexa"} | {f"orkexa.{name}" for name in core} | {f"orkexa.{name}.**" for name in core}
    names = {str(contract["name"]) for contract in _contracts()}
    assert names == {
        "Core modules do not import orkexa.backends.postgres",
        "Core modules do not import orkexa.testing",
        "Contract modules do not import orkexa.backends at runtime",  # MUP-ONA B2, see test_import_order.py
        "orkexa.testing.sqlcheck imports no other orkexa module",  # MUP-ONA B5, see tests/test_sqlcheck.py
        # QA-16 (K-33), see tests/architecture/test_forbidden_probe.py
        "Orkexa depends on no Axentra, Nodestro or Studio package (K-33)",
        "Core modules do not import tabrion until MIG-ORK-3 (E-ORK-CORE, K-33)",
    }
    for contract in _contracts():
        if str(contract["name"]).startswith("orkexa.testing.sqlcheck"):
            # The engine may import no other orkexa module: every core module and every sibling in testing.
            assert contract["source_modules"] == ["orkexa.testing.sqlcheck"]
            forbidden = set(contract["forbidden_modules"])  # type: ignore[call-overload]
            siblings = {
                f"orkexa.testing.{path.stem if path.is_file() else path.name}{suffix}"
                for path in (package / "testing").iterdir()
                if (path.suffix == ".py" and path.stem not in {"__init__", "sqlcheck"})
                or (path / "__init__.py").is_file()
                for suffix in ("", ".**")
            }
            wanted = (expected - {"orkexa"}) | siblings
            assert forbidden >= wanted, sorted(wanted - forbidden)
            continue
        if not str(contract["name"]).startswith("Core modules"):
            continue  # the B2 contract names only the contract modules
        sources = set(contract["source_modules"])  # type: ignore[call-overload]
        wanted = expected
        if str(contract["name"]).endswith(("orkexa.backends.postgres", "(E-ORK-CORE, K-33)")):
            # MUP-ONA B5: the backend itself is not core; every other backend is listed one by one. QA-16:
            # the PostgreSQL backend may import tabrion (K-33, orkexa[postgres] -> tabrion[postgres]); the
            # core may not.
            backends = {
                path.name for path in (package / "backends").iterdir() if (path / "__init__.py").is_file()
            }
            assert "postgres" in backends
            others = backends - {"postgres"}
            wanted = (expected - {"orkexa.backends.**"}) | {
                f"orkexa.backends.{name}{suffix}" for name in others for suffix in ("", ".**")
            }
            assert not {name for name in sources if name.startswith(("orkexa.backends.postgres", "orkexa.*"))}
            assert "orkexa.backends.**" not in sources
        assert sources >= wanted, sorted(wanted - sources)


# MUP-ONA T-PKG-0
def test_the_import_contracts_hold() -> None:
    pytest.importorskip("importlinter")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from importlinter.cli import lint_imports; "
            "sys.exit(lint_imports(config_filename='pyproject.toml', no_cache=True, no_logo=True))",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "6 kept, 0 broken" in result.stdout
