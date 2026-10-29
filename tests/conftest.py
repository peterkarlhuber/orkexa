"""Root test configuration.

The unchanged ``claude-agent`` 0.8.0 suite in ``tests/legacy`` imports its helpers as
``tests.conftest`` (its original location); they are re-exported here.
"""

from __future__ import annotations

from .legacy.conftest import (  # noqa: F401
    AsyncFakeClient,
    AsyncFakeMessages,
    AsyncFakeStream,
    FakeClient,
    FakeMessages,
    FakeStream,
    ParsedResponse,
    make_beta_message,
    make_message,
)
