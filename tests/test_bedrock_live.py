"""Optional live smoke test against Amazon Bedrock (costs money; never part of a normal run).

Runs only with ``pytest -m live`` and ``ORKEXA_BEDROCK_LIVE_MODEL`` set (e.g.
``eu.anthropic.claude-haiku-4-5-20251001-v1:0``) plus AWS credentials and a region in the
environment (``AWS_PROFILE``/``AWS_ACCESS_KEY_ID``, ``AWS_DEFAULT_REGION``). It verifies data
residency for the EU and then sends one tiny request.
"""

from __future__ import annotations

import os

import pytest

from orkexa import Agent, BedrockOptions

pytestmark = pytest.mark.live

MODEL = os.environ.get("ORKEXA_BEDROCK_LIVE_MODEL")
REGION = os.environ.get("AWS_DEFAULT_REGION")


@pytest.mark.skipif(not MODEL or not REGION, reason="needs ORKEXA_BEDROCK_LIVE_MODEL and AWS_DEFAULT_REGION")
def test_one_tiny_request_with_eu_residency() -> None:
    boto3 = pytest.importorskip("boto3")
    if boto3.Session().get_credentials() is None:
        pytest.skip("needs AWS credentials")
    agent = Agent(
        model=f"bedrock:{MODEL}",
        options=BedrockOptions(region_name=REGION, data_residency="eu_only"),
        system_prompt="Answer with one word.",
        max_tokens=16,
    )
    answer = agent.ask("Say OK.")
    assert isinstance(answer, str) and answer.strip()
    usage = agent.last_usage
    assert usage is not None and usage.provider == "bedrock" and usage.input_tokens > 0
