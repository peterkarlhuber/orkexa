"""Amazon Bedrock: a Claude agent in an EU region with verified data residency.

    pip install "orkexa[bedrock]"
    aws sso login --profile my-profile        # or any other AWS credential source
    AWS_PROFILE=my-profile python examples/bedrock_agent.py
    ORKEXA_BEDROCK_IMAGE=photo.png python examples/bedrock_agent.py      # adds an image turn

The provider authenticates with AWS credentials from boto3's credential chain (here
the profile named by ``AWS_PROFILE``; without it environment variables, SSO, a
container or instance role, or ``AWS_BEARER_TOKEN_BEDROCK``), never with an API key of
another provider. ``ORKEXA_BEDROCK_MODEL`` selects the model id, inference profile id
or ARN (default: the EU cross-Region inference profile of Claude Sonnet 5.5);
``ORKEXA_BEDROCK_IMAGE`` optionally names an image for a third, multimodal turn.

``data_residency="eu_only"`` checks before the first paid request that the client's
region and every destination region of the inference profile are EU regions (it calls
``bedrock:GetInferenceProfile``) and raises :class:`~orkexa.DataResidencyError`
otherwise; nothing is sent then. Every call costs money on your AWS bill. Orkexa
ships no Bedrock prices, so the cost stays unknown (``None``) until you register one.
"""

from __future__ import annotations

import os

from orkexa import Agent, BedrockOptions, DataResidencyError, Tool, UsageTracker, image

MODEL = os.environ.get("ORKEXA_BEDROCK_MODEL", "eu.anthropic.claude-sonnet-5-5")
IMAGE = os.environ.get("ORKEXA_BEDROCK_IMAGE")  # optional path of a PNG, JPEG, GIF or WebP image


@Tool
def vat_amount(net: float, rate_percent: float) -> float:
    """Calculate the value added tax of a net amount.

    Args:
        net: The net amount.
        rate_percent: The VAT rate in percent, e.g. 20.

    Returns:
        The tax amount, rounded to cents.
    """
    return round(net * rate_percent / 100, 2)


def main() -> None:
    options = BedrockOptions(
        region_name="eu-central-1",
        profile_name=os.environ.get("AWS_PROFILE"),
        data_residency="eu_only",
    )
    tracker = UsageTracker()
    agent = Agent(
        provider="bedrock",
        model=MODEL,
        options=options,
        tools=[vat_amount],
        usage_tracker=tracker,
        system_prompt="You are a concise assistant for an accounting team.",
    )
    try:
        print(agent.ask("In one sentence: what is Amazon Bedrock?"))
        print(agent.ask("What is the VAT on a net amount of 1,250 EUR at 20 %? Use the tool."))
        if IMAGE:
            print(agent.ask([image(IMAGE), "Describe this image in one sentence."]))
    except DataResidencyError as exc:
        raise SystemExit(f"Not sent, data residency could not be verified: {exc}") from None

    for record in tracker.records:
        print(
            f"  {record.provider}:{record.model} in={record.input_tokens} out={record.output_tokens} "
            f"cost={record.estimated_cost_usd} missing={list(record.missing_cost_components)}"
        )


if __name__ == "__main__":
    main()
