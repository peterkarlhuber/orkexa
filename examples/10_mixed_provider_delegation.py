"""10. A Claude agent delegates to OpenAI and Gemini agents; a judge of another provider.

All agents share one usage tracker. Every agent only receives its own provider's
credentials (from its environment variable); nothing is passed between providers.
"""

from __future__ import annotations

from _setup import MODELS, has_key

from orkexa import Agent, ClaudeAgent, Criterion, Evaluator, Tool, UsageTracker, refine


def main() -> None:
    if not all(has_key(p) for p in ("anthropic", "openai", "gemini")):
        raise SystemExit("This example needs ANTHROPIC_API_KEY, OPENAI_API_KEY and GEMINI_API_KEY.")
    usage = UsageTracker()
    researcher = Agent(model=f"openai:{MODELS['openai'][1]}", name="researcher", usage_tracker=usage)
    reviewer = Agent(model=f"gemini:{MODELS['gemini'][1]}", name="reviewer", usage_tracker=usage)

    @Tool
    def research(question: str) -> str:
        """Ask the research agent (OpenAI) a factual question.

        Args:
            question: A self-contained question.

        Returns:
            The researcher's answer.
        """
        researcher.reset()
        return researcher.ask(question)

    @Tool
    def review(draft: str) -> str:
        """Ask the review agent (Gemini) to criticize a draft.

        Args:
            draft: The text to review.

        Returns:
            The reviewer's critique.
        """
        reviewer.reset()
        return reviewer.ask(f"Criticize this draft in three bullet points:\n{draft}")

    lead = ClaudeAgent(
        model=MODELS["anthropic"][1], tools=[research, review], usage_tracker=usage, name="lead"
    )
    judge = Evaluator(
        criteria=[Criterion(name="sourced", description="Every claim names where it comes from.")],
        model=f"gemini:{MODELS['gemini'][1]}",
        usage_tracker=usage,
    )
    result = refine(
        lead, "Write a 100-word brief on the Danube's economic role. Research, draft, review.", judge
    )
    print(result.output)
    print("judge passed:", result.passed, "after", result.attempts, "answer(s)")
    for provider, summary in usage.usage_by_provider().items():
        print(f"{provider}: {summary.total_tokens} tokens")


if __name__ == "__main__":
    main()
