"""5. Pydantic structured output on any provider.

OpenAI receives a strict JSON schema (optional fields become nullable and are
restored to their defaults), Gemini an inlined response schema, Anthropic its
native structured output. The answer is validated locally in every case.
"""

from __future__ import annotations

from _setup import model
from pydantic import BaseModel, Field

from orkexa import Agent


class Company(BaseModel):
    name: str
    country: str = Field(description="ISO 3166 alpha-2 code")
    founded: int | None = None
    products: list[str] = []


def main() -> None:
    agent = Agent(model=model(), output_format=Company)
    company = agent.ask("Extract: 'Voestalpine, founded 1938 in Austria, makes steel and rails.'")
    print(company.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
