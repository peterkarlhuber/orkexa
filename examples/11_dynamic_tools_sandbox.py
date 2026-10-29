"""11. Dynamic tool creation with static check, sandbox and approval.

A Coding Agent (any provider) writes a function and its tests; the ToolBuilder
checks it statically, runs the tests in a sandbox, asks the approver and registers
it; an agent of possibly another provider then uses it. Generated code never runs
in this process. ``SubprocessSandbox`` is no security boundary; build the Docker
image (``docker build -t orkexa-sandbox -f docker/sandbox.Dockerfile .``) to use
``DockerSandbox`` automatically.
"""

from __future__ import annotations

from _setup import model

from orkexa import Agent
from orkexa.dynamic import (
    ApprovalRequest,
    CallbackApprover,
    CodingAgent,
    DockerSandbox,
    Sandbox,
    SubprocessSandbox,
    ToolBuilder,
    ToolRegistry,
)


def approve(request: ApprovalRequest) -> bool:
    print(f"--- approval requested: {request.kind} {request.name}\n{request.summary}")
    return input("approve? [y/N] ").strip().lower() in ("y", "yes")


def main() -> None:
    name = model()
    sandbox: Sandbox = DockerSandbox() if DockerSandbox.available() else SubprocessSandbox()
    builder = ToolBuilder(
        CodingAgent(model=name),
        ToolRegistry(sandbox=sandbox),
        approver=CallbackApprover(approve),
        sandbox=sandbox,
    )
    analyst = Agent(model=name, system_prompt="You are a financial analyst.")
    for tool in builder.as_tools(analyst):  # find_tools, load_tool, create_tool
        analyst.add_tool(tool)
    print(
        analyst.ask(
            "Revenue is 200m, operating income 50m. What is the operating margin? Build a tool for it."
        )
    )


if __name__ == "__main__":
    main()
