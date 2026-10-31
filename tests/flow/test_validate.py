"""Static checks of flow specifications."""

from __future__ import annotations

import pytest

from orkexa.flow import Edge, Flow, FlowError, Limits, Loop, Node, start, validate


def codes(flow: Flow) -> list[str]:
    return [issue.code for issue in validate(flow)]


def review_loop(**loop: object) -> Flow:
    return Flow(
        nodes=(Node(id="a"), Node(id="b", ports=("ok", "again"))),
        edges=(
            Edge(id="ab", source="a", target="b"),
            Edge(id="back", source="b", target="a", port="again", loop=Loop(max_iterations=3, **loop)),  # type: ignore[arg-type]
        ),
    )


def test_valid_flows_have_no_issues() -> None:
    assert validate(Flow()) == []
    assert validate(review_loop()) == []


def test_duplicate_ids() -> None:
    flow = Flow(
        nodes=(Node(id="a"), Node(id="a"), Node(id="b")),
        edges=(Edge(id="e", source="a", target="b"), Edge(id="e", source="a", target="b")),
    )
    assert codes(flow) == ["flow.duplicate_id", "flow.duplicate_id"]


def test_dangling_edge_and_self_loop() -> None:
    flow = Flow(
        nodes=(Node(id="a"),),
        edges=(Edge(id="x", source="a", target="ghost"), Edge(id="s", source="a", target="a")),
    )
    issues = validate(flow)
    assert [issue.code for issue in issues] == ["flow.edge_dangling", "flow.self_loop"]
    assert issues[0].edge == "x"


def test_ports() -> None:
    flow = Flow(
        nodes=(Node(id="plain"), Node(id="branch", ports=("yes", "no")), Node(id="t")),
        edges=(
            Edge(id="1", source="plain", target="t", port="yes"),
            Edge(id="2", source="branch", target="t"),
            Edge(id="3", source="branch", target="t", port="maybe"),
        ),
    )
    assert codes(flow) == ["flow.port_unknown", "flow.port_missing", "flow.port_unknown"]


def test_cycle_without_loop_edge() -> None:
    flow = Flow(
        nodes=(Node(id="a"), Node(id="b")),
        edges=(Edge(id="1", source="a", target="b"), Edge(id="2", source="b", target="a")),
    )
    assert codes(flow) == ["flow.cycle"]


def test_loop_edges() -> None:
    unguarded = Flow(
        nodes=(Node(id="a"), Node(id="b")),
        edges=(
            Edge(id="1", source="a", target="b"),
            Edge(id="2", source="b", target="a", loop=Loop(max_iterations=2)),
        ),
    )
    assert codes(unguarded) == ["flow.loop_unguarded"]
    forward = Flow(
        nodes=(Node(id="a", ports=("p",)), Node(id="b")),
        edges=(
            Edge(id="1", source="a", target="b", port="p"),
            Edge(id="2", source="a", target="b", port="p", loop=Loop(max_iterations=2)),
        ),
    )
    assert codes(forward) == ["flow.loop_target"]
    capped = review_loop().model_copy(update={"limits": Limits(max_iterations=2)})
    assert codes(capped) == ["flow.loop_limit"]


def test_loop_on_one_node_is_allowed() -> None:
    flow = Flow(
        nodes=(Node(id="try", ports=("done", "again")),),
        edges=(Edge(id="l", source="try", target="try", port="again", loop=Loop(max_iterations=2)),),
    )
    assert validate(flow) == []


def test_loop_heads_sources_and_overlap() -> None:
    nodes = (Node(id="a"), Node(id="b", ports=("x", "y")), Node(id="c", ports=("x", "y")))
    chain = (
        Edge(id="ab", source="a", target="b", port=None),
        Edge(id="bc", source="b", target="c", port="x"),
    )
    shared_head = Flow(
        nodes=nodes,
        edges=(
            *chain,
            Edge(id="l1", source="b", target="a", port="y", loop=Loop(max_iterations=2)),
            Edge(id="l2", source="c", target="a", port="y", loop=Loop(max_iterations=2)),
        ),
    )
    assert codes(shared_head) == ["flow.loop_head_shared"]
    shared_source = Flow(
        nodes=nodes,
        edges=(
            *chain,
            Edge(id="l1", source="c", target="a", port="y", loop=Loop(max_iterations=2)),
            Edge(id="l2", source="c", target="b", port="x", loop=Loop(max_iterations=2)),
        ),
    )
    assert codes(shared_source) == ["flow.loop_source_shared"]
    overlap = Flow(
        nodes=nodes,
        edges=(
            *chain,
            Edge(id="l1", source="b", target="a", port="y", loop=Loop(max_iterations=2)),
            Edge(id="l2", source="c", target="b", port="y", loop=Loop(max_iterations=2)),
        ),
    )
    assert codes(overlap) == ["flow.loop_overlap"]


def test_contract_version() -> None:
    assert codes(Flow(contract="2.0")) == ["flow.contract"]


def test_start_rejects_an_invalid_flow() -> None:
    with pytest.raises(FlowError) as caught:
        start(Flow(nodes=(Node(id="a"), Node(id="a"))))
    assert caught.value.code == "flow.invalid"
    assert [issue.code for issue in caught.value.issues] == ["flow.duplicate_id"]
    assert isinstance(caught.value, ValueError)


def test_loop_needs_a_limit() -> None:
    with pytest.raises(ValueError):
        Loop()  # type: ignore[call-arg]
    with pytest.raises(ValueError):
        Loop(max_iterations=1)
