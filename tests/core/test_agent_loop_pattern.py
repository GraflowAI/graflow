"""Test the agent-loop pattern: `agent >> tool >> agent` cycles through tasks
until `terminate_workflow()` is called.

This pattern is documented in the README as the Graflow equivalent of a
LangGraph agent loop with conditional edges.
"""

from graflow.core.decorators import task
from graflow.core.workflow import workflow


def test_agent_tool_loop_terminates_on_condition():
    """`agent >> tool >> agent` should loop until terminate_workflow() is called."""
    call_counts = {"agent": 0, "tool": 0}

    with workflow("agent_loop") as ctx:

        @task(inject_context=True)
        def agent(context):
            call_counts["agent"] += 1
            # Terminate after 3 agent invocations to mimic `should_continue` returning False
            if call_counts["agent"] >= 3:
                context.terminate_workflow("done")
            return f"agent_response_{call_counts['agent']}"

        @task
        def tool(out: str = "default"):
            call_counts["tool"] += 1
            return f"tool_out_{call_counts['tool']}"

        agent >> tool >> agent  # type: ignore[operator]

        ctx.execute("agent")

    # Loop fired agent → tool → agent → tool → agent (terminate)
    assert call_counts["agent"] == 3
    assert call_counts["tool"] == 2


def test_agent_tool_loop_single_pass():
    """Agent can terminate on the first pass without ever invoking the tool."""
    call_counts = {"agent": 0, "tool": 0}

    with workflow("agent_loop_single") as ctx:

        @task(inject_context=True)
        def agent(context):
            call_counts["agent"] += 1
            context.terminate_workflow("immediate exit")
            return "agent_response"

        @task
        def tool(out: str = "default"):
            call_counts["tool"] += 1
            return "tool_out"

        agent >> tool >> agent  # type: ignore[operator]

        ctx.execute("agent")

    assert call_counts["agent"] == 1
    assert call_counts["tool"] == 0
