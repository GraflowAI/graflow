"""Test the agent-loop patterns demonstrated in examples/02_workflows/agent_loop.py.

Three loop styles:
  1. agent_tool_loop  — static cycle with `>>` + terminate_workflow()
  2. loop_with_goto   — dynamic jumps with next_task(goto=True) + max_cycles
  3. loop_with_iteration — single-task self-loop with next_iteration() + max_cycles
"""

from graflow import task, workflow


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
        def tool():
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
        def tool():
            call_counts["tool"] += 1
            return "tool_out"

        agent >> tool >> agent  # type: ignore[operator]

        ctx.execute("agent")

    assert call_counts["agent"] == 1
    assert call_counts["tool"] == 0


def test_loop_with_goto_early_exit():
    """loop_with_goto: exits early when completion condition is met."""
    call_counts = {"agent": 0, "tool": 0}
    quality_threshold = 0.75

    with workflow("ralph_loop") as wf:
        state = {"score": 0.0}

        @task(inject_context=True, max_cycles=5)
        def agent(context):
            call_counts["agent"] += 1
            state["score"] += 0.2
            if state["score"] >= quality_threshold:
                context.get_channel().set("final_score", state["score"])
                return "done"
            context.next_task(tool, goto=True)

        @task(inject_context=True)
        def tool(context):
            call_counts["tool"] += 1
            context.next_task(agent, goto=True)

        _, exec_ctx = wf.execute("agent", ret_context=True)

    assert call_counts["agent"] == 4  # score: 0.2, 0.4, 0.6, 0.8 (>= 0.75)
    assert call_counts["tool"] == 3
    assert exec_ctx.channel.get("final_score") == 0.8
    assert exec_ctx.cycle_controller.get_cycle_count("agent") == 4


def test_loop_with_goto_exhausts_budget():
    """loop_with_goto: stops when max_cycles budget is exhausted."""
    call_counts = {"agent": 0, "tool": 0}

    with workflow("ralph_budget") as wf:

        @task(inject_context=True, max_cycles=3)
        def agent(context):
            call_counts["agent"] += 1
            # Never meets exit condition — relies on budget
            if context.can_iterate():
                context.next_task(tool, goto=True)
            else:
                context.get_channel().set("status", "budget_exhausted")

        @task(inject_context=True)
        def tool(context):
            call_counts["tool"] += 1
            context.next_task(agent, goto=True)

        _, exec_ctx = wf.execute("agent", ret_context=True)

    assert call_counts["agent"] == 3
    assert call_counts["tool"] == 2
    assert exec_ctx.channel.get("status") == "budget_exhausted"


def test_loop_with_iteration_early_exit():
    """loop_with_iteration: exits early when completion condition is met."""

    with workflow("self_refine") as wf:

        @task(inject_context=True, max_cycles=5)
        def refine(context, data=None):
            score = (data or {}).get("score", 0.1) + 0.2
            if score >= 0.85:
                context.get_channel().set("result", f"draft_v{context.cycle_count}")
                return "done"
            if context.can_iterate():
                context.next_iteration({"score": score})

        _, exec_ctx = wf.execute("refine", ret_context=True)

    assert exec_ctx.channel.get("result") == "draft_v4"
    assert exec_ctx.cycle_controller.get_cycle_count("refine") == 4


def test_loop_with_iteration_exhausts_budget():
    """loop_with_iteration: stops when max_cycles is reached."""

    with workflow("self_refine_budget") as wf:

        @task(inject_context=True, max_cycles=3)
        def refine(context, data=None):
            total = (data or {}).get("total", 0) + 1
            if context.can_iterate():
                context.next_iteration({"total": total})
            else:
                context.get_channel().set("total", total)

        _, exec_ctx = wf.execute("refine", ret_context=True)

    assert exec_ctx.channel.get("total") == 3
    assert exec_ctx.cycle_controller.get_cycle_count("refine") == 3


def test_loop_with_iteration_then_next_task():
    """loop_with_iteration: next_task() hands off to a downstream task after loop exits."""

    with workflow("iter_then_publish") as wf:

        @task(inject_context=True, max_cycles=3)
        def refine(context, data=None):
            draft = f"draft_v{context.cycle_count}"
            context.get_channel().set("result", draft)
            if context.can_iterate():
                context.next_iteration({"draft": draft})
            else:
                context.next_task(publish)

        @task(inject_context=True)
        def publish(context):
            result = context.get_channel().get("result")
            context.get_channel().set("published", f"published_{result}")

        _, exec_ctx = wf.execute("refine", ret_context=True)

    assert exec_ctx.cycle_controller.get_cycle_count("refine") == 3
    assert exec_ctx.channel.get("published") == "published_draft_v3"
