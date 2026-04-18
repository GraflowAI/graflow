"""
Agent Loop Pattern Example
===========================

Demonstrates how to build a cyclic "agent ↔ tool" workflow using Graflow's
`>>` operator. This is the Graflow equivalent of an agent loop with
conditional edges in other frameworks (e.g., LangGraph's
`add_conditional_edges` + `END` sentinel).

The loop runs: agent → tool → agent → tool → ... until the agent task
decides to stop by calling `context.terminate_workflow(...)`.

Concepts Covered:
-----------------
1. Building a loop with `agent >> tool >> agent`
2. Exiting a cycle with `context.terminate_workflow(message)`
3. Passing state across iterations via the workflow channel
4. Context injection (`inject_context=True`)

Expected Output:
----------------
=== Agent Loop Demo ===

[iter 1] agent: thinking...
[iter 1] tool:  running tool (step=1)
[iter 2] agent: thinking...
[iter 2] tool:  running tool (step=2)
[iter 3] agent: thinking...
[iter 3] agent: condition met → terminate

Loop finished. Agent calls: 3, Tool calls: 2
"""

from graflow import task, workflow
from graflow.core.context import TaskExecutionContext


def agent_tool_loop():
    """Run a minimal agent ↔ tool loop."""
    print("=== Agent Loop Demo ===\n")

    # Counters outside the workflow so we can inspect the result.
    # In a real agent, these would be derived from LLM output (e.g., a
    # "finish" tool call, max-token budget, or answer-quality check).
    call_counts = {"agent": 0, "tool": 0}
    MAX_TOOL_CALLS = 2

    with workflow("agent_loop") as ctx:

        @task(inject_context=True)
        def agent(context: TaskExecutionContext):
            """
            Agent task: decides whether to call a tool or terminate.

            In a real workflow this would invoke an LLM, parse its response,
            and route based on whether the model requested a tool call or
            produced a final answer.
            """
            call_counts["agent"] += 1
            iteration = call_counts["agent"]
            print(f"[iter {iteration}] agent: thinking...")

            # Termination condition: we've used enough tool calls
            if call_counts["tool"] >= MAX_TOOL_CALLS:
                print(f"[iter {iteration}] agent: condition met → terminate")
                context.terminate_workflow("budget exhausted")
                return "final_answer"

            # Otherwise, the agent wants the tool to run again.
            # The workflow will flow: agent -> tool -> agent (loop)
            return f"agent_request_{iteration}"

        @task
        def tool():
            """
            Tool task: executes an external action (search, calculation,
            API call, etc.) and returns the result back to the agent.
            """
            call_counts["tool"] += 1
            step = call_counts["tool"]
            print(f"[iter {call_counts['agent']}] tool:  running tool (step={step})")
            return f"tool_result_{step}"

        # The loop: agent runs, then tool, then agent again, repeating
        # until `agent` calls `context.terminate_workflow(...)`.
        #
        # Under the hood this creates edges:
        #   agent -> tool
        #   tool  -> agent
        # Graflow's execution engine follows the cycle at runtime
        # (define-by-run), there's no static DAG compile step.
        agent >> tool >> agent  # type: ignore[operator]

        ctx.execute("agent")

    print(f"\nLoop finished. Agent calls: {call_counts['agent']}, Tool calls: {call_counts['tool']}")


def loop_with_goto():
    """
    Dynamic Jump Loop — Multi-Task Loop with next_task(goto=True)
    ===============================================================

    An iterative loop where the agent dynamically jumps between tasks
    using `next_task(tool, goto=True)`. Inspired by the "ralph-loop"
    pattern (claude-code plugins/ralph-wiggum), where output feeds back
    as input until a completion condition is met or the budget is exhausted.

    Key APIs:
      - @task(max_cycles=N)  — hard iteration budget
      - ctx.cycle_count      — current iteration (1-based)
      - ctx.can_iterate()    — True if budget remains
      - ctx.next_task(t, goto=True) — dynamically jump to another task

    Instead of wiring the cycle statically with `agent >> tool >> agent`,
    the agent *dynamically* jumps to the tool via `next_task(tool, goto=True)`,
    and the tool jumps back. The loop exits when a quality threshold is met
    (completion condition) or when max_cycles is exhausted.

    Expected Output:
    ----------------
    === Dynamic Jump Loop Demo ===

    [cycle 1/5] agent: reflecting... (score=0.2)
               tool:  executing action (call #1)
    [cycle 2/5] agent: reflecting... (score=0.4)
               tool:  executing action (call #2)
    [cycle 3/5] agent: reflecting... (score=0.6)
               tool:  executing action (call #3)
    [cycle 4/5] agent: reflecting... (score=0.8)
    [cycle 4/5] agent: quality threshold reached — done

    Dynamic jump loop finished after 4 cycles. Final score: 0.8
    """
    print("=== Dynamic Jump Loop Demo ===\n")

    quality_threshold = 0.75
    state = {"score": 0.0, "tool_calls": 0}

    with workflow("loop_with_goto") as wf:

        @task(inject_context=True, max_cycles=5)
        def agent(context: TaskExecutionContext):
            """
            Agent decides whether to call a tool or finalize.

            Uses cycle_count / can_iterate() for budget awareness,
            and next_task(tool, goto=True) to jump to the tool.
            """
            state["score"] += 0.2
            cycle = context.cycle_count
            max_c = context.max_cycles
            print(f"[cycle {cycle}/{max_c}] agent: reflecting... (score={state['score']:.1f})")

            if state["score"] >= quality_threshold:
                print(f"[cycle {cycle}/{max_c}] agent: quality threshold reached — done")
                context.get_channel().set("final_score", state["score"])
                return f"final_answer (score={state['score']:.1f})"

            # Jump to tool; goto=True skips normal successors
            context.next_task(tool, goto=True)

        @task(inject_context=True)
        def tool(context: TaskExecutionContext):
            """Tool runs an action, then jumps back to agent."""
            state["tool_calls"] += 1
            print(f"           tool:  executing action (call #{state['tool_calls']})")
            # Jump back to agent for the next reflect cycle
            context.next_task(agent, goto=True)

        # No >> wiring needed — jumps are fully dynamic
        _, exec_ctx = wf.execute("agent", ret_context=True)

    cycles_used = exec_ctx.cycle_controller.get_cycle_count("agent")
    final_score = exec_ctx.channel.get("final_score")
    print(f"\nDynamic jump loop finished after {cycles_used} cycles. Final score: {final_score}")


def loop_with_iteration():
    """
    Self-Refinement Loop — Single-Task Iteration with next_iteration()
    ===================================================================

    The simplest loop pattern: a single task re-executes itself using
    `next_iteration(data)`. Each iteration receives the previous output
    as input, enabling accumulative refinement without a separate tool task.

    Key APIs:
      - @task(max_cycles=N)       — cap the number of iterations
      - ctx.can_iterate()         — True if cycle budget remains
      - ctx.next_iteration(data)  — re-queue this task with new input
      - ctx.cycle_count           — current iteration (1-based)

    Expected Output:
    ----------------
    === Self-Refinement Loop Demo ===

    [iter 1/5] refining... (draft_v1, score=0.3)
    [iter 2/5] refining... (draft_v2, score=0.5)
    [iter 3/5] refining... (draft_v3, score=0.7)
    [iter 4/5] refining... (draft_v4, score=0.9)
    [iter 4/5] quality threshold reached — accepting draft_v4
    [publish] publishing draft_v4

    Self-refinement finished after 4 iterations. Published: draft_v4
    """
    print("=== Self-Refinement Loop Demo ===\n")

    quality_threshold = 0.85

    with workflow("self_refine") as wf:

        @task(inject_context=True, max_cycles=5)
        def refine(context: TaskExecutionContext, data=None):
            """
            Each iteration improves the draft and checks quality.
            Stops early if quality is good enough, or when budget runs out.
            On exit, hands off to `publish` via next_task().
            """
            prev = data or {}
            version = context.cycle_count
            draft = f"draft_v{version}"
            score = prev.get("score", 0.1) + 0.2

            print(f"[iter {version}/{context.max_cycles}] refining... ({draft}, score={score:.1f})")

            if score >= quality_threshold:
                print(f"[iter {version}/{context.max_cycles}] quality threshold reached — accepting {draft}")
                context.get_channel().set("result", draft)
                context.next_task(publish)
                return draft

            # Feed output back as input for the next iteration
            if context.can_iterate():
                context.next_iteration({"draft": draft, "score": score})
            else:
                # Budget exhausted — accept best effort
                print(f"[iter {version}/{context.max_cycles}] budget exhausted — accepting {draft}")
                context.get_channel().set("result", draft)
                context.next_task(publish)

        @task(inject_context=True)
        def publish(context: TaskExecutionContext):
            """Downstream task that runs after the refinement loop finishes."""
            result = context.get_channel().get("result")
            print(f"[publish] publishing {result}")
            context.get_channel().set("published", result)

        _, exec_ctx = wf.execute("refine", ret_context=True)

    result = exec_ctx.channel.get("published")
    cycles = exec_ctx.cycle_controller.get_cycle_count("refine")
    print(f"\nSelf-refinement finished after {cycles} iterations. Published: {result}")


if __name__ == "__main__":
    agent_tool_loop()
    print()
    loop_with_goto()
    print()
    loop_with_iteration()


# ============================================================================
# Key Takeaways:
# ============================================================================
#
# 1. **Cycles are first-class**
#    Unlike compile-time DAG frameworks, Graflow lets you literally write
#    `agent >> tool >> agent` — the engine handles the cycle at runtime.
#
# 2. **Three Loop Styles**
#
#    a) **`agent >> tool >> agent` + `terminate_workflow()`** (agent_tool_loop)
#       Static cycle — edges are declared upfront. The engine drives the
#       loop until a task calls terminate_workflow(). Best when the
#       agent/tool boundary is fixed.
#
#    b) **`@task(max_cycles=N)` + `next_task(t, goto=True)`** (loop_with_goto)
#       Dynamic jumps — the agent decides *which* task to run next.
#       `max_cycles` provides a hard safety cap. Best for multi-task
#       iterative refinement where routing varies per cycle.
#
#    c) **`@task(max_cycles=N)` + `next_iteration(data)`** (loop_with_iteration)
#       Single-task self-loop — output feeds back as input. Simplest
#       pattern for accumulative refinement, polling, or optimization
#       where no separate tool task is needed.
#
# 3. **Cycle budget APIs**
#    - `ctx.cycle_count`        — how many times this task has run (1-based)
#    - `ctx.max_cycles`         — the budget set by @task(max_cycles=N)
#    - `ctx.can_iterate()       — shorthand for cycle_count < max_cycles
#    - `ctx.next_iteration(data)` — re-queue self with data as input
#
# 4. **Related Patterns**
#    - For more iteration examples with data passing, see
#      examples/07_dynamic_tasks/task_iterations.py
#    - For branching to a recovery path, use
#      `context.next_task(handler, goto=True)`.
#
# ============================================================================
