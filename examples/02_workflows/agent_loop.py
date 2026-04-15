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


def main():
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


if __name__ == "__main__":
    main()


# ============================================================================
# Key Takeaways:
# ============================================================================
#
# 1. **Cycles are first-class**
#    Unlike compile-time DAG frameworks, Graflow lets you literally write
#    `agent >> tool >> agent` — the engine handles the cycle at runtime.
#
# 2. **Exit with `terminate_workflow(message)`**
#    The loop keeps running until a task explicitly terminates the workflow.
#    The `message` argument is recorded for logging/tracing.
#
# 3. **Common termination conditions for real agents**
#    - LLM returns a "finish" / "final_answer" tool call
#    - Token or cost budget exceeded
#    - Answer passes a quality / self-consistency check
#    - Maximum iteration count reached (use `max_cycles` on the @task)
#
# 4. **Related Patterns**
#    - For parameter-driven loops (e.g., "keep optimizing until converged"),
#      prefer `context.next_iteration(data)` — see
#      examples/07_dynamic_tasks/runtime_dynamic_tasks.py
#    - For branching to a recovery path instead of looping, use
#      `context.next_task(handler, goto=True)`.
#
# ============================================================================
# Try Experimenting:
# ============================================================================
#
# 1. Add a "reasoning" channel to pass the agent's scratchpad between iterations:
#
#    @task(inject_context=True)
#    def agent(context):
#        ch = context.get_channel()
#        scratchpad = ch.get("scratchpad", [])
#        scratchpad.append(f"thought at step {len(scratchpad)}")
#        ch.set("scratchpad", scratchpad)
#
# 2. Gate the tool on the agent's decision:
#
#    @task
#    def tool(agent_request: str):
#        if agent_request.startswith("search:"):
#            return do_search(agent_request[7:])
#        return do_default()
#
# 3. Put a hard cap on iterations with `max_cycles`:
#
#    @task(inject_context=True, max_cycles=10)
#    def agent(context):
#        ...
# ============================================================================
