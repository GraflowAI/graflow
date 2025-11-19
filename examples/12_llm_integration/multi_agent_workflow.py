"""
Multi-Agent Workflow Example
=============================

Demonstrates multiple specialized agents working together in a workflow.
Each agent has different tools and responsibilities (Researcher, Analyst, Writer).

Prerequisites:
--------------
- Install Google ADK: uv add google-adk (or pip install google-adk)
- Set GOOGLE_API_KEY in .env file
- Optional: Set OPENAI_API_KEY for LLMClient fallback

Concepts Covered:
-----------------
1. Multiple agent registration in ExecutionContext
2. Specialized agents with different tool sets
3. Agent collaboration via task dependencies
4. Combining LLMClient and LLMAgent in same workflow
5. Passing data between agents through task results

Expected Output:
----------------
=== Multi-Agent Workflow Demo ===

Registering specialized agents...
✅ Researcher agent registered
✅ Analyst agent registered
✅ Writer agent registered

Task 1: Research phase (researcher)
[Researcher gathers data using tools]

Task 2: Analysis phase (analyst)
[Analyst processes data]

Task 3: Writing phase (writer)
[Writer creates final output]

Task 4: Review phase (llm client)
[Final review and summary]

✅ Multi-agent workflow completed successfully!
"""

from typing import Any

from graflow.core.decorators import task
from graflow.core.workflow import workflow
from graflow.llm.agents.base import LLMAgent
from graflow.llm.client import LLMClient


def main():
    """Run a multi-agent collaborative workflow."""
    print("=== Multi-Agent Workflow Demo ===\n")

    # Define tools for different agents
    def fetch_data(topic: str) -> str:
        """Fetch data about a topic (simulated).

        Args:
            topic: Topic to research

        Returns:
            Simulated research data
        """
        # Simulate data fetching
        data = {
            "python": "Python is a high-level programming language. Created in 1991. Popular for AI/ML.",
            "workflow": "Workflow engines manage task execution. Support DAGs, parallel execution, distributed processing.",
        }
        return data.get(topic.lower(), f"No data found for: {topic}")

    def calculate_metrics(data: str) -> str:
        """Calculate metrics from data (simulated).

        Args:
            data: Input data to analyze

        Returns:
            Calculated metrics
        """
        word_count = len(data.split())
        return f"Metrics: {word_count} words analyzed"

    def format_output(text: str, style: str = "markdown") -> str:
        """Format text in a specific style.

        Args:
            text: Text to format
            style: Output style (markdown, html, plain)

        Returns:
            Formatted text
        """
        if style == "markdown":
            return f"# Report\n\n{text}\n"
        return text

    with workflow("multi_agent") as ctx:
        # Store shared data
        workflow_data = {}

        try:
            from google.adk.types import LlmAgent  # type: ignore

            from graflow.llm.agents.adk_agent import AdkLLMAgent
        except ImportError:
            print("❌ Google ADK not installed. Install with: uv add google-adk")
            print("Skipping multi-agent demo...")
            return

        print("Registering specialized agents...")

        def make_agent_factory(agent_name: str, tools):
            def factory(exec_context):
                agent = LlmAgent(
                    name=agent_name,
                    model="gemini-2.0-flash-exp",
                    tools=tools,
                )
                wrapped = AdkLLMAgent(agent, app_name=exec_context.session_id)
                print(f"✅ {agent_name.capitalize()} agent registered")
                return wrapped

            return factory

        ctx.register_llm_agent("researcher", make_agent_factory("researcher", [fetch_data]))
        ctx.register_llm_agent("analyst", make_agent_factory("analyst", [calculate_metrics]))
        ctx.register_llm_agent("writer", make_agent_factory("writer", [format_output]))
        print()

        def _agent_output(result: Any) -> str:
            if isinstance(result, dict) and "output" in result:
                return str(result["output"])
            return str(result)

        @task(inject_llm_agent="researcher")
        def research_phase(llm_agent: LLMAgent):
            """Researcher agent gathers information."""
            print("Task 1: Research phase (researcher)")

            result = llm_agent.run(
                "Use the fetch_data tool to research 'python' and 'workflow'. "
                "Combine your findings into a brief summary."
            )

            output = _agent_output(result)
            workflow_data["research"] = output
            print(f"Research completed: {len(output)} chars\n")
            return output

        @task(inject_llm_agent="analyst")
        def analysis_phase(llm_agent: LLMAgent):
            """Analyst agent processes the research."""
            print("Task 2: Analysis phase (analyst)")

            research_data = workflow_data.get("research", "")
            result = llm_agent.run(
                f"Analyze this research data using calculate_metrics: {research_data[:200]}... "
                "Then provide key insights."
            )

            output = _agent_output(result)
            workflow_data["analysis"] = output
            print("Analysis completed\n")
            return output

        @task(inject_llm_agent="writer")
        def writing_phase(llm_agent: LLMAgent):
            """Writer agent creates final output."""
            print("Task 3: Writing phase (writer)")

            analysis = workflow_data.get("analysis", "")
            result = llm_agent.run(
                f"Based on this analysis: {analysis[:200]}... "
                "Write a concise report and format it using format_output tool with markdown style."
            )

            output = _agent_output(result)
            workflow_data["report"] = output
            print("Report written\n")
            return output

        @task(inject_llm_client=True)
        def review_phase(llm: LLMClient):
            """Use LLMClient for final review (no tools needed)."""
            print("Task 4: Review phase (llm client)")

            report = workflow_data.get("report", "")
            rating = llm.completion_text(
                model="gpt-5-mini",  # Use cheap model for simple review
                messages=[
                    {"role": "user", "content": f"Review this report and rate it (1-10): {report[:300]}..."}
                ],
                max_tokens=50
            )

            print(f"Review: {rating}\n")
            return rating

        # Define pipeline: Researcher -> Analyst -> Writer -> Reviewer
        research_phase >> analysis_phase >> writing_phase >> review_phase  # type: ignore

        # Execute the workflow
        ctx.execute("research_phase")

        print("✅ Multi-agent workflow completed successfully!")


if __name__ == "__main__":
    main()


# ============================================================================
# Key Takeaways:
# ============================================================================
#
# 1. **Multiple Agent Registration**
#    def create_agent(exec_context):
#        return AdkLLMAgent(..., app_name=exec_context.session_id)
#
#    ctx.register_llm_agent("researcher", create_agent)
#    ctx.register_llm_agent("analyst", create_agent)
#    ctx.register_llm_agent("writer", create_agent)
#
# 2. **Specialized Agent Tools**
#    # Each agent has tools for its domain
#    researcher = LlmAgent(name="researcher", tools=[fetch_data, search_web])
#    analyst = LlmAgent(name="analyst", tools=[calculate_stats, visualize])
#    writer = LlmAgent(name="writer", tools=[format_text, check_grammar])
#
# 3. **Agent Collaboration**
#    # Agents work together via task dependencies
#    @task(inject_llm_agent="researcher")
#    def research(llm_agent):
#        data = llm_agent.run("...")["output"]
#        shared_storage["data"] = data  # Pass to next agent
#        return data
#
#    @task(inject_llm_agent="analyst")
#    def analyze(llm_agent):
#        data = shared_storage["data"]  # Get from previous agent
#        result = llm_agent.run(f"Analyze: {data}")["output"]
#        return result
#
# 4. **Mixing LLMClient and LLMAgent**
#    # Use agents for tool-requiring tasks
#    @task(inject_llm_agent="researcher")
#    def complex_task(llm_agent):
#        return llm_agent.run("...")["output"]
#
#    # Use client for simple completions
#    @task(inject_llm_client=True)
#    def simple_task(llm_client):
#        return llm_client.completion(messages=[...])
#
# 5. **Next Steps**
#    ✅ Customize agents for your domain
#    ✅ Add more specialized tools
#    ✅ Implement error handling and retry logic
#    ✅ Use channels for agent communication
#
# ============================================================================
# Try Experimenting:
# ============================================================================
#
# 1. Add more agents:
#    # Validator agent
#    validator = LlmAgent(name="validator", tools=[check_facts, verify_sources])
#
#    # Coordinator agent
#    coordinator = LlmAgent(name="coordinator", tools=[assign_tasks, track_progress])
#
# 2. Use channels for communication:
#    from graflow.channels.factory import ChannelFactory
#
#    @task(inject_context=True, inject_llm_agent="researcher")
#    def research(ctx, llm_agent):
#        result = llm_agent.run("...")["output"]
#        ctx.get_channel("research_data").put(result)  # Share via channel
#
#    @task(inject_context=True, inject_llm_agent="analyst")
#    def analyze(ctx, llm_agent):
#        data = ctx.get_channel("research_data").get()  # Read from channel
#        return llm_agent.run(f"Analyze: {data}")["output"]
#
# 3. Parallel agent execution:
#    # Multiple researchers working in parallel
#    researcher1 >> analyst
#    researcher2 >> analyst
#    researcher3 >> analyst
#
#    # Analyst waits for all researchers to complete
#
# 4. Dynamic agent selection:
#    @task(inject_llm_client=True)
#    def router(llm_client, task_type):
#        # Decide which agent to use
#        if task_type == "research":
#            # Next task uses researcher agent
#            ctx.next_task(research_task)
#        elif task_type == "analysis":
#            # Next task uses analyst agent
#            ctx.next_task(analysis_task)
#
# 5. Agent hierarchies:
#    # Supervisor agent coordinates worker agents
#    @task(inject_llm_agent="supervisor")
#    def supervise(llm_agent):
#        plan = llm_agent.run("Create a plan for these tasks...")["output"]
#
#        # Supervisor delegates to workers
#        for subtask in plan:
#            ctx.next_task(create_worker_task(subtask))
#
# ============================================================================
