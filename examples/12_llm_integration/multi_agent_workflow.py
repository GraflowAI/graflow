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

from graflow.core.decorators import task
from graflow.core.workflow import workflow


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

        @task(inject_context=True)
        def setup_agents(context):
            """Register all specialized agents."""
            print("Registering specialized agents...")
            try:
                from google_adk.types import LlmAgent

                from graflow.llm.agents.adk_agent import AdkLLMAgent

                # Get ExecutionContext from TaskExecutionContext
                exec_context = context.execution_context

                # Agent 1: Researcher - has data fetching tools
                researcher_agent = LlmAgent(
                    name="researcher",
                    model="gemini-2.0-flash-exp",
                    tools=[fetch_data]
                )
                exec_context.register_llm_agent(
                    "researcher",
                    AdkLLMAgent(researcher_agent, app_name=exec_context.session_id)
                )
                print("✅ Researcher agent registered")

                # Agent 2: Analyst - has analysis tools
                analyst_agent = LlmAgent(
                    name="analyst",
                    model="gemini-2.0-flash-exp",
                    tools=[calculate_metrics]
                )
                exec_context.register_llm_agent(
                    "analyst",
                    AdkLLMAgent(analyst_agent, app_name=exec_context.session_id)
                )
                print("✅ Analyst agent registered")

                # Agent 3: Writer - has formatting tools
                writer_agent = LlmAgent(
                    name="writer",
                    model="gemini-2.0-flash-exp",
                    tools=[format_output]
                )
                exec_context.register_llm_agent(
                    "writer",
                    AdkLLMAgent(writer_agent, app_name=exec_context.session_id)
                )
                print("✅ Writer agent registered\n")
                return "agents_registered"

            except ImportError:
                print("❌ Google ADK not installed. Install with: uv add google-adk")
                print("Skipping multi-agent demo...")
                raise

        @task(inject_llm_agent="researcher")
        def research_phase(agent):
            """Researcher agent gathers information."""
            print("Task 1: Research phase (researcher)")

            result = agent.send_message(
                "Use the fetch_data tool to research 'python' and 'workflow'. "
                "Combine your findings into a brief summary."
            )

            workflow_data["research"] = result
            print(f"Research completed: {len(result)} chars\n")
            return result

        @task(inject_llm_agent="analyst")
        def analysis_phase(agent):
            """Analyst agent processes the research."""
            print("Task 2: Analysis phase (analyst)")

            research_data = workflow_data.get("research", "")
            result = agent.send_message(
                f"Analyze this research data using calculate_metrics: {research_data[:200]}... "
                "Then provide key insights."
            )

            workflow_data["analysis"] = result
            print("Analysis completed\n")
            return result

        @task(inject_llm_agent="writer")
        def writing_phase(agent):
            """Writer agent creates final output."""
            print("Task 3: Writing phase (writer)")

            analysis = workflow_data.get("analysis", "")
            result = agent.send_message(
                f"Based on this analysis: {analysis[:200]}... "
                "Write a concise report and format it using format_output tool with markdown style."
            )

            workflow_data["report"] = result
            print("Report written\n")
            return result

        @task(inject_llm_client=True)
        def review_phase(llm):
            """Use LLMClient for final review (no tools needed)."""
            print("Task 4: Review phase (llm client)")

            report = workflow_data.get("report", "")
            response = llm.completion(
                model="gpt-5-mini",  # Use cheap model for simple review
                messages=[
                    {"role": "user", "content": f"Review this report and rate it (1-10): {report[:300]}..."}
                ],
                max_tokens=50
            )

            rating = response.choices[0].message.content
            print(f"Review: {rating}\n")
            return rating

        # Define pipeline: Setup -> Researcher -> Analyst -> Writer -> Reviewer
        setup_agents >> research_phase >> analysis_phase >> writing_phase >> review_phase  # type: ignore

        # Execute the workflow
        ctx.execute("setup_agents")

        print("✅ Multi-agent workflow completed successfully!")


if __name__ == "__main__":
    main()


# ============================================================================
# Key Takeaways:
# ============================================================================
#
# 1. **Multiple Agent Registration**
#    @task(inject_context=True)
#    def setup(context):
#        exec_context = context.execution_context
#
#        # Register multiple agents
#        exec_context.register_llm_agent("researcher", researcher_agent)
#        exec_context.register_llm_agent("analyst", analyst_agent)
#        exec_context.register_llm_agent("writer", writer_agent)
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
#    def research(agent):
#        data = agent.send_message("...")
#        shared_storage["data"] = data  # Pass to next agent
#        return data
#
#    @task(inject_llm_agent="analyst")
#    def analyze(agent):
#        data = shared_storage["data"]  # Get from previous agent
#        result = agent.send_message(f"Analyze: {data}")
#        return result
#
# 4. **Mixing LLMClient and LLMAgent**
#    # Use agents for tool-requiring tasks
#    @task(inject_llm_agent="researcher")
#    def complex_task(agent):
#        return agent.send_message("...")
#
#    # Use client for simple completions
#    @task(inject_llm_client=True)
#    def simple_task(llm):
#        return llm.completion(messages=[...])
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
#    @task(inject_llm_agent="researcher")
#    def research(agent, ctx):
#        result = agent.send_message("...")
#        ctx.channel("research_data").put(result)  # Share via channel
#
#    @task(inject_llm_agent="analyst")
#    def analyze(agent, ctx):
#        data = ctx.channel("research_data").get()  # Read from channel
#        return agent.send_message(f"Analyze: {data}")
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
#    def router(llm, task_type):
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
#    def supervise(agent):
#        plan = agent.send_message("Create a plan for these tasks...")
#
#        # Supervisor delegates to workers
#        for subtask in plan:
#            ctx.next_task(create_worker_task(subtask))
#
# ============================================================================
