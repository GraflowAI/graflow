"""
LLMAgent with Google ADK Example
=================================

Demonstrates LLMAgent injection using Google's Agent Development Kit (ADK).
Shows how to use ReAct/Supervisor patterns within a single Graflow task.

Prerequisites:
--------------
- Install Google ADK: uv add google-adk (or pip install google-adk)
- Set GOOGLE_API_KEY in .env file
- Optional: Set GRAFLOW_LLM_MODEL for default model

Concepts Covered:
-----------------
1. LLMAgent injection using @task(inject_llm_agent="agent_name")
2. Agent registration in ExecutionContext
3. Google ADK LlmAgent for ReAct/Supervisor patterns
4. Tool definition for agents
5. Agent-based reasoning within tasks
6. Two patterns: passing LlmAgent (simple) or App (advanced)

Expected Output:
----------------
=== LLMAgent Demo ===

Registering research agent with tools...

Task: Research and summarize
Agent thinking: [Agent uses tools to research]
Result: [Structured research summary]

✅ Agent workflow completed successfully!
"""

from typing import TYPE_CHECKING

from graflow.core.context import ExecutionContext
from graflow.llm.agents.base import LLMAgent

if TYPE_CHECKING:
    from google.adk.agents import LlmAgent

from google.adk.agents import LlmAgent

from graflow.core.decorators import task
from graflow.core.workflow import workflow
from graflow.llm.agents.adk_agent import AdkLLMAgent


def main():
    """Run an agent-powered workflow using Google ADK."""
    print("=== LLMAgent Demo ===\n")

    # Define tools for the agent
    def get_current_time() -> str:
        """Get the current time.

        Returns:
            Current time as a string
        """
        from datetime import datetime

        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def calculate(expression: str) -> str:
        """Calculate a mathematical expression.

        Args:
            expression: Mathematical expression to evaluate (e.g., "2 + 2")

        Returns:
            Result of the calculation
        """
        try:
            # Safe eval for simple math expressions
            result = eval(expression, {"__builtins__": {}}, {})
            return str(result)
        except Exception as e:
            return f"Error: {e}"

    with workflow("llm_agent") as ctx:
        # Register the agent so it is available when the workflow executes.
        # Use a factory to access the ExecutionContext (for session_id/app_name).
        # Note: ADK may emit "app name mismatch" warnings (harmless, can be ignored).
        def create_research_agent(exec_context: ExecutionContext) -> LLMAgent:
            """Create Google ADK agent and wrap it for Graflow."""
            print("Registering research agent with tools...")
            try:
                adk_agent = LlmAgent(
                    name="research_agent", model="gemini-2.5-flash", tools=[get_current_time, calculate]
                )

                # Pattern 1: Pass LlmAgent with app_name (simpler, backward compatible)
                agent = AdkLLMAgent(adk_agent, app_name=exec_context.session_id)

                # Pattern 2: Create App and pass it (recommended, more control)
                # from google.adk.apps import App
                # app = App(name=exec_context.session_id, root_agent=adk_agent)
                # agent = AdkLLMAgent(app)

                print("✅ Agent registered successfully\n")
                return agent
            except ImportError:
                print("❌ Google ADK not installed. Install with: uv add google-adk")
                print("Skipping agent demo...")
                raise

        ctx.register_llm_agent("research_agent", create_research_agent)

        @task(inject_llm_agent="research_agent")
        def research_with_agent(llm_agent: LLMAgent):
            """Use agent to perform research with tools."""
            print("Task: Research and summarize")

            # Agent uses tools to complete the task
            result = llm_agent.run(
                """
                Please help me with the following:
                1. What is the current time?
                2. Calculate: 15 * 24
                3. Summarize your findings in one sentence.
                """
            )

            print(f"Agent response:\n{result['output']}\n")
            return result["output"]

        @task
        def process_results():
            """Process the agent's research results."""
            print("Task: Process results")
            print("✅ Results processed successfully\n")
            return "processed"

        # Define pipeline
        research_with_agent >> process_results  # type: ignore

        # Execute the workflow
        ctx.execute()

        print("✅ Agent workflow completed successfully!")


if __name__ == "__main__":
    main()


# ============================================================================
# Key Takeaways:
# ============================================================================
#
# 1. **LLMAgent Injection**
#    # Register agent before execution (optional factory)
#    def create_agent(exec_context):
#        # Use session_id for proper workflow identification
#        return AdkLLMAgent(..., app_name=exec_context.session_id)
#
#    ctx.register_llm_agent("agent_name", create_agent)
#
#    # Then inject agent into task
#    @task(inject_llm_agent="agent_name")
#    def my_task(llm_agent):
#        result = llm_agent.run("...")
#        return result["output"]
#
# 2. **Google ADK Integration**
#    from google.adk.agents import LlmAgent
#    from google.adk.apps import App
#    from graflow.llm.agents.adk_agent import AdkLLMAgent
#
#    # Create ADK agent with tools
#    adk_agent = LlmAgent(name="my_agent", model="gemini-2.5-flash", tools=[...])
#
#    # Pattern 1: Pass LlmAgent with app_name (simpler)
#    agent = AdkLLMAgent(adk_agent, app_name=exec_context.session_id)
#
#    # Pattern 2: Create and pass App (recommended, more control)
#    app = App(name=exec_context.session_id, root_agent=adk_agent)
#    agent = AdkLLMAgent(app)
#
# 3. **Tool Definition**
#    def my_tool(param: str) -> str:
#        \"\"\"Tool description for the agent.
#
#        Args:
#            param: Parameter description
#
#        Returns:
#            Result description
#        \"\"\"
#        # Tool implementation
#        return result
#
# 4. **Agent vs Client**
#    - LLMClient: Direct completion() calls, simple prompts
#    - LLMAgent: ReAct/Supervisor patterns, tool use, multi-step reasoning
#    - Use agents for complex, tool-requiring tasks
#
# 5. **Next Steps**
#    ✅ See multi_agent_workflow.py for multiple agents
#    ✅ Add custom tools for your domain
#    ✅ Check Google ADK docs for advanced patterns
#
# ============================================================================
# Try Experimenting:
# ============================================================================
#
# 1. Add more tools:
#    def search_database(query: str) -> str:
#        \"\"\"Search a database for information.\"\"\"
#        # Your search logic
#        return results
#
#    def send_email(to: str, subject: str, body: str) -> str:
#        \"\"\"Send an email.\"\"\"
#        # Your email logic
#        return "Email sent"
#
#    adk_agent = LlmAgent(
#        name="assistant",
#        model="gemini-2.5-flash",
#        tools=[search_database, send_email, ...]
#    )
#
# 2. Use different models:
#    # Light model for simple tasks
#    fast_agent = LlmAgent(name="fast", model="gemini-2.5-flash-lite", tools=[...])
#
#    # Balanced model for complex tasks
#    thinking_agent = LlmAgent(name="thinker", model="gemini-2.5-flash", tools=[...])
#
# 3. Multi-turn conversations:
#    @task(inject_llm_agent="assistant")
#    def conversation(llm_agent):
#        # First message
#        response1 = llm_agent.run("What's 2+2?")
#
#        # Follow-up (agent remembers context)
#        response2 = llm_agent.run("Now multiply that by 3")
#
#        return response2["output"]
#
# 4. Combine with LLMClient:
#    @task(inject_llm_client=True, inject_llm_agent="assistant")
#    def hybrid_task(llm_client, llm_agent):
#        # Use client for simple completion
#        summary = llm_client.completion(messages=[...])
#
#        # Use agent for complex reasoning with tools
#        result = llm_agent.run(f"Analyze this: {summary}")
#
#        return result["output"]
#
# 5. Error handling:
#    @task(inject_llm_agent="assistant")
#    def safe_agent_call(llm_agent):
#        try:
#            result = llm_agent.run("...")
#            return result["output"]
#        except Exception as e:
#            print(f"Agent error: {e}")
#            return "fallback result"
#
# ============================================================================
