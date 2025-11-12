"""
Simple LLMClient Injection Example
===================================

Demonstrates basic LLMClient injection into tasks for LLM-powered workflows.
The LLMClient is automatically created with default settings or from environment variables.

Prerequisites:
--------------
- Set OPENAI_API_KEY (or other provider API key) in .env file
- Optional: Set GRAFLOW_LLM_MODEL in .env (defaults to gpt-5-mini)

Concepts Covered:
-----------------
1. LLMClient injection using @task(inject_llm_client=True)
2. Automatic LLMClient initialization from environment
3. Basic completion() calls with LiteLLM
4. Shared LLMClient instance across tasks
5. Simple prompt engineering in workflows

Expected Output:
----------------
=== Simple LLMClient Demo ===

Task 1: Generate greeting
Hello! I'm your AI assistant. How can I help you today?

Task 2: Answer question
[AI-generated answer about Python]

Task 3: Summarize text
[AI-generated summary]

✅ LLM workflow completed successfully!
"""

from graflow.core.decorators import task
from graflow.core.workflow import workflow


def main():
    """Run a simple LLM-powered workflow."""
    print("=== Simple LLMClient Demo ===\n")

    with workflow("simple_llm") as ctx:

        @task(inject_llm_client=True)
        def generate_greeting(llm):
            """Generate a friendly greeting using LLM."""
            print("Task 1: Generate greeting")

            response = llm.completion(
                messages=[
                    {"role": "user", "content": "Say a brief, friendly greeting as an AI assistant."}
                ],
                max_tokens=50
            )

            greeting = response.choices[0].message.content
            print(f"{greeting}\n")
            return greeting

        @task(inject_llm_client=True)
        def answer_question(llm):
            """Answer a technical question using LLM."""
            print("Task 2: Answer question")

            response = llm.completion(
                messages=[
                    {"role": "user", "content": "In one sentence, what is Python?"}
                ],
                max_tokens=50
            )

            answer = response.choices[0].message.content
            print(f"{answer}\n")
            return answer

        @task(inject_llm_client=True)
        def summarize_text(llm):
            """Summarize a text using LLM."""
            print("Task 3: Summarize text")

            text = """
            Graflow is an executable task graph engine for Python workflow execution.
            It provides both local and distributed execution capabilities with support
            for task graphs, parallel execution, inter-task communication via channels,
            cycle detection, and dynamic task generation.
            """

            response = llm.completion(
                messages=[
                    {"role": "user", "content": f"Summarize in one sentence: {text}"}
                ],
                max_tokens=50
            )

            summary = response.choices[0].message.content
            print(f"{summary}\n")
            return summary

        # Define sequential pipeline
        generate_greeting >> answer_question >> summarize_text  # type: ignore

        # Execute the workflow
        ctx.execute("generate_greeting")

        print("✅ LLM workflow completed successfully!")


if __name__ == "__main__":
    main()


# ============================================================================
# Key Takeaways:
# ============================================================================
#
# 1. **LLMClient Injection**
#    @task(inject_llm_client=True)
#    def my_task(llm):
#        # llm is automatically injected
#        response = llm.completion(messages=[...])
#
# 2. **Automatic Initialization**
#    - LLMClient is auto-created from ExecutionContext
#    - Default model: GRAFLOW_LLM_MODEL env var or gpt-5-mini
#    - Requires API key in environment (e.g., OPENAI_API_KEY)
#
# 3. **Shared Instance**
#    - Same LLMClient instance is shared across all tasks
#    - Configuration is consistent throughout workflow
#    - Efficient resource usage
#
# 4. **LiteLLM Integration**
#    - llm.completion() wraps LiteLLM's completion API
#    - Supports all LiteLLM-compatible providers
#    - Consistent API across different models
#
# 5. **Next Steps**
#    ✅ See model_override.py for per-task model selection
#    ✅ See llm_agent.py for ReAct/Supervisor patterns
#    ✅ See multi_agent_workflow.py for complex agent workflows
#
# ============================================================================
# Try Experimenting:
# ============================================================================
#
# 1. Change the default model:
#    # In .env file:
#    GRAFLOW_LLM_MODEL=claude-3-5-sonnet-20241022
#
# 2. Add temperature control:
#    response = llm.completion(
#        messages=[...],
#        temperature=0.7
#    )
#
# 3. Use different providers:
#    # Set appropriate API key in .env:
#    ANTHROPIC_API_KEY=sk-ant-...
#    GOOGLE_API_KEY=...
#
#    # Then use the model:
#    GRAFLOW_LLM_MODEL=claude-3-5-sonnet-20241022
#    # or
#    GRAFLOW_LLM_MODEL=gemini-2.0-flash-exp
#
# 4. Chain LLM calls:
#    @task(inject_llm_client=True)
#    def chain_example(llm):
#        # First call
#        idea = llm.completion(messages=[{"role": "user", "content": "Suggest a topic"}])
#
#        # Second call uses result from first
#        expanded = llm.completion(messages=[
#            {"role": "user", "content": f"Explain: {idea.choices[0].message.content}"}
#        ])
#
#        return expanded.choices[0].message.content
#
# ============================================================================
