"""
Model Override Example
======================

Demonstrates per-task model override for cost/performance optimization.
Use cheaper models for simple tasks and more capable models for complex tasks.

Prerequisites:
--------------
- Set API keys in .env file (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.)
- Optional: Set GRAFLOW_LLM_MODEL for default model

Concepts Covered:
-----------------
1. Per-task model override with model parameter
2. Cost optimization strategy (cheap vs expensive models)
3. Shared LLMClient with different models per call
4. Performance vs accuracy tradeoffs
5. Multi-provider workflows

Expected Output:
----------------
=== Model Override Demo ===

Task 1: Simple classification (gpt-5-mini - cheap & fast)
Category: greeting

Task 2: Complex analysis (gpt-4o - expensive & accurate)
[Detailed analysis of workflow architecture]

Task 3: Quick summary (gpt-5-mini - cheap & fast)
[Brief summary]

✅ Optimized workflow completed!
Cost saved by using appropriate models for each task.
"""

from graflow.core.decorators import task
from graflow.core.workflow import workflow
from graflow.llm.client import LLMClient


def main():
    """Run a cost-optimized LLM workflow with model overrides."""
    print("=== Model Override Demo ===\n")

    with workflow("model_override") as ctx:

        @task(inject_llm_client=True)
        def simple_classification(llm_client: LLMClient):
            """Use cheap model for simple classification task."""
            print("Task 1: Simple classification (gpt-5-mini - cheap & fast)")

            category = llm_client.completion_text(
                model="gpt-5-mini",  # Override: use cheap model
                messages=[{"role": "user", "content": "Classify this text in one word: 'Hello, how are you?'"}],
                max_tokens=10,
            )

            print(f"Category: {category}\n")
            return category

        @task(inject_llm_client=True)
        def complex_analysis(llm_client: LLMClient):
            """Use expensive model for complex reasoning task."""
            print("Task 2: Complex analysis (gpt-4o - expensive & accurate)")

            analysis = llm_client.completion_text(
                model="gpt-4o",  # Override: use powerful model for complex task
                messages=[
                    {
                        "role": "user",
                        "content": """Analyze the architecture of a workflow engine that supports:
                        - Task graphs with dependencies
                        - Parallel execution
                        - Distributed processing via Redis
                        - Dynamic task generation

                        Explain the key design principles in 2-3 sentences.""",
                    }
                ],
                max_tokens=150,
            )

            print(f"{analysis}\n")
            return analysis

        @task(inject_llm_client=True)
        def quick_summary(llm_client: LLMClient):
            """Use cheap model for simple summarization."""
            print("Task 3: Quick summary (gpt-5-mini - cheap & fast)")

            summary = llm_client.completion_text(
                model="gpt-5-mini",  # Override: back to cheap model
                messages=[{"role": "user", "content": "In one sentence: What is a task graph?"}],
                max_tokens=50,
            )

            print(f"{summary}\n")
            return summary

        # Define sequential pipeline
        simple_classification >> complex_analysis >> quick_summary  # type: ignore

        # Execute the workflow
        ctx.execute("simple_classification")

        print("✅ Optimized workflow completed!")
        print("Cost saved by using appropriate models for each task.")


if __name__ == "__main__":
    main()


# ============================================================================
# Key Takeaways:
# ============================================================================
#
# 1. **Model Override**
#    @task(inject_llm_client=True)
#    def my_task(llm):
#        response = llm.completion(
#            model="gpt-4o",  # Override default model
#            messages=[...]
#        )
#
# 2. **Cost Optimization Strategy**
#    - Cheap models (gpt-5-mini): simple tasks, classification, formatting
#    - Expensive models (gpt-4o, claude-3-5-sonnet): complex reasoning, analysis
#    - Shared client: easy to switch models per task
#
# 3. **Common Model Choices**
#    Cheap/Fast:
#      - gpt-5-mini (OpenAI)
#      - claude-3-5-haiku (Anthropic)
#      - gemini-2.5-flash (Google)
#
#    Powerful/Accurate:
#      - gpt-4o (OpenAI)
#      - claude-3-5-sonnet-20241022 (Anthropic)
#      - gemini-2.5-flash (Google)
#
# 4. **Shared LLMClient Benefits**
#    - Single client instance across workflow
#    - Easy model switching per completion() call
#    - Consistent API regardless of model
#    - Automatic provider routing via LiteLLM
#
# 5. **Next Steps**
#    ✅ See llm_agent.py for ReAct/Supervisor patterns
#    ✅ See multi_agent_workflow.py for complex workflows
#    ✅ Check LiteLLM docs for all supported models
#
# ============================================================================
# Try Experimenting:
# ============================================================================
#
# 1. Compare model performance:
#    @task(inject_llm_client=True)
#    def compare_models(llm):
#        prompt = "Explain quantum computing in one sentence."
#
#        # Try with cheap model
#        cheap = llm.completion(model="gpt-5-mini", messages=[...])
#
#        # Try with expensive model
#        expensive = llm.completion(model="gpt-4o", messages=[...])
#
#        print(f"Cheap model: {cheap.choices[0].message.content}")
#        print(f"Expensive model: {expensive.choices[0].message.content}")
#
# 2. Use different providers for different tasks:
#    # Task 1: OpenAI for code
#    response = llm.completion(model="gpt-4o", messages=[...])
#
#    # Task 2: Anthropic for reasoning
#    response = llm.completion(model="claude-3-5-sonnet-20241022", messages=[...])
#
#    # Task 3: Google for speed
#    response = llm.completion(model="gemini-2.5-flash", messages=[...])
#
# 3. Implement cost tracking:
#    @task(inject_llm_client=True)
#    def track_costs(llm):
#        # Rough cost estimates (adjust to current pricing)
#        costs = {
#            "gpt-5-mini": 0.0001,      # per 1k tokens
#            "gpt-4o": 0.01,            # per 1k tokens
#            "claude-3-5-haiku": 0.0001,
#            "claude-3-5-sonnet-20241022": 0.005
#        }
#
#        response = llm.completion(model="gpt-5-mini", messages=[...])
#        tokens = response.usage.total_tokens
#        estimated_cost = (tokens / 1000) * costs["gpt-5-mini"]
#        print(f"Estimated cost: ${estimated_cost:.4f}")
#
# 4. Dynamic model selection:
#    @task(inject_llm_client=True)
#    def smart_routing(llm, task_complexity="simple"):
#        # Choose model based on task complexity
#        model = "gpt-4o" if task_complexity == "complex" else "gpt-5-mini"
#
#        response = llm.completion(model=model, messages=[...])
#        return response.choices[0].message.content
#
# ============================================================================
