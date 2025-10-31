"""
GPT Newspaper Workflow with Graflow
====================================

This example demonstrates a complete newspaper generation workflow using graflow's
dynamic task features. It showcases:

1. Runtime iteration with context.next_iteration() for write-critique loop
2. Conditional branching with context.next_task()
3. Parallel processing of multiple article queries
4. Channel-based state management across task iterations

Prerequisites:
--------------
- TAVILY_API_KEY environment variable
- OPENAI_API_KEY (or other LLM provider API key)

Expected Output:
----------------
Generated newspaper HTML in outputs/ directory with multiple articles.
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List

from agents import (
    CritiqueAgent,
    CuratorAgent,
    DesignerAgent,
    EditorAgent,
    PublisherAgent,
    SearchAgent,
    WriterAgent,
)

from graflow.core.context import TaskExecutionContext
from graflow.core.decorators import task
from graflow.core.workflow import workflow
from graflow.trace.langfuse import LangFuseTracer


def create_article_workflow(query: str, article_id: str, output_dir: str, tracer=None):
    """
    Create a workflow for processing a single article query.

    This workflow demonstrates graflow's dynamic task creation pattern:
    - Search -> Curate -> Write -> Critique
    - If critique has feedback: next_task(writer) then next_task(critique)
    - If critique approves: next_task(designer)

    Args:
        query: Article topic/query
        article_id: Unique identifier for this article
        output_dir: Output directory for article HTML
        tracer: Optional tracer for workflow execution tracking

    Returns:
        Workflow context
    """

    # Initialize agents
    search_agent = SearchAgent()
    curator_agent = CuratorAgent()
    writer_agent = WriterAgent()
    critique_agent = CritiqueAgent()
    designer_agent = DesignerAgent(output_dir)

    with workflow(f"article_{article_id}", tracer=tracer) as wf:

        @task(id=f"search_{article_id}")
        def search_task() -> Dict:
            """Search for news articles on the query."""
            print(f"\n[{article_id}] 🔍 Searching for: {query}")
            article = {"query": query}
            result = search_agent.run(article)
            print(f"[{article_id}] ✅ Found {len(result.get('sources', []))} sources")
            return result

        @task(id=f"curate_{article_id}", inject_context=True)
        def curate_task(context: TaskExecutionContext) -> Dict:
            """Curate the most relevant sources."""
            print(f"[{article_id}] 📋 Curating sources...")
            article_data = context.get_result(f"search_{article_id}")
            if article_data is None:
                raise ValueError(f"Search results missing for {article_id}, cannot curate.")

            result = curator_agent.run(article_data)
            channel = context.get_channel()
            channel.set("article", result)
            channel.set("iteration", 0)
            print(f"[{article_id}] ✅ Selected {len(result.get('sources', []))} sources")
            return result

        @task(id=f"write_{article_id}", inject_context=True)
        def write_task(context: TaskExecutionContext) -> Dict:
            """Write or revise the article."""
            # Get article from channel if not provided (for goto loop-back)
            channel = context.get_channel()
            article = channel.get("article")
            if article is None:
                article = context.get_result(f"curate_{article_id}")
            if article is None:
                article = context.get_result(f"search_{article_id}")
            assert article is not None, "Article data is required for writing."

            # Get iteration count from channel
            iteration = channel.get("iteration", default=0)

            if article.get("critique") is not None:
                print(f"\n[{article_id}] 📝 Revising article (iteration {iteration})...")
            else:
                print(f"\n[{article_id}] ✍️  Writing article...")

            result = writer_agent.run(article)
            print(f"[{article_id}] ✅ Article: {result.get('title', 'Untitled')}")

            # Store in channel for next tasks
            channel.set("article", result)
            channel.set("iteration", iteration)

            return result

        @task(id=f"critique_{article_id}", inject_context=True)
        def critique_task(context: TaskExecutionContext) -> Dict:
            """
            Critique the article and decide next action.

            Uses goto=True pattern: schedules writer task and lets flow continue naturally.
            When approved, naturally flows to design_task in the graph.
            """
            print(f"[{article_id}] 🔎 Critiquing article...")

            channel = context.get_channel()
            article = channel.get("article")
            if article is None:
                article = context.get_result(f"write_{article_id}")
            if article is None:
                article = context.get_result(f"curate_{article_id}")
            if article is None:
                raise ValueError("Article data is required for critique.")

            iteration = channel.get("iteration", default=0)

            result = critique_agent.run(article)

            channel.set("article", result)

            if result.get("critique") is not None:
                # Critique has feedback - need revision
                print(f"[{article_id}] 🔄 Critique feedback received, looping back to writer...")

                # Check iteration limit for safety
                if iteration >= 5:
                    print(f"[{article_id}] ⚠️  Max iterations reached, proceeding to design...")
                    result["critique"] = None
                    # Will fall through and continue to design_task
                else:
                    # Increment iteration and store article with critique feedback
                    channel.set("iteration", iteration + 1)
                    channel.set("article", result)

                    # Jump back to write_task using goto=True
                    # This will trigger: write -> critique -> (loop or design)
                    context.next_task(write_task, goto=True)
                    return result

            # Critique approved or max iterations reached
            # Natural flow continues to design_task from graph
            print(f"[{article_id}] ✅ Article approved by critique!")
            return result

        @task(id=f"design_{article_id}", inject_context=True)
        def design_task(context: TaskExecutionContext) -> Dict:
            """Design the article HTML layout."""
            channel = context.get_channel()
            article = channel.get("article")
            if article is None:
                article = context.get_result(f"critique_{article_id}")
            if article is None:
                article = context.get_result(f"write_{article_id}")
            if article is None:
                article = context.get_result(f"curate_{article_id}")
            if article is None:
                raise ValueError("Article data is required for design.")

            print(f"[{article_id}] 🎨 Designing article layout...")
            design_result = designer_agent.run(article)
            print(f"[{article_id}] ✅ Design complete: {design_result.get('path')}")
            return design_result

        # Build workflow graph: search >> curate >> write >> critique >> design
        _ = search_task >> curate_task >> write_task >> critique_task >> design_task

        return wf


def execute_article_workflow(query: str, article_id: str, output_dir: str) -> Dict:
    """
    Execute a single article workflow.

    Args:
        query: Article query/topic
        article_id: Unique article identifier
        output_dir: Output directory for article HTML

    Returns:
        Completed article dict with design, or None if not found
    """
    print(f"\n{'=' * 80}")
    print(f"Processing Article: {query}")
    print(f"{'=' * 80}")

    # Create tracer for workflow observability
    try:
        tracer = LangFuseTracer(enable_runtime_graph=True)
        print(f"[{article_id}] LangFuse tracer initialized")
    except Exception as e:
        print(f"[{article_id}] Warning: LangFuse tracer initialization failed: {e}")
        print(f"[{article_id}] Continuing without tracing...")
        tracer = None

    # Create and execute workflow
    wf = create_article_workflow(query, article_id, output_dir, tracer=tracer)
    result = wf.execute(
        f"search_{article_id}",
        max_steps=30  # Allow multiple write-critique cycles
    )

    # Shutdown tracer to flush remaining traces
    if tracer:
        tracer.shutdown()

    if isinstance(result, dict):
        return result

    # Fallback: return None if design not found
    raise ValueError(f"❌ Design task not found for {article_id}")


def run_newspaper_workflow(
    queries: List[str],
    layout: str = "layout_1.html",
    max_workers: int | None = None,
    output_dir: str | None = None,
):
    """
    Run the complete newspaper workflow for multiple queries in parallel.

    This demonstrates:
    - Parallel processing of multiple article workflows using ThreadPoolExecutor
    - Dynamic task creation with next_task()
    - Final compilation and publishing

    Args:
        queries: List of article queries/topics
        layout: Newspaper layout template
        max_workers: Max number of parallel workers (None = number of processors)
    """
    print("=" * 80)
    print("🗞️  GPT NEWSPAPER WORKFLOW")
    print("=" * 80)

    # Create output directory
    output_dir = output_dir or f"outputs/run_{int(time.time())}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n📁 Output directory: {output_dir}\n")

    # Execute article workflows in parallel using ThreadPoolExecutor
    print(f"🚀 Processing {len(queries)} articles in parallel...\n")

    # Execute the graph for each query in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Create article IDs
        article_ids = [f"article_{i}" for i in range(len(queries))]

        # Execute workflows in parallel
        completed_articles = list(
            executor.map(
                lambda args: execute_article_workflow(*args),
                zip(queries, article_ids, [output_dir] * len(queries), strict=False)
            )
        )

    # Filter out None results
    completed_articles = [a for a in completed_articles if a is not None]

    # Compile newspaper
    print(f"\n{'=' * 80}")
    print("📰 Compiling Newspaper")
    print(f"{'=' * 80}\n")

    editor_agent = EditorAgent(layout)
    newspaper_html = editor_agent.run(completed_articles)

    # Publish newspaper
    publisher_agent = PublisherAgent(output_dir)
    newspaper_path = publisher_agent.run(newspaper_html)

    print(f"\n{'=' * 80}")
    print("✨ NEWSPAPER GENERATION COMPLETE")
    print(f"{'=' * 80}")
    print(f"\n📰 Newspaper: {newspaper_path}")
    print(f"📄 Articles: {len(completed_articles)}")
    print(f"\nOpen {newspaper_path} in your browser to view!\n")

    return newspaper_path


def main():
    """Run the newspaper workflow with example queries."""

    # Check for required environment variables
    if not os.getenv("TAVILY_API_KEY"):
        print("❌ Error: TAVILY_API_KEY environment variable is required")
        print("Get your API key at: https://tavily.com/")
        return

    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY not found. Make sure your LLM provider is configured.")
        return

    # Example queries
    queries = [
        "Latest developments in artificial intelligence",
        "Climate change policy updates",
        "Technology industry trends",
    ]

    # Run workflow
    run_newspaper_workflow(
        queries=queries,
        layout="layout_1.html"
    )


if __name__ == "__main__":
    main()


# ============================================================================
# Key Graflow Patterns Demonstrated:
# ============================================================================
#
# 1. **Declarative Task Graph**
#    - All tasks defined upfront with @task decorator
#    - Clean task chain: search >> curate >> write >> critique >> design
#    - Each task is focused and independent
#    - No dynamic TaskWrapper creation needed
#
# 2. **Loop-Back Pattern with goto=True**
#    - Critique uses goto=True to jump back to existing write_task
#    - When critique approves, natural graph flow continues to design_task
#    - Flow: write >> critique >> (goto write) >> critique >> design
#    - Pattern from examples/07_dynamic_tasks/runtime_dynamic_tasks.py
#
# 3. **Channel-Based State Management**
#    - Article state persists across task iterations via channels
#    - Iteration counter tracked in channel for safety limits
#    - Write task reads from channel when called via goto
#
# 4. **Parallel Execution with ThreadPoolExecutor**
#    - Multiple article workflows execute concurrently
#    - Each workflow has isolated channel state
#    - Similar to original gpt-newspaper LangGraph implementation
#
# 5. **Safety Limits**
#    - Max 5 write-critique iterations to prevent infinite loops
#    - Configurable via max_steps parameter (default: 30)
#
# ============================================================================
# Real-World Use Cases:
# ============================================================================
#
# This pattern is useful for:
# - Content generation pipelines with quality checks
# - Multi-agent systems with feedback loops
# - Iterative refinement workflows
# - Parallel processing with final aggregation
#
# ============================================================================
