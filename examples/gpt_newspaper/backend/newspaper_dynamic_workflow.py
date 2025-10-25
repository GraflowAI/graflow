"""
GPT Newspaper Workflow with Graflow
====================================

This example demonstrates a complete newspaper generation workflow using graflow's
dynamic task features. It showcases:

1. Runtime iteration with context.next_iteration() for write-critique loop
2. Runtime task fan-out with context.next_task() for research expansion and gap filling
3. Parallel writer personas and quality gates protected by AtLeastNGroupPolicy
4. Channel-based state management across task iterations and personas

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
from typing import Any, Dict, List

from agents import (
    CritiqueAgent,
    CuratorAgent,
    DesignerAgent,
    EditorAgent,
    PublisherAgent,
    SearchAgent,
    WriterAgent,
)

from graflow.coordination.coordinator import CoordinationBackend
from graflow.core.context import TaskExecutionContext
from graflow.core.decorators import task
from graflow.core.handlers.group_policy import AtLeastNGroupPolicy, BestEffortGroupPolicy
from graflow.core.task import TaskWrapper
from graflow.core.workflow import workflow


MAX_REVISION_ITERATIONS = 3
DEFAULT_IMAGE_URL = "https://images.unsplash.com/photo-1542281286-9e0a16bb7366"


def create_article_workflow(query: str, article_id: str, output_dir: str):
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

    Returns:
        Workflow context
    """

    # Initialize agents
    search_agent = SearchAgent()
    curator_agent = CuratorAgent()
    writer_agent = WriterAgent()
    critique_agent = CritiqueAgent()
    designer_agent = DesignerAgent(output_dir)

    with workflow(f"article_{article_id}") as wf:

        def _slugify(value: str) -> str:
            return value.lower().replace(" ", "_").replace("/", "_")

        def _store_draft(channel, label: str, draft: Dict[str, Any]) -> None:
            drafts = channel.get("drafts", default=[])
            drafts.append({"label": label, "article": draft})
            channel.set("drafts", drafts)

        def _safe_get_result(context: TaskExecutionContext, task_id: str) -> Dict[str, Any] | None:
            try:
                return context.get_result(task_id)
            except KeyError:
                return None
            except ValueError:
                return None

        def _get_article_from_channel(context: TaskExecutionContext) -> Dict[str, Any]:
            channel = context.get_channel()
            article = channel.get("article")
            if article is None:
                article = _safe_get_result(context, f"curate_{article_id}")
            if article is None:
                article = {"query": query, "image": DEFAULT_IMAGE_URL}
            image = article.get("image") or channel.get("image") or DEFAULT_IMAGE_URL
            article["image"] = image
            channel.set("image", image)
            return article

        def _should_flag(task_tag: str) -> bool:
            checksum = sum(ord(c) for c in f"{article_id}:{task_tag}:{query}")
            return checksum % 7 == 0

        @task(id=f"topic_intake_{article_id}", inject_context=True)
        def topic_intake_task(context: TaskExecutionContext) -> Dict:
            """Profile the query and decide which research angles to pursue."""
            print(f"\n[{article_id}] 🧭 Profiling topic: {query}")
            lowered = query.lower()
            angles: List[str] = []
            if any(keyword in lowered for keyword in ("policy", "regulation", "bill")):
                angles.append("policy response")
            if any(keyword in lowered for keyword in ("trend", "industry", "market")):
                angles.append("market outlook")
            if any(keyword in lowered for keyword in ("climate", "energy", "emission")):
                angles.append("climate impact")
            if "ai" in lowered or "artificial" in lowered:
                angles.append("technology spotlight")
            if not angles:
                angles = ["overview", "human impact"]

            priority = "breaking" if any(k in lowered for k in ("breaking", "urgent")) else "standard"
            channel = context.get_channel()
            channel.set("angles", angles)
            channel.set("topic_profile", {"query": query, "priority": priority})
            channel.set("search_results", [])
            channel.set("expected_search_tasks", 0)
            channel.set("completed_search_tasks", 0)
            channel.set("drafts", [])
            channel.set("iteration", 0)
            channel.set("image", None)

            print(f"[{article_id}] ✅ Planned angles: {', '.join(angles)} (priority={priority})")
            return {"query": query, "angles": angles, "priority": priority}

        @task(id=f"search_router_{article_id}", inject_context=True)
        def search_router_task(context: TaskExecutionContext) -> Dict:
            """Fan out search tasks per angle using runtime task creation."""
            channel = context.get_channel()
            angles: List[str] = channel.get("angles", default=["overview"])
            channel.set("expected_search_tasks", len(angles))
            channel.set("completed_search_tasks", 0)
            channel.set("search_results", [])

            print(f"[{article_id}] 🔍 Launching {len(angles)} targeted searches...")

            for angle in angles:
                angle_id = _slugify(angle)

                def run_angle(angle_label=angle, angle_slug=angle_id):
                    task_query = f"{query} - focus on {angle_label}"
                    payload = {"query": task_query, "angle": angle_label, "base_query": query}
                    result = search_agent.run(payload)
                    sources = result.get("sources", [])
                    aggregated = channel.get("search_results", default=[])
                    aggregated.append(result)
                    channel.set("search_results", aggregated)
                    image = channel.get("image")
                    if not image:
                        channel.set("image", result.get("image") or DEFAULT_IMAGE_URL)
                    completed = channel.get("completed_search_tasks", default=0) + 1
                    channel.set("completed_search_tasks", completed)
                    print(
                        f"[{article_id}]   ↳ [{angle_slug}] completed with {len(sources)} sources (total {completed}/{len(angles)})"
                    )
                    return result

                angle_task = TaskWrapper(f"search_{article_id}_{angle_id}", run_angle)
                context.next_task(angle_task)

            return {"scheduled": len(angles)}

        @task(id=f"curate_{article_id}", inject_context=True)
        def curate_task(context: TaskExecutionContext) -> Dict:
            """Curate aggregated sources, waiting for async searches if necessary."""
            channel = context.get_channel()
            expected = channel.get("expected_search_tasks", default=0)
            completed = channel.get("completed_search_tasks", default=0)

            if expected and completed < expected:
                remaining = expected - completed
                print(f"[{article_id}] ⏳ Waiting for {remaining} search tasks to finish before curating...")
                time.sleep(0.05)
                context.next_iteration()
                return {"status": "waiting"}

            aggregated_sources = []
            for batch in channel.get("search_results", default=[]):
                aggregated_sources.extend(batch.get("sources", []))

            image = channel.get("image") or DEFAULT_IMAGE_URL
            article_data = {"query": query, "sources": aggregated_sources, "image": image}
            print(f"[{article_id}] 📋 Curating {len(aggregated_sources)} raw sources...")
            result = curator_agent.run(article_data)
            result.setdefault("image", image)

            channel.set("article", result)
            channel.set("iteration", 0)

            min_sources = 3
            if len(result.get("sources", [])) < min_sources and not channel.get("gap_fill_requested", default=False):
                channel.set("gap_fill_requested", True)
                channel.set("expected_search_tasks", expected + 1)

                def supplemental_research():
                    supplemental_payload = {
                        "query": f"{query} statistics and data",
                        "angle": "data supplement",
                        "base_query": query,
                    }
                    supplemental_result = search_agent.run(supplemental_payload)
                    aggregated = channel.get("search_results", default=[])
                    aggregated.append(supplemental_result)
                    channel.set("search_results", aggregated)
                    if not channel.get("image"):
                        channel.set("image", supplemental_result.get("image") or DEFAULT_IMAGE_URL)
                    channel.set("completed_search_tasks", channel.get("completed_search_tasks", default=0) + 1)
                    print(f"[{article_id}]   ↳ supplemental search completed with {len(supplemental_result.get('sources', []))} sources")
                    return supplemental_result

                context.next_task(TaskWrapper(f"search_{article_id}_supplemental", supplemental_research))
                print(f"[{article_id}] 🔄 Not enough sources, scheduling supplemental research...")
                time.sleep(0.05)
                context.next_iteration()
                return {"status": "gap_filling"}

            channel.set("gap_fill_requested", False)
            print(f"[{article_id}] ✅ Selected {len(result.get('sources', []))} curated sources")
            return result

        def _generate_draft(context: TaskExecutionContext, style: str, tone: str) -> Dict:
            channel = context.get_channel()
            article = channel.get("article")
            if article is None:
                article = context.get_result(f"curate_{article_id}") or {"query": query, "image": DEFAULT_IMAGE_URL}

            image = article.get("image") or channel.get("image") or DEFAULT_IMAGE_URL
            article["image"] = image

            payload = {**article, "style": style, "tone": tone}
            print(f"[{article_id}] ✍️  Writing {style} draft...")
            draft = writer_agent.run(payload)
            draft["style"] = style
            _store_draft(channel, style, draft)
            return draft

        @task(id=f"write_feature_{article_id}", inject_context=True)
        def write_feature_task(context: TaskExecutionContext) -> Dict:
            return _generate_draft(context, "feature", "in-depth")

        @task(id=f"write_brief_{article_id}", inject_context=True)
        def write_brief_task(context: TaskExecutionContext) -> Dict:
            return _generate_draft(context, "brief", "concise")

        @task(id=f"write_data_digest_{article_id}", inject_context=True)
        def write_data_digest_task(context: TaskExecutionContext) -> Dict:
            return _generate_draft(context, "data_digest", "analytical")

        writer_personas = (
            write_feature_task | write_brief_task | write_data_digest_task
        ).with_execution(
            backend=CoordinationBackend.THREADING,
            policy=BestEffortGroupPolicy(),
        )

        @task(id=f"select_draft_{article_id}", inject_context=True)
        def select_draft_task(context: TaskExecutionContext) -> Dict:
            """Pick the strongest draft emitted by the persona writers."""
            channel = context.get_channel()
            drafts = channel.get("drafts", default=[])
            if not drafts:
                # Fallback to whatever results exist
                drafts = [
                    {"label": "feature", "article": _safe_get_result(context, f"write_feature_{article_id}")},
                    {"label": "brief", "article": _safe_get_result(context, f"write_brief_{article_id}")},
                    {"label": "data_digest", "article": _safe_get_result(context, f"write_data_digest_{article_id}")},
                ]
                drafts = [d for d in drafts if d["article"] is not None]
            ranked = sorted(
                drafts,
                key=lambda item: len(item["article"].get("sources", [])) if item.get("article") else 0,
                reverse=True,
            )
            if not ranked:
                raise ValueError(f"No drafts produced for {article_id}")

            chosen = ranked[0]
            channel.set("article", chosen["article"])
            channel.set("selected_draft", chosen["label"])
            channel.set("iteration", 0)
            channel.set("drafts", [])
            print(f"[{article_id}] 🏅 Selected {chosen['label']} draft for refinement")
            return chosen["article"]

        @task(id=f"write_{article_id}", inject_context=True)
        def write_task(context: TaskExecutionContext) -> Dict:
            """Refine or revise the selected draft."""
            channel = context.get_channel()
            article = channel.get("article")
            if article is None:
                article = _safe_get_result(context, f"select_draft_{article_id}")
            if article is None:
                article = _safe_get_result(context, f"curate_{article_id}")
            if article is None:
                raise ValueError("Article data is required for writing.")

            iteration = channel.get("iteration", default=0)
            image = article.get("image") or channel.get("image") or DEFAULT_IMAGE_URL
            article["image"] = image
            channel.set("image", image)

            if article.get("critique") is not None or iteration > 0:
                print(f"\n[{article_id}] 📝 Revising article (iteration {iteration})...")
            else:
                print(f"\n[{article_id}] ✍️  Refining primary draft...")

            result = writer_agent.run(article)
            print(f"[{article_id}] ✅ Article: {result.get('title', 'Untitled')}")

            channel.set("article", result)
            channel.set("iteration", iteration)

            return result

        @task(id=f"critique_{article_id}", inject_context=True)
        def critique_task(context: TaskExecutionContext) -> Dict:
            """Critique the article and decide whether to loop back to writing."""
            print(f"[{article_id}] 🔎 Critiquing article...")

            channel = context.get_channel()
            article = _get_article_from_channel(context)
            image = article.get("image") or context.get_channel().get("image") or DEFAULT_IMAGE_URL
            article["image"] = image
            iteration = channel.get("iteration", default=0)

            result = critique_agent.run(article)

            channel.set("article", result)

            if result.get("critique") is not None:
                print(f"[{article_id}] 🔄 Critique feedback received, looping back to writer...")

                if iteration >= MAX_REVISION_ITERATIONS:
                    print(
                        f"[{article_id}] ⚠️  Max iterations ({MAX_REVISION_ITERATIONS}) reached, moving forward regardless"
                    )
                    result["critique"] = None
                else:
                    channel.set("iteration", iteration + 1)
                    channel.set("article", result)
                    context.next_task(write_task, goto=True)
                    return result

            print(f"[{article_id}] ✅ Article approved by critique!")
            return result

        @task(id=f"fact_check_{article_id}", inject_context=True)
        def fact_check_task(context: TaskExecutionContext) -> Dict:
            article = _get_article_from_channel(context)
            if _should_flag("fact"):
                raise RuntimeError("Fact discrepancy detected in references")
            print(f"[{article_id}] 🔎 Fact check cleared")
            return {"status": "ok", "check": "fact", "title": article.get("title")}

        @task(id=f"compliance_check_{article_id}", inject_context=True)
        def compliance_check_task(context: TaskExecutionContext) -> Dict:
            article = _get_article_from_channel(context)
            if _should_flag("compliance"):
                raise RuntimeError("Compliance guidelines not met")
            print(f"[{article_id}] 🛡️  Compliance check cleared")
            return {"status": "ok", "check": "compliance", "title": article.get("title")}

        @task(id=f"risk_check_{article_id}", inject_context=True)
        def risk_check_task(context: TaskExecutionContext) -> Dict:
            article = _get_article_from_channel(context)
            if _should_flag("risk") and context.get_channel().get("topic_profile", {}).get("priority") == "breaking":
                raise RuntimeError("Risk review flagged sensitive content")
            print(f"[{article_id}] 📊 Risk check cleared")
            return {"status": "ok", "check": "risk", "title": article.get("title")}

        quality_gate = (
            fact_check_task | compliance_check_task | risk_check_task
        ).with_execution(
            backend=CoordinationBackend.THREADING,
            policy=AtLeastNGroupPolicy(min_success=2),
        )

        @task(id=f"quality_gate_summary_{article_id}", inject_context=True)
        def quality_gate_summary_task(context: TaskExecutionContext) -> Dict:
            checks: List[Dict[str, Any]] = []
            for task_id in (
                f"fact_check_{article_id}",
                f"compliance_check_{article_id}",
                f"risk_check_{article_id}",
            ):
                result = _safe_get_result(context, task_id)
                if result is not None:
                    checks.append(result)
            channel = context.get_channel()
            channel.set("quality_gate", {"passed": len(checks)})
            print(f"[{article_id}] ✅ Quality gate passed with {len(checks)} checks")
            return {"checks": checks}

        @task(id=f"design_{article_id}", inject_context=True)
        def design_task(context: TaskExecutionContext) -> Dict:
            """Design the article HTML layout."""
            channel = context.get_channel()
            article = channel.get("article")
            if article is None:
                article = _safe_get_result(context, f"critique_{article_id}")
            if article is None:
                article = _safe_get_result(context, f"select_draft_{article_id}")
            if article is None:
                article = _safe_get_result(context, f"curate_{article_id}")
            if article is None:
                raise ValueError("Article data is required for design.")

            print(f"[{article_id}] 🎨 Designing article layout...")
            design_result = designer_agent.run(article)
            print(f"[{article_id}] ✅ Design complete: {design_result.get('path')}")
            return design_result

        # Build workflow graph with dynamic stages and guarded parallel groups
        topic_intake_task >> search_router_task >> curate_task >> writer_personas >> select_draft_task >> write_task >> critique_task >> quality_gate >> quality_gate_summary_task >> design_task

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

    # Create and execute workflow
    wf = create_article_workflow(query, article_id, output_dir)
    result = wf.execute(
        f"topic_intake_{article_id}",
        max_steps=30  # Allow multiple write-critique cycles
    )

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
