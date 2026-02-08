"""
GPT Newspaper Workflow with Human-in-the-Loop (HITL)
=====================================================

This example extends the dynamic newspaper workflow with a Human-in-the-Loop
editorial approval step. After the critique stage, an editor can review and
approve or reject the article via the frontend UI.

Workflow change:
    ... -> critique -> [editorial_approval (HITL)] -> quality_gate -> ...

Approve -> continues to quality gate.
Reject  -> loops back to writer with editor feedback.

Prerequisites:
--------------
- TAVILY_API_KEY environment variable
- OPENAI_API_KEY (or other LLM provider API key)

Expected Output:
----------------
Generated newspaper HTML in outputs/ directory with multiple articles.
"""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

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
from graflow.hitl.handler import FeedbackHandler
from graflow.hitl.types import FeedbackRequest, FeedbackResponse
from graflow.trace.langfuse import LangFuseTracer

logger = logging.getLogger(__name__)

MAX_REVISION_ITERATIONS = 3
DEFAULT_IMAGE_URL = "https://images.unsplash.com/photo-1542281286-9e0a16bb7366"


class WebSocketFeedbackHandler(FeedbackHandler):
    """Handler that broadcasts HITL feedback events over WebSocket via LogStreamManager.

    Sends JSON events to connected WebSocket clients so the frontend can display
    inline approval forms and status updates.
    """

    def __init__(self, log_stream_manager: Any, run_id: str) -> None:
        self._log_stream_manager = log_stream_manager
        self._run_id = run_id

    def on_request_created(self, request: FeedbackRequest) -> None:
        """Broadcast a feedback_request event to WebSocket subscribers."""
        if not self._log_stream_manager or not self._run_id:
            return
        payload = {
            "type": "feedback_request",
            "feedbackId": request.feedback_id,
            "taskId": request.task_id,
            "feedbackType": request.feedback_type.value,
            "prompt": request.prompt,
            "options": request.options,
            "metadata": request.metadata,
            "timeout": request.timeout,
            "runId": self._run_id,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        }
        self._log_stream_manager._publish(self._run_id, payload)

    def on_response_received(self, request: FeedbackRequest, response: FeedbackResponse) -> None:
        """Broadcast a feedback_resolved event."""
        if not self._log_stream_manager or not self._run_id:
            return
        payload = {
            "type": "feedback_resolved",
            "feedbackId": request.feedback_id,
            "taskId": request.task_id,
            "runId": self._run_id,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        }
        self._log_stream_manager._publish(self._run_id, payload)

    def on_request_timeout(self, request: FeedbackRequest) -> None:
        """Broadcast a feedback_timeout event."""
        if not self._log_stream_manager or not self._run_id:
            return
        payload = {
            "type": "feedback_timeout",
            "feedbackId": request.feedback_id,
            "taskId": request.task_id,
            "runId": self._run_id,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        }
        self._log_stream_manager._publish(self._run_id, payload)


def create_article_workflow(
    query: str,
    article_id: str,
    output_dir: str,
    tracer: Any = None,
    log_stream_manager: Any = None,
    run_id: Optional[str] = None,
):
    """
    Create a workflow for processing a single article query with HITL editorial approval.

    This workflow extends the dynamic workflow with a Human-in-the-Loop approval step
    between critique and quality gate:
    - Search -> Curate -> Write -> Critique -> [Editorial Approval] -> Quality Gate -> Design

    Args:
        query: Article topic/query
        article_id: Unique identifier for this article
        output_dir: Output directory for article HTML
        tracer: Optional tracer for workflow execution tracking
        log_stream_manager: Optional LogStreamManager for WebSocket broadcasting
        run_id: Optional run ID for log streaming

    Returns:
        Workflow context
    """

    # Initialize non-LLM agents
    search_agent = SearchAgent()
    designer_agent = DesignerAgent(output_dir)

    with workflow(f"article_{article_id}", tracer=tracer) as wf:

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
            print(f"\n[{article_id}] Profiling topic: {query}")
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

            print(f"[{article_id}] Planned angles: {', '.join(angles)} (priority={priority})")
            return {"query": query, "angles": angles, "priority": priority}

        @task(id=f"search_router_{article_id}", inject_context=True)
        def search_router_task(context: TaskExecutionContext) -> Dict:
            """Fan out search tasks per angle using runtime task creation."""
            channel = context.get_channel()
            angles: List[str] = channel.get("angles", default=["overview"])
            channel.set("expected_search_tasks", len(angles))
            channel.set("completed_search_tasks", 0)
            channel.set("search_results", [])

            print(f"[{article_id}] Launching {len(angles)} targeted searches...")

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
                        f"[{article_id}]   [{angle_slug}] completed with {len(sources)} sources (total {completed}/{len(angles)})"
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
                print(f"[{article_id}] Waiting for {remaining} search tasks to finish before curating...")
                time.sleep(0.05)
                context.next_iteration()
                return {"status": "waiting"}

            aggregated_sources = []
            for batch in channel.get("search_results", default=[]):
                aggregated_sources.extend(batch.get("sources", []))

            image = channel.get("image") or DEFAULT_IMAGE_URL
            article_data = {"query": query, "sources": aggregated_sources, "image": image}
            print(f"[{article_id}] Curating {len(aggregated_sources)} raw sources...")

            # Initialize curator agent with LLM client from context
            curator_agent = CuratorAgent(context.llm_client)
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
                    print(
                        f"[{article_id}]   supplemental search completed with {len(supplemental_result.get('sources', []))} sources"
                    )
                    return supplemental_result

                context.next_task(TaskWrapper(f"search_{article_id}_supplemental", supplemental_research))
                print(f"[{article_id}] Not enough sources, scheduling supplemental research...")
                time.sleep(0.05)
                context.next_iteration()
                return {"status": "gap_filling"}

            channel.set("gap_fill_requested", False)
            print(f"[{article_id}] Selected {len(result.get('sources', []))} curated sources")
            return result

        def _generate_draft(context: TaskExecutionContext, style: str, tone: str) -> Dict:
            channel = context.get_channel()
            article = channel.get("article")
            if article is None:
                article = context.get_result(f"curate_{article_id}") or {"query": query, "image": DEFAULT_IMAGE_URL}

            image = article.get("image") or channel.get("image") or DEFAULT_IMAGE_URL
            article["image"] = image

            payload = {**article, "style": style, "tone": tone}
            print(f"[{article_id}] Writing {style} draft...")

            # Initialize writer agent with LLM client from context
            writer_agent = WriterAgent(context.llm_client)
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

        writer_personas = (write_feature_task | write_brief_task | write_data_digest_task).with_execution(
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
            print(f"[{article_id}] Selected {chosen['label']} draft for refinement")
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
                print(f"\n[{article_id}] Revising article (iteration {iteration})...")
            else:
                print(f"\n[{article_id}] Refining primary draft...")

            # Initialize writer agent with LLM client from context
            writer_agent = WriterAgent(context.llm_client)
            result = writer_agent.run(article)
            print(f"[{article_id}] Article: {result.get('title', 'Untitled')}")

            channel.set("article", result)
            channel.set("iteration", iteration)

            return result

        @task(id=f"critique_{article_id}", inject_context=True)
        def critique_task(context: TaskExecutionContext) -> Dict:
            """Critique the article and decide whether to loop back to writing."""
            print(f"[{article_id}] Critiquing article...")

            channel = context.get_channel()
            article = _get_article_from_channel(context)
            image = article.get("image") or context.get_channel().get("image") or DEFAULT_IMAGE_URL
            article["image"] = image
            iteration = channel.get("iteration", default=0)

            # Initialize critique agent with LLM client from context
            critique_agent = CritiqueAgent(context.llm_client)
            result = critique_agent.run(article)

            channel.set("article", result)

            if result.get("critique") is not None:
                print(f"[{article_id}] Critique feedback received, looping back to writer...")

                if iteration >= MAX_REVISION_ITERATIONS:
                    print(
                        f"[{article_id}] Max iterations ({MAX_REVISION_ITERATIONS}) reached, moving forward regardless"
                    )
                    result["critique"] = None
                else:
                    channel.set("iteration", iteration + 1)
                    channel.set("article", result)
                    context.next_task(write_task, goto=True)
                    return result

            print(f"[{article_id}] Article approved by critique!")
            return result

        @task(id=f"editorial_approval_{article_id}", inject_context=True)
        def editorial_approval_task(context: TaskExecutionContext) -> Dict:
            """Request editorial approval via HITL before proceeding to quality gate."""
            channel = context.get_channel()
            article = _get_article_from_channel(context)

            # Try multiple sources for paragraphs: channel article, then task results
            paragraphs = article.get("paragraphs") or []
            if not paragraphs:
                for tid in [f"critique_{article_id}", f"write_{article_id}", f"select_draft_{article_id}"]:
                    fallback = _safe_get_result(context, tid)
                    if fallback and fallback.get("paragraphs"):
                        paragraphs = fallback["paragraphs"]
                        break

            title = article.get("title", "Untitled")
            summary = article.get("summary") or ""
            body = "\n\n".join(paragraphs) if isinstance(paragraphs, list) and paragraphs else ""
            body_preview = body[:500] if body else summary[:500]

            prompt = f"Editorial Review: {title}\n\n{body_preview}...\n\nApprove for publishing?"

            print(
                f"[{article_id}] [HITL] Requesting editorial approval for: {title} | DEBUG keys={list(article.keys())} paragraphs={len(paragraphs)} body={len(body)} summary={len(summary)}"
            )

            # Include full article content in metadata so the frontend can render a review panel
            sources = article.get("sources", [])
            metadata: Dict[str, Any] = {
                "article_id": article_id,
                "stage": "editorial_approval",
                "title": title,
                "body": body,
                "paragraphs": paragraphs,
                "summary": summary,
                "sources": sources[:20] if isinstance(sources, list) else sources,
                "image": article.get("image"),
                "query": article.get("query", query),
            }

            handler = WebSocketFeedbackHandler(log_stream_manager, run_id or "")
            response = context.request_feedback(
                feedback_type="approval",
                prompt=prompt,
                timeout=300.0,
                metadata=metadata,
                handler=handler,
            )

            if response.approved:
                print(f"[{article_id}] [HITL] Article approved by editor!")
                return article
            else:
                # Rejected: loop back to writer with editor feedback
                reason = response.reason or "Editor requested revisions"
                print(f"[{article_id}] [HITL] Article rejected: {reason}")
                article["critique"] = reason
                channel.set("article", article)
                channel.set("iteration", channel.get("iteration", default=0) + 1)
                context.next_task(write_task, goto=True)
                return {"status": "revision_requested", "reason": reason}

        @task(id=f"fact_check_{article_id}", inject_context=True)
        def fact_check_task(context: TaskExecutionContext) -> Dict:
            article = _get_article_from_channel(context)
            if _should_flag("fact"):
                raise RuntimeError("Fact discrepancy detected in references")
            print(f"[{article_id}] Fact check cleared")
            return {"status": "ok", "check": "fact", "title": article.get("title")}

        @task(id=f"compliance_check_{article_id}", inject_context=True)
        def compliance_check_task(context: TaskExecutionContext) -> Dict:
            article = _get_article_from_channel(context)
            if _should_flag("compliance"):
                raise RuntimeError("Compliance guidelines not met")
            print(f"[{article_id}] Compliance check cleared")
            return {"status": "ok", "check": "compliance", "title": article.get("title")}

        @task(id=f"risk_check_{article_id}", inject_context=True)
        def risk_check_task(context: TaskExecutionContext) -> Dict:
            article = _get_article_from_channel(context)
            if _should_flag("risk") and context.get_channel().get("topic_profile", {}).get("priority") == "breaking":
                raise RuntimeError("Risk review flagged sensitive content")
            print(f"[{article_id}] Risk check cleared")
            return {"status": "ok", "check": "risk", "title": article.get("title")}

        quality_gate = (fact_check_task | compliance_check_task | risk_check_task).with_execution(
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
            print(f"[{article_id}] Quality gate passed with {len(checks)} checks")
            return {"checks": checks}

        @task(id=f"design_{article_id}", inject_context=True)
        def design_task(context: TaskExecutionContext) -> Dict:
            """Design the article HTML layout."""
            channel = context.get_channel()
            article = channel.get("article")
            if article is None:
                article = _safe_get_result(context, f"editorial_approval_{article_id}")
            if article is None:
                article = _safe_get_result(context, f"critique_{article_id}")
            if article is None:
                article = _safe_get_result(context, f"select_draft_{article_id}")
            if article is None:
                article = _safe_get_result(context, f"curate_{article_id}")
            if article is None:
                raise ValueError("Article data is required for design.")

            print(f"[{article_id}] Designing article layout...")
            design_result = designer_agent.run(article)
            print(f"[{article_id}] Design complete: {design_result.get('path')}")
            return design_result

        # Build workflow graph with HITL editorial approval between critique and quality gate
        (
            topic_intake_task
            >> search_router_task
            >> curate_task
            >> writer_personas
            >> select_draft_task
            >> write_task
            >> critique_task
            >> editorial_approval_task
            >> quality_gate
            >> quality_gate_summary_task
            >> design_task
        )  # type: ignore

        return wf


def execute_article_workflow(
    query: str,
    article_id: str,
    output_dir: str,
    log_stream_manager: Any = None,
    run_id: Optional[str] = None,
) -> Dict:
    """
    Execute a single article workflow with HITL editorial approval.

    Args:
        query: Article query/topic
        article_id: Unique article identifier
        output_dir: Output directory for article HTML
        log_stream_manager: Optional LogStreamManager for WebSocket broadcasting
        run_id: Optional run ID for log streaming

    Returns:
        Completed article dict with design, or None if not found
    """
    print(f"\n{'=' * 80}")
    print(f"Processing Article (HITL): {query}")
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
    wf = create_article_workflow(
        query,
        article_id,
        output_dir,
        tracer=tracer,
        log_stream_manager=log_stream_manager,
        run_id=run_id,
    )
    result = wf.execute(
        f"topic_intake_{article_id}",
        max_steps=30,  # Allow multiple write-critique-approval cycles
    )

    # Shutdown tracer to flush remaining traces
    if tracer:
        tracer.shutdown()

    if isinstance(result, dict):
        return result

    # Fallback: return None if design not found
    raise ValueError(f"Design task not found for {article_id}")


def run_newspaper_workflow(
    queries: List[str],
    layout: str = "layout_1.html",
    max_workers: int | None = None,
    output_dir: str | None = None,
    log_stream_manager: Any = None,
    run_id: str | None = None,
):
    """
    Run the complete newspaper workflow with HITL editorial approval.

    This extends the dynamic workflow by adding an editorial approval step
    after critique. Editors can approve or reject articles via the frontend UI.

    Args:
        queries: List of article queries/topics
        layout: Newspaper layout template
        max_workers: Max number of parallel workers (None = number of processors)
        output_dir: Optional output directory override
        log_stream_manager: Optional LogStreamManager for WebSocket feedback broadcasting
        run_id: Optional run ID for log streaming
    """
    print("=" * 80)
    print("GPT NEWSPAPER WORKFLOW (HITL)")
    print("=" * 80)

    # Create output directory
    output_dir = output_dir or f"outputs/run_{int(time.time())}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}\n")

    # Execute article workflows in parallel using ThreadPoolExecutor
    print(f"Processing {len(queries)} articles in parallel...\n")

    # Execute the graph for each query in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Create article IDs
        article_ids = [f"article_{i}" for i in range(len(queries))]

        # Execute workflows in parallel
        completed_articles = list(
            executor.map(
                lambda args: execute_article_workflow(*args),
                zip(
                    queries,
                    article_ids,
                    [output_dir] * len(queries),
                    [log_stream_manager] * len(queries),
                    [run_id] * len(queries),
                    strict=False,
                ),
            )
        )

    # Filter out None results
    completed_articles = [a for a in completed_articles if a is not None]

    # Compile newspaper
    print(f"\n{'=' * 80}")
    print("Compiling Newspaper")
    print(f"{'=' * 80}\n")

    editor_agent = EditorAgent(layout)
    newspaper_html = editor_agent.run(completed_articles)

    # Publish newspaper
    publisher_agent = PublisherAgent(output_dir)
    newspaper_path = publisher_agent.run(newspaper_html)

    print(f"\n{'=' * 80}")
    print("NEWSPAPER GENERATION COMPLETE")
    print(f"{'=' * 80}")
    print(f"\nNewspaper: {newspaper_path}")
    print(f"Articles: {len(completed_articles)}")
    print(f"\nOpen {newspaper_path} in your browser to view!\n")

    return newspaper_path


def main():
    """Run the HITL newspaper workflow with example queries."""

    # Check for required environment variables
    if not os.getenv("TAVILY_API_KEY"):
        print("Error: TAVILY_API_KEY environment variable is required")
        print("Get your API key at: https://tavily.com/")
        return

    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found. Make sure your LLM provider is configured.")
        return

    # Example queries
    queries = [
        "Latest developments in artificial intelligence",
        "Climate change policy updates",
        "Technology industry trends",
    ]

    # Run workflow (no WebSocket in CLI mode)
    run_newspaper_workflow(queries=queries, layout="layout_1.html")


if __name__ == "__main__":
    main()
