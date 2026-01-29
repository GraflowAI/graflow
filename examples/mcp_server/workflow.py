"""Company Intelligence Workflow using Graflow.

This workflow implements:
Search → Curate → Write → Critique → Output

With feedback loop from Critique back to Write.

Usage:
    cd examples/mcp_server
    uv run python workflow.py "トレジャーデータ"
"""

import sys
from collections.abc import Callable
from typing import Any

from agents import CritiqueAgent, CuratorAgent, SearchAgent, WriterAgent
from agents.critique import CritiqueDecision, CritiqueResult
from agents.curator import CurationResult
from agents.search import SearchResponse
from agents.writer import CompanyReport
from config import Config

from graflow.core.context import TaskExecutionContext
from graflow.core.decorators import task
from graflow.core.workflow import workflow
from graflow.llm.client import LLMClient
from graflow.trace.langfuse import LangFuseTracer

# Type alias for progress callback
ProgressCallback = Callable[[int, int, str], None]


def _default_progress_callback(current: int, total: int, message: str) -> None:
    """Default progress callback that prints to stdout."""
    print(f"[{current}/{total}] {message}")


def create_company_intelligence_workflow(
    company_name: str,
    tracer: LangFuseTracer | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Create and execute a company intelligence workflow.

    Args:
        company_name: Name of the company to research
        tracer: Optional Langfuse tracer for observability
        progress_callback: Optional callback for reporting progress (current, total, message)

    Returns:
        Dictionary with report and metadata
    """
    # Use default callback if none provided
    report_progress = progress_callback or _default_progress_callback

    # Workflow has 5 main steps (search, curate, write, critique, output)
    # But write/critique can loop, so we use a dynamic total
    TOTAL_STEPS = 5

    # Initialize shared state
    workflow_state: dict[str, Any] = {
        "company_name": company_name,
        "search_results": None,
        "curated_context": None,
        "curation_result": None,
        "report": None,
        "critique_result": None,
        "iteration": 0,
        "final_report": None,
    }

    # Initialize agents
    llm_client = LLMClient(model=Config.DEFAULT_MODEL, enable_tracing=Config.ENABLE_TRACING)
    search_agent = SearchAgent()
    curator_agent = CuratorAgent(llm_client=llm_client)
    writer_agent = WriterAgent(llm_client=llm_client)
    critique_agent = CritiqueAgent(llm_client=llm_client)

    @task(id="search_task")
    def search_task() -> dict[str, SearchResponse]:
        """Search for company information from multiple sources."""
        report_progress(1, TOTAL_STEPS, f"Searching for information about: {company_name}")

        results = search_agent.search_all(company_name)

        total_results = sum(len(r.results) for r in results.values())
        print(f"   Found {total_results} results across {len(results)} categories")

        workflow_state["search_results"] = results
        return results

    @task(id="curate_task", inject_context=True)
    def curate_task(context: TaskExecutionContext) -> CurationResult:
        """Curate and filter search results."""
        report_progress(2, TOTAL_STEPS, "Curating search results...")

        search_results = workflow_state["search_results"]
        result = curator_agent.curate(company_name, search_results)

        workflow_state["curation_result"] = result
        workflow_state["curated_context"] = curator_agent.to_context_string(result)

        print(f"   Curated {len(result.curated_sources)} sources")
        print(f"   Categories: {result.categories}")

        return result

    @task(id="write_task", inject_context=True)
    def write_task(context: TaskExecutionContext) -> CompanyReport:
        """Write the intelligence report."""
        iteration = workflow_state["iteration"]
        feedback = None
        previous_report = None

        if iteration > 0:
            critique_result = workflow_state.get("critique_result")
            if critique_result:
                feedback = critique_agent.format_feedback(critique_result)
            previous_report = workflow_state.get("report")

        report_progress(3, TOTAL_STEPS, f"Writing report (iteration {iteration + 1})...")

        report = writer_agent.write(
            company_name=company_name,
            curated_context=workflow_state["curated_context"],
            iteration=iteration,
            feedback=feedback,
            previous_report=previous_report,
        )

        workflow_state["report"] = report
        print(f"   Generated {len(report.sections)} sections")

        return report

    @task(id="critique_task", inject_context=True)
    def critique_task(context: TaskExecutionContext) -> CritiqueResult:
        """Critique the report and decide whether to approve or revise."""
        iteration = workflow_state["iteration"]
        report = workflow_state["report"]

        report_progress(4, TOTAL_STEPS, f"Critiquing report (iteration {iteration + 1})...")

        result = critique_agent.critique(
            report=report,
            curated_context=workflow_state["curated_context"],
            iteration=iteration,
        )

        workflow_state["critique_result"] = result

        print(f"   Score: {result.overall_score:.2f}")
        print(f"   Decision: {result.decision.value}")

        if result.decision == CritiqueDecision.REVISE:
            if iteration < Config.MAX_CRITIQUE_ITERATIONS - 1:
                # Loop back to write task
                workflow_state["iteration"] = iteration + 1
                print(f"   → Requesting revision (iteration {iteration + 2})")
                context.next_task(write_task, goto=True)
            else:
                print("   → Max iterations reached, accepting current version")
                workflow_state["final_report"] = report
        else:
            print("   → Report approved!")
            workflow_state["final_report"] = report

        return result

    @task(id="output_task", inject_context=True)
    def output_task(context: TaskExecutionContext) -> dict[str, Any]:
        """Prepare final output."""
        report = workflow_state["final_report"] or workflow_state["report"]
        critique_result = workflow_state["critique_result"]

        report_progress(5, TOTAL_STEPS, "Preparing final output...")

        result = {
            "company_name": company_name,
            "report": writer_agent.to_dict(report),
            "report_markdown": writer_agent.to_markdown(report),
            "critique": critique_agent.to_dict(critique_result) if critique_result else None,
            "iterations": workflow_state["iteration"] + 1,
            "sources_count": len(report.sources),
        }

        print(f"   Report for: {company_name}")
        print(f"   Iterations: {result['iterations']}")
        print(f"   Sources: {result['sources_count']}")

        return result

    # Build and execute workflow
    with workflow("company_intelligence", tracer=tracer) as wf:
        # Define workflow graph
        search_task >> curate_task >> write_task >> critique_task >> output_task

        # Execute workflow
        print(f"\n{'=' * 60}")
        print("Starting Company Intelligence Workflow")
        print(f"Company: {company_name}")
        print(f"{'=' * 60}")

        result = wf.execute("search_task", max_steps=50)

    print(f"\n{'=' * 60}")
    print("Workflow completed!")
    print(f"{'=' * 60}\n")

    # Return the output from output_task
    # wf.execute() returns the last task's return value directly
    return result if isinstance(result, dict) else {}


def run_workflow(
    company_name: str,
    enable_tracing: bool = True,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Run the company intelligence workflow.

    Args:
        company_name: Name of the company to research
        enable_tracing: Whether to enable Langfuse tracing
        progress_callback: Optional callback for reporting progress (current, total, message)

    Returns:
        Dictionary with report and metadata
    """
    # Initialize tracer if enabled
    tracer = None
    if enable_tracing and Config.ENABLE_TRACING:
        try:
            tracer = LangFuseTracer(enable_runtime_graph=True)
            print("📊 Langfuse tracing enabled")
        except Exception as e:
            print(f"⚠️  Failed to initialize tracer: {e}")

    try:
        result = create_company_intelligence_workflow(
            company_name=company_name,
            tracer=tracer,
            progress_callback=progress_callback,
        )
        return result
    finally:
        if tracer:
            tracer.shutdown()


# Entry point for direct execution
if __name__ == "__main__":
    Config.display()

    if not Config.validate():
        print("Configuration validation failed. Exiting.")
        sys.exit(1)

    company = sys.argv[1] if len(sys.argv) > 1 else "トレジャーデータ"
    result = run_workflow(company)

    print("\n" + "=" * 60)
    print("REPORT OUTPUT")
    print("=" * 60)
    print(result.get("report_markdown", "No report generated"))
