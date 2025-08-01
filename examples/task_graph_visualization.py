#!/usr/bin/env python3
"""
Task Graph Visualization Example

This example shows how to visualize actual graflow TaskGraph objects
using the visualization utilities in graflow.utils.graph.

Demonstrates:
- Creating a workflow with tasks
- Extracting NetworkX graph from TaskGraph
- Visualizing the task dependencies
"""

from graflow.core.context import execute_in_workflow
from graflow.core.task import task
from graflow.core.workflow import get_current_workflow_context
from graflow.utils.graph import (
    draw_ascii,
    draw_mermaid,
    draw_png,
    show_graph_info,
    visualize_dependencies,
)


@task
def load_data(input_path: str) -> dict:
    """Load data from input source."""
    print(f"Loading data from {input_path}")
    return {"data": "sample_data", "rows": 1000}


@task
def validate_data(data: dict) -> dict:
    """Validate the loaded data."""
    print(f"Validating data: {data['rows']} rows")
    return {**data, "validated": True}


@task
def transform_data(data: dict) -> dict:
    """Transform the validated data."""
    print(f"Transforming validated data")
    return {**data, "transformed": True}


@task
def analyze_data(data: dict) -> dict:
    """Analyze the transformed data."""
    print(f"Analyzing data")
    return {**data, "analysis": "completed"}


@task
def generate_report(analysis: dict) -> dict:
    """Generate report from analysis."""
    print("Generating report")
    return {"report": "analysis_report.pdf", "analysis": analysis}


@task
def send_notification(report: dict) -> str:
    """Send notification about completed report."""
    print(f"Sending notification for {report['report']}")
    return "notification_sent"


def create_workflow_graph():
    """Create a workflow and extract its task graph."""
    print("Creating workflow with task dependencies...")

    def workflow():
        # Create task chain
        data = load_data("input.csv")
        validated = validate_data(data)
        transformed = transform_data(validated)
        analysis = analyze_data(transformed)
        report = generate_report(analysis)
        notification = send_notification(report)

        return notification

    # Execute in workflow context to build graph
    with execute_in_workflow():
        result = workflow()

        # Get the workflow context and extract NetworkX graph
        context = get_current_workflow_context()
        nx_graph = context.graph.nx_graph()

        return nx_graph, result


def create_parallel_workflow_graph():
    """Create a workflow with parallel execution paths."""
    print("Creating parallel workflow...")

    @task
    def split_data(data: dict) -> tuple[dict, dict]:
        """Split data for parallel processing."""
        print("Splitting data for parallel processing")
        return (
            {**data, "subset": "A"},
            {**data, "subset": "B"}
        )

    @task
    def process_subset_a(data: dict) -> dict:
        """Process subset A."""
        print("Processing subset A")
        return {**data, "processed_a": True}

    @task
    def process_subset_b(data: dict) -> dict:
        """Process subset B."""
        print("Processing subset B")
        return {**data, "processed_b": True}

    @task
    def merge_results(result_a: dict, result_b: dict) -> dict:
        """Merge parallel processing results."""
        print("Merging parallel results")
        return {
            "merged": True,
            "result_a": result_a,
            "result_b": result_b
        }

    def parallel_workflow():
        # Initial processing
        data = load_data("parallel_input.csv")
        validated = validate_data(data)

        # Split for parallel processing
        subset_a, subset_b = split_data(validated)

        # Parallel processing
        result_a = process_subset_a(subset_a)
        result_b = process_subset_b(subset_b)

        # Merge results
        merged = merge_results(result_a, result_b)

        # Final steps
        report = generate_report(merged)
        notification = send_notification(report)

        return notification

    # Execute in workflow context
    with execute_in_workflow():
        result = parallel_workflow()

        # Extract NetworkX graph
        context = get_current_workflow_context()
        nx_graph = context.graph.nx_graph()

        return nx_graph, result


def visualize_task_graph(nx_graph, title: str):
    """Visualize a task graph using all available methods."""
    print(f"\n{'='*60}")
    print(f"VISUALIZING: {title}")
    print(f"{'='*60}")

    # Graph analysis
    print(f"\n1. Graph Analysis:")
    print("-" * 40)
    show_graph_info(nx_graph)

    print(f"\n2. Task Dependencies:")
    print("-" * 40)
    visualize_dependencies(nx_graph)

    # ASCII visualization
    print(f"\n3. ASCII Visualization:")
    print("-" * 40)
    try:
        ascii_repr = draw_ascii(nx_graph)
        print(ascii_repr)
    except Exception as e:
        print(f"ASCII drawing failed: {e}")
        print("Install grandalf: pip install grandalf")

    # Mermaid diagram
    print(f"\n4. Mermaid Diagram:")
    print("-" * 40)
    mermaid_diagram = draw_mermaid(nx_graph, title=title)
    print(mermaid_diagram)

    # Save PNG if possible
    print(f"\n5. PNG Generation:")
    print("-" * 40)
    try:
        # Create custom labels for better readability
        node_labels = {}
        node_colors = {}

        for node in nx_graph.nodes():
            # Create readable labels
            label = node.replace("_", " ").title()
            node_labels[node] = label

            # Color code by task type
            if "load" in node:
                node_colors[node] = "lightblue"
            elif "validate" in node or "transform" in node:
                node_colors[node] = "lightyellow"
            elif "analyze" in node or "process" in node:
                node_colors[node] = "lightgreen"
            elif "generate" in node or "merge" in node:
                node_colors[node] = "lightcoral"
            elif "send" in node or "split" in node:
                node_colors[node] = "lightgray"
            else:
                node_colors[node] = "white"

        # Generate PNG
        png_bytes = draw_png(
            nx_graph,
            node_labels=node_labels,
            node_colors=node_colors
        )

        if png_bytes:
            # Save to file
            filename = f"/tmp/{title.lower().replace(' ', '_')}.png"
            draw_png(
                nx_graph,
                output_path=filename,
                node_labels=node_labels,
                node_colors=node_colors
            )
            print(f"Saved PNG: {filename} ({len(png_bytes)} bytes)")

    except Exception as e:
        print(f"PNG generation failed: {e}")
        print("Install pygraphviz: pip install pygraphviz")


def main():
    """Run the task graph visualization examples."""
    print("GRAFLOW TASK GRAPH VISUALIZATION")
    print("=" * 60)
    print("This example demonstrates how to visualize actual graflow")
    print("TaskGraph objects extracted from workflow execution.")

    try:
        # Create and visualize simple workflow
        print("\n>>> Creating Simple Sequential Workflow...")
        simple_graph, simple_result = create_workflow_graph()
        visualize_task_graph(simple_graph, "Simple Sequential Workflow")

        # Create and visualize parallel workflow
        print("\n>>> Creating Parallel Workflow...")
        parallel_graph, parallel_result = create_parallel_workflow_graph()
        visualize_task_graph(parallel_graph, "Parallel Processing Workflow")

        print(f"\n{'='*60}")
        print("VISUALIZATION COMPLETE")
        print(f"{'='*60}")
        print("\nWorkflow Results:")
        print(f"- Simple workflow: {simple_result}")
        print(f"- Parallel workflow: {parallel_result}")

        print("\nGenerated files:")
        print("- /tmp/simple_sequential_workflow.png")
        print("- /tmp/parallel_processing_workflow.png")

    except Exception as e:
        print(f"Error running workflow: {e}")
        print("Make sure you have the required dependencies installed.")


if __name__ == "__main__":
    main()