#!/usr/bin/env python3
"""
Graph Visualization Example

This example demonstrates how to use the various graph drawing utilities
in graflow/utils/graph.py to visualize NetworkX DiGraph objects.

The example shows:
- Creating a sample workflow graph
- Drawing ASCII representations
- Generating Mermaid diagrams
- Creating PNG visualizations
"""

import networkx as nx

from graflow.utils.graph import (
    draw_ascii,
    draw_mermaid,
    draw_mermaid_png,
    draw_png,
    show_graph_info,
    visualize_dependencies,
)


def create_sample_workflow_graph() -> nx.DiGraph:
    """Create a sample workflow graph for demonstration."""
    graph = nx.DiGraph()

    # Add nodes representing workflow tasks
    tasks = [
        "start",
        "load_data",
        "validate_data",
        "transform_data",
        "analyze_data",
        "generate_report",
        "send_email",
        "cleanup",
        "end"
    ]

    for task in tasks:
        graph.add_node(task)

    # Add edges representing task dependencies
    edges = [
        ("start", "load_data"),
        ("load_data", "validate_data"),
        ("validate_data", "transform_data"),
        ("transform_data", "analyze_data"),
        ("analyze_data", "generate_report"),
        ("generate_report", "send_email"),
        ("send_email", "cleanup"),
        ("cleanup", "end"),
        # Add some parallel paths
        ("validate_data", "analyze_data"),  # Skip transform for quick analysis
        ("generate_report", "cleanup"),     # Direct cleanup path
    ]

    graph.add_edges_from(edges)

    return graph


def create_complex_graph() -> nx.DiGraph:
    """Create a more complex graph with branching and parallel execution."""
    graph = nx.DiGraph()

    # Data processing pipeline
    nodes = [
        "input", "preprocess", "feature_extraction", "model_training",
        "validation", "hyperparameter_tuning", "final_model",
        "batch_prediction", "streaming_prediction", "model_deployment",
        "monitoring", "alerting", "output"
    ]

    graph.add_nodes_from(nodes)

    edges = [
        ("input", "preprocess"),
        ("preprocess", "feature_extraction"),
        ("feature_extraction", "model_training"),
        ("model_training", "validation"),
        ("validation", "hyperparameter_tuning"),
        ("hyperparameter_tuning", "final_model"),
        ("final_model", "batch_prediction"),
        ("final_model", "streaming_prediction"),
        ("batch_prediction", "model_deployment"),
        ("streaming_prediction", "model_deployment"),
        ("model_deployment", "monitoring"),
        ("monitoring", "alerting"),
        ("alerting", "output"),
        # Feedback loops
        ("validation", "model_training"),
        ("monitoring", "hyperparameter_tuning"),
    ]

    graph.add_edges_from(edges)

    return graph


def demonstrate_ascii_drawing():
    """Demonstrate ASCII graph drawing."""
    print("=" * 60)
    print("ASCII GRAPH VISUALIZATION")
    print("=" * 60)

    # Simple graph
    simple_graph = create_sample_workflow_graph()
    print("\n1. Simple Workflow Graph (ASCII):")
    print("-" * 40)
    try:
        ascii_output = draw_ascii(simple_graph)
        print(ascii_output)
    except Exception as e:
        print(f"Note: ASCII drawing requires 'grandalf' package: {e}")
        print("Install with: pip install grandalf")

    # Graph with no edges
    isolated_graph = nx.DiGraph()
    isolated_graph.add_nodes_from(["task1", "task2", "task3"])
    print("\n2. Graph with Isolated Nodes (ASCII):")
    print("-" * 40)
    isolated_ascii = draw_ascii(isolated_graph)
    print(isolated_ascii)

    # Empty graph
    empty_graph = nx.DiGraph()
    print("\n3. Empty Graph (ASCII):")
    print("-" * 40)
    empty_ascii = draw_ascii(empty_graph)
    print(empty_ascii)


def demonstrate_mermaid_drawing():
    """Demonstrate Mermaid diagram generation."""
    print("\n" + "=" * 60)
    print("MERMAID DIAGRAM GENERATION")
    print("=" * 60)

    # Simple workflow
    workflow_graph = create_sample_workflow_graph()

    print("\n1. Basic Mermaid Diagram:")
    print("-" * 40)
    mermaid_basic = draw_mermaid(workflow_graph, title="Workflow Pipeline")
    print(mermaid_basic)

    print("\n2. Mermaid Diagram without Styles:")
    print("-" * 40)
    mermaid_no_styles = draw_mermaid(
        workflow_graph,
        title="Workflow Pipeline (No Styles)",
        with_styles=False
    )
    print(mermaid_no_styles)

    print("\n3. Mermaid with Custom Colors:")
    print("-" * 40)
    node_colors = {
        "start": "#90EE90",      # Light green
        "load_data": "#87CEEB",  # Sky blue
        "end": "#FFB6C1"         # Light pink
    }
    mermaid_colored = draw_mermaid(
        workflow_graph,
        title="Colored Workflow",
        node_colors=node_colors
    )
    print(mermaid_colored)


def demonstrate_png_drawing():
    """Demonstrate PNG generation."""
    print("\n" + "=" * 60)
    print("PNG VISUALIZATION")
    print("=" * 60)

    workflow_graph = create_sample_workflow_graph()

    print("\n1. Basic PNG Generation:")
    print("-" * 40)
    try:
        # Generate PNG bytes (in memory)
        png_bytes = draw_png(workflow_graph)
        if png_bytes:
            print(f"Generated PNG: {len(png_bytes)} bytes")

            # Save to file
            output_path = "/tmp/workflow_graph.png"
            draw_png(workflow_graph, output_path=output_path)
            print(f"Saved PNG to: {output_path}")

    except Exception as e:
        print(f"Note: PNG drawing requires 'pygraphviz' package: {e}")
        print("Install with: pip install pygraphviz")

    print("\n2. PNG with Custom Labels and Colors:")
    print("-" * 40)
    try:
        custom_labels = {
            "start": "Begin Process",
            "load_data": "Data Loading",
            "end": "Process Complete"
        }
        custom_colors = {
            "start": "lightgreen",
            "load_data": "lightblue",
            "end": "lightcoral"
        }

        custom_png_bytes = draw_png(
            workflow_graph,
            node_labels=custom_labels,
            node_colors=custom_colors
        )

        if custom_png_bytes:
            print(f"Generated custom PNG: {len(custom_png_bytes)} bytes")

            # Save custom version
            custom_output_path = "/tmp/workflow_graph_custom.png"
            draw_png(
                workflow_graph,
                output_path=custom_output_path,
                node_labels=custom_labels,
                node_colors=custom_colors
            )
            print(f"Saved custom PNG to: {custom_output_path}")

    except Exception as e:
        print(f"PNG generation failed: {e}")


def demonstrate_mermaid_png():
    """Demonstrate Mermaid PNG generation."""
    print("\n" + "=" * 60)
    print("MERMAID PNG GENERATION")
    print("=" * 60)

    workflow_graph = create_sample_workflow_graph()

    print("\n1. Mermaid PNG via API:")
    print("-" * 40)
    try:
        # Generate using mermaid.ink API
        mermaid_png_bytes = draw_mermaid_png(
            workflow_graph,
            title="API Generated Graph",
            draw_method="api",
            background_color="white"
        )
        print(f"Generated Mermaid PNG via API: {len(mermaid_png_bytes)} bytes")

        # Save to file
        api_output_path = "/tmp/workflow_mermaid_api.png"
        draw_mermaid_png(
            workflow_graph,
            output_path=api_output_path,
            title="Workflow via Mermaid API",
            draw_method="api"
        )
        print(f"Saved Mermaid PNG to: {api_output_path}")

    except Exception as e:
        print(f"Mermaid API PNG generation failed: {e}")
        print("This might be due to network issues or missing 'requests' package")

    print("\n2. Mermaid PNG via Pyppeteer (Local):")
    print("-" * 40)
    try:
        # Generate using local browser (requires pyppeteer)
        local_png_bytes = draw_mermaid_png(
            workflow_graph,
            title="Local Generated Graph",
            draw_method="pyppeteer",
            background_color="lightgray",
            padding=20
        )
        print(f"Generated Mermaid PNG locally: {len(local_png_bytes)} bytes")

        # Save local version
        local_output_path = "/tmp/workflow_mermaid_local.png"
        draw_mermaid_png(
            workflow_graph,
            output_path=local_output_path,
            title="Workflow via Pyppeteer",
            draw_method="pyppeteer"
        )
        print(f"Saved local Mermaid PNG to: {local_output_path}")

    except Exception as e:
        print(f"Pyppeteer PNG generation failed: {e}")
        print("Install with: pip install pyppeteer")


def demonstrate_graph_analysis():
    """Demonstrate graph analysis utilities."""
    print("\n" + "=" * 60)
    print("GRAPH ANALYSIS")
    print("=" * 60)

    complex_graph = create_complex_graph()

    print("\n1. Graph Information:")
    print("-" * 40)
    show_graph_info(complex_graph)

    print("\n2. Dependencies Visualization:")
    print("-" * 40)
    visualize_dependencies(complex_graph)


def main():
    """Run all graph visualization demonstrations."""
    print("GRAFLOW GRAPH VISUALIZATION EXAMPLES")
    print("=" * 60)
    print("This example demonstrates various graph visualization methods")
    print("available in graflow.utils.graph module.")

    # Run demonstrations
    demonstrate_ascii_drawing()
    demonstrate_mermaid_drawing()
    demonstrate_png_drawing()
    demonstrate_mermaid_png()
    demonstrate_graph_analysis()

    print("\n" + "=" * 60)
    print("EXAMPLE COMPLETE")
    print("=" * 60)
    print("\nGenerated files (if successful):")
    print("- /tmp/workflow_graph.png (basic PNG)")
    print("- /tmp/workflow_graph_custom.png (custom styled PNG)")
    print("- /tmp/workflow_mermaid_api.png (Mermaid API PNG)")
    print("- /tmp/workflow_mermaid_local.png (Pyppeteer PNG)")
    print("\nNote: Some features require additional packages:")
    print("- ASCII drawing: pip install grandalf")
    print("- PNG drawing: pip install pygraphviz")
    print("- Mermaid API: pip install requests")
    print("- Mermaid local: pip install pyppeteer")


if __name__ == "__main__":
    main()