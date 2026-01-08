#!/usr/bin/env python3
"""
Graph Visualization Utilities Example
=====================================

This example demonstrates the various graph visualization utilities available
in graflow.utils.graph for working with NetworkX DiGraph objects directly.

Use this when you want to:
- Visualize custom graphs
- Create documentation diagrams
- Debug graph algorithms
- Generate reports with graph visualizations

Concepts Covered:
-----------------
1. Creating custom NetworkX graphs
2. ASCII drawing (terminal-friendly)
3. Mermaid diagram generation (markdown-friendly)
4. PNG generation (presentation-ready)
5. Graph analysis utilities

Dependencies:
-------------
Optional (for full functionality):
- pip install grandalf      # ASCII visualization
- pip install pygraphviz    # PNG generation
- pip install requests      # Mermaid PNG via API

The example works without these packages but with reduced functionality.
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


def example_1_basic_visualization():
    """Example 1: Basic graph visualization methods."""
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Basic Graph Visualization")
    print("=" * 70)

    # Create a simple graph
    graph = nx.DiGraph()
    graph.add_edges_from([("start", "task_a"), ("task_a", "task_b"), ("task_b", "task_c"), ("task_c", "end")])

    print("\n📊 Graph Structure:")
    print("-" * 70)
    show_graph_info(graph)

    print("\n📝 Dependencies:")
    print("-" * 70)
    visualize_dependencies(graph)

    print("\n🎨 ASCII Visualization:")
    print("-" * 70)
    try:
        ascii_repr = draw_ascii(graph)
        print(ascii_repr)
    except Exception as e:
        print(f"⚠️  ASCII unavailable: {e}")
        print("   Install: pip install grandalf")

    print("\n🌊 Mermaid Diagram:")
    print("-" * 70)
    mermaid = draw_mermaid(graph, title="Simple Graph")
    print(mermaid)


def example_2_branching_graph():
    """Example 2: Graph with branching and merging."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Branching and Merging Graph")
    print("=" * 70)

    graph = nx.DiGraph()

    # Create branching structure
    edges = [
        ("input", "validate"),
        ("validate", "branch_a"),
        ("validate", "branch_b"),
        ("validate", "branch_c"),
        ("branch_a", "merge"),
        ("branch_b", "merge"),
        ("branch_c", "merge"),
        ("merge", "output"),
    ]
    graph.add_edges_from(edges)

    print("\n📊 Graph Analysis:")
    print("-" * 70)
    show_graph_info(graph)

    print("\n🌊 Mermaid with Custom Colors:")
    print("-" * 70)
    node_colors = {
        "input": "#90EE90",  # Light green
        "validate": "#FFE4B5",  # Light orange
        "branch_a": "#87CEEB",  # Sky blue
        "branch_b": "#87CEEB",
        "branch_c": "#87CEEB",
        "merge": "#FFB6C1",  # Light pink
        "output": "#98FB98",  # Pale green
    }

    mermaid = draw_mermaid(graph, title="Branching Workflow", node_colors=node_colors)
    print(mermaid)

    print("\n📸 PNG with Colored Nodes:")
    print("-" * 70)
    try:
        png_colors = {
            "input": "lightgreen",
            "validate": "lightyellow",
            "branch_a": "lightblue",
            "branch_b": "lightblue",
            "branch_c": "lightblue",
            "merge": "lightcoral",
            "output": "palegreen",
        }

        png_bytes = draw_png(graph, node_colors=png_colors, output_path="/tmp/branching_graph.png")
        if png_bytes:
            print(f"✅ Saved: /tmp/branching_graph.png ({len(png_bytes)} bytes)")
    except Exception as e:
        print(f"⚠️  PNG unavailable: {e}")


def example_3_dag_visualization():
    """Example 3: Complex DAG (Directed Acyclic Graph) visualization."""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Complex DAG Visualization")
    print("=" * 70)

    # Create a data pipeline DAG
    graph = nx.DiGraph()

    edges = [
        # Data ingestion layer
        ("source_a", "extract_a"),
        ("source_b", "extract_b"),
        ("source_c", "extract_c"),
        # Validation layer
        ("extract_a", "validate_a"),
        ("extract_b", "validate_b"),
        ("extract_c", "validate_c"),
        # Transformation layer
        ("validate_a", "transform"),
        ("validate_b", "transform"),
        ("validate_c", "transform"),
        # Enrichment
        ("transform", "enrich_demographics"),
        ("transform", "enrich_behavior"),
        # Aggregation
        ("enrich_demographics", "aggregate"),
        ("enrich_behavior", "aggregate"),
        # Output
        ("aggregate", "load_warehouse"),
        ("aggregate", "create_reports"),
        ("load_warehouse", "notify"),
        ("create_reports", "notify"),
    ]
    graph.add_edges_from(edges)

    print("\n📊 Graph Statistics:")
    print("-" * 70)
    show_graph_info(graph)

    print("\n📝 Dependency Tree:")
    print("-" * 70)
    visualize_dependencies(graph)

    print("\n🌊 Mermaid Diagram (Layered by Stage):")
    print("-" * 70)

    # Color by layer
    layer_colors = {}
    sources = ["source_a", "source_b", "source_c"]
    extracts = ["extract_a", "extract_b", "extract_c"]
    validates = ["validate_a", "validate_b", "validate_c"]
    enrichments = ["enrich_demographics", "enrich_behavior"]
    outputs = ["load_warehouse", "create_reports", "notify"]

    for node in sources:
        layer_colors[node] = "#E6F3FF"  # Very light blue
    for node in extracts:
        layer_colors[node] = "#B3D9FF"  # Light blue
    for node in validates:
        layer_colors[node] = "#FFF4E6"  # Light orange
    layer_colors["transform"] = "#FFE6CC"  # Darker orange
    for node in enrichments:
        layer_colors[node] = "#E6FFE6"  # Light green
    layer_colors["aggregate"] = "#CCE6CC"  # Darker green
    for node in outputs:
        layer_colors[node] = "#F3E6FF"  # Light purple

    mermaid = draw_mermaid(graph, title="Data Pipeline DAG", node_colors=layer_colors, with_styles=True)
    print(mermaid)

    print("\n📸 PNG Generation:")
    print("-" * 70)
    try:
        png_colors = {
            "source_a": "azure",
            "source_b": "azure",
            "source_c": "azure",
            "extract_a": "lightblue",
            "extract_b": "lightblue",
            "extract_c": "lightblue",
            "validate_a": "lightyellow",
            "validate_b": "lightyellow",
            "validate_c": "lightyellow",
            "transform": "wheat",
            "enrich_demographics": "lightgreen",
            "enrich_behavior": "lightgreen",
            "aggregate": "palegreen",
            "load_warehouse": "plum",
            "create_reports": "plum",
            "notify": "plum",
        }

        png_bytes = draw_png(graph, node_colors=png_colors, output_path="/tmp/data_pipeline_dag.png")
        if png_bytes:
            print(f"✅ Saved: /tmp/data_pipeline_dag.png ({len(png_bytes)} bytes)")
    except Exception as e:
        print(f"⚠️  PNG unavailable: {e}")


def example_4_custom_labels():
    """Example 4: Custom node labels and styling."""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Custom Labels and Styling")
    print("=" * 70)

    # Create graph with abbreviated node IDs
    graph = nx.DiGraph()
    edges = [
        ("cfg", "ld"),  # config -> load
        ("ld", "val"),  # load -> validate
        ("val", "xfm"),  # validate -> transform
        ("xfm", "sav"),  # transform -> save
    ]
    graph.add_edges_from(edges)

    # Create readable labels
    labels = {
        "cfg": "Configure\nSettings",
        "ld": "Load\nData",
        "val": "Validate\nQuality",
        "xfm": "Transform\n& Enrich",
        "sav": "Save\nResults",
    }

    print("\n📊 Graph with Abbreviated IDs:")
    print("-" * 70)
    show_graph_info(graph)

    print("\n🌊 Mermaid with Full Labels:")
    print("-" * 70)
    # Mermaid uses node IDs, so we'll show both
    mermaid_without_labels = draw_mermaid(graph, title="Abbreviated View")
    print(mermaid_without_labels)

    print("\n📸 PNG with Custom Labels:")
    print("-" * 70)
    try:
        colors = {"cfg": "lightgreen", "ld": "lightblue", "val": "lightyellow", "xfm": "lightcoral", "sav": "plum"}

        png_bytes = draw_png(graph, node_labels=labels, node_colors=colors, output_path="/tmp/custom_labels.png")
        if png_bytes:
            print(f"✅ Saved: /tmp/custom_labels.png ({len(png_bytes)} bytes)")
            print("   (Node labels are expanded in the PNG)")
    except Exception as e:
        print(f"⚠️  PNG unavailable: {e}")


def example_5_edge_cases():
    """Example 5: Handle edge cases and special graphs."""
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Edge Cases and Special Graphs")
    print("=" * 70)

    # Empty graph
    print("\n1️⃣  Empty Graph:")
    print("-" * 40)
    empty_graph = nx.DiGraph()
    print(draw_ascii(empty_graph))

    # Single node
    print("\n2️⃣  Single Node:")
    print("-" * 40)
    single_node = nx.DiGraph()
    single_node.add_node("lonely_task")
    print(draw_ascii(single_node))

    # Disconnected components
    print("\n3️⃣  Disconnected Components:")
    print("-" * 40)
    disconnected = nx.DiGraph()
    disconnected.add_edges_from([("a1", "a2"), ("b1", "b2"), ("c1", "c2")])
    show_graph_info(disconnected)
    print(draw_ascii(disconnected))

    # Linear chain
    print("\n4️⃣  Long Linear Chain:")
    print("-" * 40)
    chain = nx.DiGraph()
    nodes = [f"step_{i}" for i in range(10)]
    edges = [(nodes[i], nodes[i + 1]) for i in range(9)]
    chain.add_edges_from(edges)

    print(f"Chain length: {len(nodes)} nodes")
    mermaid = draw_mermaid(chain, title="Linear Pipeline")
    print(mermaid[:500] + "..." if len(mermaid) > 500 else mermaid)


def example_6_mermaid_api():
    """Example 6: Generate PNG via Mermaid API."""
    print("\n" + "=" * 70)
    print("EXAMPLE 6: Mermaid PNG via API")
    print("=" * 70)

    graph = nx.DiGraph()
    graph.add_edges_from([("start", "process"), ("process", "validate"), ("validate", "end")])

    print("\n🌐 Generating PNG via mermaid.ink API...")
    print("-" * 70)
    try:
        png_bytes = draw_mermaid_png(graph, title="API Generated Graph", background_color="white")

        if png_bytes:
            output_path = "/tmp/mermaid_api.png"
            with open(output_path, "wb") as f:
                f.write(png_bytes)
            print(f"✅ Saved: {output_path} ({len(png_bytes)} bytes)")
            print("   (Generated via mermaid.ink API)")
    except Exception as e:
        print(f"⚠️  API generation failed: {e}")
        print("   This requires internet connection and 'requests' package")
        print("   Install: pip install requests")


def main():
    """Run all graph utility examples."""
    print("\n" + "=" * 70)
    print("GRAFLOW GRAPH VISUALIZATION UTILITIES")
    print("=" * 70)
    print("\nThis example demonstrates low-level graph visualization utilities")
    print("for working directly with NetworkX DiGraph objects.")

    # Run examples
    example_1_basic_visualization()
    example_2_branching_graph()
    example_3_dag_visualization()
    example_4_custom_labels()
    example_5_edge_cases()
    example_6_mermaid_api()

    # Summary
    print("\n" + "=" * 70)
    print("EXAMPLES COMPLETE")
    print("=" * 70)

    print("\n📁 Generated Files:")
    print("   • /tmp/branching_graph.png")
    print("   • /tmp/data_pipeline_dag.png")
    print("   • /tmp/custom_labels.png")
    print("   • /tmp/mermaid_api.png")

    print("\n🛠️  Available Methods:")
    print("   • draw_ascii(graph)           - Terminal-friendly ASCII art")
    print("   • draw_mermaid(graph)         - Mermaid markdown diagram")
    print("   • draw_png(graph)             - PNG image (requires pygraphviz)")
    print("   • draw_mermaid_png(graph)     - PNG via mermaid.ink API")
    print("   • show_graph_info(graph)      - Graph statistics")
    print("   • visualize_dependencies(g)   - Dependency tree view")

    print("\n📦 Optional Dependencies:")
    print("   • pip install grandalf      - For ASCII visualization")
    print("   • pip install pygraphviz    - For local PNG generation")
    print("   • pip install requests      - For mermaid.ink API")

    print("\n💡 Use Cases:")
    print("   • Documentation - Use Mermaid diagrams in markdown")
    print("   • Presentations - Generate PNGs for slides")
    print("   • Debugging - Use ASCII in terminal/logs")
    print("   • Analysis - Use show_graph_info for graph statistics")

    print("\n✨ Next Steps:")
    print("   • Check workflow_visualization.py for Graflow-specific examples")
    print("   • Customize colors to match your documentation style")
    print("   • Integrate visualizations into CI/CD for workflow documentation")


if __name__ == "__main__":
    main()
