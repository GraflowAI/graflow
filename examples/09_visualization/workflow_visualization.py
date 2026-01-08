#!/usr/bin/env python3
"""
Workflow Visualization Example
==============================

This example demonstrates how to visualize Graflow workflows using the built-in
visualization utilities. Learn how to:
- Extract NetworkX graphs from workflows
- Generate ASCII representations
- Create Mermaid diagrams
- Generate PNG visualizations
- Analyze workflow structure

Concepts Covered:
-----------------
1. Graph extraction from TaskGraph
2. ASCII visualization (requires grandalf)
3. Mermaid diagram generation
4. PNG generation (requires pygraphviz)
5. Workflow analysis and debugging

Dependencies:
-------------
Optional (for full functionality):
- pip install grandalf      # ASCII visualization
- pip install pygraphviz    # PNG generation
- pip install requests      # Mermaid PNG via API

The example works without these packages but with reduced functionality.
"""

from graflow.core.context import TaskExecutionContext
from graflow.core.decorators import task
from graflow.core.workflow import workflow
from graflow.utils.graph import (
    draw_ascii,
    draw_mermaid,
    draw_png,
    show_graph_info,
    visualize_dependencies,
)


def example_1_simple_pipeline():
    """Example 1: Visualize a simple sequential pipeline."""
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Simple Sequential Pipeline")
    print("=" * 70)

    with workflow("simple_pipeline") as ctx:

        @task
        def load_data():
            """Load data from source."""
            print("  📥 Loading data...")
            return {"rows": 1000, "columns": 10}

        @task
        def validate_data():
            """Validate the loaded data."""
            print("  ✓ Validating data...")
            return {"valid": True}

        @task
        def transform_data():
            """Transform the data."""
            print("  🔄 Transforming data...")
            return {"transformed": True}

        @task
        def save_results():
            """Save the results."""
            print("  💾 Saving results...")
            return {"saved": True}

        # Build pipeline
        load_data >> validate_data >> transform_data >> save_results

        # Show workflow structure
        print("\n📊 Workflow Structure:")
        ctx.show_info()

        # Extract NetworkX graph
        nx_graph = ctx.graph.nx_graph()

        # Visualize with different methods
        print("\n🔍 Graph Analysis:")
        print("-" * 70)
        show_graph_info(nx_graph)

        print("\n📝 Task Dependencies:")
        print("-" * 70)
        visualize_dependencies(nx_graph)

        print("\n🎨 ASCII Visualization:")
        print("-" * 70)
        try:
            ascii_repr = draw_ascii(nx_graph)
            print(ascii_repr)
        except Exception as e:
            print(f"⚠️  ASCII drawing unavailable: {e}")
            print("   Install with: pip install grandalf")

        print("\n🌊 Mermaid Diagram:")
        print("-" * 70)
        mermaid = draw_mermaid(nx_graph, title="Simple Pipeline")
        print(mermaid)

        print("\n📸 PNG Generation:")
        print("-" * 70)
        try:
            png_bytes = draw_png(nx_graph, output_path="/tmp/simple_pipeline.png")
            if png_bytes:
                print(f"✅ PNG saved: /tmp/simple_pipeline.png ({len(png_bytes)} bytes)")
        except Exception as e:
            print(f"⚠️  PNG generation unavailable: {e}")
            print("   Install with: pip install pygraphviz")


def example_2_parallel_workflow():
    """Example 2: Visualize a workflow with parallel execution."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Parallel Processing Workflow")
    print("=" * 70)

    with workflow("parallel_workflow") as ctx:

        @task(inject_context=True)
        def load_data(context: TaskExecutionContext):
            """Load data from source."""
            print("  📥 Loading data...")
            context.get_channel().set("data", {"items": [1, 2, 3, 4, 5]})

        @task
        def process_batch_1():
            """Process first batch."""
            print("  ⚙️  Processing batch 1...")
            return {"batch": 1, "processed": True}

        @task
        def process_batch_2():
            """Process second batch."""
            print("  ⚙️  Processing batch 2...")
            return {"batch": 2, "processed": True}

        @task
        def process_batch_3():
            """Process third batch."""
            print("  ⚙️  Processing batch 3...")
            return {"batch": 3, "processed": True}

        @task
        def aggregate_results():
            """Aggregate all batch results."""
            print("  📊 Aggregating results...")
            return {"aggregated": True, "total": 3}

        @task
        def generate_report():
            """Generate final report."""
            print("  📄 Generating report...")
            return {"report": "complete"}

        # Build parallel workflow
        load_data >> (process_batch_1 | process_batch_2 | process_batch_3)
        (process_batch_1 | process_batch_2 | process_batch_3) >> aggregate_results
        aggregate_results >> generate_report

        # Visualization
        print("\n📊 Workflow Structure:")
        ctx.show_info()

        nx_graph = ctx.graph.nx_graph()

        print("\n📝 Task Dependencies:")
        print("-" * 70)
        visualize_dependencies(nx_graph)

        print("\n🌊 Mermaid Diagram:")
        print("-" * 70)
        mermaid = draw_mermaid(
            nx_graph,
            title="Parallel Processing",
            node_colors={"load_data": "#90EE90", "aggregate_results": "#FFB6C1", "generate_report": "#87CEEB"},
        )
        print(mermaid)

        print("\n📸 PNG with Custom Colors:")
        print("-" * 70)
        try:
            node_colors = {
                "load_data": "lightgreen",
                "process_batch_1": "lightyellow",
                "process_batch_2": "lightyellow",
                "process_batch_3": "lightyellow",
                "aggregate_results": "lightcoral",
                "generate_report": "lightblue",
            }

            png_bytes = draw_png(nx_graph, node_colors=node_colors, output_path="/tmp/parallel_workflow.png")
            if png_bytes:
                print(f"✅ PNG saved: /tmp/parallel_workflow.png ({len(png_bytes)} bytes)")
        except Exception as e:
            print(f"⚠️  PNG generation unavailable: {e}")


def example_3_complex_workflow():
    """Example 3: Visualize a complex workflow with branching."""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Complex ML Training Workflow")
    print("=" * 70)

    with workflow("ml_training") as ctx:

        @task
        def load_training_data():
            """Load training dataset."""
            return {"samples": 10000}

        @task
        def preprocess_data():
            """Preprocess the data."""
            return {"preprocessed": True}

        @task
        def feature_engineering():
            """Engineer features."""
            return {"features": 50}

        @task
        def train_model():
            """Train the model."""
            return {"model": "trained"}

        @task
        def validate_model():
            """Validate model performance."""
            return {"accuracy": 0.95}

        @task
        def hyperparameter_tuning():
            """Tune hyperparameters."""
            return {"optimized": True}

        @task
        def final_training():
            """Final model training."""
            return {"final_model": True}

        @task
        def model_evaluation():
            """Evaluate final model."""
            return {"eval_metrics": "complete"}

        @task
        def model_deployment():
            """Deploy the model."""
            return {"deployed": True}

        @task
        def monitoring_setup():
            """Setup monitoring."""
            return {"monitoring": "active"}

        # Build complex workflow
        load_training_data >> preprocess_data >> feature_engineering
        feature_engineering >> train_model >> validate_model
        validate_model >> hyperparameter_tuning >> final_training
        final_training >> model_evaluation >> model_deployment
        model_deployment >> monitoring_setup

        # Visualization
        print("\n📊 Workflow Structure:")
        ctx.show_info()

        nx_graph = ctx.graph.nx_graph()

        print("\n🔍 Graph Analysis:")
        print("-" * 70)
        show_graph_info(nx_graph)

        print("\n🌊 Mermaid Diagram:")
        print("-" * 70)
        # Color code by stage
        stage_colors = {
            "load_training_data": "#E6F3FF",  # Data loading - light blue
            "preprocess_data": "#E6F3FF",
            "feature_engineering": "#FFF4E6",  # Feature eng - light orange
            "train_model": "#FFEBE6",  # Training - light red
            "validate_model": "#FFEBE6",
            "hyperparameter_tuning": "#FFEBE6",
            "final_training": "#FFEBE6",
            "model_evaluation": "#E6FFE6",  # Evaluation - light green
            "model_deployment": "#F3E6FF",  # Deployment - light purple
            "monitoring_setup": "#F3E6FF",
        }

        mermaid = draw_mermaid(nx_graph, title="ML Training Pipeline", node_colors=stage_colors)
        print(mermaid)

        print("\n📸 PNG Generation:")
        print("-" * 70)
        try:
            png_colors = {
                "load_training_data": "lightblue",
                "preprocess_data": "lightblue",
                "feature_engineering": "lightyellow",
                "train_model": "lightcoral",
                "validate_model": "lightcoral",
                "hyperparameter_tuning": "lightcoral",
                "final_training": "lightcoral",
                "model_evaluation": "lightgreen",
                "model_deployment": "plum",
                "monitoring_setup": "plum",
            }

            png_bytes = draw_png(nx_graph, node_colors=png_colors, output_path="/tmp/ml_workflow.png")
            if png_bytes:
                print(f"✅ PNG saved: /tmp/ml_workflow.png ({len(png_bytes)} bytes)")
        except Exception as e:
            print(f"⚠️  PNG generation unavailable: {e}")


def main():
    """Run all workflow visualization examples."""
    print("\n" + "=" * 70)
    print("GRAFLOW WORKFLOW VISUALIZATION EXAMPLES")
    print("=" * 70)
    print("\nThis example demonstrates various methods to visualize Graflow workflows.")
    print("Visualization helps you understand, debug, and document your workflows.")

    # Run examples
    example_1_simple_pipeline()
    example_2_parallel_workflow()
    example_3_complex_workflow()

    # Summary
    print("\n" + "=" * 70)
    print("VISUALIZATION COMPLETE")
    print("=" * 70)

    print("\n📁 Generated Files:")
    print("   • /tmp/simple_pipeline.png")
    print("   • /tmp/parallel_workflow.png")
    print("   • /tmp/ml_workflow.png")

    print("\n📦 Optional Dependencies:")
    print("   • pip install grandalf      # ASCII visualization")
    print("   • pip install pygraphviz    # PNG generation")

    print("\n💡 Tips:")
    print("   • Use show_info() for quick workflow structure overview")
    print("   • Use Mermaid diagrams for documentation (copy to markdown)")
    print("   • Use PNG for presentations and reports")
    print("   • Use ASCII for quick terminal debugging")

    print("\n✨ Next Steps:")
    print("   • Try visualizing your own workflows")
    print("   • Customize colors to indicate task types")
    print("   • Use visualizations to identify optimization opportunities")


if __name__ == "__main__":
    main()
