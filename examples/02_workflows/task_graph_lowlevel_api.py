"""
Low-Level TaskGraph API Example
=================================

This example demonstrates how to construct workflows using the low-level
TaskGraph APIs directly, without the high-level workflow context or operators.

This is useful when you need:
- Fine-grained control over graph construction
- Dynamic graph building based on runtime conditions
- Integration with external graph sources
- Custom workflow orchestration logic

Prerequisites:
--------------
None

Concepts Covered:
-----------------
1. Direct TaskGraph construction with add_node() and add_edge()
2. Manual ExecutionContext creation
3. Using WorkflowEngine for execution
4. Understanding the relationship between high-level and low-level APIs
5. Graph visualization with ASCII representation

Expected Output:
----------------
=== Example 1: Simple Linear Pipeline ===

Building graph with low-level APIs...
Graph structure:
 extract
  ↓
 transform
  ↓
 load

Executing pipeline...
📥 Extract: Loading data...
🔄 Transform: Processing data...
💾 Load: Saving results...

✅ Pipeline completed!

=== Example 2: Fan-out Pattern ===

Building graph with low-level APIs...
Graph structure:
 source
  ├─→ process_a
  ├─→ process_b
  └─→ process_c

Executing fan-out workflow...
📡 Source: Fetching data...
⚡ Process A: Processing partition 1...
⚡ Process B: Processing partition 2...
⚡ Process C: Processing partition 3...

✅ Fan-out completed!

=== Example 3: Diamond Pattern (Fan-out + Fan-in) ===

Building graph with low-level APIs...
Graph structure:
 fetch
  ├─→ transform_a
  └─→ transform_b
    ↓
 store

Executing diamond workflow...
📥 Fetch: Loading data...
🔄 Transform A: Applying transformation A...
🔄 Transform B: Applying transformation B...
💾 Store: Saving combined results...

✅ Diamond pattern completed!
"""

from graflow.core.context import ExecutionContext
from graflow.core.engine import WorkflowEngine
from graflow.core.graph import TaskGraph
from graflow.core.task import TaskWrapper


def example_1_linear_pipeline():
    """Example 1: Simple linear pipeline using low-level TaskGraph API.

    High-level equivalent:
        extract >> transform >> load
    """
    print("=== Example 1: Simple Linear Pipeline ===\n")

    # Define tasks
    extract = TaskWrapper("extract", func=lambda: print("📥 Extract: Loading data..."), register_to_context=False)
    transform = TaskWrapper(
        "transform", func=lambda: print("🔄 Transform: Processing data..."), register_to_context=False
    )
    load = TaskWrapper("load", func=lambda: print("💾 Load: Saving results..."), register_to_context=False)

    # Build graph with low-level APIs
    print("Building graph with low-level APIs...")
    graph = TaskGraph()

    # Add nodes to the graph
    graph.add_node(extract, "extract")
    graph.add_node(transform, "transform")
    graph.add_node(load, "load")

    # Add edges to define dependencies
    graph.add_edge("extract", "transform")  # transform depends on extract
    graph.add_edge("transform", "load")  # load depends on transform

    # Visualize the graph structure
    print("Graph structure:")
    print(graph)

    # Create execution context with the graph
    context = ExecutionContext.create(graph, "extract", max_steps=10)

    # Execute using WorkflowEngine
    print("Executing pipeline...")
    engine = WorkflowEngine()
    engine.execute(context)

    print("\n✅ Pipeline completed!\n")


def example_2_fan_out():
    """Example 2: Fan-out pattern using low-level TaskGraph API.

    High-level equivalent:
        source >> (process_a | process_b | process_c)
    """
    print("=== Example 2: Fan-out Pattern ===\n")

    # Define tasks
    source = TaskWrapper("source", func=lambda: print("📡 Source: Fetching data..."), register_to_context=False)
    process_a = TaskWrapper(
        "process_a", func=lambda: print("⚡ Process A: Processing partition 1..."), register_to_context=False
    )
    process_b = TaskWrapper(
        "process_b", func=lambda: print("⚡ Process B: Processing partition 2..."), register_to_context=False
    )
    process_c = TaskWrapper(
        "process_c", func=lambda: print("⚡ Process C: Processing partition 3..."), register_to_context=False
    )

    # Build graph with low-level APIs
    print("Building graph with low-level APIs...")
    graph = TaskGraph()

    # Add nodes
    graph.add_node(source, "source")
    graph.add_node(process_a, "process_a")
    graph.add_node(process_b, "process_b")
    graph.add_node(process_c, "process_c")

    # Add edges: source fans out to all three processors
    graph.add_edge("source", "process_a")
    graph.add_edge("source", "process_b")
    graph.add_edge("source", "process_c")

    # Visualize the graph structure
    print("Graph structure:")
    print(graph)

    # Create execution context and execute
    print("Executing fan-out workflow...")
    context = ExecutionContext.create(graph, "source", max_steps=10)
    engine = WorkflowEngine()
    engine.execute(context)

    print("\n✅ Fan-out completed!\n")


def example_3_diamond():
    """Example 3: Diamond pattern (fan-out + fan-in) using low-level TaskGraph API.

    High-level equivalent:
        fetch >> (transform_a | transform_b) >> store
    """
    print("=== Example 3: Diamond Pattern (Fan-out + Fan-in) ===\n")

    # Define tasks
    fetch = TaskWrapper("fetch", func=lambda: print("📥 Fetch: Loading data..."), register_to_context=False)
    transform_a = TaskWrapper(
        "transform_a", func=lambda: print("🔄 Transform A: Applying transformation A..."), register_to_context=False
    )
    transform_b = TaskWrapper(
        "transform_b", func=lambda: print("🔄 Transform B: Applying transformation B..."), register_to_context=False
    )
    store = TaskWrapper("store", func=lambda: print("💾 Store: Saving combined results..."), register_to_context=False)

    # Build graph with low-level APIs
    print("Building graph with low-level APIs...")
    graph = TaskGraph()

    # Add nodes
    graph.add_node(fetch, "fetch")
    graph.add_node(transform_a, "transform_a")
    graph.add_node(transform_b, "transform_b")
    graph.add_node(store, "store")

    # Add edges: diamond pattern
    # fetch fans out to both transforms
    graph.add_edge("fetch", "transform_a")
    graph.add_edge("fetch", "transform_b")

    # Both transforms fan in to store
    graph.add_edge("transform_a", "store")
    graph.add_edge("transform_b", "store")

    # Visualize the graph structure
    print("Graph structure:")
    print(graph)

    # Create execution context and execute
    print("Executing diamond workflow...")
    context = ExecutionContext.create(graph, "fetch", max_steps=10)
    engine = WorkflowEngine()
    engine.execute(context)

    print("\n✅ Diamond pattern completed!\n")


def main():
    """Run all low-level TaskGraph API examples."""
    example_1_linear_pipeline()
    example_2_fan_out()
    example_3_diamond()
    print("All low-level TaskGraph API examples completed! 🎉")


if __name__ == "__main__":
    main()


# ============================================================================
# Key Takeaways:
# ============================================================================
#
# 1. **TaskGraph Construction**
#    graph = TaskGraph()
#    graph.add_node(task)          # Add a task to the graph
#    graph.add_edge("from", "to")  # Define dependency: "to" depends on "from"
#
# 2. **Task Creation**
#    task = TaskWrapper("task_id", func=lambda: print("Hello"), register_to_context=False)
#    - task_id: Unique identifier for the task
#    - func: The function to execute
#    - register_to_context=False: Don't auto-register to workflow context
#
# 3. **Execution Context**
#    context = ExecutionContext.create(graph, "start_task_id", max_steps=10)
#    - Creates the execution environment with the graph
#    - start_task_id: The task to start execution from
#    - max_steps: Maximum number of tasks to execute
#    - Manages task execution state, results, and channels
#
# 4. **WorkflowEngine Execution**
#    engine = WorkflowEngine()
#    engine.execute(context)
#    - Executes tasks in topological order
#    - Starts from the task specified in ExecutionContext.create()
#
# 5. **High-Level vs Low-Level APIs**
#
#    High-level (workflow context + operators):
#    ----------------------------------------
#    with workflow("name") as ctx:
#        task_a >> task_b >> task_c
#        ctx.execute("task_a")
#
#    Low-level (TaskGraph + Engine):
#    --------------------------------
#    task_a = TaskWrapper("task_a", func=..., register_to_context=False)
#    task_b = TaskWrapper("task_b", func=..., register_to_context=False)
#    task_c = TaskWrapper("task_c", func=..., register_to_context=False)
#    graph = TaskGraph()
#    graph.add_node(task_a, "task_a")
#    graph.add_node(task_b, "task_b")
#    graph.add_node(task_c, "task_c")
#    graph.add_edge("task_a", "task_b")
#    graph.add_edge("task_b", "task_c")
#    context = ExecutionContext.create(graph, "task_a", max_steps=10)
#    engine = WorkflowEngine()
#    engine.execute(context)
#
#    Both approaches produce the same result!
#
# 6. **When to Use Low-Level APIs**
#    - Dynamic graph construction based on runtime data
#    - Integration with external workflow definitions
#    - Custom graph algorithms or analysis
#    - Fine-grained control over graph structure
#    - Building workflow tools and frameworks
#
# 7. **Graph Operations**
#    graph.nodes              # Get all node IDs
#    graph.edges              # Get all edges
#    graph.get_node(task_id)  # Get task by ID
#    graph.successors(task_id)    # Get dependent tasks
#    graph.predecessors(task_id)  # Get dependency tasks
#    graph.detect_cycles()    # Find cycles in the graph
#
# ============================================================================
# Try Experimenting:
# ============================================================================
#
# 1. Build a multi-stage pipeline dynamically:
#    graph = TaskGraph()
#    tasks = [TaskWrapper(f"stage_{i}", func=lambda i=i: print(f"Stage {i}"), register_to_context=False) for i in range(5)]
#    for task in tasks:
#        graph.add_node(task, task.task_id)
#    for i in range(len(tasks) - 1):
#        graph.add_edge(tasks[i].task_id, tasks[i+1].task_id)
#
# 2. Create a conditional graph:
#    if some_condition:
#        graph.add_edge("task_a", "task_b")
#    else:
#        graph.add_edge("task_a", "task_c")
#
# 3. Build a graph from configuration:
#    config = {
#        "nodes": ["a", "b", "c"],
#        "edges": [("a", "b"), ("b", "c")]
#    }
#    graph = TaskGraph()
#    for node in config["nodes"]:
#        task = TaskWrapper(node, func=lambda: print(f"Task {node}"), register_to_context=False)
#        graph.add_node(task, node)
#    for from_node, to_node in config["edges"]:
#        graph.add_edge(from_node, to_node)
#
# 4. Inspect graph structure before execution:
#    print(f"Nodes: {list(graph.nodes)}")
#    print(f"Edges: {graph.get_edges()}")
#    print(f"Start nodes: {graph.get_start_nodes()}")
#    cycles = graph.detect_cycles()
#    if cycles:
#        print(f"Warning: Cycles detected: {cycles}")
#
# 5. Mix high-level and low-level approaches:
#    # Build initial structure with high-level API
#    with workflow("mixed") as ctx:
#        task_a >> task_b
#
#    # Then modify graph with low-level API
#    graph = ctx.graph
#    task_c = TaskWrapper("task_c", func=lambda: print("Task C"), register_to_context=False)
#    graph.add_node(task_c, "task_c")
#    graph.add_edge("task_b", "task_c")
#
#    # Execute the modified graph
#    ctx.execute("task_a")
#
# ============================================================================
# Real-World Use Cases:
# ============================================================================
#
# **Dynamic Pipeline Generation**:
# Build workflows based on user input, configuration files, or database schemas
#
# **Workflow Analysis Tools**:
# Analyze, validate, or optimize workflow graphs before execution
#
# **Custom Execution Strategies**:
# Implement custom scheduling, resource allocation, or execution policies
#
# **Workflow Migration**:
# Convert workflows from other systems (Airflow, Prefect, etc.) to Graflow
#
# **Graph Algorithms**:
# Apply graph algorithms (shortest path, critical path, etc.) to workflows
#
# ============================================================================
