# Modular Workflow Execution Architecture (with Cyclic Task Graph Support)

This document describes a modular architecture that separates **workflow control** and **task execution** in systems that allow **cyclic task graphs** and parallel task workers.

---

## 🧠 Motivation

Traditional systems often mix:

- Task execution logic
- Dependency resolution
- Workflow state tracking

We separate these responsibilities for **clarity**, **scalability**, and **interruption-resume** support.

---

## 🎯 Goal

Build a system where:

- **WorkflowEngine** manages task graph state, scheduling, and dependencies
- **TaskWorker** executes only one task at a time
- **ResultQueue** allows workers to report back to the engine
- **Context** tracks the whole system state, supporting checkpointing

---

## 🧱 Architecture Overview

```text
+---------------------+          +---------------------+
|    WorkflowEngine   |          |     TaskWorker      |
|---------------------|          |---------------------|
| - graph             |          | - get task          |
| - context           |   ===>   | - run task          |
| - visit counts      |  task    | - put result        |
| - dependency logic  |   ===>   |   in result queue   |
+---------------------+          +---------------------+

             ^                          |
             |         +---------------------+
             |         |    Result Queue     |
             +---------+  (task_id results)  |
                       +---------------------+
```

---

## 📦 1. `WorkflowEngine`: schedules ready tasks

```python
class WorkflowEngine:
    def __init__(self, graph, context, task_queue):
        self.graph = graph
        self.context = context
        self.task_queue = task_queue

    def step(self):
        for node in self.graph.nodes:
            if self.is_ready(node):
                self.enqueue(node)

    def is_ready(self, node_id):
        return (
            self.context["visit_count"][node_id] < self.context["max_visits"] and
            self.context["pending_predecessors"][node_id] == 0 and
            node_id not in self.context["enqueued"]
        )

    def enqueue(self, node_id):
        self.context["enqueued"].add(node_id)
        task = self.graph.nodes[node_id]["task"]
        self.task_queue.put({"task_id": node_id, "task": task})

    def mark_task_done(self, task_id):
        self.context["visit_count"][task_id] += 1
        self.context["executed"].add(task_id)
        for succ in self.graph.successors(task_id):
            self.context["pending_predecessors"][succ] -= 1
```

---

## ⚙ 2. `TaskWorker`: executes only

```python
def task_worker_loop(task_queue, result_queue):
    while True:
        task_info = task_queue.get()
        if task_info is None:
            break
        task = task_info["task"]
        task_id = task_info["task_id"]
        print(f"[Worker] Running {task_id}")
        task.run({})
        result_queue.put(task_id)
        task_queue.task_done()
```

---

## 🔄 3. Handling results in the engine

```python
def handle_result(engine, result_queue):
    while not result_queue.empty():
        task_id = result_queue.get()
        engine.mark_task_done(task_id)
        result_queue.task_done()
```

---

## 🚀 4. Main Execution Loop

```python
engine = WorkflowEngine(graph, context, task_queue)

while context["steps"] < context["max_steps"]:
    engine.step()
    handle_result(engine, result_queue)
    context["steps"] += 1
```

---

## 💾 Context State Structure

```python
context = {
    "visit_count": defaultdict(int),
    "pending_predecessors": defaultdict(int),
    "executed": set(),
    "enqueued": set(),
    "steps": 0,
    "max_steps": 50,
    "max_visits": 3
}
```

---

## ✅ Advantages

| Feature | Benefit |
|--------|---------|
| Clear separation | Task execution logic decoupled from flow control |
| Distributed-ready | Workers can run on remote machines |
| State persistence | Context can be checkpointed and resumed |
| Cycle-safe | Visitation limits avoid infinite loops |
| HITL support | Workflow can pause, wait for input |

---

## 📝 Summary

By separating task scheduling from execution, this architecture supports:

- Scalable parallel workflows
- Execution over cyclic graphs
- Robust state recovery and resumption

It is especially useful for AI pipelines, ML model training loops, or human-in-the-loop workflows.

