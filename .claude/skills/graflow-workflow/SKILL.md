---
name: graflow-workflow
description: Create Python workflow pipelines using Graflow with a structured plan-implement-review process. Use when building task graphs, parallel pipelines, LLM workflows, or any Graflow-based automation. Triggers on requests for "workflow", "pipeline", "task graph", "Graflow", or when user wants to build an automated data/AI pipeline.
---

# Graflow Workflow Builder

Build executable task graphs in Python using a structured 3-phase approach.

## Workflow: Plan -> Implement -> Review

### Phase 1: Plan

**Goal**: Clarify requirements, create a design document, and get user approval through iterative feedback.

**Steps**:

#### Step 1: Requirements Gathering
Ask the user clarifying questions about the workflow:
- What is the workflow's purpose?
- What are the input sources and output destinations?
- Which tasks need to run sequentially vs. in parallel?
- Is LLM integration needed?
- Are there any dynamic/conditional branching requirements?

#### Step 2: Create Design Document
Create a design document (`workflow_design.md`) with:
- Workflow overview and purpose
- Task definitions (name, responsibility, inputs/outputs)
- Task graph structure (ASCII diagram)
- Channel data flow
- Error handling strategy

**Design Document Template**:
```markdown
# Workflow Design: {workflow_name}

## Overview
{Brief description of what this workflow accomplishes}

## Tasks

| Task ID | Responsibility | Inputs | Outputs |
|---------|---------------|--------|---------|
| task_a  | ...           | ...    | ...     |

## Task Graph
```
source >> (transform_a | transform_b) >> sink
```

## Channel Data Flow
- `config`: Set by setup, used by all tasks
- `results`: Accumulated by each task

## Error Handling
- {Strategy: fail-fast, best-effort, retry, etc.}
```

#### Step 3: Present Design to User
Present the design document to the user with a clear summary:
- Show the task graph diagram
- Highlight key design decisions
- Ask explicitly: "Does this design meet your requirements? Please provide feedback if any changes are needed."

#### Step 4: Iterate on Feedback
If the user provides feedback:
1. Update `workflow_design.md` with the requested changes
2. Summarize the changes made
3. Re-present the updated design
4. Repeat until the user is satisfied

#### Step 5: Confirm Design Approval
Before proceeding to implementation:
- Ask the user to confirm: "Is this design approved? If yes, I'll proceed to implementation."
- Only move to Phase 2 after explicit approval

### Phase 2: Implement

**Goal**: Write the workflow code based on the approved design.

**Steps**:
1. Create the workflow file following the design document
2. Use appropriate Graflow patterns (see references/workflow-patterns.md)
3. Add type hints and docstrings

**Implementation Checklist**:
- [ ] Import statements
- [ ] Task definitions with `@task` decorator
- [ ] Workflow context with `workflow()` context manager
- [ ] Task composition with `>>` and `|` operators
- [ ] Channel usage for inter-task communication
- [ ] Entry point (`if __name__ == "__main__"`)

### Phase 3: Review

**Goal**: Validate the implementation against the design and create documentation.

#### Step 1: Implementation Review

**Review Checklist**:
- [ ] All tasks from design are implemented
- [ ] Task graph matches the design diagram
- [ ] Channel data flow is correct
- [ ] Error handling is in place
- [ ] Code follows Graflow conventions
- [ ] Type hints are present
- [ ] Docstrings explain task purpose

**Common Issues to Check**:
- Missing `inject_context=True` when accessing channels
- Incorrect parameter priority (channel < bound < injection)
- Missing `set_group_name()` for parallel groups
- Incorrect execution entry point

#### Step 2: Create README.md

After implementation is complete, create a `README.md` in the workflow directory with:

**README Template**:
```markdown
# {Workflow Name}

## Overview
{Brief description of what this workflow does and its use cases}

## Requirements
- Python 3.11+
- Graflow
- {Additional dependencies}

## Usage

### Basic Execution
```bash
PYTHONPATH=. uv run python {workflow_file}.py
```

### With Custom Parameters
```python
from {workflow_module} import run_workflow

result = run_workflow(param1="value1", param2="value2")
```

## Workflow Structure

### Task Graph
```
{ASCII representation of task graph}
```

### Tasks

| Task | Description |
|------|-------------|
| task_a | {What this task does} |
| task_b | {What this task does} |

### Channel Data Flow

| Channel Key | Producer | Consumer | Description |
|-------------|----------|----------|-------------|
| config | setup | all | Configuration settings |
| results | processors | aggregator | Accumulated results |

## Configuration

{Description of configurable parameters and environment variables}

## Examples

{Usage examples with expected output}
```

#### Step 3: Final Presentation

Present the completed workflow to the user:
- Summarize what was implemented
- Show the README.md content
- Confirm the workflow is ready for use

## Quick Reference

### Core Imports
```python
from graflow.core.decorators import task
from graflow.core.workflow import workflow
from graflow.core.context import TaskExecutionContext
from graflow.core.task import parallel
```

### Task Patterns
```python
# Simple task
@task
def process() -> str:
    return "done"

# Task with context injection
@task(inject_context=True)
def with_channel(ctx: TaskExecutionContext):
    ctx.get_channel().set("key", "value")

# Task with LLM client
@task(inject_llm_client=True)
def with_llm(llm_client: LLMClient):
    return llm_client.completion_text(messages=[...])

# Task instance with bound parameters
task1 = process(task_id="task1", value=10)
```

### Composition Operators
```python
a >> b           # Sequential: a then b
a | b            # Parallel: a and b concurrently
(a | b) >> c     # Fan-in: c waits for both a and b
a >> (b | c)     # Fan-out: b and c start after a
```

## References

- **Detailed patterns**: See [references/workflow-patterns.md](references/workflow-patterns.md)
- **Advanced patterns**: See [references/advanced-patterns.md](references/advanced-patterns.md)
