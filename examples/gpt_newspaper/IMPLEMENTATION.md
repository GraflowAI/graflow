# GPT Newspaper - Implementation Summary

## Overview

This is a complete graflow implementation of the gpt-newspaper autonomous agent system, demonstrating advanced workflow patterns including dynamic task creation, parallel execution, and the `goto=True` pattern.

## Key Implementation Decisions

### 1. Task Flow Architecture

```
search >> curate >> write >> critique >> designer
```

Each agent has its own focused task:
- **Search**: Finds news articles via Tavily API
- **Curator**: Selects 5 most relevant sources
- **Writer**: Writes or revises articles
- **Critique**: Provides feedback and decides next action
- **Designer**: Creates HTML layout

### 2. Declarative Graph with goto=True (Key Innovation)

Instead of dynamically creating designer task, define all tasks upfront:

```python
# All tasks defined with @task decorator
@task(id=f"write_{article_id}", inject_context=True)
def write_task(...): ...

@task(id=f"critique_{article_id}", inject_context=True)
def critique_task(...): ...

@task(id=f"design_{article_id}")
def design_task(...): ...

# Static graph definition
_ = search_task >> curate_task >> write_task >> critique_task >> design_task

# In critique_task: loop back with goto=True
if critique_feedback:
    context.next_task(write_task, goto=True)
    return result  # Returns, but will jump to write_task

# When approved, natural flow continues to design_task
return result  # Continues to next task in graph: design_task
```

**Benefits:**
- ✅ All tasks defined upfront (no dynamic TaskWrapper creation)
- ✅ Clear, declarative workflow structure
- ✅ Design task automatically runs when critique approves
- ✅ Cleaner control flow
- ✅ Matches graflow's goto pattern from runtime_dynamic_tasks.py

### 3. Parallel Execution with ThreadPoolExecutor

Multiple article workflows execute concurrently:

```python
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    completed_articles = list(
        executor.map(
            execute_article_workflow,
            zip(queries, article_ids, [output_dir] * len(queries))
        )
    )
```

This matches the original gpt-newspaper's parallel processing approach.

### 4. Loop-Back with goto=True

When critique has feedback, jump back to existing write_task:

```python
@task(id=f"critique_{article_id}", inject_context=True)
def critique_task(context: TaskExecutionContext, article: Dict) -> Dict:
    result = critique_agent.run(article)

    if result.get("critique") is not None:
        # Has feedback - loop back to write_task
        channel.set("iteration", iteration + 1)
        channel.set("article", result)
        context.next_task(write_task, goto=True)
        return result

    # Approved - natural flow continues to design_task
    return result
```

No need to create new tasks for iterations - reuses existing write_task. Design task runs automatically when critique completes without goto.

### 5. Safety Limits

- Max 5 write-critique iterations per article
- Configurable `max_steps=30` in workflow execution
- Prevents infinite loops

## Architecture Comparison

### Original (LangGraph)

```python
workflow.add_conditional_edges(
    start_key='critique',
    condition=lambda x: "accept" if x['critique'] is None else "revise",
    conditional_edge_mapping={"accept": "design", "revise": "write"}
)
```

### Graflow Version

```python
# In critique task:
if result.get("critique") is not None:
    # Jump back to write_task with goto=True
    context.next_task(write_task, goto=True)
    return result
# If approved, natural flow continues to design_task
return result
```

## File Structure

```
examples/gpt_newspaper/
├── agents/
│   ├── __init__.py
│   ├── search.py           # Tavily web search
│   ├── curator.py          # Source selection (uses LiteLLMClient)
│   ├── writer.py           # Article writing (uses LiteLLMClient)
│   ├── critique.py         # Quality feedback (uses LiteLLMClient)
│   ├── designer.py         # HTML generation
│   ├── editor.py           # Newspaper compilation
│   └── publisher.py        # Output saving
├── utils/
│   ├── __init__.py
│   └── litellm.py          # LiteLLM wrapper utility
├── templates/
│   ├── article/index.html
│   └── newspaper/layouts/  # 3 layout options
├── newspaper_workflow.py   # Main workflow (350 lines)
├── test_workflow.py        # Structure tests
├── parallel_example.py     # Demo script
├── config.py              # Configuration
├── requirements.txt       # Dependencies
└── README.md              # Documentation
```

## Key Patterns Demonstrated

1. **Declarative Task Graph**: All tasks defined upfront with @task decorator
2. **Sequential Task Flow**: Clear separation of concerns (search >> curate >> write >> critique >> design)
3. **goto=True Loop-Back**: Jump to existing task, natural flow continues after approval
4. **Channel State Management**: State persists across iterations
5. **Parallel Execution**: ThreadPoolExecutor for concurrent workflows
6. **Safety Limits**: Max iterations prevent infinite loops

## Differences from Original

| Aspect | Original (LangGraph) | Graflow Version |
|--------|---------------------|-----------------|
| LLM Library | langchain_openai | litellm |
| Workflow Engine | LangGraph | Graflow |
| Iteration Pattern | Conditional edges | goto=True + dynamic tasks |
| State Management | LangGraph state | Graflow channels |
| Parallel Execution | ThreadPoolExecutor | ThreadPoolExecutor (same) |

## Usage

```bash
# Set environment variables
export TAVILY_API_KEY=your_key
export OPENAI_API_KEY=your_key

# Run workflow
python newspaper_workflow.py

# Or with custom settings
python -c "
from newspaper_workflow import run_newspaper_workflow
run_newspaper_workflow(
    queries=['AI news', 'Climate updates'],
    layout='layout_2.html',
    max_workers=2
)
"
```

## Testing

All tests pass:
```bash
python test_workflow.py
# ✅ Agent Imports
# ✅ Workflow Structure
# ✅ Task Dependencies
```

## Future Enhancements

Potential improvements:
- Add distributed execution with Redis queue
- Implement caching for search results
- Add more sophisticated critique criteria
- Support for more LLM providers
- Real-time progress monitoring

## Credits

Based on the original gpt-newspaper project by @assafelovic and @rotemweiss57.

Adapted to use:
- Graflow workflow engine
- litellm for LLM abstraction
- goto=True pattern for cleaner flow
