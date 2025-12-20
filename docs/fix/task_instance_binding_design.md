# Task Instance Binding Design

## Overview

Enable creating multiple task instances from a single `@task` decorated function with bound parameters, allowing function reuse with different parameter values.

## Motivation

Currently, users cannot easily create multiple task instances from the same function:

```python
@task
def ask_weather(ctx, query: str, history: list | None = None):
    ...

# Problem: Want to reuse function with different parameters
ask_tokyo = ask_weather(task_id="ask_tokyo", query="What's the weather in Tokyo?")
ask_paris = ask_weather(task_id="ask_paris", query="What's the weather in Paris?")

# Current behavior: Both variables point to the same TaskWrapper instance (broken!)
# Desired behavior: Two separate TaskWrapper instances with bound parameters
```

## Requirements

1. **Instance Creation**: `TaskWrapper.__call__()` should create new instances when called with `task_id`
2. **Parameter Binding**: Support binding keyword arguments at creation time
3. **Unique Task IDs**: Each instance must have a unique `task_id`
4. **Lazy Registration**: Follow existing `Executable._pending_registration` pattern
5. **Parameter Resolution**: Support both bound parameters and channel-based resolution

## Design Decisions

### 1. Implementation Location

**Decision**: Implement in `TaskWrapper.__call__()`

**Rationale**:
- Minimal changes to existing codebase
- Natural syntax: `task_function(task_id="...", param="...")`
- Works seamlessly with existing `@task` decorator
- No breaking changes to decorator logic

### 2. Registration Timing

**Decision**: Follow `Executable` base class pattern

**Behavior**:
- If workflow context exists → Register immediately
- If no workflow context → Set `_pending_registration = True`, register lazily when added via `>>` or `|`

**Implementation**: Reuse existing `Executable._register_to_context()` and `_ensure_registered()` methods

### 3. Bound Parameters Storage

**Decision**: Support **kwargs only** (no positional args)

**Rationale**:
- Clearer and more explicit
- Easier to merge with channel-resolved and injected kwargs
- Avoids ambiguity with positional parameter injection

**Storage**: Add `_bound_kwargs: dict` attribute to `TaskWrapper` instances

### 4. Parameter Priority/Merge Strategy

**Decision**: `channel < bound < injection`

**Priority Order** (lowest to highest):
1. **Channel kwargs** (resolved from channel via `_resolve_keyword_args_from_channel()`)
2. **Bound kwargs** (passed at task creation time) ← **Overrides channel**
3. **Injection kwargs** (system-injected: `llm_client`, `llm_agent`) ← **Highest priority**

**Rationale**:
- Bound parameters are explicit user intent → should override generic channel values
- Injection kwargs are system-level → always win
- Channel provides fallback for unbound parameters

**Merge Logic**:
```python
all_kwargs = {**resolved_kwargs, **bound_kwargs, **injection_kwargs}
```

### 5. Edge Cases

#### Case 1: Call without `task_id`
```python
result = ask_weather(query="Tokyo")
```

**Behavior**: Auto-generate task_id as `{function_name}_{uuid.uuid4().hex[:8]}`

**Example**: `ask_weather_a3f2b9c1`

**Rationale**: Allow quick task creation without explicit naming

#### Case 2: Call with only `task_id`, no bindings
```python
ask_tokyo = ask_weather(task_id="ask_tokyo")
```

**Behavior**: Valid. Create instance, resolve all parameters from channel at execution time.

**Rationale**: Support pure channel-based parameter passing

#### Case 3: Positional args
```python
ask_tokyo = ask_weather("ask_tokyo", "Tokyo")
```

**Behavior**: **Not supported**. Only keyword arguments allowed.

**Rationale**: Avoid ambiguity, keep implementation simple

## Implementation Plan

### Step 1: Modify `TaskWrapper.__call__()`

Add instance creation logic when called without execution context:

```python
def __call__(self, *args, **kwargs) -> Any:
    if not hasattr(self, '_execution_context'):
        # Instance creation mode (no execution context)

        # Extract or generate task_id
        if 'task_id' in kwargs:
            new_task_id = kwargs.pop('task_id')
        else:
            # Auto-generate task_id
            import uuid
            func_name = getattr(self.func, '__name__', 'task')
            new_task_id = f"{func_name}_{uuid.uuid4().hex[:8]}"

        # Create new TaskWrapper instance
        new_instance = TaskWrapper(
            task_id=new_task_id,
            func=self.func,
            inject_context=self.inject_context,
            inject_llm_client=self.inject_llm_client,
            inject_llm_agent=self.inject_llm_agent,
            register_to_context=True,  # Uses _register_to_context() internally
            handler_type=self.handler_type,
            resolve_keyword_args=self.resolve_keyword_args
        )

        # Store bound parameters (remaining kwargs after task_id extraction)
        if kwargs:
            new_instance._bound_kwargs = kwargs

        return new_instance

    # Execution mode (has execution context) - existing logic
    exec_context = self._execution_context
    # ... rest of existing execution logic
```

### Step 2: Modify `TaskWrapper.run()`

Update parameter merging to include bound kwargs:

```python
def run(self) -> Any:
    exec_context = self.get_execution_context()

    # Resolve keyword arguments from channel
    resolved_kwargs = self._resolve_keyword_args_from_channel(exec_context)

    # Prepare injection kwargs
    injection_kwargs = self._prepare_injection_kwargs(exec_context)

    # Get bound kwargs (creation-time parameters)
    bound_kwargs = getattr(self, '_bound_kwargs', {})

    # Merge all kwargs (priority: channel < bound < injection)
    all_kwargs = {**resolved_kwargs, **bound_kwargs, **injection_kwargs}

    # ... rest of existing logic
```

### Step 3: Update `TaskWrapper.__init__()`

Initialize `_bound_kwargs` attribute:

```python
def __init__(self, ...):
    # ... existing initialization

    # Initialize bound kwargs storage
    self._bound_kwargs = {}

    # ... existing registration logic
```

### Step 4: Update Serialization

Ensure `_bound_kwargs` is preserved during pickling:

```python
def __getstate__(self):
    state = super().__getstate__()
    # _bound_kwargs should be preserved (it's serializable dict)
    # No special handling needed
    return state
```

## Testing Strategy

### Unit Tests

1. **Test Instance Creation**
   - Create multiple instances from same function
   - Verify each has unique task_id
   - Verify bound parameters are stored

2. **Test Parameter Resolution**
   - Verify priority: channel < bound < injection
   - Test bound parameters override channel values
   - Test injection kwargs always win

3. **Test Auto-generated task_id**
   - Call without task_id
   - Verify format: `{func_name}_{8-char-hash}`
   - Verify uniqueness

4. **Test Registration**
   - Test immediate registration (with context)
   - Test lazy registration (without context)

5. **Test Serialization**
   - Pickle/unpickle instance with bound kwargs
   - Verify bound kwargs preserved

### Integration Tests

1. **Multi-instance Workflow**
   ```python
   with workflow("multi_instance") as wf:
       ask_tokyo = ask_weather(task_id="tokyo", query="Tokyo?")
       ask_paris = ask_weather(task_id="paris", query="Paris?")
       _ = ask_tokyo >> ask_paris
       wf.execute()
   ```

2. **Channel Override Test**
   ```python
   # Bound param should override channel
   channel.set("query", "fallback")
   task = ask_weather(task_id="test", query="explicit")
   # Should use "explicit", not "fallback"
   ```

3. **LLM Integration Example**
   - Verify `examples/11_llm_integration/pydantic_agent_with_tools.py` works
   - Multiple agent calls with different queries

## Backward Compatibility

**Impact**: Minimal, no breaking changes

**Existing Behavior Preserved**:
- Tasks without parameters: `task = my_task()` → auto-generated task_id
- Direct execution: `task.run()` → unchanged
- Workflow operators: `task_a >> task_b` → unchanged
- Channel resolution: Existing `resolve_keyword_args` → still works

**New Behavior**:
- With `task_id`: Creates new instance (previously returned `self`)
- Auto-generated task_id: New feature, doesn't affect existing code

## Example Usage

### Before (Current, Broken)
```python
@task
def ask_weather(ctx, query: str):
    result = agent.run(query)
    print(result)

# Problem: All variables point to same instance
ask_tokyo = ask_weather(task_id="tokyo", query="Tokyo?")
ask_paris = ask_weather(task_id="paris", query="Paris?")
# ask_tokyo.task_id == ask_paris.task_id (broken!)
```

### After (Fixed)
```python
@task
def ask_weather(ctx, query: str):
    result = agent.run(query)
    print(result)

# Solution: Creates separate instances with bound parameters
ask_tokyo = ask_weather(task_id="tokyo", query="Tokyo?")
ask_paris = ask_weather(task_id="paris", query="Paris?")

with workflow("weather") as wf:
    _ = ask_tokyo >> ask_paris
    wf.execute()
    # tokyo executes with query="Tokyo?"
    # paris executes with query="Paris?"
```

### Auto-generated task_id
```python
@task
def process_data(ctx, value: int):
    return value * 2

# Auto-generates task_id like "process_data_a3f2b9c1"
task1 = process_data(value=10)
task2 = process_data(value=20)
# Each has unique task_id
```

## Files to Modify

1. **`graflow/core/task.py`**
   - `TaskWrapper.__init__()`: Initialize `_bound_kwargs`
   - `TaskWrapper.__call__()`: Add instance creation logic
   - `TaskWrapper.run()`: Update parameter merge logic

2. **`tests/core/test_task.py`** (new tests)
   - Test instance creation
   - Test parameter binding
   - Test priority order
   - Test auto-generated task_id

3. **`tests/integration/test_multi_instance.py`** (new file)
   - Integration tests for multi-instance workflows

4. **`examples/11_llm_integration/pydantic_agent_with_tools.py`**
   - Verify existing example works with new implementation

## Success Criteria

✅ Multiple instances can be created from same `@task` function
✅ Each instance has unique `task_id`
✅ Bound parameters are stored and used at execution time
✅ Parameter priority: channel < bound < injection
✅ Auto-generated task_id works when not specified
✅ Lazy registration works correctly
✅ Serialization preserves bound parameters
✅ Existing tests pass without modification
✅ `examples/11_llm_integration/pydantic_agent_with_tools.py` works correctly

## Future Enhancements (Out of Scope)

- Support for positional arg binding (if needed)
- Partial parameter binding with `functools.partial`-like syntax
- Task instance cloning: `task2 = task1.clone(task_id="new_id", param=new_value)`
- Parameter validation at creation time


## Key Features Working ✅ 

1. **Multiple Instances from Single @task:**
   ```python
   @task
   def ask_weather(query: str) -> str:
       return f"Weather for {query}"

   ask_tokyo = ask_weather(task_id="tokyo", query="Tokyo")
   ask_paris = ask_weather(task_id="paris", query="Paris")
   # Creates two separate TaskWrapper instances
   ```

2. **Auto-generated Task IDs:**
   ```python
   task1 = ask_weather(query="Tokyo")  # task_id: ask_weather_a3f2b9c1
   task2 = ask_weather(query="Paris")  # task_id: ask_weather_b7e8f4d2
   ```

3. **Parameter Priority:** Channel < Bound < Injection
   ```python
   # Bound parameters override channel values
   with workflow("test") as wf:
       task = process(task_id="test", value=10)  # value=10 is bound
       wf.execute(task.task_id, initial_channel={"value": 100})
   # Uses value=10, not value=100
   ```

4. **Initial Channel Values:**
   ```python
   with workflow("test") as wf:
       task = process(task_id="test")
       wf.execute(task.task_id, initial_channel={"value": 42})
   # Channel is pre-populated with value=42
   ```

- ✅ Multiple instances can be created from same `@task` function
- ✅ Each instance has unique `task_id`
- ✅ Bound parameters are stored and used at execution time
- ✅ Parameter priority: channel < bound < injection
- ✅ Auto-generated task_id works when not specified
- ✅ Lazy registration works correctly
- ✅ Serialization preserves bound parameters
- ✅ All new unit tests pass (15/15)
- ✅ All new integration tests pass (9/9)
- ✅ Initial channel parameter added to workflow execution

