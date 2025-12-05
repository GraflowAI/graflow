# Workflow Control Scenario Tests

## Overview

The `test_workflow_control_scenarios.py` file contains comprehensive scenario tests for Graflow's workflow control features:
- `context.terminate_workflow(message)` - Normal early exit
- `context.cancel_workflow(message)` - Abnormal exit with exception

## Test Coverage

### ✅ 8 Scenario Tests (All Passing)

#### 1. **test_cache_hit_terminates_workflow**
**Scenario**: Cache hit allows early termination, skipping expensive operations.

**Flow**: check_cache → (terminate) → ❌ fetch_data → ❌ transform_data → ❌ load_data

**Verifies**:
- Only `check_cache` executes when cache hit
- `check_cache` is marked as completed
- Downstream tasks (fetch, transform, load) are skipped

#### 2. **test_cache_miss_executes_full_pipeline**
**Scenario**: Cache miss leads to full pipeline execution.

**Flow**: check_cache → fetch_data → transform_data

**Verifies**:
- All tasks execute when cache miss
- Normal workflow completion

#### 3. **test_data_validation_failure_cancels_workflow**
**Scenario**: Invalid data causes workflow cancellation.

**Flow**: validate_input → (cancel) → ❌ process_data → ❌ store_result

**Verifies**:
- `GraflowWorkflowCanceledError` is raised
- `validate_input` is NOT marked as completed (failure)
- Downstream tasks are not executed
- Exception contains task_id and message

#### 4. **test_data_validation_success_continues_workflow**
**Scenario**: Valid data allows workflow to continue.

**Flow**: validate_input → process_data → store_result

**Verifies**:
- All tasks execute when validation passes
- Normal workflow completion

#### 5. **test_conditional_workflow_early_completion**
**Scenario**: Workflow completes early when condition is met.

**Test 1 (COMPLETED status)**:
- Flow: check_status → (terminate) → ❌ process_step1 → ❌ process_step2
- Only check_status executes

**Test 2 (PENDING status)**:
- Flow: check_status → process_step1 → process_step2
- All tasks execute

**Verifies**:
- Conditional early termination based on channel state
- Normal flow when condition not met

#### 6. **test_parallel_tasks_with_termination**
**Scenario**: One branch terminates while others would continue.

**Flow**: start_task → (branch_a | branch_b) → ❌ final_task
- branch_a calls terminate_workflow
- final_task should not execute

**Verifies**:
- Termination works with parallel execution
- Successors after parallel group are skipped

#### 7. **test_error_detection_cancels_workflow**
**Scenario**: Critical error detected mid-workflow causes cancellation.

**Flow**: prepare_data → check_integrity → (cancel) → ❌ process_data → ❌ finalize

**Verifies**:
- `prepare_data` completes before cancellation
- `check_integrity` is NOT marked as completed
- Exception raised with appropriate message
- Downstream tasks not executed

#### 8. **test_mixed_control_flow_priority**
**Scenario**: Test control flow priority (cancel > terminate > goto > normal).

**Test 1 (Cancel priority)**:
- Task calls both cancel_workflow and terminate_workflow
- Cancel takes effect (higher priority)
- Task NOT marked as completed
- Exception raised

**Test 2 (Terminate when no cancel)**:
- Task calls terminate_workflow only
- Task IS marked as completed
- Successors skipped
- No exception

**Verifies**:
- Control flow priority: cancel > terminate > goto > normal
- Correct completion semantics for each type

## Key Behaviors Verified

### ✅ terminate_workflow (Normal Exit)
- Current task IS marked as completed ✓
- Workflow exits normally (no exception) ✓
- Successor tasks are skipped ✓
- Use case: Cache hit, early completion, condition met ✓

### ✅ cancel_workflow (Abnormal Exit)
- Current task is NOT marked as completed ✓
- GraflowWorkflowCanceledError is raised ✓
- Successor tasks are skipped ✓
- Exception contains task_id and message ✓
- Use case: Validation failure, critical error, integrity check failure ✓

### ✅ Control Flow Priority
```
1. cancel_workflow    ← Highest priority, task not completed, exception
2. terminate_workflow ← Task completed, normal exit
3. goto_called        ← Task completed, successors skipped
4. normal flow        ← Task completed, successors executed
```

## Real-World Use Cases

### 1. **ETL Pipeline with Cache**
```python
check_cache → [cache hit: terminate] → fetch_data → transform → load
              [cache miss: continue]
```

### 2. **Data Validation Pipeline**
```python
validate → [invalid: cancel] → process → store
           [valid: continue]
```

### 3. **Multi-Stage Processing**
```python
prepare → check_integrity → [fail: cancel] → process → finalize
                            [pass: continue]
```

### 4. **Conditional Workflow**
```python
check_status → [completed: terminate] → step1 → step2
               [pending: continue]
```

### 5. **Error Detection**
```python
prepare → detect_error → [error: cancel] → process → finalize
                         [ok: continue]
```

## Test Execution

```bash
# Run scenario tests
PYTHONPATH=. uv run pytest tests/scenario/test_workflow_control_scenarios.py -v

# Run with verbose output
PYTHONPATH=. uv run pytest tests/scenario/test_workflow_control_scenarios.py -v --tb=short

# Run specific test
PYTHONPATH=. uv run pytest tests/scenario/test_workflow_control_scenarios.py::test_cache_hit_terminates_workflow -v
```

## Related Files

- **Implementation**: `graflow/core/engine.py` (lines 145-190)
- **Context API**: `graflow/core/context.py` (lines 428-498)
- **Exception**: `graflow/exceptions.py` (lines 94-113)
- **Documentation**: `docs/state_machine_execution.md` (patterns 5-8)
- **Unit Tests**: `test_workflow_control.py` (root directory)

## Test Results

```
✅ All 8 scenario tests passing
✅ Control flow semantics verified
✅ Exception handling validated
✅ Task completion tracking correct
✅ Real-world use cases covered
```

## Version

- **Created**: 2025-12-05
- **Graflow Version**: 1.3+
- **Test Framework**: pytest
- **Python**: 3.11+
