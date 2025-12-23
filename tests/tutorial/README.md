# Tutorial Tests README

## Test Coverage

This directory contains comprehensive unit tests for the Tasks and Workflows Guide, covering all major features documented in the guide.

### Overview

**Total Tests:** 63 (across 3 test files)
**Pass Rate:** 100%
**Test Files:**
1. `test_tasks_and_workflows_guide.py` - Core workflow features (34 tests)
2. `test_llm_integration.py` - LLM client and agent injection (11 tests)
3. `test_hitl.py` - Human-in-the-Loop feedback (18 tests)

---

## File: `test_tasks_and_workflows_guide.py`

Comprehensive test file for core workflow features from `docs/tutorial/tasks_and_workflows_guide.md`.

**Test Statistics:**
- **Total Tests:** 34
- **Pass Rate:** 100%
- **Coverage:** Levels 1-9 of the guide (core features)

### Test Organization

The tests are organized by tutorial levels:

1. **Level 1: Your First Task** (4 tests)
   - Basic @task decorator
   - Custom task IDs via instances
   - .run() method with parameters
   - Default parameters

2. **Level 2: Your First Workflow** (1 test)
   - Simple workflow execution

3. **Level 3: Task Composition** (7 tests)
   - Sequential and parallel operators
   - chain() and parallel() helpers
   - Parallel group configuration
   - Execution policies (best_effort, AtLeastN, Critical)

4. **Level 4: Passing Parameters** (2 tests)
   - Channel communication
   - Partial parameter binding

5. **Level 5: Task Instances** (3 tests)
   - Task instances with parameters
   - Auto-generated task IDs
   - Unique task ID requirements

6. **Level 6: Channels and Context** (6 tests)
   - Basic channel operations
   - List operations (append/prepend)
   - TTL (time-to-live)
   - Typed channels
   - Context injection
   - Parameter priority

7. **Level 7: Execution Patterns** (5 tests)
   - Getting final results
   - Getting all results with ret_context
   - Auto start node detection
   - Manual start node specification
   - Result storage format

8. **Level 8: Complex Workflows** (2 tests)
   - Diamond pattern (split → parallel → merge)
   - Multi-instance pipeline

9. **Level 9: Dynamic Task Generation** (2 tests)
   - next_iteration() for convergence
   - cancel_workflow() for error handling

10. **Integration Tests** (2 tests)
    - Complex diamond with channels
    - Multi-stage ETL pipeline

---

## File: `test_llm_integration.py`

Tests for LLM client and agent injection features from the Tasks and Workflows Guide.

**Test Statistics:**
- **Total Tests:** 11
- **Pass Rate:** 100%
- **Coverage:** LLM client injection, LLM agent injection, integration patterns

### Test Organization

1. **TestLLMClientInjection** (4 tests)
   - Basic LLM client injection with `@task(inject_llm_client=True)`
   - Multiple tasks sharing LLM client
   - Combining LLM client with context injection
   - Accessing LLM client via context

2. **TestLLMAgentInjection** (4 tests)
   - Agent registration and injection with `@task(inject_llm_agent="name")`
   - Agent with tool calls
   - Multiple agents in workflow
   - Combining agent with context injection

3. **TestLLMIntegrationPatterns** (3 tests)
   - Multi-model scenario (using different models per task)
   - LLM with retry pattern
   - LLM response to channel communication

### Key Features Tested

- ✅ `@task(inject_llm_client=True)` - Direct LLM API calls
- ✅ `@task(inject_llm_agent="name")` - Agent with tools and ReAct loops
- ✅ Agent registration via `wf.register_llm_agent(name, agent_or_factory)`
- ✅ Factory pattern for agent creation
- ✅ Model overrides per task
- ✅ Combining LLM injection with context injection
- ✅ LLM response caching in channels
- ✅ Retry patterns with LLM calls

---

## File: `test_hitl.py`

Tests for Human-in-the-Loop feedback features using `ctx.request_feedback()`.

**Test Statistics:**
- **Total Tests:** 18
- **Pass Rate:** 100%
- **Coverage:** All feedback types, timeouts, channel integration, handlers

### Test Organization

1. **TestBasicApproval** (3 tests)
   - Approval request (approved)
   - Approval request (rejected)
   - Approval with reason field

2. **TestTextInput** (2 tests)
   - Basic text input
   - Text input with metadata

3. **TestSelection** (2 tests)
   - Basic selection from options
   - Selection with descriptive options

4. **TestChannelIntegration** (2 tests)
   - Auto-write feedback to channel
   - Manual channel write

5. **TestTimeoutBehavior** (2 tests)
   - Quick response within timeout
   - Timeout error handling

6. **TestWorkflowIntegration** (3 tests)
   - Multiple approval gates in pipeline
   - Rejected approval cancels workflow
   - Conditional branching with feedback

7. **TestFeedbackHandler** (1 test)
   - Custom feedback handler callbacks

8. **TestMultipleFeedbackTypes** (1 test)
   - Mixed feedback types in same workflow

9. **TestRealWorldPatterns** (2 tests)
   - Deployment approval pattern
   - Data validation pattern

### Key Features Tested

- ✅ `ctx.request_feedback()` - All feedback types (approval, text, selection)
- ✅ Timeout handling with `FeedbackTimeoutError`
- ✅ Channel integration (`write_to_channel`, `channel_key`)
- ✅ Custom metadata
- ✅ Feedback handlers (`on_request_created`, `on_response_received`, `on_request_timeout`)
- ✅ Workflow cancellation on rejection
- ✅ Conditional branching based on feedback
- ✅ Multiple feedback requests in pipelines
- ✅ Real-world deployment and validation patterns

---

## Running the Tests

### Run All Tutorial Tests
```bash
uv run pytest tests/tutorial/test_*.py -v
```

### Run Specific Test File
```bash
# Core workflows
uv run pytest tests/tutorial/test_tasks_and_workflows_guide.py -v

# LLM integration
uv run pytest tests/tutorial/test_llm_integration.py -v

# HITL feedback
uv run pytest tests/tutorial/test_hitl.py -v
```

### Run Specific Test Class
```bash
uv run pytest tests/tutorial/test_tasks_and_workflows_guide.py::TestLevel3TaskComposition -v
```

### Run Specific Test
```bash
uv run pytest tests/tutorial/test_tasks_and_workflows_guide.py::TestLevel1FirstTask::test_basic_task_decorator -v
```

---

## Key Testing Patterns

### 1. Execution Log Verification
Tests verify order of execution through append-only logs:
```python
execution_log = []

@task
def my_task():
    execution_log.append("Task executed")

assert "Task executed" in execution_log
```

### 2. Result Verification
Tests check task return values via `get_result()`:
```python
_, ctx = wf.execute(ret_context=True)
result = ctx.get_result("task_id")
assert result == expected_value
```

### 3. Channel Communication
Tests verify data passing between tasks:
```python
ctx.get_channel().set("key", "value")
value = ctx.get_channel().get("key")
```

### 4. Mocking for LLM and HITL
Tests use mocks to avoid requiring actual API keys or user interaction:
```python
mock_client = Mock(spec=LLMClient)
mock_client.completion_text.return_value = "Mocked response"
```

### 5. Threading for HITL
Tests use threading to simulate human responses:
```python
def provide_feedback_after_delay(ctx, approved=True):
    time.sleep(0.5)
    manager.provide_feedback(feedback_id, response)

thread = threading.Thread(target=provide_feedback_after_delay, daemon=True)
thread.start()
```

---

## Important Notes

### For Core Workflow Tests
- Some tests explicitly specify `start_node` to avoid graph compilation errors when using parallel groups
- Tests use `# type: ignore` comments to suppress type checker warnings for operator overloading
- All tests are independent and can run in any order
- Tests clean up after themselves (no persistent state)

### For LLM Integration Tests
- All tests use mocking to avoid requiring actual API keys
- Agent registration uses factory pattern: `lambda ctx: mock_agent`
- Type hints use `LLMAgent` for agent parameters
- Tests verify both direct injection and context access patterns

### For HITL Tests
- All tests use threading to simulate human responses
- Feedback responses provided after short delays (0.5-1.0 seconds)
- Custom handlers must use correct callback signatures
- Import `FeedbackTimeoutError` from `graflow.hitl.types`

---

## Test Coverage Summary

| Category | File | Tests | Features |
|----------|------|-------|----------|
| Core Workflows | `test_tasks_and_workflows_guide.py` | 34 | Tasks, workflows, composition, channels, context |
| LLM Integration | `test_llm_integration.py` | 11 | LLM client, agents, tools, multi-model |
| HITL Feedback | `test_hitl.py` | 18 | Approval, text, selection, timeout, handlers |
| **Total** | | **63** | **100% pass rate** |

---

## Future Enhancements

Potential areas for additional test coverage:
- **Distributed Execution**: Redis-based queue and channel tests
- **Docker Handlers**: Docker task execution tests
- **Checkpoint/Resume**: Checkpoint creation and resumption tests
- **Tracing**: Langfuse and custom tracer tests
- **Advanced Dynamic Tasks**: `next_task()` with `goto=True`, `terminate_workflow()`
- **Cycle Detection**: Workflow cycles and `CycleController` tests

---

## Contributing

When adding new tests:
1. Follow the existing organization pattern (test classes by feature)
2. Add meaningful docstrings explaining what each test verifies
3. Use mocking for external dependencies (APIs, databases, etc.)
4. Ensure tests are independent and can run in any order
5. Update this README with new test counts and coverage
6. Run `make format` and `make lint` before committing
7. Ensure all tests pass: `uv run pytest tests/tutorial/test_*.py -v`
