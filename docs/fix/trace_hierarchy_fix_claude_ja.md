# トレースの親子関係追跡の問題と改善案

## 問題の概要

`_runtime_graph`のトレースが呼び出しの上下関係（親子関係）をうまくトレースできていない。具体的には、以下の問題が発生している：

1. ParallelGroupとそのメンバータスクの親子関係が正しく記録されない
2. ネストされたタスクの親子関係が正しく追跡されない
3. `parent_task_id = context.current_task_id`が意図した値を返さない場合がある

## 問題の詳細分析

### 1. トレースフックの呼び出しフロー

#### 現在の実装

**`context.py`の`executing_task`メソッド（1193-1228行目）：**

```python
@contextmanager
def executing_task(self, task: Executable):
    task_ctx = self.create_task_context(task.task_id)

    # Call tracer hook: task start (before pushing to stack)
    # This ensures current_task_id points to parent task
    self.tracer.on_task_start(task, self)  # ← ここで親タスクIDを取得

    # Push task context to stack after tracer hook
    self.push_task_context(task_ctx)  # ← ここで新しいタスクをスタックにpush

    error: Optional[Exception] = None

    try:
        task.set_execution_context(self)
        yield task_ctx
    except Exception as e:
        error = e
        raise
    finally:
        self.tracer.on_task_end(task, self, result=None, error=error)
        self.pop_task_context()
```

**`base.py`の`on_task_start`実装（313-324行目）：**

```python
def on_task_start(self, task: Executable, context: ExecutionContext) -> None:
    parent_task_id = None
    if hasattr(context, 'current_task_id') and context.current_task_id:
        parent_task_id = context.current_task_id  # ← スタックのトップのタスクID

    self.span_start(
        task.task_id,
        parent_name=parent_task_id,
        metadata={
            "task_type": type(task).__name__,
            "handler_type": getattr(task, 'handler_type', 'direct')
        }
    )
```

この実装自体は理論的には正しい：
- `on_task_start`が呼ばれる時点では、新しいタスクはまだスタックにpushされていない
- したがって、`current_task_id`は親タスク（スタックのトップ）を指す
- `span_start`で親子関係のエッジが作成される

#### 問題点

しかし、実際には以下の問題が発生する：

### 2. ParallelGroupの実行フローにおける問題

**`ParallelGroup.run()`メソッド（task.py:541-563行目）：**

```python
def run(self) -> Any:
    context = self.get_execution_context()

    for task in self.tasks:
        task.set_execution_context(context)

    # Extract policy configuration
    policy = self._execution_config.get("policy", "strict")
    backend = self._execution_config.get("backend")
    backend_config = self._execution_config.get("backend_config", {})

    # GroupExecutor is stateless - call static method directly
    GroupExecutor.execute_parallel_group(
        self.task_id,
        self.tasks,
        context,
        backend=backend,
        backend_config=backend_config,
        policy=policy,
    )
```

**問題：**
1. **トレースフックが呼ばれていない**
   - `on_parallel_group_start`が呼ばれていない
   - メンバータスクが実行される前に親グループとの関係が記録されない

2. **`executing_task`コンテキストが使われていない**
   - ParallelGroupの`run()`が呼ばれる際、`with context.executing_task(self)`が使われていない
   - そのため、`current_task_id`がParallelGroupのIDに設定されない

**`GroupExecutor.direct_execute`（executor.py:118-179行目）：**

```python
@staticmethod
def direct_execute(
    group_id: str,
    tasks: List[Executable],
    execution_context: ExecutionContext,
    policy_instance: GroupExecutionPolicy
) -> None:
    engine = WorkflowEngine()
    results: Dict[str, TaskResult] = {}

    for task in tasks:
        try:
            # Execute single task via unified engine
            engine.execute(execution_context, start_task_id=task.task_id)
        except Exception as e:
            # エラーハンドリング...
```

**問題：**
1. **`current_task_id`の不一致**
   - 各メンバータスクが`engine.execute(execution_context, start_task_id=task.task_id)`で実行される
   - しかし、この時点で`execution_context.current_task_id`はParallelGroupを指していない
   - したがって、`on_task_start`で取得される`parent_task_id`が間違った値になる

2. **同じcontextの共有**
   - すべてのメンバータスクが同じ`execution_context`を共有
   - スタック（`_task_execution_stack`）が正しく管理されない

### 3. エッジ追加のタイミング問題

**`base.py`の`span_start`（106-134行目）：**

```python
def span_start(
    self,
    name: str,
    parent_name: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    if self.enable_runtime_graph:
        self._add_node_to_runtime_graph(
            name,
            status="running",
            metadata={"type": "span", **(metadata or {})}
        )
        if parent_name:
            self._add_edge_to_runtime_graph(parent_name, name, relation="parent-child")
```

**`_add_edge_to_runtime_graph`（620-636行目）：**

```python
def _add_edge_to_runtime_graph(
    self,
    parent_id: str,
    child_id: str,
    relation: str = "parent-child"
) -> None:
    if self._runtime_graph is not None:
        if parent_id in self._runtime_graph and child_id in self._runtime_graph:
            self._runtime_graph.add_edge(parent_id, child_id, relation=relation)
```

**問題：**
1. **両方のノードが存在する必要がある**
   - エッジを追加するには、親ノードと子ノードの両方がグラフに存在する必要がある
   - `span_start`では子ノードを追加した後にエッジを追加する
   - しかし、親ノードがまだ追加されていない場合、エッジは作成されない

2. **`on_parallel_group_start`でも同様の問題**
   - `on_parallel_group_start`（352-382行目）でも、両方のノードが存在することをチェック
   - メンバータスクのノードがまだ追加されていない場合、エッジは作成されない

```python
def on_parallel_group_start(
    self,
    group_id: str,
    member_ids: List[str],
    context: ExecutionContext
) -> None:
    if self.enable_runtime_graph and self._runtime_graph is not None:
        for member_id in member_ids:
            # Only add edge if both nodes exist
            if group_id in self._runtime_graph and member_id in self._runtime_graph:
                self._add_edge_to_runtime_graph(
                    group_id,
                    member_id,
                    relation="parallel-member"
                )
```

### 4. 並列実行時のスレッドセーフティ問題

**ThreadingCoordinatorの場合：**

複数のスレッドが同じ`execution_context`を共有するが、`_task_execution_stack`はスレッドセーフではない。これにより：

1. スタックの状態が不正確になる可能性
2. `current_task_id`が間違った値を返す可能性
3. 親子関係が正しく追跡されない

## 具体的な改善案

### 改善案1: ParallelGroup.run()でトレースフックを呼ぶ

**場所：** `graflow/core/task.py`の`ParallelGroup.run()`メソッド

**変更内容：**

```python
def run(self) -> Any:
    """Execute all tasks in this parallel group."""
    context = self.get_execution_context()

    for task in self.tasks:
        # Set execution context for each task
        task.set_execution_context(context)

    # Extract policy configuration
    policy = self._execution_config.get("policy", "strict")
    backend = self._execution_config.get("backend")
    backend_config = self._execution_config.get("backend_config", {})

    # ========== 追加: トレースフック呼び出し ==========
    # Tracer: ParallelGroup start
    member_ids = [task.task_id for task in self.tasks]
    context.tracer.on_parallel_group_start(self.task_id, member_ids, context)
    # ==============================================

    try:
        # GroupExecutor is stateless - call static method directly
        GroupExecutor.execute_parallel_group(
            self.task_id,
            self.tasks,
            context,
            backend=backend,
            backend_config=backend_config,
            policy=policy,
        )
    finally:
        # ========== 追加: トレースフック呼び出し ==========
        # Tracer: ParallelGroup end
        results = {task.task_id: context.get_result(task.task_id) for task in self.tasks}
        context.tracer.on_parallel_group_end(self.task_id, member_ids, context, results=results)
        # ==============================================
```

**効果：**
- ParallelGroupとメンバータスクの関係がトレースに記録される
- `on_parallel_group_start`で並列実行開始が明確に記録される

### 改善案2: DirectTaskHandlerでParallelGroupを特別扱い

**場所：** `graflow/core/handlers/direct.py`の`DirectTaskHandler.execute_task()`

**変更内容：**

```python
def execute_task(self, task: Executable, context: ExecutionContext) -> Any:
    """Execute task and store result in context.

    Args:
        task: Executable task to execute
        context: Execution context
    """
    task_id = task.task_id
    logger.debug(f"[DirectTaskHandler] Executing task {task_id}")

    try:
        # ========== 追加: ParallelGroupの場合の特別処理 ==========
        from graflow.core.task import ParallelGroup
        if isinstance(task, ParallelGroup):
            # ParallelGroupの場合、executing_taskコンテキストは既に設定されている
            # （engine.executeで設定済み）
            # ただし、run()内でトレースフックが呼ばれる
            result = task.run()
        else:
            # 通常のタスク実行
            result = task.run()
        # =====================================================

        # Store result in context (including None)
        context.set_result(task_id, result)
        logger.debug(f"[DirectTaskHandler] Task {task_id} completed successfully")
        return result
    except Exception as e:
        # Store exception in context
        context.set_result(task_id, e)
        logger.debug(f"[DirectTaskHandler] Task {task_id} failed: {e}")
        raise
```

**注意：** この変更は最小限ですが、実際にはParallelGroupの処理は既存のフローで問題ありません。重要なのは、`engine.execute`が`executing_task`コンテキストを設定することです。

### 改善案3: GroupExecutor.direct_executeでexecuting_taskコンテキストを明示的に設定

**場所：** `graflow/coordination/executor.py`の`GroupExecutor.direct_execute()`

**現在の問題：**
- 各メンバータスクが`engine.execute(execution_context, start_task_id=task.task_id)`で実行される
- しかし、この時点で`execution_context.current_task_id`はParallelGroupを指していない

**変更内容：**

```python
@staticmethod
def direct_execute(
    group_id: str,
    tasks: List[Executable],
    execution_context: ExecutionContext,
    policy_instance: GroupExecutionPolicy
) -> None:
    """Execute tasks using unified WorkflowEngine for consistency."""
    from graflow.core.handler import TaskResult

    task_ids = [task.task_id for task in tasks]
    logger.info(
        "Running parallel group: %s with %d tasks",
        group_id,
        len(tasks),
        extra={"group_id": group_id, "task_ids": task_ids}
    )

    # Use unified WorkflowEngine for each task
    from graflow.core.engine import WorkflowEngine

    engine = WorkflowEngine()
    results: Dict[str, TaskResult] = {}

    # ========== 変更: ParallelGroupのコンテキストをスタックに設定 ==========
    # ParallelGroupをcurrent_task_idとして設定するため、
    # ダミーのTaskExecutionContextを作成してスタックにpush
    from graflow.core.task import Task

    # ParallelGroupのタスクコンテキストを作成（既に存在する可能性があるので確認）
    if group_id not in execution_context._task_contexts:
        group_task_ctx = execution_context.create_task_context(group_id)
        execution_context.push_task_context(group_task_ctx)
        should_pop = True
    else:
        # 既に存在する場合は、スタックに再pushしない
        should_pop = False
    # ====================================================================

    try:
        for task in tasks:
            logger.debug("Executing task directly: %s", task.task_id, extra={"group_id": group_id})
            success = True
            error_message = None
            start_time = time.time()
            try:
                # Execute single task via unified engine
                # この時点で、current_task_idはgroup_idを指す
                engine.execute(execution_context, start_task_id=task.task_id)
            except Exception as e:
                logger.error(
                    "Task failed in parallel group: %s",
                    task.task_id,
                    exc_info=True,
                    extra={"group_id": group_id, "error": str(e)}
                )
                success = False
                error_message = str(e)

            results[task.task_id] = TaskResult(
                task_id=task.task_id,
                success=success,
                error_message=error_message,
                duration=time.time() - start_time,
                timestamp=time.time()
            )
    finally:
        # ========== 追加: コンテキストのクリーンアップ ==========
        if should_pop:
            execution_context.pop_task_context()
        # =====================================================

    logger.info(
        "Direct parallel group completed: %s",
        group_id,
        extra={
            "group_id": group_id,
            "task_count": len(tasks),
            "success_count": sum(1 for r in results.values() if r.success)
        }
    )

    # Use GroupExecutionPolicy directly instead of handler
    policy_instance.on_group_finished(group_id, tasks, results, execution_context)
```

**効果：**
- 各メンバータスクが実行される際、`current_task_id`がParallelGroupのIDを指す
- `on_task_start`で取得される`parent_task_id`が正しくParallelGroupのIDになる
- 親子関係が正しくトレースに記録される

### 改善案4: ThreadingCoordinatorでbranch contextを使う

**場所：** `graflow/coordination/threading_coordinator.py`

**変更内容：**

```python
def execute_group(
    self,
    group_id: str,
    tasks: List[Executable],
    exec_context: ExecutionContext,
    policy_instance: GroupExecutionPolicy
) -> None:
    """Execute tasks in parallel using threading with branch contexts."""
    task_ids = [task.task_id for task in tasks]
    logger.info(
        "Running parallel group: %s with %d tasks (threading)",
        group_id,
        len(tasks),
        extra={"group_id": group_id, "task_ids": task_ids}
    )

    results: Dict[str, TaskResult] = {}

    def execute_task_with_branch_context(task: Executable, branch_id: str) -> TaskResult:
        """Execute task in a separate branch context."""
        # ========== 追加: branch contextの作成 ==========
        branch_context = exec_context.create_branch_context(branch_id)
        # ==============================================

        success = True
        error_message = None
        start_time = time.time()

        try:
            from graflow.core.engine import WorkflowEngine
            engine = WorkflowEngine()

            # ========== 変更: branch contextを使う ==========
            engine.execute(branch_context, start_task_id=task.task_id)
            # ==============================================
        except Exception as e:
            logger.error(
                "Task failed in parallel group: %s",
                task.task_id,
                exc_info=True,
                extra={"group_id": group_id, "error": str(e)}
            )
            success = False
            error_message = str(e)
        finally:
            # ========== 追加: branch contextの結果をマージ ==========
            exec_context.merge_results(branch_context)
            # ==============================================

        return TaskResult(
            task_id=task.task_id,
            success=success,
            error_message=error_message,
            duration=time.time() - start_time,
            timestamp=time.time()
        )

    with ThreadPoolExecutor(max_workers=self._thread_count) as executor:
        futures = {
            executor.submit(execute_task_with_branch_context, task, f"{group_id}_member_{i}"): task
            for i, task in enumerate(tasks)
        }

        for future in as_completed(futures):
            task = futures[future]
            try:
                result = future.result()
                results[task.task_id] = result
            except Exception as e:
                logger.error(
                    "Unexpected error in parallel task: %s",
                    task.task_id,
                    exc_info=True
                )
                results[task.task_id] = TaskResult(
                    task_id=task.task_id,
                    success=False,
                    error_message=str(e),
                    duration=0,
                    timestamp=time.time()
                )

    logger.info(
        "Threading parallel group completed: %s",
        group_id,
        extra={
            "group_id": group_id,
            "task_count": len(tasks),
            "success_count": sum(1 for r in results.values() if r.success)
        }
    )

    policy_instance.on_group_finished(group_id, tasks, results, exec_context)
```

**効果：**
- 各スレッドが独立した`branch_context`を持つ
- `_task_execution_stack`の競合が解消される
- トレースIDは共有されるが、session_idは個別になる
- 並列実行時のトレースが正しく記録される

### 改善案5: エッジ追加の遅延処理

**場所：** `graflow/trace/base.py`

**変更内容：**

```python
class Tracer(ABC):
    def __init__(self, enable_runtime_graph: bool = True):
        self.enable_runtime_graph = enable_runtime_graph
        self._runtime_graph: Optional[nx.DiGraph] = (
            nx.DiGraph() if enable_runtime_graph else None
        )
        self._execution_order: List[str] = []
        self._current_trace_id: Optional[str] = None
        self._span_stack: List[str] = []

        # ========== 追加: 保留中のエッジを追跡 ==========
        self._pending_edges: List[tuple[str, str, str]] = []  # (parent_id, child_id, relation)
        # ==============================================

    def _add_edge_to_runtime_graph(
        self,
        parent_id: str,
        child_id: str,
        relation: str = "parent-child"
    ) -> None:
        """Add edge to runtime graph (internal helper).

        Args:
            parent_id: Parent node ID
            child_id: Child node ID
            relation: Edge relation type ("parent-child", "successor", etc.)
        """
        if self._runtime_graph is not None:
            if parent_id in self._runtime_graph and child_id in self._runtime_graph:
                self._runtime_graph.add_edge(parent_id, child_id, relation=relation)
            else:
                # ========== 変更: ノードが存在しない場合は保留 ==========
                self._pending_edges.append((parent_id, child_id, relation))
                # =====================================================

    def _add_node_to_runtime_graph(
        self,
        node_id: str,
        status: str = "running",
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add node to runtime graph (internal helper).

        Args:
            node_id: Node identifier (task_id)
            status: Initial status ("running", "completed", "failed")
            metadata: Optional metadata
        """
        if self._runtime_graph is not None:
            self._runtime_graph.add_node(
                node_id,
                status=status,
                start_time=datetime.now(),
                metadata=metadata or {}
            )
            self._execution_order.append(node_id)

            # ========== 追加: 保留中のエッジを処理 ==========
            self._process_pending_edges(node_id)
            # ==============================================

    def _process_pending_edges(self, node_id: str) -> None:
        """Process pending edges for the newly added node.

        Args:
            node_id: Newly added node ID
        """
        if self._runtime_graph is None:
            return

        # 保留中のエッジをチェックして、追加可能なものを追加
        remaining_edges = []
        for parent_id, child_id, relation in self._pending_edges:
            if parent_id == node_id or child_id == node_id:
                # このノードが関係するエッジ
                if parent_id in self._runtime_graph and child_id in self._runtime_graph:
                    # 両方のノードが存在するので、エッジを追加
                    self._runtime_graph.add_edge(parent_id, child_id, relation=relation)
                else:
                    # まだ追加できないので保留継続
                    remaining_edges.append((parent_id, child_id, relation))
            else:
                # このノードとは関係ないので保留継続
                remaining_edges.append((parent_id, child_id, relation))

        self._pending_edges = remaining_edges
```

**効果：**
- ノードが追加される前にエッジを追加しようとした場合、保留リストに保存
- ノードが追加された際に、保留中のエッジを再処理
- タイミングに関わらず、最終的にすべてのエッジが追加される

## 推奨される実装順序

1. **改善案1（ParallelGroup.run()でトレースフック）** - 最も影響が大きく、実装も簡単
2. **改善案3（direct_executeでexecuting_taskコンテキスト設定）** - 親子関係の正確性を向上
3. **改善案5（エッジ追加の遅延処理）** - タイミング問題を根本的に解決
4. **改善案4（ThreadingCoordinatorでbranch context）** - 並列実行の正確性を向上

改善案2（DirectTaskHandler）は必須ではありませんが、コードの明確性を向上させます。

## テスト計画

各改善案の実装後、以下のテストを実施すべきです：

1. **単純なParallelGroupのテスト**
   - ParallelGroupとメンバータスクの親子関係が正しく記録されることを確認

2. **ネストされたParallelGroupのテスト**
   - ParallelGroup内のParallelGroupが正しくトレースされることを確認

3. **動的タスク生成のテスト**
   - `next_task()`や`next_iteration()`で生成されたタスクの親子関係を確認

4. **ThreadingCoordinatorのテスト**
   - 並列実行時のトレースが正しく記録されることを確認

5. **runtime_graphの検証**
   - `get_runtime_graph()`で取得したグラフの構造を検証
   - すべてのノードとエッジが正しく記録されているか確認

## まとめ

トレースの親子関係追跡の問題は、主に以下の原因によるものです：

1. **ParallelGroupの実行時にトレースフックが呼ばれていない**
2. **`current_task_id`がParallelGroupを指していない**
3. **エッジ追加のタイミング問題**
4. **並列実行時のスレッドセーフティ問題**

提案した改善案を順次実装することで、これらの問題を解決し、正確な親子関係のトレースを実現できます。
