# ParallelGroup.with_execution() 実装設計書 v2

## 概要

ParallelGroupに`with_execution()`メソッドを追加し、グループごとに異なる実行バックエンド（DIRECT, THREADING, REDIS）を指定できるようにする。

**v2の主要な変更点：**
- `ExecutionContext.group_executor` を削除（GroupExecutorはステートレス）
- 各ParallelGroupが独自の設定を保持し、実行時に毎回GroupExecutorを作成
- **`TaskHandler` から group_policy 関連メソッドを削除**（不要な間接化を排除）
  - `set_group_policy()`, `get_group_policy()`, `on_group_finished()` を削除
  - `GroupExecutionPolicy` を直接 Coordinator に渡す
  - TaskHandler は `execute_task()` のみに責任を限定
- より明確でシンプルなアーキテクチャ

## 設計原則

1. **GroupExecutorはステートレス**: 毎回作成してもオーバーヘッドなし
2. **ParallelGroupが設定を保持**: 各グループが独自の実行設定を持つ
3. **ExecutionContextをシンプル化**: グローバル設定を削除
4. **TaskHandler から group_policy 関連を削除**: `GroupExecutionPolicy` を直接使用（不要な間接化を排除）
5. **責任分離**: TaskHandler = 個別タスク実行、GroupExecutionPolicy = グループ結果評価
6. **破壊的変更あり**: よりクリーンな設計のため、後方互換性は維持しない
7. **メソッドチェーン対応**: `set_group_name()`と同様にselfを返す

## アーキテクチャの比較

### v1（現在）: ExecutionContextにgroup_executorを保持

```
ExecutionContext
  ├─ group_executor: GroupExecutor  ← グローバル設定（オプション）
  └─ ...

ParallelGroup.run():
  executor = context.group_executor or GroupExecutor()
  executor.execute_parallel_group(...)
```

**問題点:**
- GroupExecutorはステートレスなのにcontextに保持する意味がない
- グローバル設定とローカル設定の優先順位が複雑
- ExecutionContextが肥大化

### v2（改訂版）: ParallelGroupが独立して設定を管理 + TaskHandler廃止

```
ExecutionContext
  └─ ... (group_executor削除)

ParallelGroup
  ├─ _execution_config: dict
  │    ├─ backend: Optional[CoordinationBackend]
  │    ├─ backend_config: dict
  │    └─ policy: str
  └─ run():
       # GroupExecutorはステートレスなのでクラスメソッドとして呼び出し
       GroupExecutor.execute_parallel_group(
           backend=self._execution_config["backend"],
           backend_config=self._execution_config["backend_config"],
           policy=self._execution_config["policy"],  # Policy を直接渡す
           ...
       )

GroupExecutor.execute_parallel_group():
    policy_instance = resolve_group_policy(policy)  # Policy インスタンス化
    coordinator = _create_coordinator(backend, config)
    coordinator.execute_group(..., policy_instance)  # Policy を直接渡す

Coordinator.execute_group(..., policy: GroupExecutionPolicy):
    # タスク実行 + 結果収集
    results = {...}
    # Policy を直接呼び出し（TaskHandler 不要）
    policy.on_group_finished(group_id, tasks, results, context)
```

**メリット:**
- シンプルで理解しやすい
- 各ParallelGroupが独立した設定を持つ
- ExecutionContextがよりシンプル
- GroupExecutorの責務が明確（ステートレスなユーティリティクラス）
- インスタンス化のオーバーヘッドなし
- **不要な間接化を排除**（TaskHandler を経由しない）
- **責任分離が明確**（Policy は結果評価のみ）

## 実装詳細

### 1. ExecutionContext から group_executor を削除

**変更前** (`graflow/core/context.py:301`):
```python
# Group execution
self.group_executor: GroupExecutor = parent_context.group_executor if parent_context else GroupExecutor()
```

**変更後**:
```python
# Group execution: 削除
# ParallelGroupが独自にGroupExecutorを作成するため不要
```

**影響箇所:**
- `ExecutionContext.__init__()` (line 301)
- `ExecutionContext.__getstate__()` (line 908) - group_executorのpop処理
- `ExecutionContext.__setstate__()` (line 982-983) - group_executorの復元処理

### 2. ParallelGroup.__init__() の変更

**現在の実装** (`graflow/core/task.py:355-367`):
```python
def __init__(self, tasks: list[Executable]) -> None:
    """Initialize a parallel group with a list of tasks."""
    super().__init__()
    self._task_id = self._get_group_name()
    self.tasks = list(tasks)

    # Execution configuration for with_execution()
    self._execution_config = {
        "backend": None,  # None = use context.group_executor or default
        "backend_config": {},
        "policy": "strict",
    }

    self._register_to_context()
```

**変更後:**
```python
def __init__(self, tasks: list[Executable]) -> None:
    """Initialize a parallel group with a list of tasks."""
    super().__init__()
    self._task_id = self._get_group_name()
    self.tasks = list(tasks)

    # Execution configuration for with_execution()
    self._execution_config = {
        "backend": None,  # None = use GroupExecutor.DEFAULT_BACKEND (THREADING)
        "backend_config": {},
        "policy": "strict",
    }

    self._register_to_context()
```

**変更内容:**
- コメントを更新: `backend: None` の意味を明確化

### 3. ParallelGroup.run() の変更

**現在の実装** (`graflow/core/task.py:445-467`):
```python
def run(self) -> Any:
    """Execute all tasks in this parallel group."""
    context = self.get_execution_context()

    executor = context.group_executor or GroupExecutor()

    for task in self.tasks:
        task.set_execution_context(context)

    policy = self._execution_config.get("policy", "strict")
    backend = self._execution_config.get("backend")
    backend_config = self._execution_config.get("backend_config", {})

    executor.execute_parallel_group(
        self.task_id,
        self.tasks,
        context,
        backend=backend,
        backend_config=backend_config,
        policy=policy,
    )
```

**変更後:**
```python
def run(self) -> Any:
    """Execute all tasks in this parallel group."""
    context = self.get_execution_context()

    for task in self.tasks:
        task.set_execution_context(context)

    policy = self._execution_config.get("policy", "strict")
    backend = self._execution_config.get("backend")
    backend_config = self._execution_config.get("backend_config", {})

    # GroupExecutorはステートレスなのでクラスメソッドとして直接呼び出し
    GroupExecutor.execute_parallel_group(
        self.task_id,
        self.tasks,
        context,
        backend=backend,
        backend_config=backend_config,
        policy=policy,
    )
```

**変更内容:**
- `context.group_executor` への参照を削除
- `GroupExecutor()` のインスタンス化を削除
- `GroupExecutor.execute_parallel_group()` をクラスメソッドとして直接呼び出し

### 4. with_execution() メソッド（変更なし）

**現在の実装** (`graflow/core/task.py:390-439`):
```python
def with_execution(
    self,
    backend: Optional[CoordinationBackend] = None,
    backend_config: Optional[dict] = None,
    policy: Union[str, GroupExecutionPolicy] = "strict",
) -> ParallelGroup:
    """Configure execution backend and group policy for this parallel group.

    Args:
        backend: Coordinator backend (DIRECT, THREADING, REDIS)
        backend_config: Backend-specific configuration
            - THREADING: {"thread_count": int}
            - REDIS: {"key_prefix": str, "host": str, "port": int, "db": int}
        policy: Group execution policy name or instance

    Returns:
        Self (for method chaining)

    Examples:
        # Default backend (THREADING)
        (task_a | task_b).with_execution()

        # Redis backend with custom key prefix
        (task_a | task_b).with_execution(
            backend=CoordinationBackend.REDIS,
            backend_config={"key_prefix": "my_workflow"}
        )

        # Threading with custom thread count
        (task_a | task_b).with_execution(
            backend=CoordinationBackend.THREADING,
            backend_config={"thread_count": 4}
        )

        # Best-effort policy
        (task_a | task_b | task_c).with_execution(policy="best_effort")

    Note:
        Individual task execution handlers should be set at task level using @task(handler="...").
        The ``policy`` parameter controls parallel group success/failure criteria.
    """
    if backend is not None:
        self._execution_config["backend"] = backend

    if backend_config is not None:
        self._execution_config["backend_config"].update(backend_config)

    from graflow.core.handlers.group_policy import canonicalize_group_policy

    self._execution_config["policy"] = canonicalize_group_policy(policy)

    return self
```

**変更内容:**
- ドキュメント文字列の例を更新（Redis backend_config に `key_prefix` を追加）
- 実装ロジックは変更なし

### 5. GroupExecutor.execute_parallel_group() を @staticmethod に変更 + TaskHandler を削除

**現在の実装** (`graflow/coordination/executor.py:72-111`):
```python
def execute_parallel_group(
    self,
    group_id: str,
    tasks: List['Executable'],
    exec_context: 'ExecutionContext',
    *,
    backend: Optional[Union[str, CoordinationBackend]] = None,
    backend_config: Optional[Dict[str, Any]] = None,
    policy: Union[str, 'GroupExecutionPolicy'] = "strict",
) -> None:
    """Execute parallel group with a configurable group policy."""
    from graflow.core.handlers.direct import DirectTaskHandler
    from graflow.core.handlers.group_policy import resolve_group_policy

    policy_instance = resolve_group_policy(policy)
    handler = DirectTaskHandler()
    handler.set_group_policy(policy_instance)

    resolved_backend = self._resolve_backend(backend)
    # ...
    coordinator.execute_group(group_id, tasks, exec_context, handler)
```

**変更後:**
```python
@staticmethod
def execute_parallel_group(
    group_id: str,
    tasks: List['Executable'],
    exec_context: 'ExecutionContext',
    *,
    backend: Optional[Union[str, CoordinationBackend]] = None,
    backend_config: Optional[Dict[str, Any]] = None,
    policy: Union[str, 'GroupExecutionPolicy'] = "strict",
) -> None:
    """Execute parallel group with a configurable group policy.

    Args:
        group_id: Parallel group identifier
        tasks: List of tasks to execute
        exec_context: Execution context
        backend: Coordination backend (name or CoordinationBackend)
        backend_config: Backend-specific configuration
        policy: Group execution policy (name or instance)
    """
    from graflow.core.handlers.group_policy import resolve_group_policy

    # Policy を直接インスタンス化（TaskHandler 不要）
    policy_instance = resolve_group_policy(policy)

    resolved_backend = GroupExecutor._resolve_backend(backend)

    # Merge context config with backend config
    # backend_config takes precedence over context config
    context_config = getattr(exec_context, 'config', {})
    config = {**context_config, **(backend_config or {})}

    if resolved_backend == CoordinationBackend.DIRECT:
        return GroupExecutor.direct_execute(group_id, tasks, exec_context, policy_instance)

    coordinator = GroupExecutor._create_coordinator(resolved_backend, config, exec_context)
    # Policy を直接渡す（handler ではなく）
    coordinator.execute_group(group_id, tasks, exec_context, policy_instance)
```

**変更内容:**
- `@staticmethod` デコレータを追加
- `self` パラメータを削除
- **`DirectTaskHandler` を削除**
- **`policy_instance` を直接 coordinator に渡す**
- `self._resolve_backend()` → `GroupExecutor._resolve_backend()`
- `self.direct_execute()` → `GroupExecutor.direct_execute()`
- `self._create_coordinator()` → `GroupExecutor._create_coordinator()`

### 6. GroupExecutor.direct_execute() を @staticmethod に変更 + policy を使用

**現在の実装** (`graflow/coordination/executor.py:113-157`):
```python
def direct_execute(
    self,
    group_id: str,
    tasks: List['Executable'],
    execution_context: 'ExecutionContext',
    handler: 'TaskHandler'
) -> None:
    """Execute tasks using unified WorkflowEngine for consistency."""
    # ...
    handler.on_group_finished(group_id, tasks, results, execution_context)
```

**変更後:**
```python
@staticmethod
def direct_execute(
    group_id: str,
    tasks: List['Executable'],
    execution_context: 'ExecutionContext',
    policy: 'GroupExecutionPolicy'
) -> None:
    """Execute tasks using unified WorkflowEngine for consistency."""
    import time

    from graflow.core.handler import TaskResult

    print(f"Running parallel group: {group_id}")
    print(f"  Direct tasks: {[task.task_id for task in tasks]}")

    from graflow.core.engine import WorkflowEngine

    engine = WorkflowEngine()
    results: Dict[str, TaskResult] = {}

    for task in tasks:
        print(f"  - Executing directly: {task.task_id}")
        success = True
        error_message = None
        start_time = time.time()
        try:
            engine.execute(execution_context, start_task_id=task.task_id)
        except Exception as e:
            print(f"    Task {task.task_id} failed: {e}")
            success = False
            error_message = str(e)

        results[task.task_id] = TaskResult(
            task_id=task.task_id,
            success=success,
            error_message=error_message,
            duration=time.time() - start_time,
            timestamp=time.time()
        )

    print(f"  Direct group {group_id} completed")

    # Policy を直接呼び出し（handler 不要）
    policy.on_group_finished(group_id, tasks, results, execution_context)
```

**変更内容:**
- `@staticmethod` デコレータを追加
- `self` パラメータを削除
- **`handler: TaskHandler` → `policy: GroupExecutionPolicy` に変更**
- **`policy.on_group_finished()` を直接呼び出し**

### 7. GroupExecutor クラスドキュメントの更新

**現在の実装** (`graflow/coordination/executor.py:17-22`):
```python
class GroupExecutor:
    """Unified executor for parallel task groups supporting multiple backends.

    This executor is stateless. It creates appropriate coordinators per execution
    request based on the specified backend and configuration.
    """
```

**変更後:**
```python
class GroupExecutor:
    """Stateless utility class for parallel task group execution.

    This class provides static methods to execute parallel task groups using
    different coordination backends (DIRECT, THREADING, REDIS). All methods
    are static - no instantiation required.

    The executor creates appropriate coordinators per execution request based
    on the specified backend and configuration.

    Usage:
        GroupExecutor.execute_parallel_group(
            group_id="my_group",
            tasks=[task1, task2, task3],
            exec_context=context,
            backend=CoordinationBackend.REDIS,
            backend_config={"redis_client": client, "key_prefix": "my_workflow"}
        )
    """
```

**変更内容:**
- クラスが完全にステートレスであることを明示
- "Stateless utility class" として説明
- すべてのメソッドが静的であることを明記
- 使用例を追加

### 8. Coordinator.execute_group() のシグネチャ変更

すべてのCoordinatorの`execute_group()`メソッドを変更して、`handler` の代わりに `policy` を受け取るようにします。

**影響を受けるCoordinator:**
- `RedisCoordinator` (`graflow/coordination/redis.py`)
- `ThreadingCoordinator` (`graflow/coordination/threading.py`)
- `DirectCoordinator` (存在しない - GroupExecutor.direct_execute() が代わり)

#### RedisCoordinator の変更

**現在の実装** (`graflow/coordination/redis.py:45-94`):
```python
def execute_group(
    self,
    group_id: str,
    tasks: List[Executable],
    execution_context: ExecutionContext,
    handler: TaskHandler
) -> None:
    """Execute parallel group with barrier synchronization."""
    # ...
    # Apply handler's group execution logic
    handler.on_group_finished(group_id, tasks, task_results, execution_context)
```

**変更後:**
```python
def execute_group(
    self,
    group_id: str,
    tasks: List[Executable],
    execution_context: ExecutionContext,
    policy: GroupExecutionPolicy
) -> None:
    """Execute parallel group with barrier synchronization.

    Args:
        group_id: Parallel group identifier
        tasks: List of tasks to execute
        execution_context: Execution context
        policy: Group execution policy for result evaluation
    """
    # ... (タスク実行と結果収集)

    # Policy を直接呼び出し
    policy.on_group_finished(group_id, tasks, task_results, execution_context)
```

#### ThreadingCoordinator の変更

**現在の実装** (`graflow/coordination/threading.py:35-152`):
```python
def execute_group(
    self,
    group_id: str,
    tasks: List['Executable'],
    execution_context: 'ExecutionContext',
    handler: 'TaskHandler'
) -> None:
    """Execute parallel group using WorkflowEngine in thread pool."""
    # ...
    # Apply handler (can raise ParallelGroupError)
    handler.on_group_finished(group_id, tasks, results, execution_context)
```

**変更後:**
```python
def execute_group(
    self,
    group_id: str,
    tasks: List['Executable'],
    execution_context: 'ExecutionContext',
    policy: 'GroupExecutionPolicy'
) -> None:
    """Execute parallel group using WorkflowEngine in thread pool.

    Args:
        group_id: Parallel group identifier
        tasks: List of tasks to execute
        execution_context: Execution context
        policy: Group execution policy for result evaluation
    """
    # ... (タスク実行と結果収集)

    # Policy を直接呼び出し
    policy.on_group_finished(group_id, tasks, results, execution_context)
```

### 9. TaskHandler から group_policy 関連メソッドを削除

**現在の実装** (`graflow/core/handler.py:108-154`):
```python
class TaskHandler(ABC):
    """Base class for task execution handlers."""

    @abstractmethod
    def execute_task(self, task: Executable, context: ExecutionContext) -> Any:
        """Execute single task."""
        pass

    def set_group_policy(self, policy: 'GroupExecutionPolicy') -> None:
        """Assign a custom group execution policy for this handler."""
        self._group_policy = policy

    def get_group_policy(self) -> 'GroupExecutionPolicy':
        """Return the group execution policy for this handler."""
        from graflow.core.handlers.group_policy import StrictGroupPolicy
        policy = getattr(self, "_group_policy", None)
        if policy is None:
            policy = StrictGroupPolicy()
            self._group_policy = policy
        return policy

    def on_group_finished(
        self,
        group_id: str,
        tasks: Sequence[Executable],
        results: Dict[str, TaskResult],
        context: ExecutionContext
    ) -> None:
        """Handle parallel group execution results."""
        policy = self.get_group_policy()
        policy.on_group_finished(group_id, tasks, results, context)
```

**変更後:**
```python
class TaskHandler(ABC):
    """Base class for task execution handlers.

    TaskHandler is responsible for individual task execution strategy.
    For group result evaluation, use GroupExecutionPolicy directly.

    Design:
    - execute_task() defines how individual tasks execute (e.g., direct, docker, async)
    - GroupExecutionPolicy defines when parallel groups succeed/fail (e.g., strict, best_effort)

    Examples:
        # Execution handler (inherits from TaskHandler)
        class DockerTaskHandler(TaskHandler):
            def execute_task(self, task, context):
                # Execute task in Docker container
                pass

        # Group policy (separate class)
        policy = BestEffortGroupPolicy()
        coordinator.execute_group(..., policy)
    """

    @abstractmethod
    def execute_task(self, task: Executable, context: ExecutionContext) -> Any:
        """Execute single task and store result in context.

        Args:
            task: Executable task to execute
            context: Execution context

        Returns:
            Task result value (or None if not available)

        Note:
            Implementation must call context.set_result(task_id, result) or
            context.set_result(task_id, exception) within the execution environment.
        """
        pass
```

**変更内容:**
- `set_group_policy()` を削除
- `get_group_policy()` を削除
- `on_group_finished()` を削除
- `execute_task()` のみ残す
- ドキュメント文字列を更新（責任分離を明確化）

### 10. GroupExecutor._create_coordinator() と _resolve_backend()（変更なし）

これらのメソッドは既に `@staticmethod` として実装されているため、変更不要です。

**`_create_coordinator()`** (`graflow/coordination/executor.py:40-70`):
```python
@staticmethod
def _create_coordinator(
    backend: CoordinationBackend,
    config: Dict[str, Any],
    exec_context: 'ExecutionContext'
) -> TaskCoordinator:
    """Create appropriate coordinator based on backend."""
    # ... (既に key_prefix を適切に処理している)
```

**`_resolve_backend()`** (`graflow/coordination/executor.py:26-37`):
```python
@staticmethod
def _resolve_backend(backend: Optional[Union[str, CoordinationBackend]]) -> CoordinationBackend:
    """Resolve backend from string or enum."""
    # ... implementation
```

**変更内容:**
- なし（既に `@staticmethod`）

## 動作の変更

### v1（現在）: グローバル設定とローカル設定

| with_execution() | context.group_executor | 使用されるExecutor |
|-----------------|------------------------|-------------------|
| ❌ なし | ❌ なし | デフォルトGroupExecutor() |
| ❌ なし | ✅ あり | context.group_executor（グローバル設定） |
| ✅ あり | ❌ なし | 設定されたGroupExecutor |
| ✅ あり | ✅ あり | 設定されたGroupExecutor（with_execution優先） |

### v2（改訂版）: ローカル設定のみ

| with_execution() | 使用されるバックエンド |
|-----------------|---------------------|
| ❌ なし | THREADING（GroupExecutor.DEFAULT_BACKEND） |
| ✅ backend=REDIS | REDIS |
| ✅ backend=THREADING | THREADING |
| ✅ backend=DIRECT | DIRECT |

**メリット:**
- シンプルで予測可能
- 設定の優先順位が明確
- グローバル設定が不要

## 使用例

### 例1: デフォルト実行（THREADING）

```python
from graflow.core.decorators import task

@task
def task_a():
    return "A"

@task
def task_b():
    return "B"

# デフォルト: THREADINGバックエンド
(task_a | task_b)  # with_execution()を呼ばなくてもOK
```

### 例2: Redis バックエンドでカスタム key_prefix と redis_client

```python
from graflow.coordination.coordinator import CoordinationBackend
import redis

# Redis クライアントを作成
redis_client = redis.Redis(host='localhost', port=6379, db=0)

# Redis バックエンドで key_prefix と redis_client を指定
parallel_extract = (extract_source_1 | extract_source_2 | extract_source_3).with_execution(
    backend=CoordinationBackend.REDIS,
    backend_config={
        "redis_client": redis_client,
        "key_prefix": "graflow:distributed_demo"
    }
)
```

**key_prefix の利点:**
- 複数ワークフローが同一 Redis インスタンスを共有可能
- キー衝突を防止（例: `graflow:workflow_a:queue`, `graflow:workflow_b:queue`）
- ワークフローごとに独立したキュー管理

### 例3: 異なるバックエンドを混在

```python
from graflow.core.workflow import workflow

with workflow("ml_pipeline") as wf:
    data_prep = task("data_prep")

    # Redis 分散実行（カスタム key_prefix）
    training = (train_a | train_b | train_c).with_execution(
        backend=CoordinationBackend.REDIS,
        backend_config={"key_prefix": "training_tasks"}
    )

    # スレッド並列実行
    validation = (validate_a | validate_b).with_execution(
        backend=CoordinationBackend.THREADING,
        backend_config={"thread_count": 2}
    )

    # 直接実行（軽量タスク）
    cleanup = (cleanup_a | cleanup_b).with_execution(
        backend=CoordinationBackend.DIRECT
    )

    data_prep >> training >> validation >> cleanup
```

### 例4: 分散ワークフローの完全な例

```python
from graflow.core.workflow import workflow
from graflow.core.decorators import task
from graflow.coordination.coordinator import CoordinationBackend
import redis

# Redis クライアントを作成
redis_client = redis.Redis(host='localhost', port=6379, db=0)

@task
def extract_source_1():
    return {"source": "S3", "data": [...]}

@task
def extract_source_2():
    return {"source": "PostgreSQL", "data": [...]}

@task
def extract_source_3():
    return {"source": "API", "data": [...]}

@task
def transform_data():
    # 並列抽出の結果を統合・変換
    return transformed_data

@task
def load_to_warehouse():
    # データウェアハウスにロード
    pass

with workflow("etl_pipeline") as wf:
    # 並列データ抽出（Redis 分散実行）
    parallel_extract = (extract_source_1 | extract_source_2 | extract_source_3).with_execution(
        backend=CoordinationBackend.REDIS,
        backend_config={
            "redis_client": redis_client,
            "key_prefix": "graflow:etl_pipeline"
        }
    )

    # 順次処理
    parallel_extract >> transform_data >> load_to_warehouse

    wf.execute()
```

**ポイント:**
- `redis_client` と `key_prefix` を同時に指定可能
- `key_prefix` でワークフローを識別（例: `graflow:etl_pipeline:queue`）
- 複数ワーカーで並列タスクを分散実行

### 例5: 複数のワークフローで異なる key_prefix

```python
# ワークフロー A: key_prefix = "graflow:workflow_a"
with workflow("workflow_a") as wf_a:
    group_a = (task_a1 | task_a2).with_execution(
        backend=CoordinationBackend.REDIS,
        backend_config={
            "redis_client": redis_client,
            "key_prefix": "graflow:workflow_a"
        }
    )

# ワークフロー B: key_prefix = "graflow:workflow_b"
with workflow("workflow_b") as wf_b:
    group_b = (task_b1 | task_b2).with_execution(
        backend=CoordinationBackend.REDIS,
        backend_config={
            "redis_client": redis_client,
            "key_prefix": "graflow:workflow_b"
        }
    )
```

**メリット:**
- 各ワークフローが独立した Redis キー空間を持つ
- 同一 Redis インスタンスで複数ワークフローを並列実行可能
- キー衝突なし（`graflow:workflow_a:queue` vs `graflow:workflow_b:queue`）

## 実装チェックリスト

### ExecutionContext の変更
- [ ] `ExecutionContext.__init__()` から `group_executor` 初期化を削除（line 301）
- [ ] `ExecutionContext.__getstate__()` から `group_executor` の `pop()` を削除（line 908）
- [ ] `ExecutionContext.__setstate__()` から `group_executor` 復元コードを削除（line 982-983）
- [ ] `ExecutionContext.create_branch_context()` から `group_executor` の継承を削除（該当箇所があれば）

### ParallelGroup の変更
- [ ] `ParallelGroup.__init__()` のコメントを更新（line 364: "None = use context.group_executor or default" → "None = use GroupExecutor.DEFAULT_BACKEND"）
- [ ] `ParallelGroup.run()` を変更
  - [ ] `executor = context.group_executor or GroupExecutor()` を削除（line 449）
  - [ ] `GroupExecutor.execute_parallel_group()` をクラスメソッドとして直接呼び出し
- [ ] `ParallelGroup.with_execution()` のドキュメント文字列を更新（Redis例に `key_prefix` と `redis_client` を追加）

### GroupExecutor の変更
- [ ] `GroupExecutor` クラスのドキュメント文字列を更新
  - [ ] "This executor is stateless." → "Stateless utility class for parallel task group execution."
  - [ ] "It creates appropriate coordinators per execution" を強調
- [ ] `GroupExecutor.execute_parallel_group()` を `@staticmethod` に変更
  - [ ] `self` パラメータを削除
  - [ ] **`DirectTaskHandler` のインポートと使用を削除**
  - [ ] **`policy_instance` を直接 coordinator に渡す**
  - [ ] `self._resolve_backend()` → `GroupExecutor._resolve_backend()`
  - [ ] `self.direct_execute(..., handler)` → `GroupExecutor.direct_execute(..., policy_instance)`
  - [ ] `self._create_coordinator()` → `GroupExecutor._create_coordinator()`
  - [ ] `coordinator.execute_group(..., handler)` → `coordinator.execute_group(..., policy_instance)`
- [ ] `GroupExecutor.direct_execute()` を `@staticmethod` に変更
  - [ ] `self` パラメータを削除
  - [ ] **`handler: TaskHandler` → `policy: GroupExecutionPolicy` に変更**
  - [ ] **`handler.on_group_finished()` → `policy.on_group_finished()` に変更**

### Coordinator の変更
- [ ] `RedisCoordinator.execute_group()` のシグネチャ変更
  - [ ] `handler: TaskHandler` → `policy: GroupExecutionPolicy`
  - [ ] `handler.on_group_finished()` → `policy.on_group_finished()`
  - [ ] ドキュメント文字列を更新
- [ ] `ThreadingCoordinator.execute_group()` のシグネチャ変更
  - [ ] `handler: TaskHandler` → `policy: GroupExecutionPolicy`
  - [ ] `handler.on_group_finished()` → `policy.on_group_finished()`
  - [ ] ドキュメント文字列を更新

### TaskHandler の変更
- [ ] `TaskHandler.set_group_policy()` を削除
- [ ] `TaskHandler.get_group_policy()` を削除
- [ ] `TaskHandler.on_group_finished()` を削除
- [ ] `TaskHandler` クラスのドキュメント文字列を更新（責任分離を明確化）
- [ ] **注:** `DirectTaskHandler` は変更不要（`execute_task()` のみ実装）

### テスト・ドキュメント
- [ ] ユニットテストを更新
  - [ ] `context.group_executor` に依存するテストを削除/修正
  - [ ] `handler.set_group_policy()` を使用するテストを修正
  - [ ] `handler.on_group_finished()` を直接呼び出すテストを修正
- [ ] 既存テストがパスすることを確認
- [ ] ドキュメント更新（使用例）
- [ ] CLAUDE.md を更新（ExecutionContext の説明から group_executor を削除）

## テスト方針

### 1. 基本機能テスト

```python
def test_parallel_group_default_backend():
    """Test default backend (THREADING) when with_execution() not called"""
    group = (task_a | task_b)
    assert group._execution_config["backend"] is None
    # run() should use GroupExecutor.DEFAULT_BACKEND (THREADING)

def test_parallel_group_with_redis_key_prefix():
    """Test Redis backend with custom key_prefix"""
    group = (task_a | task_b).with_execution(
        backend=CoordinationBackend.REDIS,
        backend_config={"key_prefix": "custom_prefix"}
    )
    assert group._execution_config["backend"] == CoordinationBackend.REDIS
    assert group._execution_config["backend_config"]["key_prefix"] == "custom_prefix"
```

### 2. ExecutionContext テスト

```python
def test_execution_context_no_group_executor():
    """Test that ExecutionContext no longer has group_executor attribute"""
    context = ExecutionContext.create(graph, start_node)
    assert not hasattr(context, 'group_executor')

def test_group_executor_is_stateless():
    """Test that GroupExecutor methods are static"""
    import inspect

    # Verify all public methods are static
    assert isinstance(inspect.getattr_static(GroupExecutor, 'execute_parallel_group'), staticmethod)
    assert isinstance(inspect.getattr_static(GroupExecutor, 'direct_execute'), staticmethod)
    assert isinstance(inspect.getattr_static(GroupExecutor, '_resolve_backend'), staticmethod)
    assert isinstance(inspect.getattr_static(GroupExecutor, '_create_coordinator'), staticmethod)
```

### 3. 統合テスト

```python
def test_multiple_workflows_different_key_prefixes():
    """Test multiple workflows with different Redis key_prefixes"""
    # Workflow A
    with workflow("workflow_a") as wf_a:
        group_a = (task_a1 | task_a2).with_execution(
            backend=CoordinationBackend.REDIS,
            backend_config={"key_prefix": "workflow_a"}
        )

    # Workflow B
    with workflow("workflow_b") as wf_b:
        group_b = (task_b1 | task_b2).with_execution(
            backend=CoordinationBackend.REDIS,
            backend_config={"key_prefix": "workflow_b"}
        )

    # Verify they use different key prefixes
    assert group_a._execution_config["backend_config"]["key_prefix"] == "workflow_a"
    assert group_b._execution_config["backend_config"]["key_prefix"] == "workflow_b"
```

### 4. 後方互換性テスト

```python
def test_backward_compatibility_without_with_execution():
    """Test existing code works without with_execution()"""
    # Existing syntax
    group = task_a | task_b | task_c

    # Should have default config
    assert group._execution_config["backend"] is None
    assert group._execution_config["backend_config"] == {}

    # Execution should work with default backend (THREADING)
    context = ExecutionContext.create(graph, start_node)
    group.set_execution_context(context)
    group.run()  # Should succeed
```

## 既存コードへの影響

### 影響を受けるコード

1. **`ExecutionContext` を直接操作するコード**:
   ```python
   # ❌ 動作しなくなる
   context.group_executor = GroupExecutor(CoordinationBackend.REDIS)
   ```

2. **`context.group_executor` に依存するテスト**:
   - `tests/` 配下で `context.group_executor` を参照するテストを修正

### 影響を受けないコード

1. **`with_execution()` を使用するコード**:
   ```python
   # ✅ そのまま動作
   (task_a | task_b).with_execution(backend=CoordinationBackend.REDIS)
   ```

2. **`with_execution()` を呼ばないコード**:
   ```python
   # ✅ そのまま動作（デフォルトバックエンドで実行）
   task_a | task_b | task_c
   ```

## マイグレーションガイド

### v1 → v2 への移行

**v1（非推奨）**:
```python
# グローバル設定
context = ExecutionContext.create(graph, start_node)
context.group_executor = GroupExecutor(CoordinationBackend.REDIS)

# ParallelGroup は context.group_executor を使用
(task_a | task_b).run()
```

**v2（推奨）**:
```python
# ローカル設定（各ParallelGroupで指定）
context = ExecutionContext.create(graph, start_node)

# ParallelGroup ごとに設定
(task_a | task_b).with_execution(
    backend=CoordinationBackend.REDIS,
    backend_config={"key_prefix": "my_workflow"}
)
```

**メリット:**
- 各 `ParallelGroup` が独立した設定を持つ
- より柔軟で明示的

## 設計判断（確定）

### ✅ 判断1: ExecutionContext.group_executor を削除

**理由:**
- `GroupExecutor` はステートレス（line 20-22: "This executor is stateless."）
- Context に保持する必要がない
- 毎回作成してもオーバーヘッドなし

### ✅ 判断2: ParallelGroup が独自に GroupExecutor を作成

**理由:**
- よりシンプルで理解しやすい
- 各グループが独立した設定を持つ
- グローバル設定が不要

### ✅ 判断3: backend_config で key_prefix をサポート

**理由:**
- 既に実装されている（`executor.py:54`）
- 複数ワークフローの並列実行に必須
- Redis キー空間の分離が可能

## GroupExecutionPolicy の挙動（重要）

### Policy の役割

`with_execution()` の `policy` パラメータは、並列タスクグループの**成功/失敗判定基準**を指定します。

**重要:** Policy の挙動は**すべてのバックエンド（DIRECT, THREADING, REDIS）で同じ**です。

### アーキテクチャ

```
ParallelGroup.with_execution(policy="strict")
  ↓
GroupExecutor.execute_parallel_group(policy="strict")
  ↓
  handler = DirectTaskHandler()  # すべてのバックエンドで使用
  policy_instance = resolve_group_policy("strict")
  handler.set_group_policy(policy_instance)
  ↓
Coordinator.execute_group(handler)
  ↓ (各バックエンドでタスク実行 + 結果収集)
  ↓
handler.on_group_finished(results)
  ↓ (Policy に委譲)
policy_instance.on_group_finished(results)
  ↓ (結果評価)
  ✅ 成功 or ❌ ParallelGroupError
```

**注:** `DirectTaskHandler` という名前は誤解を招きますが、これは「基本的なタスク実行」を意味し、REDIS/THREADING バックエンドでも使用されます。

### 利用可能な Policy

#### 1. Strict Policy（デフォルト）

**挙動:** 1つでもタスクが失敗したら `ParallelGroupError` を投げる

```python
# デフォルト（policy 未指定）
(task_a | task_b | task_c).with_execution(
    backend=CoordinationBackend.REDIS
)

# 明示的に指定
(task_a | task_b | task_c).with_execution(
    backend=CoordinationBackend.THREADING,
    policy="strict"
)
```

**動作:**
- task_a, task_b, task_c がすべて成功 → ✅ OK
- いずれか1つでも失敗 → ❌ `ParallelGroupError`

**すべてのバックエンドで同じ動作:**
- DIRECT: 順次実行後、失敗があれば例外
- THREADING: スレッドプール実行後、失敗があれば例外
- REDIS: Worker 実行後、失敗があれば例外

#### 2. Best Effort Policy

**挙動:** タスクの失敗を無視してログのみ記録

```python
(task_a | task_b | task_c).with_execution(
    backend=CoordinationBackend.REDIS,
    policy="best_effort"
)
```

**動作:**
- task_a, task_b, task_c がすべて成功 → ✅ OK
- いずれかが失敗 → ⚠️ 警告ログのみ、ワークフロー継続

**ユースケース:**
- 非クリティカルなタスク
- ベストエフォート処理（メール送信、ログ記録など）
- 失敗しても次のステップに進みたい場合

#### 3. Critical Policy

**挙動:** クリティカルタスクのみチェック、他は無視

```python
from graflow.core.handlers.group_policy import CriticalGroupPolicy

(task_a | task_b | task_c).with_execution(
    backend=CoordinationBackend.THREADING,
    policy=CriticalGroupPolicy(critical_task_ids=["task_a"])
)
```

**動作:**
- task_a（クリティカル）が失敗 → ❌ `ParallelGroupError`
- task_b or task_c（オプショナル）が失敗 → ⚠️ 警告ログのみ
- すべて成功 → ✅ OK

**ユースケース:**
- 必須タスクとオプショナルタスクの混在
- 例: データ抽出（必須）+ 統計計算（オプショナル）

#### 4. At Least N Policy

**挙動:** 最低N個のタスクが成功すればOK

```python
from graflow.core.handlers.group_policy import AtLeastNGroupPolicy

(task_a | task_b | task_c | task_d).with_execution(
    backend=CoordinationBackend.REDIS,
    policy=AtLeastNGroupPolicy(min_success=3)
)
```

**動作:**
- 3つ以上成功 → ✅ OK
- 2つ以下成功 → ❌ `ParallelGroupError`

**ユースケース:**
- 冗長性のあるタスク
- 例: 複数データソースから抽出、最低3つ成功すればOK

### 各バックエンドでの Policy 適用フロー

#### DIRECT Backend

```python
(task_a | task_b | task_c).with_execution(
    backend=CoordinationBackend.DIRECT,
    policy="strict"
)
```

**実行フロー:**
1. GroupExecutor が DIRECT backend を選択
2. タスクを**順次実行**（並列ではない）
3. 各タスクの成功/失敗を記録
4. **すべて完了後**に policy で評価
5. policy が `ParallelGroupError` を投げるか判断

**特徴:**
- 並列実行なし（デバッグ用）
- policy は最後に適用される

#### THREADING Backend

```python
(task_a | task_b | task_c).with_execution(
    backend=CoordinationBackend.THREADING,
    backend_config={"thread_count": 3},
    policy="best_effort"
)
```

**実行フロー:**
1. GroupExecutor が ThreadingCoordinator を作成
2. スレッドプールで**並列実行**
3. 各スレッドで独立した branch_context を使用
4. **すべて完了後**に結果を収集
5. policy で評価（失敗を無視）

**特徴:**
- スレッドプールで並列実行
- policy は最後に適用される
- 中間失敗は検知されない（最後にまとめて評価）

#### REDIS Backend

```python
(task_a | task_b | task_c).with_execution(
    backend=CoordinationBackend.REDIS,
    backend_config={"redis_client": redis_client, "key_prefix": "my_workflow"},
    policy="strict"
)
```

**実行フロー:**
1. GroupExecutor が RedisCoordinator を作成
2. バリアを作成（タスク数 = 3）
3. タスクを Redis キューに投入
4. Worker が各タスクを実行
5. バリア待機（**すべて完了を待つ**）
6. Redis から結果を収集
7. policy で評価（1つでも失敗したら例外）

**特徴:**
- Worker で分散並列実行
- バリア同期ですべて完了を待つ
- policy は最後に適用される

### Policy の挙動比較表

| Policy | task_a 成功 | task_b 成功 | task_c 成功 | 結果 |
|--------|-----------|-----------|-----------|------|
| **strict** | ✅ | ✅ | ✅ | ✅ OK |
| **strict** | ✅ | ❌ | ✅ | ❌ ParallelGroupError |
| **best_effort** | ✅ | ✅ | ✅ | ✅ OK |
| **best_effort** | ✅ | ❌ | ✅ | ⚠️ OK (警告ログ) |
| **critical(task_a)** | ✅ | ✅ | ✅ | ✅ OK |
| **critical(task_a)** | ❌ | ✅ | ✅ | ❌ ParallelGroupError |
| **critical(task_a)** | ✅ | ❌ | ❌ | ⚠️ OK (警告ログ) |
| **at_least_2** | ✅ | ✅ | ❌ | ✅ OK (2/3 成功) |
| **at_least_2** | ✅ | ❌ | ❌ | ❌ ParallelGroupError (1/3 成功) |

**すべてのバックエンドで同じ挙動**です。

### 重要な制約

1. **Policy は最後に適用される:**
   - すべてのタスクが完了するまで待つ
   - 中間失敗では停止しない
   - 最後にまとめて評価

2. **バックエンドに依存しない:**
   - DIRECT, THREADING, REDIS で同じ挙動
   - Coordinator は結果を収集するだけ
   - Policy が成功/失敗を判定

3. **タスクの実行自体は停止しない:**
   - `strict` policy でも、失敗タスクがあっても他のタスクは実行される
   - すべて完了後に例外が投げられる
   - 早期停止したい場合は、タスク内で実装が必要

### 使用例

#### 例1: ETL パイプラインで必須タスクとオプショナルタスクを分離

```python
from graflow.core.handlers.group_policy import CriticalGroupPolicy

# データ抽出（必須）+ 統計計算（オプショナル）
parallel_extract = (
    extract_from_db |        # 必須
    extract_from_api |       # 必須
    calculate_statistics     # オプショナル
).with_execution(
    backend=CoordinationBackend.REDIS,
    backend_config={"redis_client": redis_client},
    policy=CriticalGroupPolicy(critical_task_ids=["extract_from_db", "extract_from_api"])
)

# extract_from_db or extract_from_api が失敗 → ParallelGroupError
# calculate_statistics が失敗 → 警告ログのみ、ワークフロー継続
```

#### 例2: 複数データソースから冗長抽出

```python
from graflow.core.handlers.group_policy import AtLeastNGroupPolicy

# 5つのデータソースから、最低3つ成功すればOK
parallel_extract = (
    extract_s3 |
    extract_postgres |
    extract_api_1 |
    extract_api_2 |
    extract_backup
).with_execution(
    backend=CoordinationBackend.REDIS,
    backend_config={"redis_client": redis_client},
    policy=AtLeastNGroupPolicy(min_success=3)
)

# 3つ以上成功 → OK
# 2つ以下成功 → ParallelGroupError
```

#### 例3: ベストエフォート通知

```python
# メイン処理 + 通知（失敗しても継続）
parallel_notifications = (
    send_email |
    send_slack |
    update_dashboard
).with_execution(
    backend=CoordinationBackend.THREADING,
    policy="best_effort"
)

# いずれかが失敗しても警告ログのみ、ワークフロー継続
```

### 設計上の注意（DirectTaskHandler の名前について）

**現在の実装:**
```python
handler = DirectTaskHandler()  # すべてのバックエンドで使用
policy_instance = resolve_group_policy(policy)
handler.set_group_policy(policy_instance)
coordinator.execute_group(..., handler)
```

**注意:**
- `DirectTaskHandler` という名前は誤解を招く
- "Direct" は「直接実行」ではなく「基本的なタスク実行」を意味
- REDIS/THREADING バックエンドでも使用される
- 将来的には `BasicTaskHandler` にリネームを検討

詳細は `docs/policy_and_handler_design.md` を参照してください。

### 将来の改善案: TaskHandler を廃止して GroupExecutionPolicy を直接使用

**現在の設計:**
```python
# GroupExecutor.execute_parallel_group()
handler = DirectTaskHandler()
policy_instance = resolve_group_policy(policy)
handler.set_group_policy(policy_instance)
coordinator.execute_group(..., handler)
```

**問題点:**
- `TaskHandler` が不要な中間層として機能
- `handler.on_group_finished()` は単に `policy.on_group_finished()` に委譲するだけ
- `DirectTaskHandler` という名前が誤解を招く

**改善案:**
```python
# GroupExecutor.execute_parallel_group()
policy_instance = resolve_group_policy(policy)
coordinator.execute_group(..., policy_instance)
```

**変更内容:**
1. `Coordinator.execute_group()` のシグネチャを変更:
   ```python
   # 変更前
   def execute_group(self, group_id, tasks, execution_context, handler: TaskHandler):
       # ...
       handler.on_group_finished(group_id, tasks, results, execution_context)

   # 変更後
   def execute_group(self, group_id, tasks, execution_context, policy: GroupExecutionPolicy):
       # ...
       policy.on_group_finished(group_id, tasks, results, execution_context)
   ```

2. `TaskHandler` を2つのクラスに分離:
   - `TaskExecutionHandler`: 個別タスクの実行（`execute_task()` のみ）
   - `GroupExecutionPolicy`: グループ結果の評価（`on_group_finished()` のみ）← 既存

**メリット:**
- シンプルで理解しやすい
- 責任分離が明確（Single Responsibility Principle）
- 名前が実際の役割を反映
- `DirectTaskHandler` の混乱を解消

**デメリット:**
- 破壊的変更（既存コードに影響）
- マイグレーションコストがかかる

**実装タイミング:**
- v2実装後のリファクタリングとして検討
- または v3 での破壊的変更として実施

## key_prefix による Redis 名前空間の分離

### key_prefix の役割

`key_prefix` は Redis 上のすべてのキーに適用されるプレフィックスで、複数のワークフローを同一 Redis インスタンス上で完全に分離するために使用されます。

### 生成される Redis キー

```python
key_prefix = "graflow:distributed_demo"
group_id = "parallel_extract"
graph_hash = "abc123"

# 生成されるキー
queue_key       = "graflow:distributed_demo:queue"
barrier_key     = "graflow:distributed_demo:barrier:parallel_extract"
completion_key  = "graflow:distributed_demo:completions:parallel_extract"
graph_key       = "graflow:distributed_demo:graph:abc123"
```

### Worker との対応関係（重要）

**原則:** ワークフローで指定した `key_prefix` と Worker の `--redis-key-prefix` は**必ず一致**させる必要があります。

#### ケース1: デフォルト key_prefix を使用（推奨）

**ワークフロー側:**
```python
(task_a | task_b | task_c).with_execution(
    backend=CoordinationBackend.REDIS,
    backend_config={
        "redis_client": redis_client
        # key_prefix 未指定 → デフォルト "graflow" を使用
    }
)
```

**Worker 起動:**
```bash
# デフォルト key_prefix で起動
python -m graflow.worker.main --redis-key-prefix=graflow
```

**動作:**
- タスクは `graflow:queue` にエンキュー
- Worker は `graflow:queue` から取得
- ✅ 正常に動作

#### ケース2: カスタム key_prefix を使用

**ワークフロー側:**
```python
(extract_source_1 | extract_source_2 | extract_source_3).with_execution(
    backend=CoordinationBackend.REDIS,
    backend_config={
        "redis_client": redis_client,
        "key_prefix": "graflow:distributed_demo"
    }
)
```

**Worker 起動:**
```bash
# 同じ key_prefix で起動（必須）
python -m graflow.worker.main --redis-key-prefix=graflow:distributed_demo
```

**動作:**
- タスクは `graflow:distributed_demo:queue` にエンキュー
- Worker は `graflow:distributed_demo:queue` から取得
- ✅ 正常に動作

#### ケース3: key_prefix のミスマッチ（エラー）

**ワークフロー側:**
```python
(task_a | task_b).with_execution(
    backend=CoordinationBackend.REDIS,
    backend_config={
        "redis_client": redis_client,
        "key_prefix": "graflow:workflow_a"
    }
)
```

**Worker 起動:**
```bash
# 異なる key_prefix で起動
python -m graflow.worker.main --redis-key-prefix=graflow
```

**動作:**
- タスクは `graflow:workflow_a:queue` にエンキュー
- Worker は `graflow:queue` から取得しようとする
- ❌ **タスクが処理されない！**

### 複数ワークフローの分離パターン

#### パターン1: 複数ワークフローを完全に分離

```python
# ワークフロー A
with workflow("etl_pipeline_a") as wf_a:
    parallel_a = (task_a1 | task_a2 | task_a3).with_execution(
        backend=CoordinationBackend.REDIS,
        backend_config={
            "redis_client": redis_client,
            "key_prefix": "graflow:pipeline_a"
        }
    )
    wf_a.execute()

# ワークフロー B
with workflow("etl_pipeline_b") as wf_b:
    parallel_b = (task_b1 | task_b2 | task_b3).with_execution(
        backend=CoordinationBackend.REDIS,
        backend_config={
            "redis_client": redis_client,
            "key_prefix": "graflow:pipeline_b"
        }
    )
    wf_b.execute()
```

**Worker 起動:**
```bash
# ワークフロー A 用の Worker
python -m graflow.worker.main --worker-id=worker-a-1 --redis-key-prefix=graflow:pipeline_a

# ワークフロー B 用の Worker
python -m graflow.worker.main --worker-id=worker-b-1 --redis-key-prefix=graflow:pipeline_b
```

**Redis キー:**
```
graflow:pipeline_a:queue              ← Worker A が監視
graflow:pipeline_a:barrier:*
graflow:pipeline_a:completions:*
graflow:pipeline_a:graph:*

graflow:pipeline_b:queue              ← Worker B が監視
graflow:pipeline_b:barrier:*
graflow:pipeline_b:completions:*
graflow:pipeline_b:graph:*
```

**メリット:**
- 完全なワークフロー分離
- 異なるリソース要件に対応可能（Worker A: 4並列、Worker B: 8並列など）
- 片方のワークフローの障害が他方に影響しない

#### パターン2: 共通 key_prefix で複数ワークフローを実行

```python
# ワークフロー A
with workflow("etl_pipeline_a") as wf_a:
    parallel_a = (task_a1 | task_a2).with_execution(
        backend=CoordinationBackend.REDIS,
        backend_config={"redis_client": redis_client}  # デフォルト key_prefix
    )

# ワークフロー B
with workflow("etl_pipeline_b") as wf_b:
    parallel_b = (task_b1 | task_b2).with_execution(
        backend=CoordinationBackend.REDIS,
        backend_config={"redis_client": redis_client}  # デフォルト key_prefix
    )
```

**Worker 起動:**
```bash
# 共通 Worker が両方のワークフローを処理
python -m graflow.worker.main --worker-id=worker-1 --max-concurrent-tasks=8
python -m graflow.worker.main --worker-id=worker-2 --max-concurrent-tasks=8
```

**Redis キー:**
```
graflow:queue                         ← 全 Worker が監視（両ワークフローのタスクが混在）
graflow:barrier:parallel_a            ← group_id で区別
graflow:barrier:parallel_b
graflow:completions:parallel_a
graflow:completions:parallel_b
graflow:graph:*
```

**メリット:**
- Worker の管理が簡単
- リソースの柔軟な活用（アイドル Worker が他ワークフローを処理）

**デメリット:**
- ワークフロー間でリソース競合が発生する可能性

### 推奨される使い分け

| ユースケース | key_prefix 設定 | Worker 構成 |
|------------|----------------|------------|
| **単一ワークフロー** | デフォルト (`graflow`) | 複数 Worker で共有 |
| **開発/検証環境** | デフォルト (`graflow`) | 1つの Worker |
| **本番環境（複数ワークフロー）** | ワークフローごとに異なる key_prefix | 専用 Worker を起動 |
| **リソース分離が重要** | ワークフローごとに異なる key_prefix | 専用 Worker を起動 |
| **リソース効率重視** | 共通 key_prefix | 共有 Worker プール |

### 設計判断（確定）

#### ✅ 判断: key_prefix による完全な名前空間分離をサポート

**理由:**
- 複数ワークフローの完全な分離が可能
- 本番環境でのリソース管理が容易
- Worker の障害影響範囲を限定できる

**制約:**
- ユーザーは `with_execution()` の `key_prefix` と Worker の `--redis-key-prefix` を一致させる責任がある
- ドキュメントで明確に説明する

**ドキュメント化すべき内容:**
1. `key_prefix` と Worker の対応関係
2. 複数ワークフロー分離パターン
3. ミスマッチ時のエラー動作

## まとめ

### 主要な変更

1. **`ExecutionContext.group_executor` を削除**
   - グローバル設定を廃止
   - ExecutionContext をシンプル化
   - 各 ParallelGroup が独立して動作

2. **`GroupExecutor` を完全にステートレスなユーティリティクラスに変更**
   - すべてのメソッドを `@staticmethod` に変更
   - インスタンス化不要（`GroupExecutor.execute_parallel_group()` として直接呼び出し）
   - オーバーヘッドゼロ

3. **`TaskHandler` から group_policy 関連メソッドを削除（重要な破壊的変更）**
   - `set_group_policy()`, `get_group_policy()`, `on_group_finished()` を削除
   - TaskHandler は `execute_task()` のみに責任を限定
   - `GroupExecutionPolicy` を直接 Coordinator に渡す
   - 不要な間接化を排除
   - 責任分離が明確に

4. **Coordinator のシグネチャ変更**
   - `execute_group(..., handler: TaskHandler)` → `execute_group(..., policy: GroupExecutionPolicy)`
   - Policy を直接呼び出し
   - RedisCoordinator, ThreadingCoordinator, GroupExecutor.direct_execute() すべてで統一

5. **`ParallelGroup.run()` で GroupExecutor をクラスメソッドとして直接呼び出し**
   - `executor = GroupExecutor()` を削除
   - `GroupExecutor.execute_parallel_group()` として呼び出し
   - よりシンプルで明確

6. **`backend_config` で `redis_client` と `key_prefix` をサポート**
   - 既存実装を活用
   - 複数ワークフローの並列実行が可能
   - キー衝突を防止

### メリット

- **シンプル**: グローバル設定が不要、インスタンス化不要、不要な間接化なし
- **明確**: 各 ParallelGroup が独自の設定を持つ、責任分離が明確
- **柔軟**: グループごとに異なるバックエンドや key_prefix を使用可能
- **効率的**: インスタンス化のオーバーヘッドなし
- **保守性向上**: クリーンなアーキテクチャ、理解しやすいコード

### アーキテクチャの改善

**Before (v1):**
```python
context.group_executor = GroupExecutor()  # グローバル設定
executor = context.group_executor or GroupExecutor()  # インスタンス化
executor.execute_parallel_group(...)  # メソッド呼び出し
```

**After (v2):**
```python
# グローバル設定なし
GroupExecutor.execute_parallel_group(...)  # クラスメソッドとして直接呼び出し
```

### 次のステップ

1. 実装変更（上記チェックリスト）
2. テスト更新
3. ドキュメント更新
4. 既存テストの確認
