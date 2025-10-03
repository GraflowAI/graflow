# ParallelGroup.with_execution() 実装設計書

## 概要

ParallelGroupに`with_execution()`メソッドを追加し、グループごとに異なる実行バックエンド（DIRECT, THREADING, REDIS）を指定できるようにする。

## 設計原則

1. **ParallelGroupの設定を変更する**: `with_execution()`はParallelGroupインスタンス自身の実行設定を変更
2. **後方互換性の維持**: 既存コード（`with_execution()`を呼ばないコード）は影響を受けない
3. **メソッドチェーン対応**: `set_group_name()`と同様にselfを返す
4. **シンプルなAPI**: 必要最小限のパラメータのみ提供

## 実装詳細

### 1. 初期化時のデフォルト設定

`_execution_config`はデフォルト設定で初期化する（`None`ではない）。

```python
def __init__(self, tasks: list[Executable]) -> None:
    """Initialize a parallel group with a list of tasks."""
    self._task_id = self._get_group_name()
    self.tasks = list(tasks)

    # Default execution configuration
    self._execution_config = {
        "backend": None,  # with_execution()未呼び出し時はNone
        "backend_config": {}
    }

    self._register_to_context()
    for task in self.tasks:
        self._add_dependency_edge(self._task_id, task.task_id)
```

**設計判断:**
- `backend: None`で初期化し、`with_execution()`呼び出しの有無を判断
- `None`の場合は既存動作（`context.group_executor`優先）を維持

### 2. with_execution()メソッド

ParallelGroupの実行設定を変更するメソッド。

```python
def with_execution(
    self,
    backend: Optional[CoordinationBackend] = None,
    backend_config: Optional[dict] = None
) -> 'ParallelGroup':
    """Configure execution backend for this parallel group.

    このメソッドはParallelGroup自身の実行設定を変更します。
    メソッドチェーンをサポートしています。

    Args:
        backend: 実行バックエンド (DIRECT, THREADING, REDIS)
        backend_config: バックエンド固有の設定
            - THREADING: {"thread_count": int}
            - MULTIPROCESSING: {"process_count": int}
            - REDIS: 追加設定なし（将来拡張可能）

    Returns:
        Self (メソッドチェーン用)

    Examples:
        >>> # 基本的な使用
        >>> (task_a | task_b).with_execution(backend=CoordinationBackend.THREADING)

        >>> # 詳細設定
        >>> (task_a | task_b).with_execution(
        ...     backend=CoordinationBackend.THREADING,
        ...     backend_config={"thread_count": 4}
        ... )

        >>> # メソッドチェーン
        >>> group = (task_a | task_b | task_c) \\
        ...     .with_execution(backend=CoordinationBackend.REDIS) \\
        ...     .set_group_name("training_tasks")
    """
    # Update backend if specified
    if backend is not None:
        self._execution_config["backend"] = backend

    # Update backend_config if specified (merge)
    if backend_config is not None:
        self._execution_config["backend_config"].update(backend_config)

    return self
```

**設計判断:**
- `backend_config`は**マージ**（`update()`）: 複数回呼び出した場合に設定を積み重ねられる
- 両方のパラメータが`Optional`: 柔軟な使用を可能にする
- バリデーションは行わない（GroupExecutorに任せる）

### 3. _create_configured_executor()メソッド

設定に基づいてGroupExecutorを生成するヘルパーメソッド。

```python
def _create_configured_executor(self) -> GroupExecutor:
    """Create GroupExecutor based on execution configuration.

    Returns:
        Configured GroupExecutor instance
    """
    backend = self._execution_config["backend"]
    backend_config = self._execution_config["backend_config"]

    # backendがNoneの場合はデフォルトGroupExecutor
    if backend is None:
        return GroupExecutor()
    else:
        return GroupExecutor(backend, backend_config)
```

**設計判断:**
- privateメソッド（`_`プレフィックス）: 内部実装の詳細
- `backend is None`の場合はデフォルトGroupExecutorを返す

### 4. run()メソッドの変更

実行時に設定されたGroupExecutorを使用するように変更。

**変更前:**
```python
def run(self) -> Any:
    """Execute all tasks in this parallel group."""
    context = self.get_execution_context()
    executor = context.group_executor or GroupExecutor()

    for task in self.tasks:
        task.set_execution_context(context)

    executor.execute_parallel_group(self.task_id, self.tasks, context)
```

**変更後:**
```python
def run(self) -> Any:
    """Execute all tasks in this parallel group."""
    context = self.get_execution_context()

    # backend is None かつ context.group_executor がある場合のみ既存動作
    # それ以外は設定されたexecutorを使用
    if self._execution_config["backend"] is None and context.group_executor:
        executor = context.group_executor
    else:
        executor = self._create_configured_executor()

    for task in self.tasks:
        task.set_execution_context(context)

    executor.execute_parallel_group(self.task_id, self.tasks, context)
```

**設計判断（確定）:**

この実装により以下の動作が実現されます：

| `with_execution()`呼び出し | `context.group_executor` | 使用されるExecutor |
|-------------------------|------------------------|------------------|
| ❌ なし | ❌ なし | デフォルトGroupExecutor() |
| ❌ なし | ✅ あり | context.group_executor（既存動作） |
| ✅ あり | ❌ なし | 設定されたGroupExecutor |
| ✅ あり | ✅ あり | 設定されたGroupExecutor（`with_execution()`優先） |

**メリット:**
- `backend`の値自体が状態フラグ（追加のフラグ変数不要）
- `with_execution()`未使用時は既存動作を完全に維持
- `with_execution()`使用時はそちらを優先
- シンプルで理解しやすいロジック

**既存動作との互換性:**
- `with_execution()`を使わないコードは影響を受けない
- `context.group_executor`設定も引き続き機能する

### 5. Import文の追加

ファイルトップに以下を追加:

```python
from typing import Any, Optional
from graflow.coordination.coordinator import CoordinationBackend
from graflow.coordination.executor import GroupExecutor
```

## 後方互換性の保証

### 既存コードへの影響

**ケース1: with_execution()を呼ばない場合**
```python
# 既存コード
task_a | task_b | task_c
```

**動作:**
- `backend is None` → `context.group_executor`があればそれを使用
- `context.group_executor`がなければデフォルトGroupExecutor()

**ケース2: context.group_executorが設定されている場合**
```python
context = ExecutionContext(graph)
context.group_executor = GroupExecutor(CoordinationBackend.REDIS)

# with_execution()を呼ばない
(task_a | task_b).run()
```

**動作:**
- `backend is None` かつ `context.group_executor`あり → REDISで実行（既存動作維持）

**ケース3: with_execution()を呼んだ場合**
```python
context = ExecutionContext(graph)
context.group_executor = GroupExecutor(CoordinationBackend.REDIS)

# with_execution()を呼ぶ
(task_a | task_b).with_execution(backend=CoordinationBackend.THREADING).run()
```

**動作:**
- `backend is not None` → THREADINGで実行（`with_execution()`が優先）
- `context.group_executor`は無視される

## 使用例

### 例1: 基本的な使用

```python
from graflow.core.decorators import task
from graflow.coordination.coordinator import CoordinationBackend

@task
def task_a():
    return "A"

@task
def task_b():
    return "B"

# THREADINGバックエンドで実行
(task_a | task_b).with_execution(backend=CoordinationBackend.THREADING)
```

### 例2: スレッド数を指定

```python
(task_a | task_b | task_c).with_execution(
    backend=CoordinationBackend.THREADING,
    backend_config={"thread_count": 4}
)
```

### 例3: 異なるバックエンドを混在

```python
from graflow.core.workflow import workflow

with workflow("ml_pipeline") as wf:
    data_prep = task("data_prep")

    # スレッド並列実行
    training = (train_a | train_b | train_c).with_execution(
        backend=CoordinationBackend.THREADING,
        backend_config={"thread_count": 3}
    )

    # 直接実行（軽量タスク）
    validation = (validate_a | validate_b).with_execution(
        backend=CoordinationBackend.DIRECT
    )

    # Redis分散実行
    heavy_processing = (process_a | process_b).with_execution(
        backend=CoordinationBackend.REDIS
    )

    data_prep >> training >> validation >> heavy_processing
```

### 例4: メソッドチェーン

```python
group = (task_a | task_b | task_c) \
    .with_execution(backend=CoordinationBackend.REDIS) \
    .set_group_name("my_parallel_tasks")
```

## テスト方針

### 1. 基本機能テスト

```python
def test_with_execution_backend():
    """Test basic backend configuration"""
    group = (task_a | task_b).with_execution(backend=CoordinationBackend.DIRECT)
    assert group._execution_config["backend"] == CoordinationBackend.DIRECT

def test_with_execution_backend_config():
    """Test backend_config setting"""
    group = (task_a | task_b).with_execution(
        backend_config={"thread_count": 4}
    )
    assert group._execution_config["backend_config"]["thread_count"] == 4
```

### 2. メソッドチェーンテスト

```python
def test_with_execution_method_chaining():
    """Test method chaining returns self"""
    group = task_a | task_b
    result = group.with_execution(backend=CoordinationBackend.THREADING)
    assert result is group
```

### 3. 設定マージテスト

```python
def test_with_execution_config_merge():
    """Test backend_config merging on multiple calls"""
    group = (task_a | task_b)
    group.with_execution(backend_config={"thread_count": 4})
    group.with_execution(backend_config={"timeout": 30})

    config = group._execution_config["backend_config"]
    assert config["thread_count"] == 4
    assert config["timeout"] == 30
```

### 4. 実行テスト

```python
def test_with_execution_creates_correct_executor(mocker):
    """Test that configured executor is used"""
    mock_executor_class = mocker.patch('graflow.core.task.GroupExecutor')

    group = (task_a | task_b).with_execution(
        backend=CoordinationBackend.THREADING,
        backend_config={"thread_count": 4}
    )

    # Execute
    context = ExecutionContext(TaskGraph())
    group.set_execution_context(context)
    group.run()

    # Verify correct executor was created
    mock_executor_class.assert_called_once_with(
        CoordinationBackend.THREADING,
        {"thread_count": 4}
    )
```

### 5. 後方互換性テスト

```python
def test_backward_compatibility_without_with_execution():
    """Test existing code works without with_execution()"""
    # Existing syntax
    group = task_a | task_b | task_c

    # Should have default config
    assert group._execution_config["backend"] == CoordinationBackend.THREADING
    assert group._execution_config["backend_config"] == {}
```

## 実装チェックリスト

- [ ] `ParallelGroup.__init__`に`_execution_config`を追加
- [ ] Import文を追加（`CoordinationBackend`, `GroupExecutor`, `Optional`）
- [ ] `with_execution()`メソッドを実装
- [ ] `_create_configured_executor()`メソッドを実装
- [ ] `run()`メソッドを変更
- [ ] ユニットテストを追加
- [ ] ドキュメント更新（使用例）
- [ ] 既存テストがパスすることを確認

## Handler統合設計（将来拡張）

### 現在のHandler実装状況

#### TaskHandlerアーキテクチャ

```python
# graflow/worker/handler.py
class TaskHandler(ABC):
    """Abstract base class for task processing handlers."""

    @abstractmethod
    def _process_task(self, task: Executable) -> bool:
        pass

class DirectTaskExecutor(TaskHandler):
    """Task executor that runs tasks directly in the worker process."""

    def _process_task(self, task: Executable) -> bool:
        execution_context = task.get_execution_context()
        task_id = task.task_id
        self.engine.execute(execution_context, start_task_id=task_id)
        return True
```

#### 現在の実行フロー

**REDISバックエンド使用時:**
```
ParallelGroup.run()
  ↓
GroupExecutor.execute_parallel_group()
  ↓
RedisCoordinator.execute_group()
  ↓
RedisCoordinator.dispatch_task() → TaskQueue
  ↓
TaskWorker (別プロセス)
  ↓
TaskHandler.process_task()
```

**重要な制約:**
- `TaskWorker`は別プロセスで起動され、`handler: TaskHandler`を`__init__`で受け取る
- Handlerは**ワーカーレベル**で固定（1つのワーカーは1つのhandlerのみ使用）
- `ParallelGroup`や`with_execution()`から直接handlerを制御できない

### Handler設定の設計方針（確定）

**責任分離:**
- **`with_execution()`** → Coordinator（backend）選択のみ
- **`@task`デコレータ** → Handler設定（タスクレベル）

**設計原則:**
```
ParallelGroup.with_execution() → どのCoordinatorで実行するか (DIRECT/THREADING/REDIS)
@task(handler="...")           → どのHandlerで実行するか (direct/docker/async)
```

この設計により、以下のような明確な分離が実現されます：

| レイヤー | 設定場所 | 対象 | 例 |
|---------|---------|------|---|
| **並列実行制御** | `with_execution()` | Coordinator/Backend | `THREADING`, `REDIS` |
| **タスク実行制御** | `@task(handler=...)` | TaskHandler | `docker`, `async` |

### Phase 1実装: with_execution()はCoordinatorのみ

**設計方針:**
- `QueueTaskSpec`にhandler情報を追加
- TaskWorkerが複数のhandlerを保持し、タスクごとに選択
- `with_execution()`および`@task`デコレータからhandler指定可能

**Phase 1での`with_execution()`実装:**
```python
def with_execution(
    self,
    backend: Optional[CoordinationBackend] = None,
    backend_config: Optional[dict] = None
) -> 'ParallelGroup':
    """Configure execution backend (coordinator) for this parallel group.

    Args:
        backend: Coordinator backend (DIRECT, THREADING, REDIS)
        backend_config: Backend-specific configuration
            - THREADING: {"thread_count": int}
            - REDIS: 将来の拡張用

    Returns:
        Self (for method chaining)

    Note:
        Handler設定は@taskデコレータで行います。
        with_execution()はCoordinator（並列実行方法）の設定のみを行います。
    """
    if backend is not None:
        self._execution_config["backend"] = backend
    if backend_config is not None:
        self._execution_config["backend_config"].update(backend_config)
    return self
```

### Phase 2実装: @taskデコレータでHandler設定

**必要な変更:**

1. **QueueTaskSpec拡張**
```python
class QueueTaskSpec:
    def __init__(self, executable, execution_context, handler_type=None):
        self.executable = executable
        self.execution_context = execution_context
        self.handler_type = handler_type  # NEW: "direct", "docker", "async"
```

2. **TaskWrapper拡張（@task統合）**
```python
@dataclass
class TaskWrapper(Executable):
    _handler_type: Optional[str] = None  # NEW

    # @taskデコレータで指定可能
    @task(handler="docker")
    def gpu_training():
        pass
```

3. **TaskWorker拡張**
```python
class TaskWorker:
    def __init__(self, queue: TaskQueue, handlers: Dict[str, TaskHandler], ...):
        self.handlers = handlers  # 複数のhandlerを保持

    def _process_task_wrapper(self, task_spec: TaskSpec):
        # タスクごとにhandlerを選択
        handler_type = task_spec.handler_type or "direct"
        handler = self.handlers.get(handler_type)
        success = handler.process_task(task_wrapper)
```

4. **RedisCoordinator拡張**
```python
def dispatch_task(self, executable: 'Executable', group_id: str) -> None:
    # handler情報を取得
    handler_type = self._get_handler_type(executable)

    queue_task_spec = QueueTaskSpec(
        executable=executable,
        execution_context=executable.get_execution_context(),
        handler_type=handler_type  # NEW
    )
    self.task_queue.enqueue(queue_task_spec)

def _get_handler_type(self, executable: 'Executable') -> Optional[str]:
    # TaskWrapper._handler_type または backend_config から取得
    if hasattr(executable, '_handler_type'):
        return executable._handler_type
    return None
```

**Handler選択ロジック:**
- タスクレベル設定（`@task(handler="docker")`）が存在すればそれを使用
- なければデフォルト（"direct"）を使用
- **グループレベルでのhandler設定はサポートしない**（責任分離の原則）

**メリット:**
- タスクごとに異なるhandlerを使用可能
- `@task`デコレータで直感的に指定
- グループレベルでも一括指定可能

**デメリット:**
- 複数コンポーネントの変更が必要
- TaskWorkerの複雑性が増す
- Phase 2以降の実装

### Handler実装例（将来）

#### Docker Handler
```python
class DockerTaskHandler(TaskHandler):
    """Execute tasks in Docker containers."""

    def _process_task(self, task: Executable) -> bool:
        # Docker container内でタスク実行
        container = docker.run(image="pytorch/pytorch:latest", ...)
        result = container.exec(task.task_id)
        return result.success
```

#### Async Handler
```python
class AsyncTaskHandler(TaskHandler):
    """Execute async tasks."""

    async def _process_task_async(self, task: Executable) -> bool:
        # 非同期実行
        result = await task.run_async()
        return result
```

### @taskデコレータとの統合（将来拡張）

**Phase 2以降の実装イメージ:**

```python
# 個別タスクでhandler指定
@task(handler="docker")
def gpu_training_task():
    # このタスクはDockerコンテナ内で実行される
    pass

@task(handler="async")
async def async_api_call():
    # このタスクは非同期で実行される
    pass

@task  # デフォルト（DirectTaskExecutor）
def normal_task():
    pass

# Coordinatorとhandlerの組み合わせ
with workflow("ml_pipeline") as wf:
    # REDIS Coordinator + 各タスクで個別にhandler設定
    training = (
        train_a |  # @task → DirectTaskExecutor (デフォルト)
        gpu_training_task |  # @task(handler="docker") → DockerTaskHandler
        async_api_call  # @task(handler="async") → AsyncTaskHandler
    ).with_execution(
        backend=CoordinationBackend.REDIS  # Coordinator選択のみ
    )

    # THREADING Coordinator + デフォルトhandler
    validation = (validate_a | validate_b).with_execution(
        backend=CoordinationBackend.THREADING,
        backend_config={"thread_count": 4}
    )
```

### Phase実装ロードマップ

**Phase 1: with_execution()によるCoordinator選択**
- ✅ `with_execution(backend, backend_config)`実装
- ✅ ParallelGroupレベルでのCoordinator選択
- ❌ Handler設定は含めない（タスクレベルの責任）

**Phase 2: @taskデコレータでのHandler設定**
- QueueTaskSpec、TaskWrapper、TaskWorkerの拡張
- `@task(handler="docker")`のサポート
- TaskWorkerでの動的handler選択

**設計の明確化:**
```python
# ❌ 非推奨: with_execution()でhandler設定はしない
(task_a | task_b).with_execution(
    backend=CoordinationBackend.REDIS,
    backend_config={"handler_type": "docker"}  # ← これはしない
)

# ✅ 推奨: Coordinatorとhandlerは分離
@task(handler="docker")  # Handler設定はタスクレベル
def gpu_task():
    pass

(task_a | gpu_task).with_execution(
    backend=CoordinationBackend.REDIS  # Coordinator設定はグループレベル
)
```

## 設計判断（確定）

### ✅ 判断1: run()メソッドの実装

**採用した方法:**
```python
if self._execution_config["backend"] is None and context.group_executor:
    executor = context.group_executor
else:
    executor = self._create_configured_executor()
```

- `backend`の値自体を状態判断に使用（追加フラグ不要）
- 既存動作を完全に維持しつつ、新機能も提供

### ✅ 判断2: デフォルトbackend値

**採用した値:** `None`

- `with_execution()`未呼び出しを明示的に表現
- 既存の`context.group_executor`動作を維持可能

### ✅ 判断3: バリデーション

**採用した方針:** 実行時（`GroupExecutor`）に任せる

- `with_execution()`はシンプルに値を設定するのみ
- バリデーションはGroupExecutorが責任を持つ
- エラーは実行時に適切に報告される
