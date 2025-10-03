# ParallelGroup Execution Mode Implementation Plan

## 概要

ParallelGroupにチェーンメソッド形式のexecution mode制御機能を段階的に実装する。現在のGroupExecutor/TaskWorkerシステムとの統合を重視し、後方互換性を維持しながら柔軟な実行制御を提供する。

## 現在の実装状況

### 既存のGroupExecutor機能
- **CoordinationBackend**: REDIS、THREADING、DIRECTの3つのバックエンドをサポート
- **バックエンド切り替え**: GroupExecutor(backend, backend_config)で実行方法を制御
- **統合エンジン**: DIRECTバックエンドではWorkflowEngineを使用した統一実行

### 現在のParallelGroup機能
- **基本並列実行**: `|`演算子でタスクを並列グループ化
- **自動実行**: `context.group_executor`またはデフォルトGroupExecutorを使用
- **依存関係管理**: `>>`、`<<`演算子でワークフロー内の依存関係を設定

## 設計原則

1. **後方互換性**: 既存の`|`演算子使用コードは無変更で動作
2. **段階的実装**: 最小限の変更から開始し、徐々に機能拡張
3. **既存インフラ活用**: GroupExecutor/RedisCoordinator/TaskWorkerを最大活用
4. **責任分離**: TaskWorkerプロセス管理は独立した課題として扱う
5. **シンプルAPI**: 必要最小限のパラメータのみ提供

## 実装フェーズ

### Phase 1: 基本的なwith_execution()メソッド追加

#### 1.1 ParallelGroup拡張

```python
from graflow.coordination.coordinator import CoordinationBackend

class ParallelGroup(Executable):
    def __init__(self, tasks: list[Executable]) -> None:
        # 既存実装（変更なし）
        self._task_id = self._get_group_name()
        self.tasks = list(tasks)

        # 新規追加：実行設定
        self._execution_config = {
            "backend": CoordinationBackend.THREADING,  # 現在のデフォルト
            "backend_config": {}  # GroupExecutorのbackend_configに対応
        }

        # 既存の依存関係設定（変更なし）
        self._register_to_context()
        for task in self.tasks:
            self._add_dependency_edge(self._task_id, task.task_id)

    def with_execution(self, backend: CoordinationBackend = None,
                      backend_config: dict = None) -> 'ParallelGroup':
        """Configure execution backend for this parallel group.

        Args:
            backend: CoordinationBackend (DIRECT, THREADING, REDIS)
            backend_config: Backend-specific configuration dict

        Returns:
            Self for method chaining
        """
        if backend is not None:
            self._execution_config["backend"] = backend
        if backend_config is not None:
            self._execution_config["backend_config"] = backend_config
        return self
```

#### 1.2 実行ロジック統合

```python
    def run(self) -> Any:
        """Execute all tasks in this parallel group."""
        context = self.get_execution_context()

        # 実行設定に基づいてGroupExecutor作成
        executor = self._create_configured_executor(context)

        # 各タスクに実行コンテキストを設定（既存実装）
        for task in self.tasks:
            task.set_execution_context(context)

        # GroupExecutorで並列実行
        executor.execute_parallel_group(self.task_id, self.tasks, context)

    def _create_configured_executor(self, context: 'ExecutionContext') -> GroupExecutor:
        """Create GroupExecutor based on execution configuration."""
        backend = self._execution_config["backend"]
        backend_config = self._execution_config["backend_config"]

        # 設定されたバックエンドでGroupExecutorを作成
        return GroupExecutor(backend, backend_config)
```

### Phase 2: TaskWorker Handler設定の統合 (将来拡張)

#### 2.1 現在の実装状況
- GroupExecutorは既にCoordinationBackendを使った柔軟なバックエンド切り替えをサポート
- RedisCoordinatorは統一されたTaskWorkerシステムを利用
- TaskWorkerのhandler設定は現在DirectTaskExecutorが担当

#### 2.2 将来の拡張方向
TaskWorkerでの実行ハンドラー（docker、async等）を細かく制御する場合：

```python
# backend_configでhandler情報を指定
(task_a | task_b).with_execution(
    backend=CoordinationBackend.REDIS,
    backend_config={"handler_type": "docker"}
)
```

この機能は必要に応じて以下のステップで実装可能：
1. TaskSpecへのhandler_type情報追加
2. RedisCoordinatorでのhandler情報転送
3. TaskWorkerでのhandler動的選択

### Phase 3: 使用例とテスト

#### 3.1 基本使用例

```python
from graflow.coordination.coordinator import CoordinationBackend

# 既存コード（変更なし）
train_a | train_b | train_c

# 直接実行（WorkflowEngine使用）
(light_task_a | light_task_b).with_execution(
    backend=CoordinationBackend.DIRECT
)

# マルチスレッド並列実行
(cpu_task_a | cpu_task_b | cpu_task_c).with_execution(
    backend=CoordinationBackend.THREADING,
    backend_config={"thread_count": 4}
)

# Redis分散実行
(heavy_task_a | heavy_task_b | heavy_task_c).with_execution(
    backend=CoordinationBackend.REDIS
)

# Redis + カスタムハンドラー（将来拡張）
(gpu_task_a | gpu_task_b).with_execution(
    backend=CoordinationBackend.REDIS,
    backend_config={"handler_type": "docker"}
)
```

#### 3.2 複合使用例

```python
from graflow.core.workflow import workflow
from graflow.core.task import Task
from graflow.coordination.coordinator import CoordinationBackend

with workflow("ml_pipeline") as wf:
    data_prep = Task("data_prep")

    # Redis分散実行
    training = (train_a | train_b | train_c).with_execution(
        backend=CoordinationBackend.REDIS
    )

    # 直接実行（軽量タスク）
    validation = (validate_a | validate_b).with_execution(
        backend=CoordinationBackend.DIRECT
    )

    # マルチスレッド実行
    analysis = (analyze_a | analyze_b | analyze_c | analyze_d).with_execution(
        backend=CoordinationBackend.THREADING,
        backend_config={"thread_count": 2}
    )

    evaluation = Task("evaluation")

    data_prep >> training >> validation >> analysis >> evaluation
```

## 実装順序

### 優先度 HIGH

1. **ParallelGroup._execution_config追加** ⚠️ 未実装
2. **with_execution()メソッド実装** ⚠️ 未実装
3. **_create_configured_executor()メソッド実装** ⚠️ 未実装
4. **既存動作の後方互換性テスト** ⚠️ 必要

### 優先度 MEDIUM

5. **backend_config検証機能** - CoordinationBackend用設定の妥当性チェック
6. **エラーハンドリング強化** - 無効なbackend設定時の適切なエラーメッセージ
7. **使用例とドキュメント作成** - 新しいAPIの使用方法

### 優先度 LOW

8. **TaskWorker handler統合** - 将来的なhandler動的選択機能
9. **パフォーマンステスト** - 各バックエンドの性能比較
10. **デバッグツール** - 実行モード別のログ出力強化

### ✅ 実装済み（活用可能）

- ✅ GroupExecutorの多バックエンド対応
- ✅ CoordinationBackend列挙型
- ✅ Redis/Threading/Directバックエンド
- ✅ 統一WorkflowEngine（Directバックエンド）

## 技術的考慮事項

### 1. 設定値の妥当性

```python
from graflow.coordination.coordinator import CoordinationBackend

# 有効なバックエンド（enum値で保証済み）
VALID_BACKENDS = [
    CoordinationBackend.DIRECT,
    CoordinationBackend.THREADING,
    CoordinationBackend.REDIS
]

# backend_config の妥当性チェック例
def validate_backend_config(backend: CoordinationBackend, config: dict):
    if backend == CoordinationBackend.THREADING:
        if "thread_count" in config and not isinstance(config["thread_count"], int):
            raise ValueError("thread_count must be an integer")
    elif backend == CoordinationBackend.REDIS:
        # Redis設定の検証（将来拡張）
        pass
```

### 2. エラーハンドリング

- **BackendConfig検証**: 各バックエンド用設定の妥当性チェック
- **Redis接続失敗**: 適切なエラーメッセージとフォールバック対応
- **TaskWorker不在**: Redis分散実行時のワーカープロセス状態確認

### 3. 既存コードとの統合ポイント

- **ExecutionContext.group_executor**: with_execution()は個別設定、未設定時は既存のgroup_executorを使用
- **CoordinationBackend活用**: 既存のGroupExecutorインフラをそのまま利用
- **後方互換性**: 既存の`|`演算子コードは無変更で動作

## テスト戦略

### 1. 後方互換性テスト

```python
def test_existing_parallel_syntax():
    """既存の|演算子が正常動作することを確認"""
    with workflow("test") as wf:
        result = task_a | task_b | task_c
        # 既存動作が変わらないことを検証
        assert isinstance(result, ParallelGroup)

def test_default_execution_behavior():
    """デフォルト実行が既存動作と同じことを確認"""
    # context.group_executorまたはデフォルトGroupExecutorの動作を検証
```

### 2. 新機能テスト

```python
def test_direct_execution_backend():
    """DIRECTバックエンドの動作確認（WorkflowEngine使用）"""
    group = (task_a | task_b).with_execution(
        backend=CoordinationBackend.DIRECT
    )
    # 統一エンジンでの実行確認

def test_threading_execution_backend():
    """THREADINGバックエンドの動作確認"""
    group = (task_a | task_b).with_execution(
        backend=CoordinationBackend.THREADING,
        backend_config={"thread_count": 2}
    )

def test_redis_execution_backend():
    """REDISバックエンドの動作確認"""
    group = (task_a | task_b).with_execution(
        backend=CoordinationBackend.REDIS
    )

def test_backend_config_validation():
    """backend_config設定の妥当性チェック"""
    # 無効な設定値でのエラー確認
```

## 将来拡張

### 1. 高度なbackend_config設定

```python
# 複雑な設定が必要になった場合の階層化
(task_a | task_b).with_execution(
    backend=CoordinationBackend.REDIS,
    backend_config={
        "handler_type": "docker",
        "docker_config": {
            "image": "pytorch/pytorch:latest",
            "gpu": True,
            "memory_limit": "4g"
        },
        "redis_config": {
            "key_prefix": "ml_pipeline",
            "timeout": 300
        }
    }
)
```

### 2. 実行時動的切り替え

```python
# 環境変数や実行時条件による動的切り替え
def get_execution_backend():
    if os.getenv("USE_REDIS"):
        return CoordinationBackend.REDIS
    elif os.getenv("USE_THREADING"):
        return CoordinationBackend.THREADING
    else:
        return CoordinationBackend.DIRECT

pipeline = (task_a | task_b).with_execution(backend=get_execution_backend())
```

## まとめ

この実装計画により、ParallelGroupに柔軟な実行制御機能を段階的に追加できる。既存のGroupExecutor/CoordinationBackendインフラを最大活用し、後方互換性を維持しながら、ユーザーフレンドリーなAPIを提供する設計となっている。

実装時は最小限の変更から開始し、テストを充実させながら段階的に機能を追加していく。現在の統一エンジンアーキテクチャとの整合性も保たれている。