# ParallelGroup Execution Mode Implementation Plan

## 概要

ParallelGroupにチェーンメソッド形式のexecution mode制御機能を段階的に実装する。現在のGroupExecutor/TaskWorkerシステムとの統合を重視し、後方互換性を維持しながら柔軟な実行制御を提供する。

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
class ParallelGroup(Executable):
    def __init__(self, tasks: list[Executable]) -> None:
        # 既存実装（変更なし）
        self._task_id = self._get_group_name()
        self.tasks = list(tasks)
        
        # 新規追加：実行設定
        self._execution_config = {
            "mode": "auto",        # "auto" | "local" | "distributed"
            "backend": "redis",    # "redis" | "multiprocessing" 
            "handler": "inprocess" # TaskWorker向け設定
        }
        
        # 既存の依存関係設定（変更なし）
        self._register_to_context()
        for task in self.tasks:
            self._add_dependency_edge(self._task_id, task.task_id)

    def with_execution(self, mode: str = None, backend: str = None, 
                      handler: str = None) -> 'ParallelGroup':
        """Configure execution mode for this parallel group.
        
        Args:
            mode: "auto", "local", "distributed"
            backend: "redis", "multiprocessing" 
            handler: "inprocess", "async", "docker"
        
        Returns:
            Self for method chaining
        """
        if mode is not None:
            self._execution_config["mode"] = mode
        if backend is not None:
            self._execution_config["backend"] = backend  
        if handler is not None:
            self._execution_config["handler"] = handler
        return self
```

#### 1.2 実行ロジック統合

```python
    def run(self) -> Any:
        """Execute all tasks in this parallel group."""
        context = self.get_execution_context()
        
        # 実行設定に基づいてGroupExecutor作成
        executor = self._create_configured_executor(context)
        
        # 既存のTaskSpec作成ロジック（変更なし）
        task_specs: List[TaskSpec] = []
        # ... 既存実装をそのまま維持 ...
        
        executor.execute_parallel_group(self.task_id, task_specs, context)

    def _create_configured_executor(self, context: 'ExecutionContext') -> GroupExecutor:
        """Create GroupExecutor based on execution configuration."""
        mode = self._execution_config["mode"]
        
        if mode == "local":
            return GroupExecutor(CoordinationBackend.DIRECT)
            
        elif mode == "distributed":
            backend_name = self._execution_config["backend"]
            if backend_name == "redis":
                backend_config = {"handler": self._execution_config["handler"]}
                return GroupExecutor(CoordinationBackend.REDIS, backend_config)
            elif backend_name == "multiprocessing":
                return GroupExecutor(CoordinationBackend.MULTIPROCESSING)
            else:
                raise ValueError(f"Unsupported backend: {backend_name}")
                
        else:  # mode == "auto"
            # 既存動作を維持
            return context.group_executor or GroupExecutor()
```

### Phase 2: TaskHandler設定の統合

#### 2.1 GroupExecutor拡張

```python
class GroupExecutor:
    def execute_parallel_group(self, group_id: str, tasks: List[TaskSpec], 
                              exec_context: 'ExecutionContext') -> None:
        if self.backend == CoordinationBackend.REDIS:
            coordinator = self._create_coordinator(self.backend, self.backend_config, exec_context)
            
            # handler設定をCoordinatorに渡す
            handler_type = self.backend_config.get("handler", "inprocess")
            coordinator.execute_group_with_handler(group_id, tasks, handler_type)
        else:
            # 既存実装
            coordinator = self._create_coordinator(self.backend, self.backend_config, exec_context)
            coordinator.execute_group(group_id, tasks)
```

#### 2.2 RedisCoordinator拡張

```python
class RedisCoordinator:
    def execute_group_with_handler(self, group_id: str, tasks: List[TaskSpec], 
                                  handler_type: str) -> None:
        """Execute group with specific TaskWorker handler type."""
        # TaskSpecにhandler情報を付与してdispatch
        for task_spec in tasks:
            task_spec.handler_type = handler_type  # TaskSpecを拡張
            self.dispatch_task(task_spec, group_id)
        
        # 既存のバリア同期処理
        self.wait_for_group_completion(group_id, len(tasks))
```

### Phase 3: 使用例とテスト

#### 3.1 基本使用例

```python
# 既存コード（変更なし）
train_a | train_b | train_c

# ローカル並列実行
(light_task_a | light_task_b).with_execution(mode="local")

# Redis分散実行
(heavy_task_a | heavy_task_b | heavy_task_c).with_execution(
    mode="distributed",
    backend="redis"
)

# Docker環境での分散実行
(gpu_task_a | gpu_task_b).with_execution(
    mode="distributed", 
    backend="redis",
    handler="docker"
)
```

#### 3.2 複合使用例

```python
with workflow("ml_pipeline") as wf:
    data_prep = Task("data_prep")
    
    # 分散Docker実行
    training = (train_a | train_b | train_c).with_execution(
        mode="distributed",
        backend="redis",
        handler="docker"
    )
    
    # ローカル並列実行
    validation = (validate_a | validate_b).with_execution(mode="local")
    
    evaluation = Task("evaluation")
    
    data_prep >> training >> validation >> evaluation
```

## 実装順序

### 優先度 HIGH

1. **ParallelGroup._execution_config追加**
2. **with_execution()メソッド実装**
3. **_create_configured_executor()メソッド実装**
4. **既存動作の後方互換性テスト**

### 優先度 MEDIUM

5. **GroupExecutorのhandler設定対応**
6. **RedisCoordinatorのhandler統合**
7. **TaskSpecのhandler情報拡張**

### 優先度 LOW

8. **エラーハンドリング強化**
9. **設定値検証機能**
10. **ドキュメント更新**

## 技術的考慮事項

### 1. 設定値の妥当性

```python
VALID_MODES = {"auto", "local", "distributed"}
VALID_BACKENDS = {"redis", "multiprocessing"}
VALID_HANDLERS = {"inprocess", "async", "docker"}
```

### 2. エラーハンドリング

- 無効な設定値の検証
- Redis接続失敗時のフォールバック
- TaskWorkerプロセス不在時の適切なエラーメッセージ

### 3. 既存コードとの統合ポイント

- `ExecutionContext.group_executor`との関係
- `CoordinationBackend`列挙型の活用
- `TaskSpec`構造との整合性

## テスト戦略

### 1. 後方互換性テスト

```python
def test_existing_parallel_syntax():
    """既存の|演算子が正常動作することを確認"""
    with workflow("test") as wf:
        result = task_a | task_b | task_c
        # 既存動作が変わらないことを検証

def test_auto_mode_default():
    """デフォルトがautoモードで既存動作と同じことを確認"""
    # ...
```

### 2. 新機能テスト

```python
def test_local_execution_mode():
    """ローカル実行モードの動作確認"""
    # ...

def test_distributed_execution_mode():
    """分散実行モードの動作確認"""
    # ...

def test_handler_configuration():
    """handler設定がTaskWorkerに正しく渡されることを確認"""
    # ...
```

## 将来拡張

### 1. 設定の階層化

```python
# 将来的に複雑な設定が必要になった場合
(task_a | task_b).with_execution(
    mode="distributed",
    backend="redis",
    config={
        "handler": "docker",
        "docker": {
            "image": "pytorch/pytorch:latest",
            "gpu": True
        },
        "redis": {
            "key_prefix": "ml_pipeline"
        }
    }
)
```

### 2. 実行時動的切り替え

```python
# 環境変数や実行時条件による動的切り替え
def get_execution_mode():
    return "distributed" if os.getenv("USE_DISTRIBUTED") else "local"

pipeline = (task_a | task_b).with_execution(mode=get_execution_mode())
```

## まとめ

この実装計画により、ParallelGroupに柔軟な実行制御機能を段階的に追加できる。既存システムとの統合を重視し、後方互換性を維持しながら、TaskWorkerシステムの活用を最大化する設計となっている。

実装時は最小限の変更から開始し、テストを充実させながら段階的に機能を追加していく。