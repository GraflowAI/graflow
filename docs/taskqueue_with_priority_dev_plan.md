# TaskQueue 抽象化設計ドキュメント

## 概要

現在の `ExecutionContext.queue` (deque) を抽象化し、Memory、Redis等の複数のバックエンドに対応したTaskQueueアーキテクチャを設計する。

## 現状分析

### 現在のキュー使用パターン（実装調査結果）

**ExecutionContext（context.py:133-180）**:
```python
self.queue = deque([start_node])  # 初期化
self.queue.append(node)          # add_to_queue()
self.queue.popleft()             # get_next_node()
not self.queue                   # is_completed()
```

**WorkflowEngine（engine.py:37-65）**:
```python
while not context.is_completed():
    node = context.get_next_node()
    # ... task execution ...
    context.add_to_queue(succ)  # successor nodes
```

**テストコード**:
```python
exec_context.add_to_queue("task_a")
exec_context.add_to_queue("task_b")
```

### 現在の実装の特徴
- **シンプルなFIFO**: `deque`でノードID（文字列）のみを管理
- **文字列ベース**: TaskSpecやメタデータは使用せず、純粋にnode_idのみ
- **直接的なAPI**: `add_to_queue(str)`, `get_next_node() -> str`, `is_empty() -> bool`
- **最小限の機能**: 優先度、依存関係、リトライ機能なし

### 目標
1. **段階的移行**: 既存のdeque実装との完全互換性を最優先
2. **最小限の抽象化**: 既存の文字列ベースAPIを保持
3. **将来の拡張性**: 後からRedis等のバックエンドを追加可能
4. **実用主義**: 複雑な機能は実際に必要になってから追加

## アーキテクチャ設計

### 1. 抽象基底クラス（最小限のインターフェース）

```python
# graflow/core/queue/base.py
from abc import ABC, abstractmethod
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..context import ExecutionContext

class AbstractTaskQueue(ABC):
    """TaskQueueの抽象基底クラス（TaskSpec対応）"""
    
    def __init__(self, execution_context: 'ExecutionContext'):
        self.execution_context = execution_context
        self._task_specs: Dict[str, TaskSpec] = {}
    
    # === 必須のコアインターフェース ===
    @abstractmethod
    def enqueue(self, task_spec: TaskSpec) -> bool:
        """TaskSpecをキューに追加"""
        pass
    
    @abstractmethod  
    def dequeue(self) -> Optional[TaskSpec]:
        """次のTaskSpecを取得"""
        pass
    
    @abstractmethod
    def is_empty(self) -> bool:
        """キューが空かどうか"""
        pass
    
    # === 既存API互換メソッド ===
    def add_node(self, node_id: str, priority: TaskPriority = TaskPriority.NORMAL) -> None:
        """ノードIDをキューに追加（ExecutionContext.add_to_queue互換）"""
        task_spec = TaskSpec(
            node_id=node_id,
            execution_context=self.execution_context,
            priority=priority
        )
        self.enqueue(task_spec)
    
    def add_node_with_priority(self, node_id: str, priority: int) -> None:
        """整数値で優先度指定してノードを追加"""
        priority_enum = TaskPriority.from_int(priority)
        self.add_node(node_id, priority_enum)
    
    def get_next_node(self) -> Optional[str]:
        """次の実行ノードIDを取得（ExecutionContext.get_next_node互換）"""
        task_spec = self.dequeue()
        return task_spec.node_id if task_spec else None
    
    # === 拡張用オプションインターフェース ===
    def size(self) -> int:
        """待機中のノード数"""
        return 0
    
    def peek_next_node(self) -> Optional[str]:
        """次のノードを削除せずに確認"""
        return None
    
    def get_task_spec(self, node_id: str) -> Optional[TaskSpec]:
        """TaskSpecを取得"""
        return self._task_specs.get(node_id)

# === TaskSpec関連（Phase 1から導入） ===
from dataclasses import dataclass, field
from enum import IntEnum
import time
from typing import Any, Dict, List

class TaskPriority(IntEnum):
    """Task priority levels with type safety"""
    VERY_LOW = 1    # Lowest priority, minimal time weight
    LOW = 2         # Low priority
    NORMAL = 3      # Standard priority
    HIGH = 4        # High priority
    VERY_HIGH = 5   # Very high priority, maximum time weight
    CRITICAL = 6    # Critical priority, uses separate FIFO queue
    
    @property
    def weight(self) -> float:
        """Get priority weight for score calculation (1-5 only)"""
        if self == TaskPriority.CRITICAL:
            raise ValueError("Critical priority does not use weight calculation")
        return self.value / 5.0
    
    @property
    def is_critical(self) -> bool:
        """Check if priority is critical"""
        return self == TaskPriority.CRITICAL
    
    @classmethod
    def from_int(cls, value: int) -> "TaskPriority":
        """Create Priority from integer with validation"""
        try:
            return cls(value)
        except ValueError:
            raise ValueError(f"Invalid priority value: {value}. Must be 1-6.")

class TaskStatus(Enum):
    """タスクの状態管理"""
    BLOCKED = "blocked"    # 依存待ち
    READY = "ready"        # 実行可能
    RUNNING = "running"    # 実行中
    SUCCESS = "success"    # 成功完了
    ERROR = "error"        # エラー

@dataclass
class TaskSpec:
    """タスク仕様とメタデータ（Phase 1から導入）"""
    node_id: str
    execution_context: 'ExecutionContext'
    priority: TaskPriority = TaskPriority.NORMAL
    status: TaskStatus = TaskStatus.READY
    dependencies: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def calculate_score(self) -> float:
        """Calculate priority score for normal queue ordering
        
        Higher score = higher priority for processing
        Formula: created_at + (priority_weight * elapsed_time)
        This prevents starvation while respecting priority levels
        """
        if self.priority.is_critical:
            raise ValueError("Critical tasks do not use score calculation")
        
        current_time = time.time()
        elapsed_time = current_time - self.created_at
        priority_weight = self.priority.weight
        
        return self.created_at + (priority_weight * elapsed_time)
    
    def __lt__(self, other: 'TaskSpec') -> bool:
        """優先度キューでのソート用"""
        # CRITICALタスクは常に最優先
        if self.priority.is_critical != other.priority.is_critical:
            return self.priority.is_critical
        
        if self.priority.is_critical and other.priority.is_critical:
            # CRITICAL同士はFIFO（古いものが先）
            return self.created_at < other.created_at
        
        # 通常優先度はスコアベース（高スコアが先）
        return self.calculate_score() > other.calculate_score()
```

### 2. Memory実装（deque完全互換）

```python
# graflow/core/queue/memory.py
from collections import deque
from typing import Optional
from .base import AbstractTaskQueue

class SimpleMemoryQueue(AbstractTaskQueue):
    """既存のdeque完全互換クラス（TaskSpec対応・Phase 1実装）"""
    
    def __init__(self, execution_context, start_node: Optional[str] = None):
        super().__init__(execution_context)
        self._queue = deque()
        if start_node:
            task_spec = TaskSpec(
                node_id=start_node,
                execution_context=execution_context
            )
            self._queue.append(task_spec)
            self._task_specs[start_node] = task_spec
    
    def enqueue(self, task_spec: TaskSpec) -> bool:
        """TaskSpecをキューに追加（FIFO）"""
        self._queue.append(task_spec)
        self._task_specs[task_spec.node_id] = task_spec
        return True
    
    def dequeue(self) -> Optional[TaskSpec]:
        """次のTaskSpecを取得"""
        if self._queue:
            task_spec = self._queue.popleft()
            task_spec.status = TaskStatus.RUNNING
            return task_spec
        return None
    
    def is_empty(self) -> bool:
        """キューが空かどうか"""
        return len(self._queue) == 0
    
    def size(self) -> int:
        """待機中のTaskSpec数"""
        return len(self._queue)
    
    def peek_next_node(self) -> Optional[str]:
        """次のノードを削除せずに確認"""
        return self._queue[0].node_id if self._queue else None

# === Phase 2以降で実装予定 ===
class PriorityMemoryQueue(AbstractTaskQueue):
    """優先度機能付きメモリキュー（Phase 2実装予定）"""
    
    def __init__(self, execution_context):
        super().__init__(execution_context)
        from queue import PriorityQueue
        self._normal_queue = PriorityQueue()  # VERY_LOW～VERY_HIGH用
        self._critical_queue = deque()        # CRITICAL専用FIFO
        
    def enqueue(self, task_spec: TaskSpec) -> bool:
        """TaskSpecを優先度付きで追加"""
        self._task_specs[task_spec.node_id] = task_spec
        
        if task_spec.priority.is_critical:
            # CRITICALタスクは専用FIFO
            self._critical_queue.append(task_spec)
        else:
            # 通常優先度はスコアベース
            # PriorityQueueは小さい値が先なので、負の値を使用
            score = -task_spec.calculate_score()
            self._normal_queue.put((score, task_spec.created_at, task_spec))
        
        return True
    
    def dequeue(self) -> Optional[TaskSpec]:
        """優先度を考慮して次のTaskSpecを取得"""
        # CRITICALタスクが最優先
        if self._critical_queue:
            task_spec = self._critical_queue.popleft()
            task_spec.status = TaskStatus.RUNNING
            return task_spec
        
        # 通常優先度タスク
        if not self._normal_queue.empty():
            _, _, task_spec = self._normal_queue.get()
            task_spec.status = TaskStatus.RUNNING
            return task_spec
        
        return None
    
    def is_empty(self) -> bool:
        """キューが空かどうか"""
        return len(self._critical_queue) == 0 and self._normal_queue.empty()
    
    def size(self) -> int:
        """待機中のTaskSpec数"""
        return len(self._critical_queue) + self._normal_queue.qsize()
```

### 3. Redis実装（Phase 3実装予定）

```python
# graflow/core/queue/redis.py
import redis
from typing import Optional
from .base import AbstractTaskQueue

class RedisTaskQueue(AbstractTaskQueue):
    """Redis分散タスクキュー（Phase 3実装予定）"""
    
    def __init__(self, execution_context, redis_client: Optional[redis.Redis] = None, 
                 key_prefix: str = "graflow"):
        super().__init__(execution_context)
        
        self.redis = redis_client or redis.Redis(
            host='localhost', port=6379, db=0, decode_responses=True
        )
        self.key_prefix = key_prefix
        self.session_id = execution_context.session_id
        
        # Redis keys
        self.queue_key = f"{key_prefix}:queue:{self.session_id}"
    
    def add_node(self, node_id: str) -> None:
        """RedisリストにノードIDを追加（FIFO）"""
        self.redis.rpush(self.queue_key, node_id)
    
    def get_next_node(self) -> Optional[str]:
        """Redisリストから次のノードIDを取得"""
        result = self.redis.lpop(self.queue_key)
        return result if result else None
    
    def is_empty(self) -> bool:
        """Redisキューが空かどうか"""
        return self.redis.llen(self.queue_key) == 0
    
    def size(self) -> int:
        """Redisキューのサイズ"""
        return self.redis.llen(self.queue_key)
    
    def peek_next_node(self) -> Optional[str]:
        """次のノードを削除せずに確認"""
        result = self.redis.lindex(self.queue_key, 0)
        return result if result else None
    
    def cleanup(self) -> None:
        """セッション終了時のクリーンアップ"""
        self.redis.delete(self.queue_key)
```

### 4. Factory パターン

```python
# graflow/core/queue/factory.py
from enum import Enum
from typing import Optional, Dict, Any
from .base import AbstractTaskQueue
from .memory import SimpleMemoryQueue

class QueueBackend(Enum):
    """利用可能なキューバックエンド"""
    SIMPLE_MEMORY = "simple_memory"    # deque互換（Phase 1）
    # MEMORY = "memory"                # 拡張メモリキュー（Phase 2）
    # REDIS = "redis"                  # Redis分散キュー（Phase 3）

class TaskQueueFactory:
    """TaskQueueのファクトリークラス"""
    
    @staticmethod
    def create(backend: QueueBackend, execution_context, **kwargs) -> AbstractTaskQueue:
        """指定されたバックエンドでTaskQueueを作成"""
        
        if backend == QueueBackend.SIMPLE_MEMORY:
            start_node = kwargs.get('start_node')
            return SimpleMemoryQueue(execution_context, start_node)
        
        # Phase 2以降で実装予定
        # elif backend == QueueBackend.MEMORY:
        #     from .memory import MemoryTaskQueue
        #     return MemoryTaskQueue(execution_context)
        #     
        # elif backend == QueueBackend.REDIS:
        #     from .redis import RedisTaskQueue
        #     redis_config = kwargs.get('redis_config', {})
        #     return RedisTaskQueue(execution_context, **redis_config)
        
        else:
            raise ValueError(f"Unsupported backend: {backend}")
    
    @staticmethod
    def create_from_config(execution_context, config: Dict[str, Any]) -> AbstractTaskQueue:
        """設定辞書からTaskQueueを作成"""
        backend_name = config.get('backend', 'simple_memory')
        backend = QueueBackend(backend_name)
        
        # Extract backend-specific config
        backend_config = config.get('config', {})
        
        return TaskQueueFactory.create(backend, execution_context, **backend_config)
```

### 5. ExecutionContext統合（Phase 1：最小限の変更）

```python
# graflow/core/context.py への統合コード（段階的移行）
from typing import Optional, Union
from .queue.factory import TaskQueueFactory, QueueBackend
from .queue.base import AbstractTaskQueue

class ExecutionContext:
    def __init__(
        self,
        graph: TaskGraph,
        start_node: Optional[str] = None,
        max_steps: int = 10,
        default_max_cycles: int = 10,
        default_max_retries: int = 3,
        steps: int = 0,
        # Phase 1: オプショナルパラメータとして追加
        queue_backend: Union[QueueBackend, str] = QueueBackend.SIMPLE_MEMORY,
        queue_config: Optional[Dict[str, Any]] = None
    ):
        """Initialize ExecutionContext with configurable queue backend."""
        session_id = str(uuid.uuid4().int)
        self.session_id = session_id
        self.graph = graph
        self.start_node = start_node
        self.max_steps = max_steps
        self.default_max_retries = default_max_retries
        self.steps = steps
        self.executed = []

        # Phase 1: 既存のdequeを抽象化
        if isinstance(queue_backend, str):
            queue_backend = QueueBackend(queue_backend)
            
        queue_config = queue_config or {}
        if start_node:
            queue_config['start_node'] = start_node
            
        self.task_queue: AbstractTaskQueue = TaskQueueFactory.create(
            queue_backend, self, **queue_config
        )
        
        # 既存の初期化コード（変更なし）
        self.cycle_controller = CycleController(default_max_cycles)
        self.channel = MemoryChannel(session_id)
        self._task_execution_stack: list[TaskExecutionContext] = []
        self._task_contexts: dict[str, TaskExecutionContext] = {}
    
    @property
    def queue(self):
        """後方互換性：既存のdeque風アクセス"""
        # Phase 1では完全互換性を保持
        if hasattr(self.task_queue, '_queue'):
            return self.task_queue._queue
        else:
            # フォールバック（将来の実装用）
            return self.task_queue
    
    def add_to_queue(self, node: str) -> None:
        """既存のadd_to_queueメソッドの完全互換"""
        self.task_queue.add_node(node)
    
    def get_next_node(self) -> Optional[str]:
        """既存のget_next_nodeメソッドの完全互換"""
        return self.task_queue.get_next_node()
    
    def is_completed(self) -> bool:
        """既存のis_completedメソッドの完全互換"""
        return self.task_queue.is_empty() or self.steps >= self.max_steps
```

## 使用例

### 1. 既存コード（後方互換）

```python
# 既存のコードは変更なしで動作
context = ExecutionContext(graph, start_node="task1")
context.execute()
```

### 2. Memory拡張モード

```python
# 優先度とリトライ機能付き
context = ExecutionContext(
    graph, 
    start_node="task1",
    queue_backend=QueueBackend.MEMORY
)
context.execute()
```

### 3. Redis分散モード

```python
# Redis分散実行
redis_config = {
    'host': 'redis-server',
    'port': 6379,
    'db': 0,
    'key_prefix': 'myworkflow'
}

context = ExecutionContext(
    graph,
    start_node="task1", 
    queue_backend=QueueBackend.REDIS,
    queue_config={'redis_config': redis_config}
)
context.execute()
```

### 4. 設定ファイルベース

```python
# YAML/JSONの設定から作成
config = {
    'backend': 'redis',
    'config': {
        'redis_config': {
            'host': 'localhost',
            'port': 6379,
            'key_prefix': 'workflow'
        }
    }
}

queue = TaskQueueFactory.create_from_config(execution_context, config)
```

## 段階的実装ロードマップ（優先度・状態管理を見越した設計）

### Phase 1: 基礎実装・完全互換性確保 (1週間)
**目標**: 既存deque実装との100%互換性

**実装内容**:
1. `AbstractTaskQueue`基底クラス（最小限API）
   - `add_node(node_id: str)`
   - `get_next_node() -> Optional[str]`
   - `is_empty() -> bool`

2. `SimpleMemoryQueue`（deque完全互換）
   - 内部で`collections.deque`を使用
   - 文字列ベースのFIFOキュー

3. ExecutionContextの最小限統合
   - 既存APIを完全保持
   - `queue_backend`オプションパラメータ追加

4. 基本テスト作成
   - 既存テストが全て通過することを確認

**成功指標**: 全既存テストがpassし、実行時間が変わらない

### Phase 2: 拡張インターフェース設計 (1週間)
**目標**: 将来の優先度・状態管理機能の基盤構築

**実装内容**:
1. 拡張抽象インターフェース追加
```python
class ExtendedTaskQueue(AbstractTaskQueue):
    """将来の拡張機能用インターフェース"""
    
    def add_node_with_priority(self, node_id: str, priority: int = 2) -> None:
        """優先度付きでノードを追加"""
        # Phase 2ではデフォルト実装でadd_nodeに委譲
        self.add_node(node_id)
    
    def get_node_status(self, node_id: str) -> str:
        """ノードの状態を取得（準備用）"""
        return "unknown"  # Phase 2では固定値
    
    def set_node_dependencies(self, node_id: str, deps: List[str]) -> None:
        """依存関係設定（準備用）"""
        pass  # Phase 2では何もしない
```

2. `MemoryTaskQueue`（拡張版）の骨格実装
   - 内部的に優先度キュー準備
   - Phase 2では従来のFIFO動作を維持

3. TaskSpec・TaskStatus・TaskPriorityクラス設計
   - データ構造定義のみ（実際の使用はPhase 3以降）

**成功指標**: 拡張インターフェース経由でも既存動作が保持される

### Phase 3: 優先度機能実装 (2週間)
**目標**: 優先度ベースのタスクスケジューリング

**実装内容**:
1. TaskSpec完全実装
```python
@dataclass
class TaskSpec:
    node_id: str
    priority: TaskPriority = TaskPriority.NORMAL
    status: TaskStatus = TaskStatus.READY
    dependencies: List[str] = field(default_factory=list)
    enqueue_time: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
```

2. `PriorityMemoryQueue`実装
   - `queue.PriorityQueue`を使用
   - 優先度・時間ベースのソート

3. ExecutionContextの拡張統合
```python
# 新しい使用パターン
context = ExecutionContext(
    graph,
    start_node="task1",
    queue_backend="priority_memory"
)

# 優先度付きでタスクを追加
context.add_to_queue_with_priority("urgent_task", priority=TaskPriority.HIGH)
```

4. 後方互換性維持
   - `add_to_queue(node)`は従来通り動作
   - `queue_backend="simple_memory"`がデフォルト

**成功指標**: 優先度機能が動作し、既存コードも変更なしで動作

### Phase 4: 状態管理・依存関係 (2週間)  
**目標**: タスクの状態追跡と依存関係管理

**実装内容**:
1. 状態管理システム
```python
class TaskStatus:
    PENDING = "pending"      # キュー待機中
    READY = "ready"         # 実行可能
    RUNNING = "running"     # 実行中  
    COMPLETED = "completed" # 完了
    FAILED = "failed"       # 失敗
    BLOCKED = "blocked"     # 依存待ち
```

2. 依存関係エンジン
   - ノード間の依存関係チェック
   - 依存解決後の自動キューイング

3. `StatefulMemoryQueue`実装
   - タスク状態の永続化
   - 依存関係ベースのスケジューリング

**成功指標**: 複雑な依存関係を持つワークフローが正しく実行される

### Phase 5: Redis分散実装 (2週間)
**目標**: 分散環境での優先度・状態管理

**実装内容**:
1. `RedisTaskQueue`完全実装
   - Redis Sorted Set (ZADD/ZPOPMIN) で優先度キュー
   - Redis Hash で状態管理
   - Redis Set で依存関係管理

2. 分散状態同期
   - 複数ワーカー間での状態一貫性
   - リーダー選択とロック機構

3. 障害回復機能
   - ワーカー障害時の自動回復
   - 未完了タスクの再キューイング

**成功指標**: 複数ノードでの分散実行が安定動作

### Phase 6: 高度な機能・最適化 (2週間)
**目標**: プロダクション使用に向けた機能完成

**実装内容**:
1. リトライ機能
   - 指数バックオフによる再実行
   - 最大リトライ回数制御

2. デッドレター機能
   - 失敗タスクの隔離
   - 手動での再実行機能

3. メトリクス・監視
   - キューサイズ、処理時間等の計測
   - ダッシュボード用API

4. パフォーマンス最適化
   - バッチ処理
   - 接続プール

**成功指標**: 実際の本番環境で安定稼動

## 段階的移行戦略

### 互換性レベル
- **Level 0**: 既存dequeのdrop-in replacement (Phase 1)
- **Level 1**: オプション機能として拡張API (Phase 2-3)  
- **Level 2**: 新機能をデフォルト有効化 (Phase 4-5)
- **Level 3**: 旧APIのdeprecation (Phase 6)

### 設定駆動の段階的有効化
```python
# Phase 1: デフォルト（変更なし）
context = ExecutionContext(graph, start_node="task1")

# Phase 2: オプション機能
context = ExecutionContext(
    graph, start_node="task1",
    queue_backend="priority_memory",  # 優先度機能
    queue_config={"enable_priority": True}
)

# Phase 3: 状態管理
context = ExecutionContext(
    graph, start_node="task1", 
    queue_backend="stateful_memory",
    queue_config={
        "enable_priority": True,
        "enable_status_tracking": True,
        "enable_dependencies": True
    }
)

# Phase 4: 分散実行
context = ExecutionContext(
    graph, start_node="task1",
    queue_backend="redis",
    queue_config={
        "redis_config": {"host": "redis-cluster"},
        "enable_all_features": True
    }
)
```

## 利点

1. **段階的移行**: 既存コードを壊さずに新機能を追加
2. **スケーラビリティ**: Redis等でクラスタ環境に対応
3. **拡張性**: 新しいバックエンド（Database、SQS等）を容易に追加
4. **柔軟性**: 用途に応じて最適なキューを選択可能
5. **テスタビリティ**: モックしやすい抽象化設計