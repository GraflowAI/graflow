# Graflow Architecture

Graflowは分散タスク実行とワークフロー管理のためのPythonフレームワークです。この文書では、主要コンポーネントとそれらの関係について説明します。

## 全体アーキテクチャ

```mermaid
graph TB
    subgraph "Core Components"
        EC[ExecutionContext]
        WE[WorkflowEngine]
        TG[TaskGraph]
    end
    
    subgraph "Task Management"
        TQ[TaskQueue]
        TS[TaskSpec]
        TH[TaskHandler]
    end
    
    subgraph "Communication"
        CH[Channel]
        RC[RedisChannel]
        MC[MemoryChannel]
    end
    
    subgraph "Worker & Coordination"
        TW[TaskWorker]
        RCO[RedisCoordinator]
        GE[GroupExecutor]
    end
    
    subgraph "Backend Storage"
        Redis[(Redis)]
        Memory[(Memory)]
    end
    
    WE --> EC
    EC --> TQ
    EC --> CH
    EC --> GE
    TW --> TQ
    TW --> TH
    RCO --> TQ
    RCO --> Redis
    RC --> Redis
    CH --> RC
    CH --> MC
    TQ --> Redis
    TQ --> Memory
```

## コンポーネント詳細

### ExecutionContext
ワークフロー実行の中心的なオーケストレーター

- **役割**: タスクの実行状態管理、キューイング、結果保存
- **主要機能**:
  - TaskQueueとChannelの管理
  - タスクの実行コンテキスト管理
  - サイクル制御とリトライ処理
  - 動的タスク生成とgoto機能

### TaskQueue
タスクのキューイング管理の抽象基底クラス

- **実装**: 
  - `MemoryTaskQueue`: インメモリ実装
  - `RedisTaskQueue`: Redis分散実装
- **機能**:
  - TaskSpecの enqueue/dequeue
  - メトリクス収集
  - リトライ処理

### Channel
タスク間通信の抽象基底クラス

- **実装**:
  - `MemoryChannel`: インメモリ実装
  - `RedisChannel`: Redis分散実装
- **機能**:
  - キー・バリューストア
  - TTL対応
  - タスク結果の保存・取得

### TaskWorker
キューからタスクを取得して実行するワーカー

- **機能**:
  - 並行タスク処理
  - メトリクス収集
  - グレースフルシャットダウン
  - タイムアウト処理

### RedisCoordinator
Redis基盤の分散タスク調整

- **機能**:
  - 並列グループ実行
  - バリア同期
  - タスクディスパッチ

## Redis分散アーキテクチャ

```mermaid
graph TB
    subgraph "Application Layer"
        APP[Application]
        WE[WorkflowEngine]
    end
    
    subgraph "Execution Layer"
        EC[ExecutionContext]
        RCO[RedisCoordinator]
    end
    
    subgraph "Queue Layer"
        RTQ[RedisTaskQueue]
        TS[TaskSpec]
    end
    
    subgraph "Worker Layer"
        TW1[TaskWorker-1]
        TW2[TaskWorker-2]
        TW3[TaskWorker-N]
        TH[TaskHandler]
    end
    
    subgraph "Communication Layer"
        RC[RedisChannel]
        Redis[(Redis Server)]
    end
    
    APP --> WE
    WE --> EC
    EC --> RCO
    EC --> RTQ
    EC --> RC
    
    RCO --> RTQ
    RCO --> Redis
    
    RTQ --> Redis
    RC --> Redis
    
    TW1 --> RTQ
    TW2 --> RTQ
    TW3 --> RTQ
    
    TW1 --> TH
    TW2 --> TH
    TW3 --> TH
    
    TW1 --> RC
    TW2 --> RC
    TW3 --> RC
```

## データフロー

### 基本タスク実行フロー

```mermaid
sequenceDiagram
    participant App as Application
    participant WE as WorkflowEngine
    participant EC as ExecutionContext
    participant TQ as TaskQueue
    participant CH as Channel
    participant TW as TaskWorker
    participant TH as TaskHandler
    participant Redis as Redis Server
    
    App->>WE: execute(workflow)
    WE->>EC: execute()
    
    Note over EC: タスクの初期化
    EC->>TQ: enqueue(TaskSpec)
    TQ->>Redis: LPUSH task_queue
    
    Note over TW: ワーカーループ開始
    loop Task Processing
        TW->>TQ: dequeue()
        TQ->>Redis: BRPOP task_queue
        Redis-->>TQ: TaskSpec
        TQ-->>TW: TaskSpec
        
        Note over TW: タスク実行開始
        TW->>TW: _submit_task(TaskSpec)
        TW->>TH: process_task(task)
        
        alt Task Success
            TH->>TH: _process_task(task)
            TH-->>TW: True
            TW->>CH: set(result_key, result)
            CH->>Redis: SET result_key
            TW->>TW: _update_metrics(success=True)
        else Task Failure
            TH->>TH: _process_task(task)
            TH-->>TW: False/Exception
            TW->>TW: _update_metrics(success=False)
            
            alt Retry Enabled
                TW->>TQ: enqueue(TaskSpec) [retry]
                TQ->>Redis: LPUSH task_queue
            end
        end
        
        Note over TW: タスク完了処理
        TW->>TW: _task_completed()
    end
    
    Note over EC: 結果取得
    EC->>CH: get(result_key)
    CH->>Redis: GET result_key
    Redis-->>CH: result
    CH-->>EC: result
    EC-->>WE: execution_result
    WE-->>App: workflow_result
```

### 並列グループ実行フロー

```mermaid
sequenceDiagram
    participant EC as ExecutionContext
    participant RCO as RedisCoordinator
    participant TQ as TaskQueue
    participant TW1 as TaskWorker-1
    participant TW2 as TaskWorker-2
    participant Redis as Redis Server
    
    EC->>RCO: execute_group(group_id, tasks)
    
    Note over RCO: バリア作成
    RCO->>Redis: create_barrier(group_id, count)
    Redis-->>RCO: barrier_created
    
    Note over RCO: タスク分散
    loop For each task
        RCO->>TQ: enqueue(TaskSpec)
        TQ->>Redis: LPUSH task_queue
    end
    
    Note over TW1,TW2: 並列実行
    par TaskWorker-1
        TW1->>TQ: dequeue()
        TQ->>Redis: BRPOP task_queue
        Redis-->>TW1: TaskSpec-1
        TW1->>TW1: process_task()
        TW1->>Redis: signal_barrier(group_id)
        TW1->>Redis: SET result_key
    and TaskWorker-2  
        TW2->>TQ: dequeue()
        TQ->>Redis: BRPOP task_queue
        Redis-->>TW2: TaskSpec-2
        TW2->>TW2: process_task()
        TW2->>Redis: signal_barrier(group_id)
        TW2->>Redis: SET result_key
    end
    
    Note over RCO: バリア待機
    RCO->>Redis: wait_barrier(group_id)
    Redis->>Redis: INCR barrier_counter
    Redis->>Redis: PUBLISH barrier_done
    Redis-->>RCO: barrier_complete
    
    RCO->>Redis: cleanup_barrier(group_id)
    RCO-->>EC: group_completed
```

## バックエンド設定

```mermaid
graph LR
    subgraph "Queue Backends"
        QF[QueueFactory]
        MTQ[MemoryTaskQueue]
        RTQ[RedisTaskQueue]
    end
    
    subgraph "Channel Backends"
        CF[ChannelFactory]
        MC[MemoryChannel]
        RC[RedisChannel]
    end
    
    subgraph "Configuration"
        CONFIG[Config]
        QB[queue_backend]
        CB[channel_backend]
    end
    
    CONFIG --> QB
    CONFIG --> CB
    
    QB --> QF
    CB --> CF
    
    QF --> MTQ
    QF --> RTQ
    
    CF --> MC
    CF --> RC
    
    RTQ --> Redis
    RC --> Redis
```

## 並列グループ実行

```mermaid
graph TB
    subgraph "Coordination"
        RCO[RedisCoordinator]
        BR[Barrier]
    end
    
    subgraph "Task Distribution"
        RTQ[RedisTaskQueue]
        TS1[TaskSpec-1]
        TS2[TaskSpec-2]
        TS3[TaskSpec-N]
    end
    
    subgraph "Worker Pool"
        TW1[TaskWorker-1]
        TW2[TaskWorker-2]
        TW3[TaskWorker-N]
    end
    
    subgraph "Synchronization"
        Redis[(Redis Server)]
        BC[Barrier Counter]
        PS[PubSub Channel]
    end
    
    RCO --> BR
    RCO --> RTQ
    
    RTQ --> TS1
    RTQ --> TS2
    RTQ --> TS3
    
    TS1 --> TW1
    TS2 --> TW2
    TS3 --> TW3
    
    TW1 --> BC
    TW2 --> BC
    TW3 --> BC
    
    BC --> PS
    PS --> BR
    
    BR --> Redis
    BC --> Redis
    PS --> Redis
```

## 主要な設計パターン

### 1. ファクトリーパターン
- `TaskQueueFactory`: バックエンドに応じたTaskQueue実装を生成
- `ChannelFactory`: バックエンドに応じたChannel実装を生成

### 2. 抽象基底クラスパターン
- `TaskQueue`: キューの抽象インターface
- `Channel`: チャネルの抽象インターface

### 3. ワーカープールパターン
- `TaskWorker`: 複数タスクの並行処理
- スレッドプールによる並行実行

### 4. オブザーバーパターン
- Redis PubSubによるイベント通知
- バリア同期での完了通知

## スケーラビリティ

### 水平スケーリング
- 複数TaskWorkerによる分散処理
- Redisを介したタスク分散
- ステートレスワーカー設計

### 垂直スケーリング
- TaskWorker内での並行タスク処理
- 設定可能な同時実行数
- スレッドプールによるリソース管理

## 設定例

### インメモリ設定
```python
context = ExecutionContext.create(
    graph=task_graph,
    start_node="start",
    queue_backend=QueueBackend.IN_MEMORY,
    channel_backend="memory"
)
```

### Redis分散設定
```python
redis_config = {
    "host": "localhost",
    "port": 6379,
    "db": 0
}

context = ExecutionContext.create(
    graph=task_graph,
    start_node="start", 
    queue_backend=QueueBackend.REDIS,
    channel_backend="redis",
    config=redis_config
)
```

## エラー処理と信頼性

### リトライメカニズム
- TaskSpecレベルでのリトライ制御
- 設定可能なリトライ回数
- 指数バックオフ対応

### メトリクス収集
- タスク処理統計
- 実行時間測定
- 成功率監視

### グレースフルシャットダウン
- シグナルハンドリング
- アクティブタスクの完了待機
- リソースクリーンアップ

## コンテナ化アーキテクチャ

TaskHandlerとTaskWorkerをコンテナ化することで、分散処理とスケーラビリティを実現できます。

### コンテナ化設計

```mermaid
graph TB
    subgraph "Producer Container"
        APP[Application]
        WE[WorkflowEngine]
        EC[ExecutionContext]
    end
    
    subgraph "Redis Infrastructure"
        Redis[(Redis Server)]
        RTQ[RedisTaskQueue]
        RC[RedisChannel]
    end
    
    subgraph "Worker Container 1"
        TW1[TaskWorker-1]
        TH1[TaskHandler-1]
        CTX1[TaskContext-1]
    end
    
    subgraph "Worker Container 2"
        TW2[TaskWorker-2]
        TH2[TaskHandler-2]
        CTX2[TaskContext-2]
    end
    
    subgraph "Worker Container N"
        TWN[TaskWorker-N]
        THN[TaskHandler-N]
        CTXN[TaskContext-N]
    end
    
    APP --> WE
    WE --> EC
    EC --> RTQ
    EC --> RC
    
    RTQ --> Redis
    RC --> Redis
    
    TW1 --> RTQ
    TW2 --> RTQ
    TWN --> RTQ
    
    TW1 --> TH1
    TW2 --> TH2
    TWN --> THN
    
    TH1 --> RC
    TH2 --> RC
    THN --> RC
```

### コンテナ間通信フロー

```mermaid
sequenceDiagram
    participant Producer as Producer Container
    participant Redis as Redis Server
    participant Worker1 as Worker Container 1
    participant Worker2 as Worker Container 2
    
    Producer->>Redis: enqueue(TaskSpec)
    
    par Worker 1 Processing
        Worker1->>Redis: dequeue()
        Redis-->>Worker1: TaskSpec
        Worker1->>Worker1: TaskHandler.process_task()
        Worker1->>Redis: set_result(key, value)
    and Worker 2 Processing
        Worker2->>Redis: dequeue()
        Redis-->>Worker2: TaskSpec
        Worker2->>Worker2: TaskHandler.process_task()
        Worker2->>Redis: set_result(key, value)
    end
    
    Producer->>Redis: get_result(key)
    Redis-->>Producer: result
```

### コンテナ設定例

#### Docker Compose設定

```yaml
version: '3.8'

services:
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data

  producer:
    build: .
    environment:
      - REDIS_HOST=redis
      - REDIS_PORT=6379
      - GRAFLOW_MODE=producer
    depends_on:
      - redis
    volumes:
      - ./workflows:/app/workflows

  worker:
    build: .
    environment:
      - REDIS_HOST=redis
      - REDIS_PORT=6379
      - GRAFLOW_MODE=worker
      - WORKER_ID=${WORKER_ID:-worker-1}
      - MAX_CONCURRENT_TASKS=4
    depends_on:
      - redis
    scale: 3  # 3つのワーカーコンテナを起動

volumes:
  redis_data:
```

#### ワーカーコンテナエントリーポイント

```python
# worker_main.py
import os
import redis
from graflow.worker.worker import TaskWorker
from graflow.worker.handler import InProcessTaskExecutor
from graflow.queue.redis import RedisTaskQueue
from graflow.core.context import ExecutionContext

def create_worker_from_env():
    """環境変数からワーカー設定を生成"""
    redis_host = os.getenv('REDIS_HOST', 'localhost')
    redis_port = int(os.getenv('REDIS_PORT', '6379'))
    worker_id = os.getenv('WORKER_ID', 'worker-1')
    max_concurrent = int(os.getenv('MAX_CONCURRENT_TASKS', '4'))
    
    # Redis接続
    redis_client = redis.Redis(
        host=redis_host, 
        port=redis_port, 
        decode_responses=True
    )
    
    # ダミーExecutionContext（ワーカー専用）
    from graflow.core.graph import TaskGraph
    dummy_graph = TaskGraph()
    context = ExecutionContext.create(
        graph=dummy_graph,
        start_node="dummy",
        queue_backend="redis",
        channel_backend="redis",
        config={"redis_client": redis_client}
    )
    
    # TaskQueue作成
    task_queue = RedisTaskQueue(context, redis_client=redis_client)
    
    # TaskHandler作成
    handler = InProcessTaskExecutor()
    
    # ワーカー作成
    worker = TaskWorker(
        queue=task_queue,
        handler=handler,
        worker_id=worker_id,
        max_concurrent_tasks=max_concurrent
    )
    
    return worker

if __name__ == "__main__":
    worker = create_worker_from_env()
    try:
        worker.start()
        # Keep running until signal
        while worker.is_running:
            time.sleep(1)
    except KeyboardInterrupt:
        worker.stop()
```

### カスタムTaskHandlerの実装

コンテナ化では、特定のタスクタイプに特化したTaskHandlerを実装できます：

```python
class ContainerizedTaskHandler(TaskHandler):
    """コンテナ環境用のタスクハンドラー"""
    
    def __init__(self, task_registry: Dict[str, Callable]):
        self.task_registry = task_registry
        
    def _process_task(self, task: Any) -> bool:
        task_type = getattr(task, 'task_type', None)
        if task_type and task_type in self.task_registry:
            func = self.task_registry[task_type]
            try:
                result = func(task)
                return True
            except Exception as e:
                logger.error(f"Task {task.task_id} failed: {e}")
                return False
        else:
            # デフォルト処理
            return task() if callable(task) else False
```

### スケーリング戦略

```mermaid
graph LR
    subgraph "Producer"
        P[Producer Container]
    end
    
    subgraph "Redis Cluster"
        R1[Redis-1]
        R2[Redis-2]
        R3[Redis-3]
    end
    
    subgraph "Worker Pool"
        W1[Worker-1]
        W2[Worker-2]
        W3[Worker-3]
        WN[Worker-N]
    end
    
    P --> R1
    P --> R2
    P --> R3
    
    W1 --> R1
    W2 --> R1
    W3 --> R2
    WN --> R3
```

この設計により、Graflowは単一プロセスでの実行から大規模分散環境まで柔軟にスケールできる実行基盤を提供します。