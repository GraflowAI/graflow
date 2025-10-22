# Graflow独自機能とオリジナリティ
## LangGraph、LangChain、Celery、Airflowとの包括的比較

**ドキュメントバージョン**: 1.0
**最終更新**: 2025-10-22
**著者**: Graflowチーム

---

## 目次

1. [エグゼクティブサマリー](#エグゼクティブサマリー)
2. [コア独自機能](#コア独自機能)
3. [詳細機能分析](#詳細機能分析)
4. [比較分析](#比較分析)
5. [ユースケースガイドライン](#ユースケースガイドライン)
6. [本番環境デプロイメント](#本番環境デプロイメント)
7. [今後のロードマップ](#今後のロードマップ)

---

## エグゼクティブサマリー

Graflowは、**汎用ワークフロー実行エンジン**として、タスクオーケストレーションシステムの長所を組み合わせ、**動的グラフ変更**、**ワーカーフリート管理**、**プラグ可能な実行戦略**、**Pythonic DSL設計**における独自の革新を実現しています。

### 主要な差別化要因

| 機能 | 説明 | 競合ツール |
|------|------|-----------|
| **ワーカーフリート管理** | オートスケーリング機能を持つ分散実行用の組み込みCLI | Celery、Airflow（部分的） |
| **ランタイム動的タスク** | 実行中にワークフローグラフを変更 | なし（LangGraph: コンパイル時のみ） |
| **ステートマシン実行** | next_iteration()による第一級のステートマシン | なし |
| **Pythonic演算子DSL** | DAG構築のための数学的構文（`>>`、`\|`） | なし |
| **プラグ可能タスクハンドラー** | カスタム実行戦略（GPU、SSH、クラウド、Docker） | Celery/Airflowで限定的 |
| **Docker実行** | 組み込みコンテナ化タスク実行 | 外部ツールが必要 |
| **細粒度エラーポリシー** | 5つの組み込み並列グループエラーハンドリングモード | なし |
| **シームレスなローカル/分散** | 1行でのバックエンド切り替え | なし（ほとんどがインフラ必要） |
| **チャンネルベース通信** | Pub/Subスタイルのタスク間メッセージング | XCom（Airflow）、State（LangGraph） |

---

## コア独自機能

### 概要

Graflowの独自機能は**8つの主要カテゴリ**に分類できます：

1. **ワーカーフリート管理** - 組み込みCLIによる分散実行
2. **ランタイム動的タスク** - 実行中のグラフ変更
3. **ステートマシン実行** - 第一級のステートマシンサポート
4. **プラグ可能タスクハンドラー** - カスタム実行戦略（Docker含む）
5. **細粒度エラーポリシー** - 柔軟な並列グループエラーハンドリング
6. **Pythonic演算子DSL** - 数学的DAG構文
7. **シームレスなローカル/分散** - バックエンド切り替え
8. **チャンネル通信** - Pub/Subメッセージング

### 1. ワーカーフリート管理 🚀

**実装**: `graflow/worker/worker.py`、`examples/05_distributed/redis_worker.py`

#### アーキテクチャ

```
┌─────────────┐
│メインプロセス│  RedisキューへタスクをSubmit
└─────┬───────┘
      │
      ▼
┌─────────────────────────────────────────┐
│         Redisタスクキュー                │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐  │
│  │タスク1  │ │タスク2  │ │タスク3  │  │
│  └─────────┘ └─────────┘ └─────────┘  │
└───────┬──────────┬──────────┬──────────┘
        │          │          │
   ┌────▼───┐ ┌───▼────┐ ┌──▼─────┐
   │Worker 1│ │Worker 2│ │Worker 3│
   │  4 CPUs│ │  8 CPUs│ │ 16 CPUs│
   └────────┘ └────────┘ └────────┘
```

#### 主要機能

##### a. 組み込みCLIワーカー

```bash
# 別ターミナルまたはサーバーでワーカーを起動
python -m graflow.worker.main --worker-id worker-1 --max-concurrent-tasks 4
python -m graflow.worker.main --worker-id worker-2 --max-concurrent-tasks 8
python -m graflow.worker.main --worker-id worker-3 --max-concurrent-tasks 16
```

##### b. 自律的なライフサイクル管理

- **Graceful Shutdown**: SIGTERM/SIGINTシグナルに応答
- **実行中タスクの完了**: 停止前に処理中タスクを完了
- **設定可能なタイムアウト**: `graceful_shutdown_timeout`パラメータ
- **ThreadPoolExecutor**: ワーカーごとの並行タスク処理

##### c. 組み込みメトリクス

```python
worker.tasks_processed      # 実行タスク総数
worker.tasks_succeeded      # 成功完了数
worker.tasks_failed         # 失敗タスク数
worker.total_execution_time # 累積実行時間
```

##### d. 水平スケーリング

- **線形スケーリング**: ワーカー追加でスループット向上
- **コーディネーション不要**: ワーカーは独立してRedisをポーリング
- **地理的分散**: データセンター間でワーカーをデプロイ
- **特化ワーカー**: GPUワーカー、I/Oワーカー、計算ワーカー

##### e. 本番環境デプロイ対応

**Systemdサービス** (`examples/05_distributed/redis_worker.py:381-391`):
```ini
[Unit]
Description=Graflow Worker

[Service]
ExecStart=/usr/bin/python3 -m graflow.worker.main --worker-id worker-1
Restart=always

[Install]
WantedBy=multi-user.target
```

**Dockerデプロイメント**:
```dockerfile
FROM python:3.11
RUN pip install graflow redis
CMD ["python", "-m", "graflow.worker.main", "--worker-id", "worker-1"]
```

**Kubernetesデプロイメント**:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: graflow-workers
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: worker
        image: graflow-worker:latest
        env:
        - name: REDIS_HOST
          value: redis-service
        - name: MAX_CONCURRENT_TASKS
          value: "4"
```

#### 競合ツールとの比較

| 機能 | Graflow | Celery | LangGraph | Airflow |
|------|---------|--------|-----------|---------|
| **組み込みCLI** | ✅ `python -m graflow.worker.main` | ✅ `celery worker` | ❌ なし | ✅ `airflow worker` |
| **Graceful Shutdown** | ✅ SIGTERM/SIGINT | ✅ | N/A | ✅ |
| **メトリクス** | ✅ 組み込み | ⚠️ Flower必要 | ❌ | ✅ |
| **オートスケーリング** | ✅ 動的にワーカー追加 | ✅ | ❌ | ⚠️ 限定的 |
| **ステート共有** | ✅ Redisチャンネル | ⚠️ ブローカー経由 | ⚠️ Stateオブジェクト | ⚠️ XCom |
| **タスクルーティング** | ✅ FIFOキュー | ✅ ルーティングキー | N/A | ✅ DAGベース |

---

### 2. ランタイム動的タスク生成 🔄

**実装**: `examples/07_dynamic_tasks/runtime_dynamic_tasks.py`

#### 独自性

コンパイル時のタスク生成（ループ、ファクトリー）とは異なり、Graflowは**実行中のワークフローグラフ変更**を可能にします。

#### コアAPI

##### a. `context.next_task(task)` - ランタイムで新タスク追加

```python
@task(inject_context=True)
def adaptive_processor(context: TaskExecutionContext):
    data_quality = check_quality()

    if data_quality < 0.5:
        # その場でクリーンアップタスクを作成
        cleanup_task = TaskWrapper("cleanup", cleanup_low_quality_data)
        context.next_task(cleanup_task)
    elif data_quality > 0.9:
        # 強化タスクを作成
        enhance_task = TaskWrapper("enhance", enhance_high_quality_data)
        context.next_task(enhance_task)
    else:
        # 標準処理
        process_task = TaskWrapper("process", standard_processing)
        context.next_task(process_task)
```

##### b. `context.next_iteration(data)` - 収束まで自己ループ

```python
@task(inject_context=True)
def optimize_hyperparameters(context: TaskExecutionContext, params=None):
    if params is None:
        params = {"learning_rate": 0.1, "accuracy": 0.5}

    # 訓練ステップ
    new_accuracy = train_model(params)

    if new_accuracy >= 0.95:
        # 収束 - 完了処理
        save_task = TaskWrapper("save_model", lambda: save_model(params))
        context.next_task(save_task)
    else:
        # 最適化を続行
        params["accuracy"] = new_accuracy
        params["learning_rate"] *= 0.95
        context.next_iteration(params)
```

#### 実世界のユースケース

1. **MLハイパーパラメータチューニング**（収束ベース）:
   ```python
   while accuracy < target:
       context.next_iteration(optimize_step())
   ```

2. **データ分類パイプライン**（条件分岐）:
   ```python
   if data_type == "image":
       context.next_task(TaskWrapper("image_proc", process_image))
   elif data_type == "text":
       context.next_task(TaskWrapper("text_proc", process_text))
   ```

3. **耐障害性API呼び出し**（リトライロジック）:
   ```python
   try:
       result = api_call()
   except Exception:
       if attempts < max_retries:
           context.next_iteration({"attempts": attempts + 1})
       else:
           context.next_task(TaskWrapper("log_error", handle_error))
   ```

4. **プログレッシブエンハンスメント**（品質向上）:
   ```python
   if current_quality < target_quality:
       context.next_iteration(apply_enhancement())
   else:
       context.next_task(TaskWrapper("finalize", save_result))
   ```

#### 競合ツールとの比較

| 機能 | Graflow | LangGraph | Celery | Airflow |
|------|---------|-----------|--------|---------|
| **ランタイムタスク作成** | ✅ `next_task()` | ❌ コンパイル時のみ | ❌ | ⚠️ Dynamic DAG（限定的） |
| **自己ループ（イテレーション）** | ✅ `next_iteration()` | ⚠️ 条件付きエッジ経由 | ❌ | ❌ |
| **条件分岐** | ✅ 実行結果に基づく | ⚠️ 事前定義ブランチ | ❌ | ⚠️ BranchPythonOperator |
| **グラフイントロスペクション** | ✅ 実行中 | ❌ | N/A | ⚠️ 限定的 |
| **最大イテレーション** | ✅ `max_steps`パラメータ | ⚠️ 手動追跡 | N/A | N/A |

**主要な利点**: Graflowのグラフは**真に動的** - 事前定義ブランチだけでなく、ランタイム条件に基づいて成長・変化します。

---

### 3. ステートマシン実行 🔄

**実装**: `examples/07_dynamic_tasks/runtime_dynamic_tasks.py:265-319`

#### 独自性

Graflowは、`next_iteration()`とチャンネルベースのステート永続化を使用して**真のステートマシン実装**を可能にし、明示的なステートマシンライブラリなしで複雑な制御フローを実現します。

#### ステートマシンパターン

```python
@task(inject_context=True)
def state_controller(context: TaskExecutionContext):
    """現在の状態に基づいてステート遷移を制御"""
    channel = context.get_channel()
    current_state = channel.get("state", default="START")
    data = channel.get("data", default=0)

    # ステート遷移ロジック
    if current_state == "START":
        channel.set("state", "PROCESSING")
        channel.set("data", data + 1)
        context.next_iteration()  # state_controllerにループバック

    elif current_state == "PROCESSING":
        if data < threshold:
            # 現在の状態を維持
            channel.set("data", data + 1)
            context.next_iteration()
        else:
            # 次の状態に遷移
            channel.set("state", "FINALIZING")
            context.next_iteration()

    elif current_state == "FINALIZING":
        # ENDに遷移し、最終タスクを作成
        channel.set("state", "END")
        final_task = TaskWrapper("end_state", finalize_handler)
        context.next_task(final_task)

    return {"state": current_state, "data": data}
```

#### ステートマシンの可視化

```
┌─────────┐
│  START  │
└────┬────┘
     │ next_iteration()
     ▼
┌────────────┐
│ PROCESSING │◄───┐ data < thresholdの間ループ
└─────┬──────┘    │ next_iteration()
      │           │
      │ data >= threshold
      ▼           │
┌────────────┐    │
│ FINALIZING │────┘
└─────┬──────┘
      │ next_task()
      ▼
┌─────────┐
│   END   │
└─────────┘
```

#### 主要コンポーネント

##### a. チャンネル経由のステート永続化
```python
# ステートを保存
channel.set("state", "PROCESSING")
channel.set("data", current_data)

# 次のイテレーションでステートを取得
current_state = channel.get("state", default="START")
```

##### b. `next_iteration()`による自己ループ
```python
# 更新されたステートで同じタスクが再実行
context.next_iteration()
```

##### c. ステートベースの分岐
```python
if current_state == "STATE_A":
    # State Aのロジック
    context.next_iteration()
elif current_state == "STATE_B":
    # State Bのロジック
    context.next_task(next_handler)
```

#### 実世界のユースケース

##### 1. プロトコルステートマシン（ネットワーク、API）
```python
@task(inject_context=True)
def connection_fsm(context):
    channel = context.get_channel()
    state = channel.get("state", "DISCONNECTED")

    if state == "DISCONNECTED":
        establish_connection()
        channel.set("state", "CONNECTING")
        context.next_iteration()

    elif state == "CONNECTING":
        if connection_ready():
            channel.set("state", "CONNECTED")
            context.next_iteration()
        else:
            time.sleep(1)
            context.next_iteration()

    elif state == "CONNECTED":
        if should_disconnect():
            channel.set("state", "DISCONNECTING")
            context.next_iteration()
        else:
            handle_messages()
            context.next_iteration()

    elif state == "DISCONNECTING":
        close_connection()
        return "DONE"
```

##### 2. 注文処理ステートマシン
```python
@task(inject_context=True)
def order_processor(context):
    channel = context.get_channel()
    order_state = channel.get("order_state", "NEW")
    order_data = channel.get("order_data")

    if order_state == "NEW":
        validate_order(order_data)
        channel.set("order_state", "VALIDATED")
        context.next_iteration()

    elif order_state == "VALIDATED":
        reserve_inventory(order_data)
        channel.set("order_state", "RESERVED")
        context.next_iteration()

    elif order_state == "RESERVED":
        process_payment(order_data)
        channel.set("order_state", "PAID")
        context.next_iteration()

    elif order_state == "PAID":
        ship_order(order_data)
        channel.set("order_state", "SHIPPED")
        # 通知タスクを作成
        context.next_task(TaskWrapper("notify", send_notification))
```

##### 3. ML訓練ステートマシン
```python
@task(inject_context=True)
def training_fsm(context):
    channel = context.get_channel()
    phase = channel.get("phase", "INIT")
    model = channel.get("model")

    if phase == "INIT":
        model = initialize_model()
        channel.set("model", model)
        channel.set("phase", "TRAINING")
        context.next_iteration()

    elif phase == "TRAINING":
        metrics = train_epoch(model)
        if metrics["accuracy"] >= target:
            channel.set("phase", "VALIDATING")
        context.next_iteration()

    elif phase == "VALIDATING":
        val_metrics = validate(model)
        if val_metrics["accuracy"] >= target:
            channel.set("phase", "SAVING")
        else:
            channel.set("phase", "TRAINING")  # 訓練に戻る
        context.next_iteration()

    elif phase == "SAVING":
        save_model(model)
        return "COMPLETE"
```

##### 4. ゲームステートマシン
```python
@task(inject_context=True)
def game_loop(context):
    channel = context.get_channel()
    game_state = channel.get("game_state", "MENU")

    if game_state == "MENU":
        choice = show_menu()
        channel.set("game_state", "PLAYING" if choice == "start" else "QUIT")
        context.next_iteration()

    elif game_state == "PLAYING":
        player_action = get_player_action()
        game_result = process_turn(player_action)

        if game_result == "won":
            channel.set("game_state", "WIN")
        elif game_result == "lost":
            channel.set("game_state", "LOSE")
        context.next_iteration()

    elif game_state in ["WIN", "LOSE"]:
        show_result(game_state)
        channel.set("game_state", "MENU")
        context.next_iteration()

    elif game_state == "QUIT":
        return "GAME_OVER"
```

#### 従来のステートマシンに対する利点

| 側面 | 従来のFSMライブラリ | Graflowステートマシン |
|------|---------------------|----------------------|
| **ステート保存** | インメモリオブジェクト | ✅ 永続的チャンネル（Redis） |
| **分散** | ❌ 単一プロセス | ✅ ワーカーがステートを引き継げる |
| **可視化** | ⚠️ 別ツール必要 | ✅ ワークフローグラフでフロー表示 |
| **デバッグ** | ⚠️ カスタムロギング | ✅ タスク実行履歴 |
| **統合** | ⚠️ スタンドアロン | ✅ ワークフローの一部 |
| **副作用** | ⚠️ 手動管理必要 | ✅ ハンドラー付きタスクとして |

#### 競合ツールとの比較

| 機能 | Graflow | LangGraph | Celery | Airflow |
|------|---------|-----------|--------|---------|
| **ステートマシンパターン** | ✅ `next_iteration()` + チャンネル | ⚠️ グラフサイクル経由 | ❌ | ❌ |
| **ステート永続化** | ✅ チャンネル（Memory/Redis） | ⚠️ Memory/Checkpointer | ❌ | ⚠️ XCom |
| **分散ステート** | ✅ Redisチャンネル | ❌ | ❌ | ⚠️ DBバックエンド |
| **ステート遷移** | ✅ 動的（ランタイム） | ⚠️ 静的（グラフ定義） | N/A | N/A |
| **最大イテレーション** | ✅ `max_steps` | ⚠️ 手動追跡 | N/A | N/A |

**主要な利点**: Graflowは、分散ステート永続化と他タスクとのシームレスな統合により、**複雑なステートマシンをワークフローとして実装**できます。

#### ベストプラクティス

1. **常に`max_steps`を設定**して無限ループを防止:
   ```python
   ctx.execute("state_controller", max_steps=100)
   ```

2. **説明的なステート名を使用**:
   ```python
   # ✅ 良い例
   channel.set("state", "WAITING_FOR_PAYMENT")

   # ❌ 悪い例
   channel.set("state", "S3")
   ```

3. **ステート遷移をログ記録**:
   ```python
   logger.info(f"State transition: {old_state} → {new_state}")
   ```

4. **型安全性のためにenumを使用**:
   ```python
   from enum import Enum

   class OrderState(Enum):
       NEW = "NEW"
       VALIDATED = "VALIDATED"
       PAID = "PAID"

   channel.set("state", OrderState.NEW.value)
   ```

5. **予期しないステートを処理**:
   ```python
   else:
       raise ValueError(f"Unknown state: {current_state}")
   ```

---

### 4. プラグ可能タスクハンドラーシステム 🔌

**実装**: `graflow/core/handler.py`、`examples/04_execution/custom_handler.py`

#### アーキテクチャ

```python
# ハンドラーインターフェース
class TaskHandler(ABC):
    @abstractmethod
    def execute_task(self, task: Executable, context: ExecutionContext) -> None:
        """カスタム実行ロジックを実装"""
        pass

# カスタム実装
class GPUHandler(TaskHandler):
    def execute_task(self, task, context):
        with gpu_lock:
            result = task.run()  # GPU上で実行
        context.set_result(task.task_id, result)

# 登録
engine = WorkflowEngine()
engine.register_handler("gpu", GPUHandler())

# 使用
@task(handler="gpu")
def train_model():
    ...
```

#### 組み込みハンドラーパターン

##### a. リトライハンドラー
```python
class RetryHandler(TaskHandler):
    def __init__(self, max_retries=3):
        self.max_retries = max_retries

    def execute_task(self, task, context):
        for attempt in range(self.max_retries):
            try:
                result = task.run()
                context.set_result(task.task_id, result)
                return
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise
                logger.warning(f"Retry {attempt + 1}/{self.max_retries}")
```

##### b. キャッシングハンドラー
```python
class CachingHandler(TaskHandler):
    def __init__(self):
        self.cache = {}

    def execute_task(self, task, context):
        if task.task_id in self.cache:
            context.set_result(task.task_id, self.cache[task.task_id])
            return

        result = task.run()
        self.cache[task.task_id] = result
        context.set_result(task.task_id, result)
```

##### c. レート制限ハンドラー
```python
class RateLimitHandler(TaskHandler):
    def __init__(self, min_interval=1.0):
        self.min_interval = min_interval
        self.last_execution = 0

    def execute_task(self, task, context):
        elapsed = time.time() - self.last_execution
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)

        result = task.run()
        self.last_execution = time.time()
        context.set_result(task.task_id, result)
```

##### d. モニタリングハンドラー
```python
class MonitoringHandler(TaskHandler):
    def __init__(self):
        self.metrics = []

    def execute_task(self, task, context):
        start = time.time()
        try:
            result = task.run()
            status = "success"
            context.set_result(task.task_id, result)
        except Exception as e:
            status = "failed"
            raise
        finally:
            self.metrics.append({
                "task": task.task_id,
                "duration": time.time() - start,
                "status": status
            })
```

#### 高度なユースケース

1. **SSHリモート実行**:
   ```python
   class SSHHandler(TaskHandler):
       def execute_task(self, task, context):
           with ssh_client.connect(remote_host):
               result = execute_remotely(task)
           context.set_result(task.task_id, result)
   ```

2. **クラウド関数実行**（AWS Lambda、GCP Functions）:
   ```python
   class LambdaHandler(TaskHandler):
       def execute_task(self, task, context):
           payload = serialize_task(task)
           response = lambda_client.invoke(FunctionName="executor", Payload=payload)
           result = deserialize_response(response)
           context.set_result(task.task_id, result)
   ```

3. **GPUキュー管理**:
   ```python
   class GPUQueueHandler(TaskHandler):
       def execute_task(self, task, context):
           gpu_id = gpu_pool.acquire()
           try:
               result = task.run(device=f"cuda:{gpu_id}")
           finally:
               gpu_pool.release(gpu_id)
           context.set_result(task.task_id, result)
   ```

#### 組み込みDockerハンドラー 🐳

**実装**: `graflow/core/handlers/docker.py`、`examples/04_execution/docker_handler.py`

Graflowは、コンテナ化タスク実行のための**本番環境対応Dockerハンドラー**を含んでいます。

##### 基本的な使用法

```python
from graflow.core.handlers.docker import DockerTaskHandler
from graflow.core.engine import WorkflowEngine

# Dockerハンドラーを登録
engine = WorkflowEngine()
engine.register_handler("docker", DockerTaskHandler(image="python:3.11-slim"))

# タスクで使用
@task(handler="docker")
def isolated_task():
    """完全に隔離されたDockerコンテナで実行"""
    import sys
    return sys.version
```

##### 高度な設定

**環境変数**:
```python
DockerTaskHandler(
    image="python:3.11-slim",
    environment={"API_KEY": "secret", "DEBUG": "1"}
)
```

**ボリュームマウント**:
```python
DockerTaskHandler(
    image="python:3.11-slim",
    volumes={
        "/host/data": {"bind": "/container/data", "mode": "ro"},
        "/host/output": {"bind": "/output", "mode": "rw"}
    }
)
```

**GPUサポート**:
```python
from docker.types import DeviceRequest

DockerTaskHandler(
    image="tensorflow/tensorflow:latest-gpu",
    device_requests=[DeviceRequest(count=1, capabilities=[["gpu"]])]
)
```

**リソース制限**（Docker API経由）:
```python
# CPUとメモリの制限
container = client.containers.run(
    image="python:3.11",
    mem_limit="512m",      # 512MB RAM
    cpu_period=100000,
    cpu_quota=50000,       # 50% CPU
)
```

**ネットワーク設定**:
```python
DockerTaskHandler(
    image="python:3.11",
    network_mode="bridge"  # または "host"、"none"
)
```

##### 主要機能

1. **完全な隔離**:
   - 別プロセス空間
   - 隔離されたファイルシステム
   - ネットワーク隔離（設定可能）
   - ホストリソースへの直接アクセス不可

2. **Cloudpickleシリアライゼーション**:
   - タスク関数をcloudpickle経由でシリアライズ
   - ExecutionContextをコンテナに渡す
   - 結果をホストにデシリアライズ
   - ラムダとクロージャをサポート

3. **再現可能な環境**:
   - 正確なDockerイメージバージョンを固定
   - 一貫した依存関係バージョン
   - クロスプラットフォーム互換性

4. **マルチバージョンテスト**:
   ```python
   # 複数のPythonバージョンに対してテスト
   engine.register_handler("py39", DockerTaskHandler(image="python:3.9"))
   engine.register_handler("py311", DockerTaskHandler(image="python:3.11"))
   engine.register_handler("py312", DockerTaskHandler(image="python:3.12"))

   @task(handler="py39")
   def test_on_py39():
       ...

   @task(handler="py311")
   def test_on_py311():
       ...
   ```

##### 実世界のユースケース

###### 1. セキュリティサンドボックス（信頼できないコード）
```python
@task(handler="docker")
def execute_user_code(user_code: str):
    """ユーザー送信コードを安全に実行"""
    # 隔離実行 - ホストに害を与えない
    exec(user_code)
    return "Executed safely"
```

###### 2. CI/CDテスト
```python
# クリーンな環境でテスト
@task(handler="docker")
def run_integration_tests():
    """隔離されたコンテナでテスト実行"""
    import subprocess
    return subprocess.run(["pytest", "tests/"], capture_output=True)
```

###### 3. データサイエンス実験
```python
# 異なるMLフレームワークバージョン
engine.register_handler("tf2", DockerTaskHandler(image="tensorflow/tensorflow:2.13.0"))
engine.register_handler("tf1", DockerTaskHandler(image="tensorflow/tensorflow:1.15.5"))

@task(handler="tf2")
def train_with_tf2():
    import tensorflow as tf
    # TensorFlow 2.xでモデル訓練
    ...

@task(handler="tf1")
def train_with_tf1():
    import tensorflow as tf
    # TensorFlow 1.xでモデル訓練
    ...
```

###### 4. レガシーコード実行
```python
# Python 2.7で古いコードを実行
engine.register_handler("py27", DockerTaskHandler(image="python:2.7"))

@task(handler="py27")
def legacy_processing():
    """レガシーPython 2コードを実行"""
    # Python 3では動作しない古いコード
    print "Hello from Python 2"  # Python 2構文
```

##### パフォーマンスの考慮事項

| 側面 | Directハンドラー | Dockerハンドラー |
|------|------------------|-----------------|
| **起動時間** | ~1ms | ~500-2000ms |
| **シリアライゼーション** | なし | ~10-100ms |
| **実行** | ネイティブ | ネイティブ（同じ） |
| **総オーバーヘッド** | 最小限 | 大きい |

**推奨**: Dockerハンドラーを使用すべき場合:
- ✅ セキュリティクリティカルなタスク（信頼できないコード）
- ✅ 再現可能な環境
- ✅ マルチバージョンテスト
- ✅ 長時間実行タスク（オーバーヘッドが償却される）

避けるべき場合:
- ❌ パフォーマンスクリティカルなコード
- ❌ 頻繁な短時間タスク
- ❌ 開発イテレーション

##### 独自の利点

| 機能 | Graflow Dockerハンドラー | 競合ツール |
|------|-------------------------|-----------|
| **組み込みサポート** | ✅ コア機能 | ⚠️ 外部ツール |
| **同一ワークフロー** | ✅ Docker + Directを混在 | ❌ 別システム |
| **Cloudpickle** | ✅ ラムダ/クロージャサポート | ❌ |
| **GPUサポート** | ✅ DeviceRequest | ⚠️ 限定的 |
| **マルチイメージ** | ✅ 複数ハンドラー | ⚠️ 限定的 |

**主要な利点**: 同じDSLとコンテキスト管理を使用して、コンテナ化実行を同一ワークフロー内に**シームレスに統合**。

#### 競合ツールとの比較

| 機能 | Graflow | LangGraph | Celery | Airflow |
|------|---------|-----------|--------|---------|
| **プラグ可能ハンドラー** | ✅ `TaskHandler`インターフェース | ❌ ハードコードノード | ⚠️ Taskクラス | ⚠️ Operator |
| **ハンドラー登録** | ✅ `register_handler()` | N/A | ⚠️ デコレーター経由 | ⚠️ プラグイン経由 |
| **タスク毎ハンドラー** | ✅ `@task(handler="...")` | N/A | ⚠️ タスクルーティング | ⚠️ Operator選択 |
| **カスタムロジック** | ✅ 完全制御 | ❌ | ⚠️ 限定的 | ⚠️ 限定的 |
| **組み合わせ可能性** | ✅ デコレータパターン | N/A | ❌ | ❌ |

**主要な利点**: Graflowは、コアエンジンロジックを変更せずに**タスク実行の完全なカスタマイズ**を可能にします。

---

### 5. 細粒度並列グループエラーポリシー ⚠️

**実装**: `examples/11_error_handling/parallel_group_strict_mode.py`

#### 問題点

並列でタスクを実行する場合（`task_a | task_b | task_c`）、1つのタスクが失敗したらどうすべきか？

#### Graflowのソリューション: 5つの組み込みポリシー

##### a. Strictモード（デフォルト）
```python
parallel = (task_a | task_b | task_c)  # 全て成功が必須
```
- **動作**: 失敗があれば→ `ParallelGroupError`
- **ユースケース**: 重要な検証、アトミック操作

##### b. Best Effortモード
```python
parallel = (task_a | task_b | task_c).with_policy(
    ErrorHandlingPolicy.BEST_EFFORT
)
```
- **動作**: 失敗を無視、成功結果で続行
- **ユースケース**: 複数ソースからのデータ集約（一部利用不可の可能性）

##### c. At Least Nモード
```python
parallel = (task_a | task_b | task_c).with_policy(
    ErrorHandlingPolicy.AT_LEAST_N(n=2)
)
```
- **動作**: 少なくともN個のタスクが成功すれば続行
- **ユースケース**: 冗長データ取得（3つ中2つで十分）

##### d. Critical Tasksモード
```python
parallel = (task_a | task_b | task_c).with_policy(
    ErrorHandlingPolicy.CRITICAL_TASKS(["task_a"])
)
```
- **動作**: 指定したタスクのみ成功が必須
- **ユースケース**: プライマリデータソースは必須、他はオプション

##### e. カスタムポリシー
```python
class CustomPolicy(ErrorHandlingPolicy):
    def should_fail(self, failed_tasks, successful_tasks):
        # カスタムロジック
        return len(failed_tasks) > len(successful_tasks)

parallel.with_policy(CustomPolicy())
```

#### エラー情報

```python
try:
    engine.execute(context)
except ParallelGroupError as e:
    print(f"グループ: {e.group_id}")
    print(f"失敗: {e.failed_tasks}")  # [(task_id, error_msg), ...]
    print(f"成功: {e.successful_tasks}")  # [task_id, ...]
```

#### 競合ツールとの比較

| 機能 | Graflow | LangGraph | Celery | Airflow |
|------|---------|-----------|--------|---------|
| **Strictモード** | ✅ デフォルト | ❌ | ⚠️ 暗黙的 | ⚠️ trigger_rule経由 |
| **Best Effort** | ✅ ポリシー | ❌ | ❌ | ⚠️ `all_done`トリガー |
| **At Least N** | ✅ ポリシー | ❌ | ❌ | ❌ |
| **Critical Tasks** | ✅ ポリシー | ❌ | ❌ | ❌ |
| **カスタムポリシー** | ✅ 拡張可能 | ❌ | ❌ | ⚠️ 限定的 |
| **エラー詳細** | ✅ `ParallelGroupError` | N/A | ⚠️ result backend経由 | ⚠️ ログ経由 |

**主要な利点**: 一般的なシナリオをカバーする**5つの柔軟なポリシー**、さらにカスタムロジックの拡張性。

---

### 6. Pythonic演算子DSL 📐

**実装**: `examples/02_workflows/operators_demo.py`

#### 構文

```python
# 逐次実行
task_a >> task_b >> task_c

# 並列実行
task_a | task_b | task_c

# 組み合わせ（ダイヤモンドパターン）
fetch >> (transform_a | transform_b) >> store

# マルチステージ
(load_a | load_b) >> validate >> (process_a | process_b) >> (save_db | save_file)

# 名前付き並列グループ
(task_a | task_b | task_c).set_group_name("parallel_tasks") >> aggregate
```

#### 演算子のセマンティクス

| 演算子 | 意味 | グラフ表現 |
|--------|------|-----------|
| `a >> b` | 逐次: bはaに依存 | `a → b` |
| `a \| b` | 並列: aとbは独立 | `a`、`b`（エッジなし） |
| `(a \| b) >> c` | Fan-in: cはaとbの両方に依存 | `a → c`、`b → c` |
| `a >> (b \| c)` | Fan-out: bとcはaに依存 | `a → b`、`a → c` |

#### 一般的なDAGパターン

##### 線形パイプライン
```python
extract >> validate >> transform >> load
```

##### Fan-out（1対多）
```python
source >> (process_region_1 | process_region_2 | process_region_3)
```

##### Fan-in（多対1）
```python
(fetch_db | fetch_api | fetch_file) >> aggregate
```

##### ダイヤモンド（Fan-out + Fan-in）
```python
fetch >> (transform_a | transform_b) >> merge >> store
```

##### マルチステージパイプライン
```python
(extract_a | extract_b) >> validate >> (transform_a | transform_b) >> (load_db | load_s3)
```

#### 競合ツールとの比較

**LangGraph**:
```python
# 冗長、命令的
graph = StateGraph(...)
graph.add_node("a", node_a)
graph.add_node("b", node_b)
graph.add_edge("a", "b")
graph.add_conditional_edges("b", router, {"c": "c", "d": "d"})
```

**Airflow**:
```python
# オペレーターベース、直感的でない
task_a = BashOperator(task_id="a", ...)
task_b = BashOperator(task_id="b", ...)
task_a >> task_b
# しかし並列には明示的なグループが必要
```

**Celery**:
```python
# 関数API、視覚的でない
chain(task_a.s(), task_b.s(), task_c.s())
group(task_a.s(), task_b.s(), task_c.s())
# グループのチェーンは面倒
```

**Graflow**:
```python
# 数学的、宣言的
fetch >> (transform_a | transform_b) >> store
```

**主要な利点**: 数学的表記法に着想を得た**最も簡潔で直感的な**DAG構文。

---

### 7. シームレスなローカル/分散実行 🌐

**実装**: `examples/05_distributed/distributed_workflow.py`

#### 問題点

ほとんどのオーケストレーションツールは分散実行にインフラストラクチャが必要で、ローカル開発が困難です。

#### Graflowのソリューション: バックエンド切り替え

```python
# 開発環境: ローカル実行（インフラ不要）
context = ExecutionContext.create(
    graph,
    queue_backend=QueueBackend.MEMORY,
    channel_backend="memory"
)

# 本番環境: 分散実行（同じコード！）
context = ExecutionContext.create(
    graph,
    queue_backend=QueueBackend.REDIS,
    channel_backend="redis",
    config={"redis_client": redis_client}
)
```

#### コーディネーションバックエンド

```python
# 並列実行用
parallel = (task_a | task_b | task_c).with_execution(
    backend=CoordinationBackend.DIRECT        # 逐次（デバッグ用）
    # backend=CoordinationBackend.THREADING   # スレッドベース並列
    # backend=CoordinationBackend.MULTIPROCESSING  # プロセスベース並列
    # backend=CoordinationBackend.REDIS       # 分散ワーカー
)
```

#### 環境ベース設定

```python
import os

backend = QueueBackend.REDIS if os.getenv("ENV") == "production" else QueueBackend.MEMORY

context = ExecutionContext.create(graph, queue_backend=backend)
```

#### 競合ツールとの比較

| ツール | ローカル実行 | 分散実行 | 切り替え |
|--------|------------|---------|---------|
| **Graflow** | ✅ MEMORYバックエンド | ✅ REDISバックエンド | ✅ 1行 |
| **Celery** | ❌ 常にブローカー必要 | ✅ | ❌ |
| **LangGraph** | ✅ インメモリ | ❌ 外部ツール必要 | ❌ |
| **Airflow** | ❌ 常にDB必要 | ✅ | ❌ |

**主要な利点**: シームレスな本番デプロイメントを備えた**ゼロインフラストラクチャのローカル開発**。

---

### 8. チャンネルベース通信 📡

**実装**: `examples/03_data_flow/channels_basic.py`

#### Pub/Subスタイルメッセージング

```python
# プロデューサータスク
@task(inject_context=True)
def producer(context):
    channel = context.get_channel()
    channel.set("config", {"batch_size": 100})
    channel.set("metrics", {"processed": 0})

# コンシューマータスク（疎結合）
@task(inject_context=True)
def consumer(context):
    channel = context.get_channel()
    config = channel.get("config")
    metrics = channel.get("metrics", default={})
```

#### チャンネル操作

```python
channel.set(key, value)              # 値を保存
channel.get(key, default=None)       # 値を取得
channel.keys()                       # 全キーをリスト
channel.clear()                      # 全データをクリア
```

#### ユースケース

##### 共有設定
```python
@task(inject_context=True)
def setup(context):
    config = load_config()
    context.get_channel().set("config", config)

# 後続の全タスクが設定にアクセス
@task(inject_context=True)
def process(context):
    config = context.get_channel().get("config")
```

##### メトリクス集約
```python
@task(inject_context=True)
def task_with_metrics(context):
    channel = context.get_channel()
    metrics = channel.get("metrics", [])
    metrics.append({"task": "...", "duration": 1.5})
    channel.set("metrics", metrics)
```

##### 進捗追跡
```python
@task(inject_context=True)
def track_progress(context):
    channel = context.get_channel()
    progress = channel.get("progress", 0)
    channel.set("progress", progress + 10)
```

#### 分散チャンネル（Redis）

```python
context = ExecutionContext.create(
    graph,
    channel_backend="redis"  # ワーカーがRedis経由でステート共有
)
```

#### 競合ツールとの比較

| ツール | タスク間通信 | 分散サポート | APIスタイル |
|--------|------------|------------|-----------|
| **Graflow** | ✅ チャンネル | ✅ Redis | Pub/Sub |
| **Airflow** | ⚠️ XCom | ⚠️ メタデータDB経由 | Key-value |
| **LangGraph** | ⚠️ Stateオブジェクト | ❌ | 共有state |
| **Celery** | ❌ (result backend経由) | ⚠️ 限定的 | N/A |

**主要な利点**: 分散サポートを備えた**疎結合なPub/Subスタイル通信**。

---

## 比較分析

### 機能マトリックス

| 機能 | Graflow | LangGraph | Celery | Airflow |
|------|---------|-----------|--------|---------|
| **Pythonic DSL** | ✅ `>>`、`\|` | ❌ | ⚠️ 部分的 | ⚠️ 部分的 |
| **ランタイム動的タスク** | ✅ `next_task()` | ❌ | ❌ | ⚠️ Dynamic DAG |
| **ステートマシン実行** | ✅ `next_iteration()` + チャンネル | ⚠️ グラフサイクル | ❌ | ❌ |
| **ワーカーフリートCLI** | ✅ 組み込み | ❌ | ✅ | ✅ |
| **カスタムハンドラー** | ✅ プラグ可能 | ❌ | ⚠️ Taskクラス | ⚠️ Operator |
| **Docker実行** | ✅ 組み込みハンドラー | ❌ | ⚠️ Operator経由 | ⚠️ DockerOperator |
| **並列エラーポリシー** | ✅ 5モード + カスタム | ❌ | ⚠️ 基本的 | ⚠️ trigger_rule |
| **ローカル/分散切替** | ✅ 1行 | ❌ | ❌ | ❌ |
| **チャンネル通信** | ✅ Pub/Sub | ⚠️ State | ❌ | ⚠️ XCom |
| **Graceful Shutdown** | ✅ 組み込み | N/A | ✅ | ✅ |
| **メトリクス収集** | ✅ ワーカーレベル | ❌ | ⚠️ Flower | ✅ |
| **サイクル検出** | ✅ 組み込み | ⚠️ 手動 | N/A | ❌ |
| **コンテキストマネージャー** | ✅ `with workflow()` | ❌ | ❌ | ❌ |
| **型安全性** | ✅ TypedChannel | ✅ Pydantic | ❌ | ❌ |

### パフォーマンス特性

| メトリック | Graflow | LangGraph | Celery | Airflow |
|----------|---------|-----------|--------|---------|
| **ローカルオーバーヘッド** | 低（インプロセス） | 低 | 高（ブローカー） | 高（DB） |
| **分散レイテンシ** | 中（Redis） | N/A | 中 | 高（ポーリング） |
| **スループット** | 高（並列ワーカー） | 低（単一プロセス） | 高 | 中 |
| **メモリフットプリント** | 中 | 低 | 中 | 高 |

---

## ユースケースガイドライン

### Graflowを使用すべき場合 ✅

1. **汎用データパイプライン**
   - ETLワークフロー
   - データ処理パイプライン
   - バッチ処理ジョブ
   - データ分析ワークフロー

2. **動的ワークフロー**
   - ランタイムデータに基づく条件実行
   - 収束アルゴリズム（ML訓練、最適化）
   - 適応的データ処理
   - リトライ付きエラーリカバリー

3. **分散実行**
   - 水平スケーリング要件
   - ワーカーフリート管理
   - 地理的分散
   - リソース特化ワーカー（GPU、大容量メモリ）

4. **カスタム実行戦略**
   - リモート実行（SSH、クラウド関数）
   - 特殊ハードウェア（GPU、TPU）
   - レート制限API呼び出し
   - リトライとキャッシングロジック

5. **開発の俊敏性**
   - インフラなしのローカル開発
   - 迅速なプロトタイピング
   - シームレスな本番デプロイメント

6. **ステートマシン実装**
   - プロトコルステートマシン（ネットワーク、API）
   - 注文処理ワークフロー
   - ゲームループとインタラクティブシステム
   - ML訓練ステート管理

7. **コンテナ化実行**
   - セキュリティサンドボックス（信頼できないコード）
   - マルチバージョンテスト（Python、依存関係）
   - 再現可能な環境
   - レガシーコード実行（Python 2.7、古いライブラリ）

### LangGraphを使用すべき場合 ✅

- LLMエージェントのオーケストレーション
- 会話型AIアプリケーション
- チェックポイントとリプレイ
- Human-in-the-loopワークフロー

### Celeryを使用すべき場合 ✅

- 既存のRabbitMQ/Redisインフラ
- バックグラウンドジョブ処理（メール、通知）
- キューによるタスクルーティング
- Fire-and-forgetタスク

### Airflowを使用すべき場合 ✅

- スケジュールベースのバッチ処理
- データウェアハウスETL
- 複雑なDAG可視化
- 監査ログ要件

---

## 本番環境デプロイメント

### インフラ要件

#### 最小構成（単一サーバー）
```
┌─────────────────────────────────┐
│       単一サーバー               │
│                                 │
│  ┌──────────┐  ┌────────────┐  │
│  │  Redis   │  │ Graflow    │  │
│  │ (Queue)  │  │ Worker x3  │  │
│  └──────────┘  └────────────┘  │
│                                 │
│  ┌──────────────────────────┐  │
│  │  メインアプリケーション   │  │
│  └──────────────────────────┘  │
└─────────────────────────────────┘
```

#### スケーラブル構成（マルチサーバー）
```
┌──────────────┐
│Redisクラスター│
│  (HA構成)    │
└──────┬───────┘
       │
   ┌───┴───┬───────┬────────┐
   │       │       │        │
┌──▼───┐ ┌─▼────┐ ┌─▼─────┐ ┌─▼─────┐
│Server1│ │Server2│ │Server3│ │Server4│
│4 Work.│ │8 Work.│ │2 Work.│ │ GPU   │
└───────┘ └──────┘ └───────┘ └───────┘
```

### デプロイメント戦略

#### Docker Compose
```yaml
version: '3.8'
services:
  redis:
    image: redis:7.2
    ports:
      - "6379:6379"

  worker:
    image: graflow-worker:latest
    environment:
      - REDIS_HOST=redis
      - MAX_CONCURRENT_TASKS=4
    deploy:
      replicas: 3
    depends_on:
      - redis
```

#### Kubernetes Helmチャート
```yaml
# values.yaml
workers:
  replicas: 3
  resources:
    requests:
      cpu: "1"
      memory: "2Gi"
    limits:
      cpu: "2"
      memory: "4Gi"

redis:
  enabled: true
  cluster:
    enabled: true
    nodes: 3
```

### モニタリングと可観測性

#### 追跡すべきメトリクス
- ワーカーヘルス（処理タスク数、成功率）
- キュー深度（保留中タスク）
- タスク実行時間（P50、P95、P99）
- タスクタイプ別エラー率
- ワーカーリソース使用率（CPU、メモリ）

#### 推奨ツール
- **Prometheus**: メトリクス収集
- **Grafana**: 可視化
- **Redis Commander**: キュー検査
- **カスタムダッシュボード**: ワーカーメトリクスAPI

---

## 今後のロードマップ

### 計画中の機能

1. **強化されたモニタリング**
   - Prometheusエクスポーター
   - Grafanaダッシュボードテンプレート
   - リアルタイムタスク追跡UI

2. **高度なスケジューリング**
   - Cronベースのタスクスケジューリング
   - 優先度キュー
   - ワークフロー間のタスク依存関係

3. **チェックポイント**
   - ワークフローステート永続化
   - 障害からの再開
   - 部分実行リプレイ

4. **ワークフロー合成**
   - ネストワークフロー
   - ワークフローテンプレート
   - ワークフローライブラリ

5. **統合**
   - Kubernetesネイティブ実行
   - AWS Lambdaバックエンド
   - Apache Beam互換性

---

## 結論

Graflowは**新世代のワークフローオーケストレーション**を代表し、以下を組み合わせています：

- **Pythonicエレガンス**（演算子DSL）
- **本番環境の堅牢性**（ワーカーフリート、エラーポリシー）
- **開発の俊敏性**（ローカル/分散切り替え）
- **拡張性**（カスタムハンドラー、ポリシー）
- **動的機能**（ランタイムタスク生成）

軽量ツール（LangGraph）と重量級インフラ（Airflow）の間のギャップを埋め、**現代のデータエンジニアリングとMLワークフローにとって最適な選択肢**を提供します。

---

**ドキュメント管理者**: Graflowチーム
**最終レビュー**: 2025-10-22
**次回レビュー**: 四半期ごと
