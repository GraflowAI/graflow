# Graflow コード改善提案（統合版）

**最終更新日:** 2025-12-08
**バージョン:** 3.0（統合版）
**対象リリース:** v0.3.0+

## エグゼクティブサマリー

本文書は、Claude、Codex、Geminiの3つの独立したコードレビューを統合し、Graflowコードベースの改善提案を優先度順にまとめたものです。プロジェクトはHITL、チェックポイント、トレーシング、分散実行などの本番対応機能を備えていますが、信頼性、保守性、パフォーマンスを向上させるための改善余地があります。

### 現状分析

**強み:**
- ✅ 包括的な機能セット（HITL、チェックポイント、トレーシング、分散実行）
- ✅ モダンなモジュール構造と明確な関心の分離
- ✅ 適切なログ実装（print()文なし）
- ✅ アクティブな開発と最近の改善
- ✅ 良好なテストカバレッジ

**改善が必要な領域:**
- ⚠️ 広範な例外ハンドラ: 6-73個
- ⚠️ Redisの本番対応の問題（KEYS使用、ヘルスチェック不足）
- ⚠️ HITLタイムアウト時のトレース未完了
- ⚠️ 分散キューの耐久性不足（DLQ未実装）
- ⚠️ ワーカークラッシュ時のリカバリ遅延（30秒タイムアウト）
- ⚠️ ExecutionContextの肥大化（~1400行）
- ⚠️ 新機能の統合テスト不足

---

## 優先度別改善提案

### 🔴 最優先（High Priority）

#### 1. HITLタイムアウト時のトレース終了

**問題:**
- `WorkflowEngine.execute`が`FeedbackTimeoutError`で早期リターンする際、`tracer.on_workflow_end`を呼ばずに終了
- ルートスパンが開いたままになり、Langfuseトレースが不完全に
- チェックポイント作成時にクロージングイベントが記録されない

**影響:**
- インシデント対応時のトレース分析が困難
- 分散トレーシングの整合性が崩れる

**対策:**

```python
# graflow/core/engine.py の execute() メソッド

def execute(
    self,
    context: ExecutionContext,
    start_task_id: Optional[str] = None
) -> Any:
    """Execute workflow or single task using the provided context."""
    assert context.graph is not None, "Graph must be set before execution"

    workflow_name = getattr(context.graph, 'name', None) or f"workflow_{context.session_id[:8]}"

    # トレース開始（ネストされたコンテキストではスキップ）
    if context.parent_context is None:
        context.tracer.on_workflow_start(workflow_name, context)

    try:
        # ... 既存の実行ロジック ...

        # FeedbackTimeoutError処理
        except Exception as e:
            from graflow.hitl.types import FeedbackTimeoutError

            if isinstance(e, FeedbackTimeoutError):
                self._handle_feedback_timeout(e, task_id, task, context)

                # ========== 追加: タイムアウト時のトレース終了 ==========
                if context.parent_context is None:
                    context.tracer.on_workflow_end(
                        workflow_name,
                        context,
                        result=None,
                        metadata={
                            "status": "timeout",
                            "feedback_id": e.feedback_id,
                            "checkpoint_path": context.last_checkpoint_path
                        }
                    )
                # =====================================================
                return None

            raise exceptions.as_runtime_error(e)

    finally:
        # ========== 追加: 必ず on_workflow_end を呼ぶ ==========
        # 既に呼ばれている場合はスキップ（二重呼び出し防止）
        # または、トレーサー側で二重呼び出しを防ぐフラグを実装
        # ===================================================
        pass
```

**追加テスト:**

```python
# tests/trace/test_hitl_timeout_tracing.py

def test_tracer_completes_on_hitl_timeout(tmp_path, mock_tracer):
    """HITLタイムアウト時もトレーサーが正しく終了することを確認."""
    with workflow("hitl_timeout_test") as wf:
        @task(inject_context=True)
        def timeout_task(ctx):
            ctx.request_approval("Approve?", timeout=0.1)

        wf.add_task(timeout_task)

        context = wf.get_context()
        context.tracer = mock_tracer

        # タイムアウトで終了
        try:
            wf.execute()
        except FeedbackTimeoutError:
            pass

        # on_workflow_end が呼ばれたことを確認
        assert mock_tracer.on_workflow_end.called
        call_args = mock_tracer.on_workflow_end.call_args
        assert call_args[1]["metadata"]["status"] == "timeout"
```

**工数:** 1日
**優先度:** 最優先 - 本番環境での障害分析に影響

---

#### 2. 広範な例外ハンドラの置き換え

**現状:** 6-73個の広範な`except Exception:`または`except BaseException:`ブロック

**主な箇所:**
- `trace/langfuse.py`: 9個
- `hitl/backend/redis.py`: 6個
- `coordination/threading_coordinator.py`: 3個
- `worker/worker.py`: 4個
- `llm/client.py`: 4個
- `channels/redis_channel.py`: ping処理
- `core/engine.py`: execute catch-all

**問題:**
- エラーをログして握りつぶすため、データロスが発生
- デバッグが困難

**対策:**

```python
# 悪い例（現状）
try:
    result = execute_task()
except Exception as e:  # 広すぎる！
    logger.error(f"Error: {e}")
    return None

# 良い例（改善後）
from graflow.exceptions import TaskExecutionError, TaskTimeoutError
import redis
import json

try:
    result = execute_task()
except TaskTimeoutError as e:
    logger.warning(
        "Task timed out, will retry",
        extra={"task_id": task.task_id, "timeout": e.timeout}
    )
    return retry_task(task)
except TaskExecutionError as e:
    logger.error(
        "Task execution failed",
        extra={"task_id": task.task_id, "error": str(e)},
        exc_info=True
    )
    raise
except redis.RedisError as e:
    logger.error(
        "Redis connection error",
        extra={"host": redis_config.host, "error": str(e)}
    )
    raise TaskExecutionError(f"Redis error: {e}") from e
except json.JSONDecodeError as e:
    logger.error(
        "Invalid JSON in task payload",
        extra={"payload": payload[:100], "error": str(e)}
    )
    raise TaskExecutionError(f"JSON decode error: {e}") from e
# Exception や BaseException は使わない
```

**クリーンアップコード向けの例外処理:**

```python
def shutdown(self):
    """リソースのクリーンアップ（例外を握りつぶす必要がある場合のみ）."""
    try:
        self._cleanup_resources()
    except BaseException as e:
        logger.error("Error during shutdown cleanup", exc_info=True)
        # KeyboardInterrupt と SystemExit は再度raiseする
        if isinstance(e, (KeyboardInterrupt, SystemExit)):
            raise
```

**目標:** クリーンアップ専用以外の広範な例外ハンドラを0に

**工数:** 2-3日
**優先度:** 最優先 - 信頼性とデバッグ性向上

---

#### 3. Redisの本番対応（ヘルスシグナル追加）

**問題:**

2. **ヘルスチェック不足:**
   - `RedisChannel.ping()`: 全例外を握りつぶして`False`を返すだけ
   - 障害が隠蔽される

3. **効率の悪いリスト操作:**
   - フィードバックリクエスト一覧を`keys() + GET + JSON parse`で取得
   - OOMやストールのリスク

**対策:**

**2) フィードバックリストをソート済みセットで管理:**

```python
# graflow/hitl/backend/redis.py

class RedisHITLBackend:
    """Redisバックエンド（最適化版）."""

    def create_request(self, request: FeedbackRequest) -> str:
        """フィードバックリクエストを作成."""
        request_id = request.feedback_id

        # リクエストをJSON保存
        key = f"{self.key_prefix}:request:{request_id}"
        self._redis.setex(
            key,
            self.request_ttl,
            json.dumps(request.to_dict(), default=str)
        )

        # ========== 追加: インデックスに追加 ==========
        # スコア: 作成時刻（タイムスタンプ）
        # メンバー: feedback_id
        index_key = f"{self.key_prefix}:index:requests"
        self._redis.zadd(
            index_key,
            {request_id: request.created_at.timestamp()}
        )
        # ===========================================

        return request_id

    def list_pending_requests(
        self,
        limit: int = 100,
        offset: int = 0
    ) -> List[FeedbackRequest]:
        """保留中のリクエストを取得（ページング対応）."""
        index_key = f"{self.key_prefix}:index:requests"

        # ========== 最適化: ZREVRANGE でページング取得 ==========
        # 最新のものから取得（降順）
        request_ids = self._redis.zrevrange(
            index_key,
            offset,
            offset + limit - 1
        )
        # ====================================================

        requests = []
        for request_id in request_ids:
            if isinstance(request_id, bytes):
                request_id = request_id.decode('utf-8')

            request = self.get_request(request_id)
            if request and request.status == FeedbackStatus.PENDING:
                requests.append(request)

        return requests
```

**3) pingのロギング改善:**

```python
# graflow/channels/redis_channel.py

def ping(self) -> bool:
    """Redis接続の確認（ロギング付き）."""
    try:
        result = self._redis.ping()
        return result
    except redis.RedisError as e:
        logger.error(
            "Redis ping failed",
            extra={
                "host": self._redis.connection_pool.connection_kwargs.get("host"),
                "port": self._redis.connection_pool.connection_kwargs.get("port"),
                "error": str(e)
            }
        )
        return False
    except Exception as e:
        logger.error(
            "Unexpected error during Redis ping",
            extra={"error": str(e)},
            exc_info=True
        )
        return False

def health_check(self) -> Dict[str, Any]:
    """ヘルスチェック情報を取得."""
    try:
        info = self._redis.info()
        return {
            "status": "healthy",
            "connected_clients": info.get("connected_clients"),
            "used_memory": info.get("used_memory_human"),
            "uptime_seconds": info.get("uptime_in_seconds")
        }
    except redis.RedisError as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }
```

**目標:** Redis `KEYS`使用箇所を0に

**工数:** 3-4日
**優先度:** 最優先 - 本番環境でのスケーラビリティ

---

#### 4. 分散キューの耐久性追加（DLQ実装）

**問題:**
- `DistributedTaskQueue.dequeue`がパース不可能なアイテムを警告後に破棄
- グラフストア不在時もエラーログのみで破棄
- 可視性がなく、マルチワーカー環境でのデバッグが困難

**対策:**

```python
# graflow/queue/distributed.py

class DistributedTaskQueue:
    """分散タスクキュー（DLQ対応版）."""

    def __init__(
        self,
        redis_client: redis.Redis,
        key_prefix: str = "graflow",
        dlq_ttl: int = 86400 * 7  # 7日間保持
    ):
        self._redis = redis_client
        self._key_prefix = key_prefix
        self._queue_key = f"{key_prefix}:queue"
        self._dlq_key = f"{key_prefix}:dlq"  # Dead Letter Queue
        self._dlq_ttl = dlq_ttl

        # ========== 追加: メトリクスカウンター ==========
        self._metrics = {
            "dequeued": 0,
            "decoded": 0,
            "dropped": 0,
            "dlq_sent": 0
        }
        # ==========================================

    def _send_to_dlq(
        self,
        payload: Union[str, bytes],
        reason: str,
        error: Optional[Exception] = None
    ) -> None:
        """Dead Letter Queueにアイテムを送信."""
        dlq_item = {
            "payload": payload if isinstance(payload, str) else payload.decode('utf-8'),
            "reason": reason,
            "error": str(error) if error else None,
            "timestamp": datetime.now().isoformat(),
            "queue_key": self._queue_key
        }

        # DLQに追加（TTL付き）
        dlq_key = f"{self._dlq_key}:{uuid.uuid4().hex}"
        self._redis.setex(
            dlq_key,
            self._dlq_ttl,
            json.dumps(dlq_item)
        )

        self._metrics["dlq_sent"] += 1

        logger.error(
            "Task sent to DLQ",
            extra={
                "dlq_key": dlq_key,
                "reason": reason,
                "error": str(error) if error else None,
                "payload_preview": str(payload)[:200]
            }
        )

    def dequeue(self) -> Optional[TaskSpec]:
        """タスクをデキュー（DLQ対応版）."""
        payload = self._redis.lpop(self._queue_key)
        if not payload:
            return None

        self._metrics["dequeued"] += 1

        try:
            data = json.loads(payload)
            self._metrics["decoded"] += 1
        except json.JSONDecodeError as e:
            self._metrics["dropped"] += 1
            # ========== 変更: DLQに送信 ==========
            self._send_to_dlq(payload, "json_decode_error", e)
            # ==================================
            return None

        # グラフストアチェック
        graph_store = get_global_graph_store()
        if graph_store is None:
            self._metrics["dropped"] += 1
            # ========== 変更: DLQに送信 ==========
            self._send_to_dlq(
                payload,
                "graph_store_not_available",
                RuntimeError("Graph store not initialized")
            )
            # ==================================
            return None

        # ... 既存のTaskSpec作成ロジック ...

    def get_metrics(self) -> Dict[str, int]:
        """メトリクスを取得."""
        return self._metrics.copy()

    def list_dlq_items(
        self,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """DLQアイテムを一覧表示."""
        pattern = f"{self._dlq_key}:*"
        items = []

        for key in self._redis.scan_iter(match=pattern, count=100):
            if len(items) >= limit:
                break

            data = self._redis.get(key)
            if data:
                try:
                    items.append(json.loads(data))
                except json.JSONDecodeError:
                    continue

        return items
```

**テスト追加:**

```python
# tests/queue/test_distributed_dlq.py

def test_malformed_json_sent_to_dlq(redis_client):
    """不正なJSONがDLQに送られることを確認."""
    queue = DistributedTaskQueue(redis_client, key_prefix="test")

    # 不正なJSONをキューに追加
    redis_client.rpush(queue._queue_key, "{invalid json}")

    # デキュー試行
    result = queue.dequeue()
    assert result is None

    # DLQに送られたことを確認
    dlq_items = queue.list_dlq_items()
    assert len(dlq_items) == 1
    assert dlq_items[0]["reason"] == "json_decode_error"

    # メトリクスを確認
    metrics = queue.get_metrics()
    assert metrics["dlq_sent"] == 1
    assert metrics["dropped"] == 1
```

**目標:** DLQとメトリクスの実装完了

**工数:** 2-3日
**優先度:** 最優先 - 本番環境での可視性とデバッグ性

---

#### 5. ワーカーハートビート実装

**問題:**
- `RedisCoordinator.wait_barrier`がワーカーの完了を盲目的に待機
- ワーカープロセスがクラッシュ（OOM/Segfault）した場合、30秒のタイムアウトまで待機
- 早期検出メカニズムがない

**対策:**

```python
# graflow/worker/heartbeat.py （新規作成）

import threading
import time
import redis
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class WorkerHeartbeat:
    """ワーカーハートビート管理."""

    def __init__(
        self,
        redis_client: redis.Redis,
        task_id: str,
        key_prefix: str = "graflow",
        interval: int = 5,  # 5秒ごと
        ttl: int = 15  # 15秒のTTL
    ):
        self._redis = redis_client
        self._task_id = task_id
        self._key = f"{key_prefix}:heartbeat:{task_id}"
        self._interval = interval
        self._ttl = ttl
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        """ハートビートを開始."""
        if self._thread is not None:
            logger.warning("Heartbeat already started")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._heartbeat_loop,
            daemon=True,
            name=f"heartbeat-{self._task_id}"
        )
        self._thread.start()
        logger.debug(f"Heartbeat started for task: {self._task_id}")

    def stop(self) -> None:
        """ハートビートを停止."""
        if self._thread is None:
            return

        self._stop_event.set()
        self._thread.join(timeout=self._interval + 1)
        self._thread = None

        # ハートビートキーを削除
        try:
            self._redis.delete(self._key)
        except redis.RedisError as e:
            logger.warning(f"Failed to delete heartbeat key: {e}")

        logger.debug(f"Heartbeat stopped for task: {self._task_id}")

    def _heartbeat_loop(self) -> None:
        """ハートビートループ（バックグラウンドスレッド）."""
        while not self._stop_event.is_set():
            try:
                # ハートビートキーを更新（TTL付き）
                self._redis.setex(
                    self._key,
                    self._ttl,
                    time.time()
                )
            except redis.RedisError as e:
                logger.error(f"Failed to update heartbeat: {e}")

            # インターバル待機（早期停止可能）
            self._stop_event.wait(self._interval)

    def __enter__(self):
        """コンテキストマネージャー開始."""
        self.start()
        return self

    def __exit__(self, *args):
        """コンテキストマネージャー終了."""
        self.stop()
```

**ワーカー側での使用:**

```python
# graflow/worker/worker.py

def execute_task(self, task_spec: TaskSpec) -> None:
    """タスクを実行（ハートビート付き）."""
    task_id = task_spec.executable.task_id

    # ========== 追加: ハートビート開始 ==========
    with WorkerHeartbeat(
        self._redis,
        task_id,
        key_prefix=self._key_prefix
    ):
        # タスク実行
        try:
            result = task_spec.executable.run()
            self._report_success(task_id, result)
        except Exception as e:
            self._report_failure(task_id, e)
            raise
    # ハートビート自動停止
    # ========================================
```

**コーディネーター側でのチェック:**

```python
# graflow/coordination/redis_coordinator.py

def wait_barrier(
    self,
    group_id: str,
    task_count: int,
    timeout: float = 30.0
) -> bool:
    """バリア待機（ハートビートチェック付き）."""
    start_time = time.time()
    check_interval = 1.0  # 1秒ごとにチェック
    heartbeat_timeout = 10.0  # 10秒間ハートビートなしで失敗

    while time.time() - start_time < timeout:
        # 完了数をチェック
        completion_count = self._get_completion_count(group_id)
        if completion_count >= task_count:
            return True

        # ========== 追加: ハートビートチェック ==========
        # 実行中タスクのハートビートを確認
        running_tasks = self._get_running_tasks(group_id)
        for task_id in running_tasks:
            heartbeat_key = f"{self._key_prefix}:heartbeat:{task_id}"

            # ハートビートキーの存在確認
            if not self._redis.exists(heartbeat_key):
                logger.error(
                    f"Task {task_id} heartbeat missing - worker may have crashed",
                    extra={"group_id": group_id, "task_id": task_id}
                )
                # 即座に失敗（タイムアウト待たない）
                return False
        # ===========================================

        time.sleep(check_interval)

    # タイムアウト
    logger.error(f"Barrier timeout for group: {group_id}")
    return False
```

**目標:** ワーカークラッシュ時のリカバリ時間を30秒→10秒未満に短縮

**工数:** 3日
**優先度:** 最優先 - 本番環境での回復力向上

---

#### 6. 統合テストの拡充

**現状:** HITL、チェックポイント、トレーシング機能の統合テストが不足

**不足している領域:**
- HITLフィードバックタイムアウトとレジューム
- Redisバックエンドでのチェックポイント作成・復元
- Langfuseトレーシング統合のエンドツーエンド
- LLMエージェント統合（モック使用）
- フィードバック送信APIエンドポイント

**対策:**

```python
# tests/integration/test_hitl_redis_integration.py

import pytest
import redis
from graflow.hitl.manager import FeedbackManager
from graflow.hitl.types import FeedbackType, FeedbackStatus
from graflow.core.checkpoint import CheckpointManager

@pytest.mark.integration
class TestHITLRedisIntegration:
    """HITL Redis統合テスト."""

    @pytest.fixture
    def redis_client(self):
        """Redisクライアントフィクスチャ."""
        client = redis.Redis(
            host="localhost",
            port=6379,
            decode_responses=True
        )
        # テスト前にクリーンアップ
        for key in client.scan_iter("test:*"):
            client.delete(key)
        yield client
        # テスト後にクリーンアップ
        for key in client.scan_iter("test:*"):
            client.delete(key)

    def test_timeout_checkpoint_resume_with_redis(
        self,
        redis_client,
        tmp_path
    ):
        """Redis使用時のタイムアウト→チェックポイント→レジュームを確認."""

        # 1. フィードバックマネージャーをRedisバックエンドで作成
        feedback_manager = FeedbackManager(
            backend="redis",
            backend_config={"redis_client": redis_client, "key_prefix": "test"}
        )

        # 2. タイムアウトするワークフローを作成
        with workflow("redis_hitl_test") as wf:
            @task(inject_context=True)
            def approval_task(ctx):
                ctx.feedback_manager = feedback_manager
                response = ctx.request_approval(
                    prompt="Approve deployment?",
                    timeout=1.0  # 1秒でタイムアウト
                )
                return response

            wf.add_task(approval_task)
            context = wf.get_context()

            # 3. 実行してタイムアウト
            try:
                wf.execute()
                pytest.fail("Should have timed out")
            except FeedbackTimeoutError as e:
                feedback_id = e.feedback_id

            # 4. チェックポイントが作成されたことを確認
            assert context.last_checkpoint_path is not None
            checkpoint_path = context.last_checkpoint_path
            assert Path(checkpoint_path).exists()

            # 5. Redisにフィードバックリクエストがあることを確認
            request = feedback_manager.backend.get_request(feedback_id)
            assert request is not None
            assert request.status == FeedbackStatus.PENDING

        # 6. 外部からフィードバックを提供（別ワーカー想定）
        feedback_manager.respond_to_feedback(
            feedback_id=feedback_id,
            response={"approved": True}
        )

        # 7. チェックポイントからレジューム
        resumed_context, metadata = CheckpointManager.resume_from_checkpoint(
            checkpoint_path
        )
        resumed_context.feedback_manager = feedback_manager

        # 8. 実行完了を確認
        engine = WorkflowEngine()
        engine.execute(resumed_context)

        # 9. 結果を確認
        result = resumed_context.get_result(approval_task.task_id)
        assert result is True

        # 10. Redisのフィードバックステータスが完了になっていることを確認
        request = feedback_manager.backend.get_request(feedback_id)
        assert request.status == FeedbackStatus.COMPLETED
```

```python
# tests/integration/test_langfuse_tracing.py

@pytest.mark.integration
class TestLangfuseTracingIntegration:
    """Langfuseトレーシング統合テスト."""

    def test_distributed_tracing_with_parallel_group(self, mock_langfuse):
        """並列グループ実行時の分散トレーシングを確認."""
        # ... テスト実装 ...
```

**目標:** 統合テストカバレッジを包括的に

**工数:** 2週間
**優先度:** 最優先 - 本番環境での信頼性

---

### 🟡 中優先（Medium Priority）

#### 7. ExecutionContextの分解

**現状:** ExecutionContextが約1400行で多くの責務を持つ

**問題:**
- テスタビリティの低下
- リファクタリングの困難さ
- 複数のサブシステムへの結合

**対策:**

```python
# graflow/core/context_managers.py （新規作成）

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from pathlib import Path

@dataclass
class CheckpointState:
    """チェックポイント状態管理."""
    last_checkpoint_path: Optional[Path] = None
    checkpoint_metadata: Dict[str, Any] = field(default_factory=dict)
    checkpoint_requested: bool = False
    checkpoint_request_metadata: Optional[Dict[str, Any]] = None
    checkpoint_request_path: Optional[str] = None
    completed_tasks: List[str] = field(default_factory=list)

    def request_checkpoint(
        self,
        metadata: Optional[Dict] = None,
        path: Optional[str] = None
    ) -> None:
        """チェックポイントをリクエスト."""
        self.checkpoint_requested = True
        self.checkpoint_request_metadata = dict(metadata) if metadata else {}
        self.checkpoint_request_path = path

    def clear_request(self) -> None:
        """チェックポイントリクエストをクリア."""
        self.checkpoint_requested = False
        self.checkpoint_request_metadata = None
        self.checkpoint_request_path = None

    def mark_task_completed(self, task_id: str) -> None:
        """タスク完了を記録."""
        if task_id not in self.completed_tasks:
            self.completed_tasks.append(task_id)

@dataclass
class LLMRegistry:
    """LLMクライアントとエージェントの管理."""
    _llm_client: Optional[Any] = None
    _llm_agents: Dict[str, Any] = field(default_factory=dict)
    _llm_agents_yaml: Dict[str, str] = field(default_factory=dict)

    def register_agent(self, name: str, agent: Any) -> None:
        """エージェントを登録."""
        self._llm_agents[name] = agent

        # ADKエージェントの場合はYAMLシリアライゼーション
        try:
            from graflow.llm.agents.adk_agent import AdkLLMAgent
            from graflow.llm.serialization import agent_to_yaml

            if isinstance(agent, AdkLLMAgent):
                yaml_str = agent_to_yaml(agent._adk_agent)
                self._llm_agents_yaml[name] = yaml_str
        except (ImportError, AttributeError):
            pass

    def get_agent(self, name: str) -> Any:
        """エージェントを取得（遅延復元対応）."""
        if name in self._llm_agents:
            return self._llm_agents[name]

        # YAMLから復元
        if name in self._llm_agents_yaml:
            try:
                from graflow.llm.agents.adk_agent import AdkLLMAgent
                from graflow.llm.serialization import yaml_to_agent

                adk_agent = yaml_to_agent(self._llm_agents_yaml[name])
                agent = AdkLLMAgent._from_adk_agent(adk_agent, "")
                self._llm_agents[name] = agent
                return agent
            except (ImportError, Exception):
                pass

        raise KeyError(f"LLMAgent '{name}' not found in registry")

    @property
    def llm_client(self) -> Any:
        """LLMクライアントを取得（遅延初期化）."""
        if self._llm_client is None:
            from graflow.llm.client import LLMClient
            import os

            default_model = os.getenv("GRAFLOW_LLM_MODEL", "gpt-5-mini")
            self._llm_client = LLMClient(model=default_model)

        return self._llm_client

@dataclass
class GraphNavigator:
    """グラフトラバーサルとキュー管理."""
    graph: Any  # TaskGraph
    queue: Any  # LocalTaskQueue
    start_node: Optional[str] = None

    def get_next_task(self) -> Optional[str]:
        """次のタスクを取得."""
        return self.queue.get_next_task()

    def add_to_queue(self, executable: Any) -> None:
        """タスクをキューに追加."""
        from graflow.queue.base import TaskSpec

        task_spec = TaskSpec(
            executable=executable,
            execution_context=None,  # コンテキストは後で設定
            trace_id="",  # 後で設定
            parent_span_id=None
        )
        self.queue.enqueue(task_spec)
```

**ExecutionContextでの使用:**

```python
# graflow/core/context.py

class ExecutionContext:
    """実行コンテキスト（リファクタリング版）."""

    def __init__(self, ...):
        # ... 既存の初期化 ...

        # ========== 追加: 専門マネージャーの使用 ==========
        self._checkpoint_state = CheckpointState()
        self._llm_registry = LLMRegistry()
        self._graph_navigator = GraphNavigator(self.graph, self.task_queue, start_node)
        # ==============================================

    # デリゲートメソッド
    def request_checkpoint(
        self,
        metadata: Optional[Dict] = None,
        path: Optional[str] = None
    ) -> None:
        """チェックポイントをリクエスト."""
        return self._checkpoint_state.request_checkpoint(metadata, path)

    def mark_task_completed(self, task_id: str) -> None:
        """タスク完了を記録."""
        return self._checkpoint_state.mark_task_completed(task_id)

    def register_llm_agent(self, name: str, agent: Any) -> None:
        """LLMエージェントを登録."""
        return self._llm_registry.register_agent(name, agent)

    def get_llm_agent(self, name: str) -> Any:
        """LLMエージェントを取得."""
        return self._llm_registry.get_agent(name)

    @property
    def llm_client(self) -> Any:
        """LLMクライアントを取得."""
        return self._llm_registry.llm_client
```

**目標:** ExecutionContextを800行未満に削減

**工数:** 1週間
**優先度:** 中 - 保守性向上（現状は許容範囲）

---

#### 8. グラフシリアライゼーション最適化（CAS実装）

**問題:**
- `RedisCoordinator.execute_group`がグループ実行ごとに無条件でTaskGraphをRedisに保存
- 大規模グラフや頻繁な並列ステップでI/Oオーバーヘッド

**対策:**

```python
# graflow/core/graph.py

class TaskGraph:
    """タスクグラフ（ハッシュ計算対応版）."""

    def calculate_hash(self) -> str:
        """グラフ構造のハッシュを計算（Content-Addressable Storage用）."""
        import hashlib
        import json

        # グラフ構造をシリアライズ
        structure = {
            "nodes": sorted(self._graph.nodes()),
            "edges": sorted([
                (u, v, self._graph.edges[u, v].get("relation", ""))
                for u, v in self._graph.edges()
            ])
        }

        # JSON文字列化してハッシュ計算
        json_str = json.dumps(structure, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()
```

**RedisCoordinatorでの最適化:**

```python
# graflow/coordination/redis_coordinator.py

class RedisCoordinator:
    """Redisコーディネーター（CAS最適化版）."""

    def execute_group(
        self,
        group_id: str,
        tasks: List[Executable],
        exec_context: ExecutionContext,
        policy_instance: GroupExecutionPolicy
    ) -> None:
        """並列グループ実行（グラフアップロード最適化版）."""

        # ========== 追加: Content-Addressable Storage ==========
        # グラフハッシュを計算
        graph = exec_context.graph
        graph_hash = graph.calculate_hash()

        # コンテキストに保存されているハッシュと比較
        if not hasattr(exec_context, 'graph_hash'):
            exec_context.graph_hash = None

        # ハッシュが異なるか、Redisに存在しない場合のみアップロード
        if (graph_hash != exec_context.graph_hash or
            not self.graph_store.exists(graph_hash)):

            logger.debug(
                f"Uploading graph to Redis (hash: {graph_hash[:8]}...)",
                extra={"group_id": group_id}
            )
            self.graph_store.save(graph, key=graph_hash)
            exec_context.graph_hash = graph_hash
        else:
            logger.debug(
                f"Using cached graph (hash: {graph_hash[:8]}...)",
                extra={"group_id": group_id}
            )
        # ====================================================

        # ... 既存のバリアとタスク送信ロジック ...
```

**GraphStoreの拡張:**

```python
# graflow/queue/graph_store.py

class RedisGraphStore:
    """Redisグラフストア（exists対応版）."""

    def exists(self, key: str) -> bool:
        """グラフが存在するかチェック."""
        graph_key = f"{self._key_prefix}:graph:{key}"
        return self._redis.exists(graph_key) > 0

    def save(self, graph: TaskGraph, key: Optional[str] = None) -> str:
        """グラフを保存（キー指定可能版）."""
        if key is None:
            key = str(uuid.uuid4())

        graph_key = f"{self._key_prefix}:graph:{key}"
        # ... 保存ロジック ...
        return key
```

**目標:** グループ実行ごとのグラフアップロード率を5%未満に

**工数:** 2日
**優先度:** 中 - パフォーマンス向上

---

#### 9. LLMクライアントの回復力強化

**問題:**
- `LLMClient.completion`がタイムアウト/リトライ制御なしでLiteLLMに委譲
- `extract_text`がException全般をキャッチして空文字列を返す（エラー隠蔽）

**対策:**

```python
# graflow/llm/client.py

from typing import Optional, Dict, Any, List
import logging
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

class LLMClient:
    """LLMクライアント（回復力強化版）."""

    def __init__(
        self,
        model: str = "gpt-5-mini",
        temperature: float = 0.7,
        timeout: float = 30.0,  # 追加: タイムアウト
        max_retries: int = 3  # 追加: リトライ回数
    ):
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True
    )
    def completion(
        self,
        messages: List[Dict[str, str]],
        **kwargs: Any
    ) -> Any:
        """Completion APIを呼び出す（リトライ付き）."""
        import litellm

        # タイムアウト設定
        kwargs.setdefault("timeout", self.timeout)
        kwargs.setdefault("temperature", self.temperature)

        try:
            response = litellm.completion(
                model=self.model,
                messages=messages,
                **kwargs
            )
            return response
        except litellm.Timeout as e:
            logger.error(
                "LLM completion timeout",
                extra={
                    "model": self.model,
                    "timeout": self.timeout,
                    "error": str(e)
                }
            )
            raise
        except litellm.APIError as e:
            logger.error(
                "LLM API error",
                extra={
                    "model": self.model,
                    "status_code": getattr(e, 'status_code', None),
                    "error": str(e)
                }
            )
            raise

    def extract_text(self, response: Any) -> str:
        """レスポンスからテキストを抽出（エラーハンドリング改善版）."""
        try:
            # Choicesからテキスト抽出
            if hasattr(response, 'choices') and len(response.choices) > 0:
                choice = response.choices[0]
                if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                    return choice.message.content or ""

            # フォールバック: 文字列化
            logger.warning(
                "Unexpected response structure, falling back to str()",
                extra={"response_type": type(response).__name__}
            )
            return str(response)

        except AttributeError as e:
            # ========== 変更: エラーをログして空文字列ではなく例外を再raise ==========
            logger.error(
                "Failed to extract text from response",
                extra={"error": str(e), "response": str(response)[:200]},
                exc_info=True
            )
            raise ValueError(f"Cannot extract text from response: {e}") from e
        # ========================================================================
```

**テスト追加:**

```python
# tests/llm/test_client_resilience.py

def test_completion_retries_on_timeout(mock_litellm):
    """タイムアウト時にリトライすることを確認."""
    import litellm

    # 最初の2回はタイムアウト、3回目は成功
    mock_litellm.completion.side_effect = [
        litellm.Timeout("Timeout 1"),
        litellm.Timeout("Timeout 2"),
        {"choices": [{"message": {"content": "Success"}}]}
    ]

    client = LLMClient(max_retries=3)
    response = client.completion([{"role": "user", "content": "test"}])

    assert response["choices"][0]["message"]["content"] == "Success"
    assert mock_litellm.completion.call_count == 3

def test_extract_text_raises_on_invalid_response():
    """不正なレスポンスで例外を発生させることを確認."""
    client = LLMClient()

    with pytest.raises(ValueError, match="Cannot extract text"):
        client.extract_text({"invalid": "structure"})
```

**工数:** 2日
**優先度:** 中 - 信頼性向上

---

#### 10. パフォーマンスベンチマーク追加

**現状:** 体系的なパフォーマンスベンチマークなし

**不足している領域:**
- スループットベンチマーク（tasks/second）
- レイテンシ測定（P50、P95、P99）
- メモリプロファイリング
- スケーラビリティテスト（1 vs 10 vs 100ワーカー）

**対策:**

```python
# tests/performance/test_benchmarks.py

import pytest
import time
from statistics import mean, median, quantiles
from graflow.core.workflow import workflow
from graflow.core.decorators import task
from graflow.core.task import ParallelGroup

@pytest.mark.benchmark
class TestWorkflowPerformance:
    """ワークフロー実行のパフォーマンスベンチマーク."""

    def test_throughput_1000_simple_tasks(self, benchmark):
        """1000個の単純タスクのスループットを測定."""

        @task
        def noop_task(i: int) -> int:
            """何もしないタスク."""
            return i

        def run_workflow():
            with workflow("perf_test") as wf:
                tasks = [noop_task.clone(f"task-{i}") for i in range(1000)]
                parallel = ParallelGroup(tasks)
                wf.add_task(parallel)
                wf.execute()

        result = benchmark(run_workflow)

        # パフォーマンス目標をアサート
        tasks_per_second = 1000 / result.stats.mean
        assert tasks_per_second > 100, f"Too slow: {tasks_per_second:.1f} tasks/sec"

        print(f"\nThroughput: {tasks_per_second:.1f} tasks/sec")
        print(f"Mean latency: {result.stats.mean*1000:.1f}ms")

    def test_latency_distribution(self):
        """タスク実行レイテンシ分布を測定."""

        @task
        def single_task() -> str:
            return "done"

        latencies = []
        for i in range(100):
            with workflow(f"latency_test_{i}") as wf:
                wf.add_task(single_task)

                start = time.perf_counter()
                wf.execute()
                latencies.append(time.perf_counter() - start)

        latencies.sort()
        p50 = median(latencies)
        p95 = quantiles(latencies, n=20)[18]  # 95パーセンタイル
        p99 = quantiles(latencies, n=100)[98]  # 99パーセンタイル

        print(f"\nLatency Distribution:")
        print(f"  P50: {p50*1000:.1f}ms")
        print(f"  P95: {p95*1000:.1f}ms")
        print(f"  P99: {p99*1000:.1f}ms")

        # SLA目標をアサート
        assert p50 < 0.010, f"P50 latency too high: {p50*1000:.1f}ms"
        assert p95 < 0.050, f"P95 latency too high: {p95*1000:.1f}ms"
        assert p99 < 0.100, f"P99 latency too high: {p99*1000:.1f}ms"

    def test_memory_usage(self):
        """メモリ使用量を測定."""
        import tracemalloc

        @task
        def memory_task(i: int) -> List[int]:
            # 少しメモリを使う
            return list(range(i * 100))

        tracemalloc.start()

        with workflow("memory_test") as wf:
            tasks = [memory_task.clone(f"task-{i}") for i in range(100)]
            parallel = ParallelGroup(tasks)
            wf.add_task(parallel)
            wf.execute()

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        print(f"\nMemory Usage:")
        print(f"  Current: {current / 1024 / 1024:.1f} MB")
        print(f"  Peak: {peak / 1024 / 1024:.1f} MB")

        # メモリ目標をアサート
        assert peak / 1024 / 1024 < 100, f"Peak memory too high: {peak / 1024 / 1024:.1f} MB"
```

**CI/CDでの実行:**

```bash
# .github/workflows/performance.yml

name: Performance Benchmarks

on:
  pull_request:
  schedule:
    - cron: '0 0 * * 0'  # 毎週日曜日

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run benchmarks
        run: |
          uvx pytest tests/performance/ --benchmark-only --benchmark-autosave
      - name: Store benchmark result
        uses: benchmark-action/github-action-benchmark@v1
        with:
          tool: 'pytest'
          output-file-path: .benchmarks/benchmark.json
```

**目標:** パフォーマンスベースラインの確立

**工数:** 1週間
**優先度:** 中 - パフォーマンス回帰の追跡

---

#### 11. ドキュメントの一貫性改善

**現状:** ドキュメントは存在するが一貫性に欠ける

**対策:**

**1) モジュールレベルのdocstringを全ファイルに追加:**

```python
# graflow/hitl/manager.py

"""Human-in-the-Loop フィードバック管理.

このモジュールはワークフロー実行中の人間フィードバック
リクエスト処理のためのFeedbackManagerクラスを提供します。
対応機能:

- 複数のフィードバックタイプ（承認、テキスト入力、選択など）
- チェックポイント統合によるインテリジェントなタイムアウト処理
- ユニバーサル通知システム（Slack、webhook、コンソール）
- Redis経由の分散フィードバック永続化

Example:
    >>> from graflow.hitl.manager import FeedbackManager
    >>> manager = FeedbackManager(backend="redis")
    >>> manager.request_feedback(...)

See Also:
    - :mod:`graflow.hitl.types`: フィードバック型定義
    - :mod:`graflow.hitl.notification`: 通知システム
"""
```

**2) Architecture Decision Records (ADR)の追加:**

```markdown
# docs/adr/0001-redis-distributed-coordination.md

# Redis使用による分散コーディネーション

日付: 2025-01-15

## ステータス

承認済み

## コンテキスト

並列グループ実行のワーカー間分散コーディネーションが必要。

## 決定

バリアとpub/subを使用したRedisベースのコーディネーションを採用。

## 結果

**ポジティブ:**
- 実績のある信頼性の高い技術
- 低レイテンシのpub/sub
- 組み込みの永続化

**ネガティブ:**
- 単一障害点（Redis Clusterで緩和）
- ネットワーク依存
- メモリベースストレージ

## 実装ノート

- ハートビートによるワーカー監視を追加（2025-12-08）
- KEYSからSCANへの移行（2025-12-08）
```

**工数:** 2週間
**優先度:** 中 - 開発者エクスペリエンス向上

---

### 🟢 低優先（Low Priority）

#### 12. プロパティベーステスト

**推奨:** Hypothesisを使用したコアアルゴリズムのプロパティベーステストを追加

```python
# tests/property/test_graph_properties.py

from hypothesis import given, strategies as st
from graflow.core.graph import TaskGraph

@given(st.lists(st.text(min_size=1, max_size=20), min_size=0, max_size=100))
def test_topological_sort_preserves_all_nodes(task_ids):
    """トポロジカルソートは全ノードを正確に1回含む."""
    # 重複を除去
    task_ids = list(set(task_ids))

    graph = TaskGraph()
    for task_id in task_ids:
        graph.add_node(create_dummy_task(task_id), task_id)

    sorted_ids = graph.topological_sort()
    assert set(sorted_ids) == set(task_ids)
    assert len(sorted_ids) == len(task_ids)

@given(st.lists(st.tuples(st.text(min_size=1), st.text(min_size=1)), min_size=1))
def test_cycle_detection_is_deterministic(edges):
    """サイクル検出は決定的である."""
    graph = TaskGraph()
    for src, dst in edges:
        graph.add_edge(src, dst)

    has_cycle1 = graph.has_cycle()
    has_cycle2 = graph.has_cycle()
    assert has_cycle1 == has_cycle2
```

**工数:** 2週間
**優先度:** 低 - 堅牢性向上

---

#### 13. 分散トレーシングコンテキスト伝播

**推奨:** 分散シナリオでトレースコンテキストが適切に伝播されることを確保

```python
# graflow/trace/propagation.py （新規作成）

from typing import Dict, Optional
import uuid

class TraceContext:
    """分散トレーシング用のトレースコンテキスト."""

    def __init__(
        self,
        trace_id: str,
        span_id: str,
        parent_span_id: Optional[str] = None
    ):
        self.trace_id = trace_id
        self.span_id = span_id
        self.parent_span_id = parent_span_id

    def to_headers(self) -> Dict[str, str]:
        """HTTPヘッダーに変換（伝播用）."""
        headers = {
            "X-Trace-Id": self.trace_id,
            "X-Span-Id": self.span_id,
        }
        if self.parent_span_id:
            headers["X-Parent-Span-Id"] = self.parent_span_id
        return headers

    @classmethod
    def from_headers(cls, headers: Dict[str, str]) -> "TraceContext":
        """ヘッダーからトレースコンテキストを抽出."""
        return cls(
            trace_id=headers.get("X-Trace-Id", str(uuid.uuid4())),
            span_id=headers.get("X-Span-Id", str(uuid.uuid4())),
            parent_span_id=headers.get("X-Parent-Span-Id")
        )
```

**工数:** 1週間
**優先度:** 低 - 現在のトレーシングは動作している（拡張機能）

---

#### 14. AsyncIO対応インターフェース

**問題:**
- `wait_barrier`の`time.sleep`ブロッキングがエンジンスレッドを停止
- ハートビートチェックやシグナル処理を並行実行できない

**推奨:** 将来のマイグレーション用に`async def execute(...)`インターフェースを定義、または非ブロッキング`select`/`poll`メカニズムを使用

```python
# graflow/core/engine_async.py （将来の拡張）

import asyncio
from typing import Optional, Any

class AsyncWorkflowEngine:
    """非同期ワークフローエンジン（将来の実装）."""

    async def execute(
        self,
        context: ExecutionContext,
        start_task_id: Optional[str] = None
    ) -> Any:
        """ワークフローを非同期実行."""
        # ... async/await ベースの実装 ...
```

**工数:** 1週間（広範なリファクタリング）
**優先度:** 低 - 将来の高並行性対応

---

## 実装ロードマップ

### フェーズ1: 信頼性とエラーハンドリング（2-3週間）

**Week 1-2: トレーシングとRedis対応**
- HITLタイムアウト時のトレース終了パッチ
- Redis KEYS → SCAN置き換え
- ワーカーハートビート実装
- 分散キューDLQ実装

**Week 3: エラーハンドリング**
- 広範な例外ハンドラを特定の例外に置き換え
- 構造化ログ追加
- 失敗伝播のテスト追加

### フェーズ2: テストとパフォーマンス（2週間）

**Week 5: 統合テスト**
- Redis HITL統合テスト
- チェックポイント/レジュームテスト
- Langfuseトレーシングテスト

**Week 6: パフォーマンスベンチマーク**
- pytest-benchmark セットアップ
- ベースライン測定作成
- CIパイプラインに追加

### フェーズ3: リファクタリングとドキュメント（2週間）

**Week 7: アーキテクチャリファクタリング**
- ExecutionContextから専門マネージャーを抽出
- グラフシリアライゼーション最適化（CAS）
- LLMクライアント回復力強化

**Week 8: ドキュメント**
- モジュールdocstring追加
- ADRドキュメント作成
- APIリファレンス更新

### フェーズ4: オプショナル拡張（継続的）

- プロパティベーステスト
- 分散トレーシング改善
- アーキテクチャリファクタリング（必要に応じて）

---

## 成功指標

| 指標 | 現状 | 目標 | タイムライン |
|------|------|------|--------------|
| 広範な例外ハンドラ | 6-73 | <10（クリーンアップのみ） | 1週間 |
| Redis KEYS使用箇所 | 3 | 0（全てSCAN） | 1週間 |
| ワーカークラッシュリカバリ時間 | 30秒 | <10秒（ハートビート） | 3日 |
| DLQ可視性 | なし | DLQ+カウンター+テスト | 3日 |
| HITLタイムアウト時のトレース完了 | 不足 | 100%（テストカバー） | 1日 |
| 統合テストカバレッジ | 限定的 | 包括的 | 2週間 |
| パフォーマンスベースライン | なし | 確立済み | 1週間 |
| ExecutionContext LOC | ~1400 | <800 | 1週間 |
| グラフアップロード率 | 100% | <5%（変更時のみ） | 2日 |

---

## メンテナンス

- **四半期ごとにレビュー**
- **主要機能追加後に更新**
- **GitHub issuesで進捗追跡**
- **CONTRIBUTING.mdからリンク**

---

**文書バージョン:** 3.0（統合版）
**最終更新:** 2025-12-08
**次回レビュー:** 2026-03-08
**ステータス:** アクティブ

---

## 付録: 主要な違いと統合ノート

**Claude版との違い:**
- HITLタイムアウト時のトレース終了を追加（Codex提案）
- Redisの本番対応を詳細化（Codex提案）
- ワーカーハートビートを追加（Gemini提案）
- グラフシリアライゼーション最適化を追加（Gemini提案）

**Codex版との違い:**
- ExecutionContext分解を中優先に含める（全て共通）
- より実践的なコード例を提供
- 日本語による詳細な説明

**Gemini版との違い:**
- AsyncIO対応を低優先に含める
- Schedulerの抽出を中優先に含める
- より段階的な実装ロードマップ

**統合の利点:**
- 3つのレビューから最も重要な提案を統合
- 実装優先度を明確化
- 日本語による包括的な説明
