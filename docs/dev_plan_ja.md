# Graflow 開発計画

**日付:** 2025-12-08
**現在のバージョン:** v0.2.x
**目標バージョン:** v0.3.0+

## エグゼクティブサマリー

Graflowは、HITL、チェックポイント、トレーシング、分散実行などの包括的な機能を備えた成熟したワークフロー実行エンジンへと進化しました。本開発計画は、プロダクション対応、開発者体験、エコシステム統合に焦点を当てた次のフェーズの改善について概説します。

---

## 現状評価

### 実装済み機能 ✅

**コアワークフローエンジン:**
- サイクル検出機能付きタスクグラフ実行
- シーケンシャル（`>>`）およびパラレル（`|`）オペレータ
- 動的タスク生成（`next_task()`、`next_iteration()`）
- Redis経由の分散実行
- ローカルおよびRedisベースのタスクキューとチャネル

**プロダクション機能:**
- **チェックポイント/再開:** メタデータ付き3ファイルチェックポイントシステム
- **HITL:** Slack/webhookによる通知機能を持つマルチタイプフィードバックシステム
- **トレーシング:** 可観測性のためのLangfuse統合
- **LLM統合:** LLMクライアントとエージェント管理
- **API:** ワークフロー管理とフィードバック用REST API
- **ハンドラ:** ダイレクト、Docker、カスタム実行戦略

**開発ツール:**
- 包括的なテストスイート
- mypyによる型チェック
- ruffによるリンティング
- 豊富なサンプルコレクション
- ビジュアライゼーションツール（ASCII、Mermaid、PNG）

### ギャップ分析

**未実装または未発達:**
1. **プロダクション監視:** ビルトインメトリクス/ダッシュボードなし
2. **レジリエンス:** サーキットブレーカー、リトライ戦略が限定的
3. **セキュリティ:** タスク署名なし、Redis認証はオプション
4. **パフォーマンス:** 体系的なベンチマークなし
5. **DX:** IDE統合、デバッグツールが限定的
6. **エコシステム:** 事前構築された統合（Airflow、Prefectなど）なし

---

## 開発ロードマップ

### フェーズ1: プロダクション強化（v0.3.0）- 6-8週間

**目標:** Graflowをミッションクリティカルなワークフローに対応できるプロダクションレディにする

#### 1.1 可観測性と監視（2週間）

**メトリクス収集:**
```python
# graflow/observability/metrics.py

from typing import Protocol, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime

class MetricsCollector(Protocol):
    """メトリクス収集のプロトコル。"""
    def record_task_execution(self, task_id: str, duration: float, status: str) -> None: ...
    def record_workflow_execution(self, workflow_id: str, duration: float, tasks_count: int) -> None: ...
    def record_queue_depth(self, queue_name: str, depth: int) -> None: ...
    def increment_counter(self, metric_name: str, value: int = 1, tags: Dict[str, str] = None) -> None: ...

@dataclass
class PrometheusMetrics:
    """Prometheusメトリクスエクスポーター。"""
    registry: Any  # prometheus_client.CollectorRegistry
    _task_duration: Any = None
    _workflow_duration: Any = None
    _queue_depth: Any = None
    _task_counter: Any = None

    def __post_init__(self):
        from prometheus_client import Histogram, Gauge, Counter

        self._task_duration = Histogram(
            'graflow_task_duration_seconds',
            'タスク実行時間',
            ['task_type', 'status'],
            registry=self.registry
        )

        self._workflow_duration = Histogram(
            'graflow_workflow_duration_seconds',
            'ワークフロー実行時間',
            ['workflow_name'],
            registry=self.registry
        )

        self._queue_depth = Gauge(
            'graflow_queue_depth',
            '現在のキュー深度',
            ['queue_name'],
            registry=self.registry
        )

        self._task_counter = Counter(
            'graflow_tasks_total',
            '実行されたタスクの総数',
            ['status'],
            registry=self.registry
        )

    def record_task_execution(self, task_id: str, duration: float, status: str) -> None:
        self._task_duration.labels(task_type=task_id.split('_')[0], status=status).observe(duration)
        self._task_counter.labels(status=status).inc()
```

**ヘルスチェック:**
```python
# graflow/observability/health.py

from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Callable

class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"

@dataclass
class HealthCheck:
    name: str
    check_fn: Callable[[], bool]
    critical: bool = False

class HealthMonitor:
    """システムヘルスの監視。"""

    def __init__(self):
        self._checks: List[HealthCheck] = []

    def register_check(self, name: str, check_fn: Callable[[], bool], critical: bool = False):
        """ヘルスチェックを登録。"""
        self._checks.append(HealthCheck(name, check_fn, critical))

    def get_health(self) -> Dict[str, Any]:
        """システム全体のヘルスを取得。"""
        results = {}
        status = HealthStatus.HEALTHY

        for check in self._checks:
            try:
                is_healthy = check.check_fn()
                results[check.name] = {"status": "ok" if is_healthy else "fail"}

                if not is_healthy:
                    if check.critical:
                        status = HealthStatus.UNHEALTHY
                    elif status == HealthStatus.HEALTHY:
                        status = HealthStatus.DEGRADED
            except Exception as e:
                results[check.name] = {"status": "error", "error": str(e)}
                status = HealthStatus.UNHEALTHY

        return {
            "status": status.value,
            "checks": results
        }
```

**成果物:**
- [ ] Prometheusメトリクスエクスポーター
- [ ] ヘルスチェックエンドポイント
- [ ] Grafanaダッシュボードテンプレート
- [ ] 構造化ロギングの改善
- [ ] 分散トレーシングのコンテキスト伝播

---

#### 1.2 セキュリティ強化（2週間）

**タスク署名:**
```python
# graflow/security/signing.py

import hmac
import hashlib
from typing import Callable
from graflow.exceptions import SecurityError

class TaskSigner:
    """タスクの署名と検証を行う。"""

    def __init__(self, secret_key: bytes):
        if len(secret_key) < 32:
            raise ValueError("シークレットキーは最低32バイト必要です")
        self.secret_key = secret_key

    def sign_task(self, task_data: bytes) -> bytes:
        """タスクデータに署名する。"""
        signature = hmac.new(
            self.secret_key,
            task_data,
            hashlib.sha256
        ).digest()
        return signature + task_data

    def verify_and_load(self, signed_data: bytes) -> bytes:
        """署名を検証してタスクデータを返す。"""
        if len(signed_data) < 32:
            raise SecurityError("無効な署名済みデータ")

        signature = signed_data[:32]
        task_data = signed_data[32:]

        expected_sig = hmac.new(
            self.secret_key,
            task_data,
            hashlib.sha256
        ).digest()

        if not hmac.compare_digest(signature, expected_sig):
            raise SecurityError("署名検証に失敗しました")

        return task_data
```

**Redis認証:**
```python
# 設定でRedis認証を強制
REDIS_CONFIG = {
    "require_auth": True,  # プロダクションでは必須
    "password": os.getenv("REDIS_PASSWORD"),
    "ssl": True,  # SSL/TLSを有効化
    "ssl_cert_reqs": "required",
}
```

**成果物:**
- [ ] タスク署名検証
- [ ] Redis認証の強制
- [ ] Redis接続のSSL/TLS
- [ ] セキュリティ監査ドキュメント
- [ ] セキュリティベストプラクティスガイド

---

#### 1.3 レジリエンス機能（2週間）

**サーキットブレーカー:**
```python
# graflow/resilience/circuit_breaker.py

from enum import Enum
from datetime import datetime, timedelta
from typing import Callable, Any
from functools import wraps

class CircuitState(Enum):
    CLOSED = "closed"  # 通常動作
    OPEN = "open"  # 失敗中、呼び出しを拒否
    HALF_OPEN = "half_open"  # 回復をテスト中

class CircuitBreaker:
    """サーキットブレーカーパターンの実装。"""

    def __init__(
        self,
        failure_threshold: int = 5,
        timeout: float = 60.0,
        recovery_timeout: float = 30.0
    ):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.recovery_timeout = recovery_timeout

        self._state = CircuitState.CLOSED
        self._failures = 0
        self._last_failure_time = None
        self._last_success_time = None

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """サーキットブレーカー保護付きで関数を呼び出す。"""
        if self._state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self._state = CircuitState.HALF_OPEN
            else:
                raise CircuitBreakerError("サーキットブレーカーがOPEN状態です")

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise

    def _on_success(self):
        """成功した呼び出しを処理。"""
        self._failures = 0
        self._last_success_time = datetime.now()
        if self._state == CircuitState.HALF_OPEN:
            self._state = CircuitState.CLOSED

    def _on_failure(self):
        """失敗した呼び出しを処理。"""
        self._failures += 1
        self._last_failure_time = datetime.now()

        if self._failures >= self.failure_threshold:
            self._state = CircuitState.OPEN

    def _should_attempt_reset(self) -> bool:
        """回復を試みるべきかチェック。"""
        if self._last_failure_time is None:
            return True

        elapsed = (datetime.now() - self._last_failure_time).total_seconds()
        return elapsed >= self.recovery_timeout
```

**リトライ戦略:**
```python
# graflow/resilience/retry.py

from typing import Callable, Type, Tuple
import time
import random

class RetryStrategy:
    """設定可能なリトライ戦略。"""

    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        exceptions: Tuple[Type[Exception], ...] = (Exception,)
    ):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.exceptions = exceptions

    def execute(self, func: Callable, *args, **kwargs):
        """リトライ付きで関数を実行。"""
        last_exception = None

        for attempt in range(self.max_attempts):
            try:
                return func(*args, **kwargs)
            except self.exceptions as e:
                last_exception = e

                if attempt < self.max_attempts - 1:
                    delay = self._calculate_delay(attempt)
                    time.sleep(delay)

        raise last_exception

    def _calculate_delay(self, attempt: int) -> float:
        """リトライ試行の遅延を計算。"""
        delay = min(
            self.base_delay * (self.exponential_base ** attempt),
            self.max_delay
        )

        if self.jitter:
            delay *= (0.5 + random.random())

        return delay
```

**成果物:**
- [ ] サーキットブレーカー実装
- [ ] 指数バックオフリトライ戦略
- [ ] 失敗タスク用のデッドレターキュー
- [ ] グレースフルデグラデーションサポート
- [ ] レジリエンステストスイート

---

#### 1.4 パフォーマンス最適化（2週間）

**コネクションプーリング:**
```python
# graflow/utils/redis_pool.py

import redis
from typing import Dict, Optional
import threading

class RedisConnectionPool:
    """シングルトンRedisコネクションプール。"""

    _pools: Dict[str, redis.ConnectionPool] = {}
    _lock = threading.Lock()

    @classmethod
    def get_pool(
        cls,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        max_connections: int = 50,
        **kwargs
    ) -> redis.ConnectionPool:
        """コネクションプールを取得または作成。"""
        key = f"{host}:{port}:{db}"

        if key not in cls._pools:
            with cls._lock:
                if key not in cls._pools:
                    cls._pools[key] = redis.ConnectionPool(
                        host=host,
                        port=port,
                        db=db,
                        max_connections=max_connections,
                        decode_responses=True,
                        **kwargs
                    )

        return cls._pools[key]

    @classmethod
    def create_client(cls, **kwargs) -> redis.Redis:
        """共有プールを使用してRedisクライアントを作成。"""
        pool = cls.get_pool(**kwargs)
        return redis.Redis(connection_pool=pool)
```

**ブロッキングキュー操作:**
```python
# DistributedTaskQueueでポーリングをBLPOPに置き換え
def dequeue(self, timeout: int = 1) -> Optional[TaskSpec]:
    """ブロッキングpopでデキュー。"""
    result = self.redis_client.blpop(self.queue_key, timeout=timeout)
    if result is None:
        return None
    _, data = result
    return self._deserialize_task_spec(data)
```

**成果物:**
- [ ] Redisコネクションプーリング
- [ ] ブロッキングキュー操作（BLPOP）
- [ ] パフォーマンスベンチマークスイート
- [ ] メモリプロファイリングと最適化
- [ ] 負荷テストフレームワーク

---

### フェーズ2: 開発者体験（v0.4.0）- 4-6週間

#### 2.1 IDE統合（2週間）

**VS Code拡張機能:**
```json
{
  "name": "graflow-vscode",
  "features": [
    "ワークフロー定義のシンタックスハイライト",
    "タスクデコレータの自動補完",
    "ワークフローグラフのインラインビジュアライゼーション",
    "デバッグアダプタプロトコルサポート",
    "テストランナー統合"
  ]
}
```

**デバッグアダプタ:**
```python
# graflow/debug/adapter.py

class GraflowDebugAdapter:
    """デバッグアダプタプロトコル実装。"""

    def set_breakpoint(self, task_id: str):
        """タスクにブレークポイントを設定。"""
        ...

    def step_over(self):
        """次のタスクを実行。"""
        ...

    def inspect_context(self) -> Dict[str, Any]:
        """現在の実行コンテキストを検査。"""
        ...

    def evaluate_expression(self, expr: str) -> Any:
        """現在のコンテキストで式を評価。"""
        ...
```

**成果物:**
- [ ] VS Code拡張機能
- [ ] デバッグアダプタプロトコル
- [ ] インタラクティブワークフロービルダー（Web UI）
- [ ] ワークフロー検証CLI
- [ ] 自動生成型スタブ

---

#### 2.2 テストユーティリティ（1週間）

**テストフィクスチャ:**
```python
# graflow/testing/fixtures.py

import pytest
from graflow.core.workflow import workflow
from graflow.core.context import ExecutionContext

@pytest.fixture
def workflow_context():
    """テストワークフローコンテキストを作成。"""
    with workflow("test") as wf:
        yield wf

@pytest.fixture
def execution_context(tmp_path):
    """テスト実行コンテキストを作成。"""
    return ExecutionContext.create(
        graph=...,
        start_node="test",
        checkpoint_dir=tmp_path
    )

@pytest.fixture
def mock_redis():
    """テスト用モックRedisクライアント。"""
    from fakeredis import FakeRedis
    return FakeRedis(decode_responses=True)
```

**テストヘルパー:**
```python
# graflow/testing/helpers.py

def assert_workflow_completed(context: ExecutionContext):
    """ワークフローが正常に完了したことをアサート。"""
    assert context.is_completed()
    assert all(task_id in context.completed_tasks for task_id in context.graph.nodes)

def assert_checkpoint_valid(checkpoint_path: Path):
    """チェックポイントが有効であることをアサート。"""
    assert checkpoint_path.exists()
    assert (checkpoint_path.parent / f"{checkpoint_path.stem}.state.json").exists()
    assert (checkpoint_path.parent / f"{checkpoint_path.stem}.meta.json").exists()
```

**成果物:**
- [ ] テストフィクスチャライブラリ
- [ ] Redis、LLMなどのモックヘルパー
- [ ] ワークフローテストユーティリティ
- [ ] 統合テストテンプレート
- [ ] パフォーマンステストヘルパー

---

#### 2.3 ドキュメントと例（2週間）

**インタラクティブチュートリアル:**
```markdown
# docs/tutorials/01-getting-started.md

シンプルなETLパイプラインを構築するウォークスルー:
- APIからのデータ抽出
- データ変換
- データベースへのデータロード
- エラーハンドリング
- チェックポイント
```

**APIリファレンス:**
```bash
# sphinxでAPIドキュメントを生成
make docs
# docs.graflow.devでホスト
```

**サンプルライブラリ:**
```
examples/
  11_production/
    - production_etl_pipeline.py
    - ml_training_with_checkpoints.py
    - distributed_batch_processing.py
  12_integrations/
    - airflow_integration.py
    - prefect_migration.py
    - langchain_integration.py
```

**成果物:**
- [ ] インタラクティブチュートリアル（5-10個）
- [ ] 完全なAPIリファレンス
- [ ] プロダクションサンプルライブラリ
- [ ] 移行ガイド（Airflow、Prefectなどから）
- [ ] ビデオチュートリアル

---

### フェーズ3: エコシステム統合（v0.5.0）- 4-6週間

#### 3.1 事前構築された統合（3週間）

**LangChain統合:**
```python
# graflow/integrations/langchain.py

from langchain.chains import LLMChain
from graflow.core.decorators import task

class LangChainTask:
    """GraflowのLangChain統合。"""

    @staticmethod
    @task
    def run_chain(chain: LLMChain, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """LangChainチェーンをGraflowタスクとして実行。"""
        return chain.run(**inputs)
```

**Airflow互換性:**
```python
# graflow/integrations/airflow.py

from airflow.models import BaseOperator
from graflow.core.task import Task

class GraflowOperator(BaseOperator):
    """GraflowワークフローのためのAirflowオペレータ。"""

    def __init__(self, workflow_definition, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.workflow = workflow_definition

    def execute(self, context):
        """AirflowからGraflowワークフローを実行。"""
        return self.workflow.execute()
```

**MLflowトラッキング:**
```python
# graflow/integrations/mlflow.py

import mlflow
from graflow.core.decorators import task

@task
def track_experiment(experiment_name: str, params: Dict, metrics: Dict):
    """MLflowでML実験をトラッキング。"""
    with mlflow.start_run(run_name=experiment_name):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
```

**成果物:**
- [ ] LangChain統合
- [ ] Airflow互換性レイヤー
- [ ] Prefect移行ツール
- [ ] MLflowトラッキング統合
- [ ] Weights & Biases統合
- [ ] データパイプライン用dbt統合

---

#### 3.2 クラウドデプロイメント（2週間）

**AWSデプロイメント:**
```yaml
# cloudformation/graflow-stack.yaml
AWSTemplateFormatVersion: '2010-09-09'
Resources:
  GraflowECS:
    Type: AWS::ECS::Cluster
  GraflowService:
    Type: AWS::ECS::Service
  ElastiCache:
    Type: AWS::ElastiCache::ReplicationGroup
```

**Kubernetes Helmチャート:**
```yaml
# helm/graflow/values.yaml
replicaCount: 3
redis:
  enabled: true
  sentinel: true
workers:
  min: 2
  max: 10
  autoscaling: true
```

**成果物:**
- [ ] AWS CloudFormationテンプレート
- [ ] Kubernetes Helmチャート
- [ ] GCPデプロイメントガイド
- [ ] Azureデプロイメントガイド
- [ ] Docker Composeプロダクションセットアップ
- [ ] Terraformモジュール

---

### フェーズ4: 高度な機能（v0.6.0+）- 継続中

#### 4.1 高度なワークフローパターン

- **条件分岐:** 動的パス選択
- **Fan-out/fan-in:** スケーラブルな並列処理
- **Sagaパターン:** 分散トランザクションサポート
- **イベント駆動ワークフロー:** 外部イベントへの反応
- **サブワークフロー:** 再利用可能なワークフローコンポーネント

#### 4.2 エンタープライズ機能

- **マルチテナンシー:** テナントごとの分離されたワークフロー実行
- **RBAC:** ロールベースアクセス制御
- **監査ログ:** 完全な監査証跡
- **コンプライアンス:** GDPR、SOC2サポート
- **SLA監視:** SLAの追跡と強制

#### 4.3 ML/AI機能強化

- **AutoML統合:** 自動化されたモデルトレーニング
- **モデルバージョニング:** モデルバージョンの追跡
- **A/Bテスト:** ビルトイン実験フレームワーク
- **Feature Store統合:** フィーチャーストアへの接続
- **リアルタイム推論:** 低レイテンシーモデルサービング

---

## 成功指標

| フェーズ | 指標 | 現在 | 目標 |
|-------|--------|---------|--------|
| 1（プロダクション） | 稼働率 | N/A | 99.9% |
| 1（プロダクション） | P95レイテンシー | N/A | <100ms |
| 1（プロダクション） | セキュリティスコア | C | A |
| 2（DX） | セットアップ時間 | 30分 | <5分 |
| 2（DX） | 最初のワークフローまでの時間 | 1時間 | <15分 |
| 3（エコシステム） | 統合数 | 2 | 10+ |
| 3（エコシステム） | クラウドプラットフォーム | 0 | 3（AWS、GCP、Azure） |

---

## 実装優先度

### Must Have（v0.3.0）
- 可観測性（メトリクス、ヘルスチェック）
- セキュリティ強化
- レジリエンス機能
- パフォーマンス最適化

### Should Have（v0.4.0）
- IDE統合
- テストユーティリティ
- ドキュメント改善

### Nice to Have（v0.5.0+）
- 事前構築された統合
- クラウドデプロイメントテンプレート
- 高度なワークフローパターン

---

## リソース要件

**チーム:**
- 2-3人のフルタイムエンジニア
- 1人のDevOpsエンジニア（パートタイム）
- 1人のテクニカルライター（パートタイム）

**タイムライン:**
- フェーズ1: 6-8週間
- フェーズ2: 4-6週間
- フェーズ3: 4-6週間
- フェーズ4: 継続中

**予算:**
- 開発: $150k-200k（フェーズ1-3）
- インフラストラクチャ: $5k-10k/月
- ツールとサービス: $2k-5k/月

---

## リスク軽減

| リスク | 発生確率 | 影響 | 軽減策 |
|------|------------|--------|--------------|
| 破壊的変更 | 中 | 高 | セマンティックバージョニング、非推奨警告 |
| パフォーマンス低下 | 低 | 中 | 継続的ベンチマーク、負荷テスト |
| セキュリティ脆弱性 | 中 | クリティカル | 定期的監査、依存関係スキャン |
| 採用の課題 | 中 | 中 | より良いドキュメント、サンプル、移行ガイド |

---

## 結論

本開発計画は、Graflowをプロダクションレディで開発者フレンドリーなワークフロー実行エンジンとして位置づけます。まずプロダクション強化に焦点を当てることで、信頼性を確保します。その後の開発者体験とエコシステム統合のフェーズにより、採用を促進し、GraflowをPythonワークフローオーケストレーションの第一選択肢にします。

**次のステップ:**
1. 開発計画のレビューと承認
2. プロジェクトトラッキングのセットアップ（GitHub Projects/Jira）
3. リソース配分
4. フェーズ1実装の開始

---

**ドキュメントバージョン:** 1.0
**最終更新:** 2025-12-08
**次回レビュー:** 2026-01-08
**ステータス:** アクティブ
