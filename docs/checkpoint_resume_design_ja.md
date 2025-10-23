# チェックポイント／レジューム設計ドキュメント

**ドキュメント版**: 1.5
**日付**: 2025-01-23
**ステータス**: Draft

**変更履歴**:
- v1.5: CheckpointManager をクラスメソッド構成に変更し、パス形式からストレージバックエンドを推測する仕様に更新
- v1.4: `mark_task_completed()` が `completed_tasks` セットに task_id を追加するよう再度有効化
- v1.3: アーキテクチャ図から残っていた `__workflow_state__` 参照を削除
- v1.2: TaskQueue の永続化戦略（InMemoryTaskQueue と RedisTaskQueue の違い）を明文化
- v1.1: CheckpointManager 抽象化を導入し、3 ファイル構成へ移行、`__workflow_state__` を排除
- v1.0: 初版

---

## 設計サマリー

本ドキュメントでは、Graflow のチェックポイント／レジューム機能によるワークフロー状態の永続化と復旧について説明します。

### 主要な設計判断

1. **明示的なチェックポイント**: ユーザーが `context.checkpoint()` を呼び出したときのみ保存（自動チェックポイントなし）
2. **CheckpointManager**: すべてクラスメソッドで提供し、パス形式からストレージバックエンドを推測
   - ローカル: "checkpoints/session_123.pkl"
   - Redis: "redis://session_123"（将来対応）
   - S3: "s3://bucket/session_123.pkl"（将来対応）
3. **3 ファイル構成**:
   - `.pkl`（ExecutionContext の pickle）
   - `.state.json`（pending_tasks を含むチェックポイント状態）
   - `.meta.json`（メタデータ）
4. **キュー状態の永続化**:
   - InMemoryTaskQueue: pending_tasks をチェックポイントに保存し、レジューム時に再キューイング
   - RedisTaskQueue: キューは Redis に残るため再キューイング不要
5. **開始ノードのみ**: チャネルに `__workflow_state__` を保持しない（設計を簡素化）
6. **バックエンド対応**: MemoryChannel と RedisChannel の両方で動作

---

## 目次

1. [概要](#概要)
2. [モチベーション](#モチベーション)
3. [要求事項](#要求事項)
4. [設計目標](#設計目標)
5. [アーキテクチャ](#アーキテクチャ)
6. [API 設計](#api-設計)
7. [実装詳細](#実装詳細)
8. [バックエンド対応](#バックエンド対応)
9. [利用例](#利用例)
10. [エッジケースと制約](#エッジケースと制約)
11. [今後の拡張](#今後の拡張)

---

## 概要

本ドキュメントは、Graflow ワークフローにおけるチェックポイント／レジューム機能の設計を説明します。この機能により、ユーザーは以下を実現できます。

- 任意のタイミングでワークフロー状態を保存（チェックポイント）
- 保存したチェックポイントから、停止・障害後に実行を再開
- Memory/Redis チャネルのどちらでも一貫した動作

設計は Graflow の既存ステートマシン実行モデルとチャネルベースの状態管理を活用します。

---

## モチベーション

### ユースケース

1. **長時間ワークロード**: 数時間〜数日に及ぶ ML 学習やデータ処理パイプライン
2. **フォールトトレランス**: インフラ障害、OOM、クラッシュ後に途中から再開
3. **反復開発**: 中間状態を保存してデバッグやテストを容易化
4. **分散実行**: ワーカー再起動後もチェックポイントから作業を継続
5. **コスト最適化**: 高コストな処理を一時停止し、後から再開

### 現状の課題

チェックポイント／レジュームがない場合:
- 失敗時に最初から再実行する必要がある
- 中間進捗を永続化する仕組みがない
- 長時間ワークフローのデバッグが困難
- クラッシュ後にワーカーが再開できない

---

## 要求事項

### 機能要求

**FR1**: ユーザーはタスク内から `context.checkpoint()` を呼び出してチェックポイントを作成できること

**FR2**: チェックポイントはセッション ID、グラフ、ステップ数、完了したタスク、サイクル回数を含む完全なワークフロー状態を保存すること

**FR3**: `ExecutionContext.resume(path)` がチェックポイントファイルからワークフローを復元すること

**FR4**: MemoryChannel と RedisChannel の両バックエンドで透過的に動作すること

**FR5**: レジューム後は次の pending タスクから実行が再開されること

### 非機能要求

**NFR1**: チェックポイント作成は軽量であり、一般的なワークフローで 1 秒未満を目標とする

**NFR2**: チェックポイントファイルはマシン間で移動可能であること（RedisChannel 利用時は同一 Redis インスタンスを共有）

**NFR3**: チェックポイントを使わない既存ワークフローを破壊しない

**NFR4**: API はシンプルで直感的であること

---

## 設計目標

1. **明示的制御**: ユーザーが `context.checkpoint()` を呼び出したときのみ保存する
2. **バックエンド非依存**: Memory/Redis どちらのチャネルでも同じ API で利用可能
3. **オーバーヘッド最小化**: 既存のシリアライゼーション（`__getstate__`/`__setstate__`）を活用
4. **ステートマシン適合**: `next_iteration()` などの状態遷移処理と自然に統合
5. **メタデータ対応**: チェックポイントに任意メタデータを付与可能

---

## アーキテクチャ

### ハイレベルアーキテクチャ

```
┌─────────────────────────────────────────────────────────────┐
│                    ユーザータスクコード                     │
│                                                              │
│  @task(inject_context=True)                                 │
│  def my_task(context):                                      │
│      # ... processing ...                                   │
│      context.checkpoint(metadata={"stage": "step1"})  ◄─────┼─── 明示的チェックポイント
│      # ... more processing ...                              │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              TaskExecutionContext                           │
│  .checkpoint(path, metadata) ──────────────────────┐        │
└────────────────────────────────────────────────────┼────────┘
                                                     │
                                                     ▼
┌─────────────────────────────────────────────────────────────┐
│              ExecutionContext                               │
│                                                              │
│  .mark_task_completed(task_id)                              │
│    └─> completed_tasks セットへ task_id を追加              │
│                                                              │
│  .checkpoint(path, metadata) ───────────────────────────┐   │
│    1. checkpoint_metadata を更新                        │   │
│    2. セッション ID・グラフなどの状態を収集             │   │
│    3. self.save(path) を呼ぶ ──────────────────────────┼───┼─> cloudpickle
│                                                          │   │
│  .resume(path) ◄─────────────────────────────────────────┘   │
│    1. pickle をロード（__setstate__ を呼ぶ）               │
│    2. completed_tasks/cycle_counts を復元                  │
│    3. pending_tasks をキューへ復元（InMemoryTaskQueue）     │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              チャネル（Memory または Redis）               │
│                                                              │
│  Memory: __getstate__ でチャネルデータを保存               │
│  Redis:  セッション ID のみ保存（データは Redis 常駐）    │
└─────────────────────────────────────────────────────────────┘
```

### コンポーネントの相互作用

```
ユーザータスク
    │
    │ context.checkpoint()
    ▼
TaskExecutionContext.checkpoint()
    │
    │ タスクメタデータを追加
    ▼
ExecutionContext.checkpoint()
    │
    ├─> チェックポイント状態を収集（session_id, graph, steps など）
    │
    └─> self.save(path)
            │
            ├─> __getstate__()
            │       └─> MemoryChannel: チャネルデータを保存
            │       └─> RedisChannel: session_id のみ保存
            │
            └─> cloudpickle.dump()
```

---

## API 設計

### CheckpointManager API

```python
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

@dataclass
class CheckpointMetadata:
    """チェックポイントのメタデータ"""
    checkpoint_id: str
    session_id: str
    created_at: datetime
    steps: int
    start_node: str
    backend: dict[str, str]  # {"queue": "memory", "channel": "memory"}
    user_metadata: dict[str, Any]  # ユーザー定義メタデータ

class CheckpointManager:
    """ストレージバックエンド抽象化付きでチェックポイントの作成・復元を管理する。"""

    # パス形式でストレージバックエンドを判別:
    # - ローカル: "checkpoints/session_123.pkl"
    # - Redis: "redis://session_123"（将来対応）
    # - S3: "s3://bucket/session_123.pkl"（将来対応）

    @classmethod
    def create_checkpoint(
        cls,
        context: ExecutionContext,
        path: Optional[str] = None,
        metadata: Optional[dict] = None
    ) -> tuple[str, CheckpointMetadata]:
        """ExecutionContext からチェックポイントを生成する。"""

    @classmethod
    def resume_from_checkpoint(
        cls,
        checkpoint_path: str
    ) -> tuple[ExecutionContext, CheckpointMetadata]:
        """チェックポイントから再開する。"""

    @classmethod
    def list_checkpoints(
        cls,
        path_pattern: Optional[str] = None
    ) -> list[CheckpointMetadata]:
        """利用可能なチェックポイント一覧（将来拡張）。"""

    @classmethod
    def _infer_backend_from_path(cls, path: Optional[str]) -> str:
        """パス形式からストレージバックエンドを推測する。"""
```

### ExecutionContext API

```python
class ExecutionContext:
    # 追加属性
    completed_tasks: set[str]           # 完了したタスク ID の集合
    checkpoint_metadata: dict[str, Any] # 直近のチェックポイントメタデータ

    # 追加メソッド
    def mark_task_completed(self, task_id: str) -> None:
        """チェックポイント追跡のため、タスクを完了済みにマークする。"""

    def get_checkpoint_state(self) -> dict[str, Any]:
        """チェックポイントに必要な状態を取得する。"""
```

### TaskExecutionContext API

```python
class TaskExecutionContext:
    def checkpoint(
        self,
        path: Optional[str] = None,
        metadata: Optional[dict] = None
    ) -> tuple[str, CheckpointMetadata]:
        """タスク内からチェックポイントを作成する。"""
```

### WorkflowEngine との統合

```python
class WorkflowEngine:
    def execute(self, context: ExecutionContext, start_task_id: Optional[str] = None):
        # 各タスク実行後
        context.mark_task_completed(task_id)  # NEW: 完了トラッキング
        context.increment_step()
```

---

## 実装詳細

### CheckpointManager の実装例

```python
class CheckpointManager:
    @classmethod
    def create_checkpoint(cls, context, path=None, metadata=None):
        # 1. パスからバックエンドを推測し、必要ならパスを自動生成
        backend = cls._infer_backend_from_path(path)
        checkpoint_id = cls._generate_checkpoint_id(context)
        if path is None:
            path = cls._generate_path(checkpoint_id, backend)

        # 2. チェックポイント状態を収集
        checkpoint_state = {
            "session_id": context.session_id,
            "start_node": context.start_node,
            "steps": context.steps,
            "completed_tasks": list(context.completed_tasks),
            "cycle_counts": dict(context.cycle_controller.cycle_counts),
            "pending_tasks": cls._get_pending_tasks(context),
            "backend": {
                "queue": context._queue_backend_type,
                "channel": context._channel_backend_type
            }
        }

        # 3. メタデータを生成
        metadata_obj = CheckpointMetadata(
            checkpoint_id=checkpoint_id,
            session_id=context.session_id,
            created_at=datetime.now(),
            steps=context.steps,
            start_node=context.start_node,
            backend=checkpoint_state["backend"],
            user_metadata=metadata or {}
        )

        # 4. コンテキストを保存（Memory チャネルの場合はチャネルデータも含む）
        context.save(path)

        # 5. チェックポイント状態を別ファイルに保存
        cls._save_checkpoint_state(path, checkpoint_state, backend)

        # 6. メタデータを保存
        cls._save_metadata(path, metadata_obj, backend)

        return path, metadata_obj

    @classmethod
    def resume_from_checkpoint(cls, checkpoint_path):
        # 1. パスからバックエンドを推測
        backend = cls._infer_backend_from_path(checkpoint_path)

        # 2. チェックポイント状態をロード
        checkpoint_state = cls._load_checkpoint_state(checkpoint_path, backend)

        # 3. pickle から ExecutionContext を復元
        context = ExecutionContext.load(checkpoint_path)

        # 4. バックエンド別に pending_tasks を復元
        if context._queue_backend_type != "redis":
            for task_id in checkpoint_state["pending_tasks"]:
                task = context.graph.get_node(task_id)
                context.add_to_queue(task)
        # Redis の場合は既にキューが Redis に存在

        # 5. 完了タスク追跡を復元
        context.completed_tasks = set(checkpoint_state["completed_tasks"])

        # 6. サイクル回数を復元
        context.cycle_controller.cycle_counts.update(
            checkpoint_state["cycle_counts"]
        )

        # 7. メタデータをロード
        metadata = cls._load_metadata(checkpoint_path, backend)

        return context, metadata

    @classmethod
    def _infer_backend_from_path(cls, path: Optional[str]) -> str:
        """パス形式からバックエンドを推測する。"""
        if path is None:
            return "local"
        if path.startswith("redis://"):
            return "redis"
        if path.startswith("s3://"):
            return "s3"
        return "local"

    @classmethod
    def _get_pending_tasks(cls, context: ExecutionContext) -> list[str]:
        """キューから pending タスクを抽出する。"""
        if hasattr(context.task_queue, 'get_pending_tasks'):
            return context.task_queue.get_pending_tasks()
        return []
```

### 状態永続化戦略

**MemoryChannel**:

```python
# チェックポイント作成時
__getstate__():
    - channel.keys() を走査
    - state['_channel_data'] にキー・値を保存
    - __workflow_state__ は不要（別途管理）

# 復元時
__setstate__():
    - 新しい MemoryChannel を生成
    - state['_channel_data'] をすべて復元
```

**RedisChannel**:

```python
# チェックポイント作成時
__getstate__():
    - session_id を保存（データは Redis に常駐）
    - チャネルデータは保存不要
    - __workflow_state__ は不要（別途管理）

# 復元時
__setstate__():
    - session_id を使って Redis に再接続
    - 同じチャネル名でデータを取得
```

### チェックポイント状態スキーマ

ExecutionContext の pickle とは別ファイルに保存:

```python
{
    "session_id": "12345",                           # セッション ID
    "start_node": "my_task",                        # レジューム時の開始タスク
    "steps": 42,                                     # 実行済みステップ数
    "completed_tasks": ["task1", "task2"],         # 完了タスク ID
    "cycle_counts": {
        "task1": 3,
        "task2": 1
    },
    "pending_tasks": ["task4", "task5"],           # キュー上のタスク
    "backend": {
        "queue": "memory",
        "channel": "memory"
    }
}
```

### チェックポイントメタデータスキーマ

`ExecutionContext.checkpoint_metadata` に保存:

```python
{
    "created_at": 1234567890.123,      # Unix タイムスタンプ
    "session_id": "12345",             # ワークフローセッション ID
    "start_node": "my_task",           # 開始タスク
    "steps": 42,                        # チェックポイント時点のステップ数
    "completed_tasks": 3,               # 完了タスク数
    "backend": {
        "queue": "memory",             # キューバックエンド
        "channel": "memory"            # チャネルバックエンド
    },
    # ユーザー定義メタデータ
    "stage": "processing",
    "task_id": "ml_training",
    "cycle_count": 5,
    "epoch": 10
}
```

### ファイル命名規則

自動生成されるチェックポイントパス:

```
checkpoints/session_{session_id}_step_{steps}_{timestamp}.pkl
```

例:

```
checkpoints/session_12345_step_42_1706024400.pkl
```

---

## バックエンド対応

### 比較表

| 機能 | MemoryChannel | RedisChannel |
|------|---------------|--------------|
| **状態永続化** | pickle ファイルへ保存 | Redis に永続化済み |
| **チェックポイントサイズ** | 大きい（チャネルデータを含む） | 小さい（session_id のみ） |
| **可搬性** | ファイルに依存 | Redis 永続化に依存 |
| **マルチワーカー再開** | ❌ 非対応 | ✅ 対応 |
| **耐久性** | ファイルシステムに依存 | Redis 設定に依存 |

### キュー状態の永続化戦略

バックエンドごとに pending_tasks の扱いが異なります。

| キューバックエンド | 永続化戦略 | チェックポイントでの扱い |
|------------------|--------------|---------------------------|
| **InMemoryTaskQueue** | 未永続化 | `pending_tasks` を保存し、レジューム時に再キューイング |
| **RedisTaskQueue** | Redis に永続化 | `session_id` だけ保存し、再接続で自動復元 |

**InMemoryTaskQueue**
- キュー状態はメモリ上のみでプロセス終了時に消える
- `get_pending_tasks()` が現在の待機タスク ID を返す
- `{path}.state.json` の `pending_tasks` に保存
- レジューム時に `context.add_to_queue()` で再キューイング
- チェックポイント／レジュームでも未処理タスクを失わない

**RedisTaskQueue**
- キュー状態は Redis にセッション単位で保持（例: `session_12345:queue`）
- `get_pending_tasks()` は検証用に現在の状態を返せる
- pending_tasks を保存せずとも同じセッション ID で復元可能
- レジューム時も再キューイング不要、ワーカーは即座にタスク取得可能

**実装ノート**: `CheckpointManager._get_pending_tasks()` は両ケースを考慮:

```python
def _get_pending_tasks(self, context: ExecutionContext) -> list[str]:
    """バックエンド別に pending タスクを取得。"""
    if context._queue_backend_type == "redis":
        return context.task_queue.get_pending_tasks() if hasattr(context.task_queue, 'get_pending_tasks') else []
    else:
        return context.task_queue.get_pending_tasks()
```

### MemoryChannel ワークフロー

```
1. 実行:
   task1 → channel.set("state", "A")
   task2 → channel.set("state", "B")

2. チェックポイント:
   CheckpointManager.create_checkpoint() が以下のファイルを生成

   ファイル1: {checkpoint_path}.pkl（ExecutionContext の pickle）
   - _channel_data = {"state": "B", "order_data": {...}}
   - session_id, graph, backend 設定

   ファイル2: {checkpoint_path}.state.json（チェックポイント状態）
   - session_id, start_node, steps
   - completed_tasks, cycle_counts
   - pending_tasks（キュー状態）

   ファイル3: {checkpoint_path}.meta.json（メタデータ）
   - checkpoint_id, created_at
   - user_metadata

3. レジューム:
   CheckpointManager.resume_from_checkpoint() →
   - ExecutionContext を pickle から復元
   - MemoryChannel に _channel_data を再投入
   - pending_tasks をキューへ復元
   - メタデータを読み込み
   - 処理を継続
```

### RedisChannel ワークフロー

```
1. 実行（ワーカー1）:
   task1 → redis.set("session_12345:state", "A")
   task2 → redis.set("session_12345:state", "B")

2. チェックポイント:
   CheckpointManager.create_checkpoint() が以下のファイルを生成

   ファイル1: {checkpoint_path}.pkl（ExecutionContext の pickle）
   - session_id = "12345"（チャネルデータは保存しない）
   - graph, backend 設定

   ファイル2: {checkpoint_path}.state.json（チェックポイント状態）
   - session_id, start_node, steps
   - completed_tasks, cycle_counts
   - pending_tasks（検証用）

   ファイル3: {checkpoint_path}.meta.json（メタデータ）
   - checkpoint_id, created_at
   - user_metadata

3. レジューム（ワーカー2）:
   CheckpointManager.resume_from_checkpoint() →
   - ExecutionContext を pickle から復元
   - session_id="12345" で Redis に再接続
   - redis.get("session_12345:state") → "B"（データは Redis に常駐）
   - pending_tasks をロード（再キュー不要）
   - メタデータを読み込み
   - 処理を継続
```

### 保存情報サマリー

**ExecutionContext Pickle**（`{path}.pkl`）:
- `session_id`: セッション識別子
- `graph`: TaskGraph 構造（ノード・エッジ）
- `start_node`: 開始タスク
- `cycle_controller`: サイクル管理状態
- `_queue_backend_type`: キューバックエンド種別
- `_channel_backend_type`: チャネルバックエンド種別
- `_original_config`: バックエンド設定
- `_channel_data`: チャネルデータ（MemoryChannel のみ）

**Checkpoint State**（`{path}.state.json`）:
- `session_id`: セッション ID
- `start_node`: レジューム開始タスク
- `steps`: 実行済みステップ数
- `completed_tasks`: 完了タスク ID
- `cycle_counts`: タスクごとのサイクル数
- `pending_tasks`: キュー上のタスク
- `backend`: キュー／チャネルバックエンド情報

**Metadata**（`{path}.meta.json`）:
- `checkpoint_id`: チェックポイント識別子
- `session_id`: セッション ID
- `created_at`: タイムスタンプ
- `steps`: チェックポイント時点のステップ数
- `start_node`: 開始タスク
- `backend`: バックエンド設定
- `user_metadata`: ユーザー定義メタデータ

**追加情報（必要に応じて）**:
- タスク固有状態: ユーザーがチャネルに保存
- 結果: `{task_id}.__result__` キーでチャネルに保存
- カスタムワークフローデータ: ユーザーがチャネルに保存

---

## 利用例

### 例1: 状態マシンワークフローでのチェックポイント

```python
from graflow.core.decorators import task
from graflow.core.workflow import workflow
from graflow.checkpoint import CheckpointManager

with workflow("order_processing") as ctx:

    @task(inject_context=True)
    def process_order(context):
        channel = context.get_channel()
        state = channel.get("order_state", default="NEW")
        order_data = channel.get("order_data")

        if state == "NEW":
            validate_order(order_data)
            channel.set("order_state", "VALIDATED")

            # バリデーション後にチェックポイント
            checkpoint_path, metadata = context.checkpoint(
                metadata={
                    "stage": "validation_complete",
                    "order_id": order_data["id"]
                }
            )
            print(f"Checkpoint saved: {checkpoint_path}")

            context.next_iteration()

        elif state == "VALIDATED":
            process_payment(order_data)
            channel.set("order_state", "PAID")

            # 支払い後にチェックポイント
            checkpoint_path, metadata = context.checkpoint(
                metadata={
                    "stage": "payment_complete",
                    "amount": order_data["amount"]
                }
            )
            print(f"Checkpoint saved: {checkpoint_path}")

            context.next_iteration()

        elif state == "PAID":
            ship_order(order_data)
            return "ORDER_COMPLETE"

    # 初回実行
    channel = ctx.execution_context.get_channel()
    channel.set("order_data", {"id": "ORD123", "amount": 100})

    try:
        ctx.execute("process_order", max_steps=10)
    except Exception as e:
        print(f"Error: {e}")
        # 最後に成功したステージでチェックポイント済み

# 障害後に再開
from graflow.core.engine import WorkflowEngine

context, metadata = CheckpointManager.resume_from_checkpoint(
    checkpoint_path="checkpoints/session_12345_step_5.pkl"
)
print(f"Resuming from step {metadata.steps}, stage: {metadata.user_metadata.get('stage')}")

engine = WorkflowEngine()
engine.execute(context)  # VALIDATED 状態から続行
```

### 例2: 周期的にチェックポイントを取る ML 学習

```python
from graflow.checkpoint import CheckpointManager

@task(inject_context=True)
def ml_training(context):
    channel = context.get_channel()
    epoch = channel.get("epoch", default=0)
    model = channel.get("model")

    if epoch == 0:
        model = initialize_model()
        channel.set("model", model)

    # 学習ステップ
    metrics = train_epoch(model, epoch)
    channel.set("epoch", epoch + 1)
    channel.set("metrics", metrics)

    # 10 エポックごとにチェックポイント
    if (epoch + 1) % 10 == 0:
        checkpoint_path, checkpoint_metadata = context.checkpoint(
            metadata={
                "epoch": epoch + 1,
                "loss": metrics["loss"],
                "accuracy": metrics["accuracy"]
            }
        )
        print(f"Checkpoint saved: {checkpoint_path}")

    # 収束判定
    if metrics["accuracy"] >= 0.95:
        save_model(model)
        return "TRAINING_COMPLETE"
    else:
        context.next_iteration()

# 学習の起動
with workflow("ml_training") as ctx:
    ctx.execute("ml_training", max_steps=100)

# 中断後の再開
from graflow.core.engine import WorkflowEngine

context, metadata = CheckpointManager.resume_from_checkpoint(
    checkpoint_path="checkpoints/session_67890_step_30.pkl"
)
print(f"Resuming from epoch {metadata.user_metadata.get('epoch')}")

engine = WorkflowEngine()
engine.execute(context)  # エポック30から再開
```

### 例3: Redis を使った分散実行

```python
# ワーカー1: 実行開始
import redis
from graflow.checkpoint import CheckpointManager

redis_client = redis.Redis(host='localhost', port=6379, db=0)

with workflow("distributed_workflow") as ctx:
    @task(inject_context=True)
    def distributed_task(context):
        # データ処理
        process_data()

        # チェックポイント（パスは自動生成）
        checkpoint_path, metadata = context.checkpoint(
            metadata={"worker": "worker-1"}
        )
        print(f"Checkpoint saved: {checkpoint_path}")
        print(f"Session ID: {context.session_id}")

        # 続きの処理
        more_processing()

    exec_context = ExecutionContext.create(
        ctx.graph,
        start_node="distributed_task",
        queue_backend="redis",
        channel_backend="redis",
        config={"redis_client": redis_client}
    )

    try:
        exec_context.execute()
    except Exception as e:
        print(f"Worker 1 crashed: {e}")
        # チェックポイント済みなので他ワーカーで再開可能

# ワーカー2: チェックポイントから再開
from graflow.core.engine import WorkflowEngine

context, metadata = CheckpointManager.resume_from_checkpoint(
    checkpoint_path="checkpoints/session_12345_step_10.pkl"
)
# context.session_id == "12345" → 同じ Redis チャネルに接続
print(f"Worker 2 resuming session {metadata.session_id} from step {metadata.steps}")

engine = WorkflowEngine()
engine.execute(context)  # Worker 1 の続きを実行
```

---

## エッジケースと制約

### エッジケース

1. **並列実行中のチェックポイント**:
   - 呼び出し元タスクのコンテキストのみ保存される
   - 並列実行中の兄弟タスクは継続中の可能性がある
   - レジューム後は完了済みの兄弟タスクを再実行しない

2. **複数のチェックポイント**:
   - 各チェックポイントは独立
   - 古いチェックポイントの自動クリーンアップなし
   - ファイル管理はユーザーに委ねる

3. **グラフ構造の変更**:
   - チェックポイント後にワークフローコードを変更すると結果が未定義
   - チェックポイントに保存されたグラフ構造と互換性を保つ必要あり

4. **キュー状態**（解決済み）:
   - pending_tasks を保存済み
   - レジューム時にキューへ復元
   - 未完了タスクから実行を継続可能

### 制約

**L1**: ~~タスクキュー状態が永続化されない~~（解決済み: pending_tasks を保存）

**L2**: チェックポイントの自動クリーンアップがない
- **影響**: ファイルが増え続ける
- **回避策**: ユーザーが手動で削除

**L3**: チェックポイントにバージョン管理がない
- **影響**: 比較やマージができない
- **回避策**: メタデータで関係性を管理

**L4**: MemoryChannel のチェックポイントは他ワーカーへ移行できない
- **影響**: 別ワーカーで再開不可
- **回避策**: 分散実行には RedisChannel を使用

**L5**: グラフ構造は互換性を維持する必要がある
- **影響**: コード変更後の再開で失敗する可能性
- **回避策**: ワークフローコードの後方互換性を保つ

---

## 今後の拡張

### フェーズ 2: 自動チェックポイント管理

```python
class ExecutionContext:
    def __init__(self, ..., auto_checkpoint_interval: Optional[int] = None):
        """
        Args:
            auto_checkpoint_interval: N ステップごとに自動チェックポイント
        """

    # 実行ループ内:
    if self.auto_checkpoint_interval and self.steps % self.auto_checkpoint_interval == 0:
        self.checkpoint(metadata={"auto": True})
```

### フェーズ 3: Checkpoint Manager 拡張

```python
class CheckpointManager:
    """保持ポリシーを含むチェックポイント管理。"""

    def list_checkpoints(self, session_id: Optional[str] = None) -> list[CheckpointMetadata]:
        """チェックポイント一覧を取得。"""

    def get_latest(self, session_id: str) -> str:
        """特定セッションの最新チェックポイントを取得。"""

    def cleanup_old_checkpoints(self, keep_last_n: int = 5):
        """最新 N 件だけ残してそれ以外を削除。"""
```

### フェーズ 4: 増分チェックポイント

- 前回からの差分のみを保存するデルタチェックポイント
- 大規模ワークフローでファイルサイズを削減
- チェックポイントチェーン管理が必要

### フェーズ 5: クラウドストレージ統合

```python
class ExecutionContext:
    def checkpoint(self, storage: str = "local"):
        """
        Args:
            storage: "local", "s3", "gcs", "azure"
        """
```

### フェーズ 6: チェックポイント検証

- レジューム前に整合性を検証
- グラフ互換性のチェック
- チャネルデータ一貫性の検証

---

## 実装チェックリスト

### フェーズ 1: コア機能

- [x] `ExecutionContext.__init__` に `completed_tasks` と `checkpoint_metadata` を追加
- [x] `ExecutionContext.mark_task_completed()` を実装
- [ ] `graflow/checkpoint.py` モジュールを作成
- [ ] `CheckpointMetadata` データクラスを実装
- [ ] `CheckpointManager` クラスを実装
  - [ ] `__init__(storage_backend)`
  - [ ] `create_checkpoint(context, path, metadata)`
  - [ ] `resume_from_checkpoint(checkpoint_path, ...)`
  - [ ] `_get_pending_tasks(context)`
  - [ ] `_save_checkpoint_state(path, state)`
  - [ ] `_load_checkpoint_state(path)`
  - [ ] `_save_metadata(path, metadata)`
  - [ ] `_load_metadata(path)`
- [ ] `ExecutionContext.get_checkpoint_state()` を実装
- [ ] `TaskExecutionContext.checkpoint()` を実装
- [ ] TaskQueue インターフェースに `get_pending_tasks()` を追加
  - [ ] MemoryTaskQueue に実装
  - [ ] RedisTaskQueue に実装
- [ ] `WorkflowEngine.execute()` で `mark_task_completed()` を呼び出すよう変更
- [ ] CheckpointManager のユニットテストを作成
- [ ] MemoryChannel のチェックポイント／レジュームテストを作成
- [ ] RedisChannel のチェックポイント／レジュームテストを作成
- [ ] ステートマシンワークフローの統合テストを作成
- [ ] ドキュメントとサンプルを更新

### フェーズ 2: テストと検証

- [ ] ネストしたワークフローでのチェックポイント／レジュームをテスト
- [ ] ParallelGroup でのチェックポイント／レジュームをテスト
- [ ] 動的タスクを用いたチェックポイント／レジュームをテスト
- [ ] エラー発生時のチェックポイント／レジュームをテスト
- [ ] パフォーマンスベンチマークを実施
- [ ] 大規模ワークフローでの負荷テスト

### フェーズ 3: ドキュメント

- [ ] API リファレンスドキュメント
- [ ] 使い方ガイドとサンプル
- [ ] ベストプラクティスガイド
- [ ] 既存ワークフロー向け移行ガイド

---

## 結論

このチェックポイント／レジューム設計は、ワークフロー状態の永続化をシンプルかつバックエンド非依存に実現します。既存のシリアライゼーション基盤とチャネルベースの状態管理を活用し、ステートマシン実行モデルと自然に統合しながら、最小限の複雑さとオーバーヘッドで実行できます。

設計はユーザー制御（明示的なチェックポイント呼び出し）、可搬性（Memory と Redis の両対応）、シンプルさ（`__getstate__`/`__setstate__` の活用）を重視しており、ローカル開発からプロダクションの分散実行まで幅広いシナリオで有用です。

---

**ドキュメントステータス**: Draft
**次回レビュー**: 実装フェーズ
**レビュワー**: Graflow チーム
