# チェックポイント/レジューム機能の例

このディレクトリには、Graflowのチェックポイント/レジューム機能を使用したワークフローの状態保存と復元を実演する例が含まれています。

## 概要

チェックポイント機能により以下が可能になります：
- 実行中の特定時点でワークフローの状態を保存
- 中断や障害発生後、保存したチェックポイントから実行を再開
- 長時間実行されるワークフローのフォールトトレランスを実装
- 反復的なチェックポイントを使ったステートマシンワークフローのサポート

## 主要な概念

### チェックポイントの構造
各チェックポイントは3つのファイルで構成されます：
- `.pkl` - シリアライズされたExecutionContext（グラフ、チャネルデータ）
- `.state.json` - 実行状態（ステップ数、完了済みタスク、保留中のタスク、サイクルカウント）
- `.meta.json` - メタデータ（チェックポイントID、タイムスタンプ、ユーザー定義メタデータ）

### チェックポイントの作成
チェックポイントは2段階で作成されます：
1. **リクエスト**: タスクが `task_ctx.checkpoint(metadata={...})` を呼び出してフラグをセット
2. **作成**: タスクが正常に完了した**後**にエンジンがチェックポイントを作成

この遅延実行により、チェックポイント内でタスクが完了済みとしてマークされることが保証されます。

### レジューム（再開）
`CheckpointManager.resume_from_checkpoint(path)` を使用してワークフローの状態を復元し、実行を継続します。

## サンプル一覧

### 1. 基本的なチェックポイント/レジューム (`01_basic_checkpoint.py`)
チェックポイント/レジュームワークフローの基礎を示す**最もシンプルな例**。
- タスク完了後にチェックポイントを作成
- チェックポイントから再開
- 保存された状態から実行を継続

**使用場面**: チェックポイントの基礎を学ぶ、シンプルなワークフロー

```bash
uv run python examples/13_checkpoints/01_basic_checkpoint.py
```

### 2. チェックポイント付きステートマシン (`02_state_machine_checkpoint.py`)
各状態遷移でチェックポイントを作成する状態ベースワークフローの**本番環境向けパターン**。
- 注文処理: NEW → VALIDATED → PAID → SHIPPED
- 各状態遷移後にチェックポイント
- 任意の状態から再開

**使用場面**: 注文処理、承認ワークフロー、多段階パイプライン

```bash
uv run python examples/13_checkpoints/02_state_machine_checkpoint.py
```

### 3. 定期的なチェックポイント (`03_periodic_checkpoint.py`)
定期的なチェックポイントを使った**長時間実行ワークフロー**。
- 反復的なチェックポイントを使ったML学習のシミュレーション
- N回の反復ごとにチェックポイント
- 最新のチェックポイントから再開

**使用場面**: ML学習、バッチ処理、データパイプライン

```bash
uv run python examples/13_checkpoints/03_periodic_checkpoint.py
```

### 4. 障害復旧 (`04_fault_recovery.py`)
模擬的な障害を使った**フォールトトレランス**のデモンストレーション。
- 障害発生の可能性があるワークフローを実行
- 高コストな処理の前にチェックポイントを作成
- 模擬障害後に再開

**使用場面**: 不安定なインフラ、高コストな処理、本番環境のワークフロー

```bash
uv run python examples/13_checkpoints/04_fault_recovery.py
```

## べき等性の重要性（ユーザー責任）

### チェックポイント/レジュームにおけるタスク実行の仕組み

**重要**: チェックポイントから再開すると、**タスクは常に最初から再実行されます**。

これは設計上の仕様であり、以下の理由によります：
- タスクの途中状態を保存することは複雑でエラーの原因となる
- タスク関数のローカル変数やスタック状態を完全に復元することは困難
- 一貫性のある状態（タスク完了前 or 完了後）のみを保存することでシンプルに保つ

### ユーザーの責任：べき等なタスク設計

**タスクをべき等に設計することはユーザーの責任です。**

べき等性とは：同じタスクを何度実行しても、1回実行した場合と同じ結果になる性質。

#### べき等でない場合の問題例

```python
# ❌ 悪い例：べき等でない
from graflow.core.context import TaskExecutionContext
from graflow.core.decorators import task

@task(inject_context=True)
def process_orders(task_ctx: TaskExecutionContext):
    # チェックポイントから再開すると、同じ注文を二重処理してしまう
    orders = fetch_new_orders()
    for order in orders:
        charge_customer(order)  # 再実行で二重課金！
        ship_product(order)     # 再実行で二重出荷！
    task_ctx.checkpoint()
```

このタスクが途中でクラッシュしてチェックポイントから再開すると：
- すでに課金した顧客に再度課金
- すでに出荷した商品を再度出荷
- データの不整合が発生

#### べき等性を実現する方法

### 1. チャネルベースの状態管理（推奨）

```python
# ✅ 良い例：チャネルで状態を管理
from graflow.core.context import TaskExecutionContext
from graflow.core.decorators import task

@task(inject_context=True)
def process_orders(task_ctx: TaskExecutionContext):
    channel = task_ctx.get_channel()
    processed_order_ids = channel.get("processed_order_ids", set())

    orders = fetch_new_orders()
    for order in orders:
        # すでに処理済みならスキップ
        if order.id in processed_order_ids:
            continue

        charge_customer(order)
        ship_product(order)

        # 処理済みとしてマーク
        processed_order_ids.add(order.id)
        channel.set("processed_order_ids", processed_order_ids)

    task_ctx.checkpoint()
```

**ポイント**:
- チャネルに処理済みの情報を保存
- 再実行時、チャネルから状態を読み取り、すでに処理済みの作業をスキップ
- チャネルデータはチェックポイントに含まれるため、再開後も保持される

### 2. ステートマシンパターン

```python
# ✅ 良い例：状態ベースの実行制御
from graflow.core.context import TaskExecutionContext
from graflow.core.decorators import task

@task(inject_context=True)
def multi_stage_task(task_ctx: TaskExecutionContext):
    channel = task_ctx.get_channel()
    state = channel.get("state", "INIT")

    if state == "INIT":
        initialize_resources()
        channel.set("state", "PROCESSING")
        task_ctx.checkpoint()
        task_ctx.next_iteration()
    elif state == "PROCESSING":
        process_data()
        channel.set("state", "FINALIZING")
        task_ctx.checkpoint()
        task_ctx.next_iteration()
    elif state == "FINALIZING":
        finalize()
        return "COMPLETE"
```

**ポイント**:
- 各ステージの開始時に状態をチェック
- すでに完了したステージはスキップ
- 未完了のステージから再開

### 3. 外部システムのべき等性API利用

```python
# ✅ 良い例：べき等なAPI呼び出し
import uuid
from graflow.core.context import TaskExecutionContext
from graflow.core.decorators import task

@task(inject_context=True)
def call_external_api(task_ctx: TaskExecutionContext):
    channel = task_ctx.get_channel()

    # べき等キーを使用（同じキーでの再実行は冪等）
    idempotency_key = channel.get("idempotency_key")
    if not idempotency_key:
        idempotency_key = str(uuid.uuid4())
        channel.set("idempotency_key", idempotency_key)

    # 多くの決済APIなどはidempotency keyをサポート
    result = payment_api.charge(
        amount=100,
        idempotency_key=idempotency_key  # 同じキーなら重複実行されない
    )

    task_ctx.checkpoint()
    return result
```

### べき等性チェックリスト

タスクを実装する際、以下を確認してください：

- [ ] タスクが再実行されても安全か？
- [ ] 外部リソース（DB、API、ファイル）への書き込みは重複しないか？
- [ ] チャネルで処理済み状態を追跡しているか？
- [ ] 外部システムの冪等性機能（idempotency key等）を利用しているか？
- [ ] 金銭的な処理（課金、決済等）が含まれる場合、二重実行を防いでいるか？

### まとめ

**Graflowの責任**:
- チェックポイントの作成と復元
- タスクグラフとチャネルデータの保存
- チェックポイントからの正確な再開

**ユーザーの責任**:
- **タスクをべき等に設計すること**
- チャネルを使った状態管理の実装
- 外部システムとの安全な統合

チェックポイント機能を正しく活用するには、べき等なタスク設計が不可欠です。

## ベストプラクティス

### 1. 高コストな処理の後にチェックポイント
```python
@task(inject_context=True)
def expensive_task(task_ctx):
    expensive_computation()
    task_ctx.checkpoint(metadata={"stage": "computation_complete"})
    # ここでワークフローがクラッシュしても、計算結果は保存済み
```

### 2. 意味のあるメタデータを含める
```python
task_ctx.checkpoint(metadata={
    "stage": "validation_complete",
    "records_processed": 10000,
    "timestamp": time.time()
})
```

### 3. 自動生成パス vs カスタムパス
```python
# 自動生成（ほとんどの場合に推奨）
task_ctx.checkpoint()  # パス: checkpoints/{session_id}/...

# カスタムパス（特定の要件がある場合）
task_ctx.checkpoint(path="/tmp/my_checkpoint")
```

## アーキテクチャに関する注意事項

### 遅延チェックポイント実行
タスク実行中に `task_ctx.checkpoint()` が呼ばれると：
1. `context.checkpoint_requested = True` をセット
2. ユーザーメタデータを `context.checkpoint_request_metadata` に保存
3. タスクが完了まで継続
4. **タスク完了後にエンジンがチェックポイントを作成**
5. タスクはチェックポイント状態で完了済みとしてマーク

これにより以下が保証されます：
- タスクは完了済みまたは保留中のいずれか（「実行中」状態はなし）
- チェックポイントは一貫した状態を表す
- 再開時は次の保留中タスクから継続

### 保存される内容

**ExecutionContext (.pkl)**:
- タスクグラフ構造
- チャネルデータ（MemoryChannel）またはセッションID（RedisChannel）
- バックエンド設定

**Checkpoint State (.state.json)**:
- 完了済みタスクID
- 保留中のTaskSpec（タスクIDだけでなく完全なタスク仕様）
- サイクルカウント
- 実行済みステップ数
- スキーマバージョン

**Metadata (.meta.json)**:
- チェックポイントID、セッションID
- 作成タイムスタンプ（ISO 8601形式）
- ユーザー定義メタデータ

### 結果の保存
タスクの結果は**チャネル**に `{task_id}.__result__` というキーで保存されます：
- MemoryChannel: `.pkl` ファイルに保存
- RedisChannel: Redisに自動的に永続化

## よくあるパターン

### パターン1: 多段階パイプライン
```python
stage1 >> checkpoint >> stage2 >> checkpoint >> stage3
```
復旧のため、各主要ステージ後にチェックポイント。

### パターン2: 承認ワークフロー
```python
process >> request_approval >> [timeout時にチェックポイント] >> deploy
```
人間の承認待ち時にチェックポイント。

### パターン3: 反復的な学習
```python
for epoch in epochs:
    train()
    if epoch % 10 == 0:
        checkpoint()
```
長時間実行される学習中に定期的なチェックポイント。

## トラブルシューティング

### チェックポイントファイルが作成されない
- タスクが正常に完了することを確認（チェックポイントは完了**後**に作成される）
- チェックポイントディレクトリの書き込み権限を確認
- タスク実行中に例外が発生していないか確認

### 再開が失敗する
- 3つのチェックポイントファイルすべてが存在するか確認（`.pkl`, `.state.json`, `.meta.json`）
- ファイルが破損していないか確認
- 同じPython環境とGraflowバージョンを使用しているか確認

### 再開時にタスクが再実行される
- これは期待される動作：タスクは再開時に最初から再実行される
- すでに完了した作業をスキップするためにチャネルベースの状態を使用
- ステートマシンパターンの例を参照

## 関連ドキュメント

- 設計: `docs/checkpoint/checkpoint_resume_design.md`
- 実装: `docs/checkpoint/checkpoint_implementation_summary.md`
- ユニットテスト: `tests/core/test_checkpoint.py`
- シナリオテスト: `tests/scenario/test_checkpoint_scenarios.py`
