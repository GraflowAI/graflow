# 並列実行修正: session_id によるキュー分離

> **ステータス**: 設計ドキュメント（キュー分離戦略）  
> **アプローチ**: 既存の session_id メカニズムを利用し、グラフ構造を共有しつつ各並列ブランチに独立したタスクキューとチャネルを割り当てる  
> **問題**: ダイヤモンド型のワークフローで終端タスクが 1 回ではなく 4 回実行されてしまう  
> **テストケース**: `tests/scenario/test_parallel_diamond.py`

---

## エグゼクティブサマリー

本設計では、各並列ブランチに独立したタスクキューとチャネルインスタンスを割り当てる **キュー名前空間分離** を導入する。これにより、グラフは共有しながらもブランチごとにキューとチャネルを隔離でき、次のメリットが得られる。

- ✅ **シンプル**: 複雑な依存カウンタやマージ処理が不要
- ✅ **正確**: キュー競合を自然に回避
- ✅ **高性能**: キュー作成コストは小さい
- ✅ **保守性**: 責務境界が明確

---

## 1. 基本コンセプト: キュー名前空間の分離

### 1.1 現状の問題

```
┌────────────────────────────────────────┐
│ ExecutionContext                       │
│  ├─ graph (shared)                     │
│  ├─ channel (shared)                   │
│  └─ task_queue (SHARED - PROBLEMATIC!) │◄── 両スレッドが同じキューを消費
└────────────────────────────────────────┘
         ↑                    ↑
    Thread 1            Thread 2
  (transform_a)      (transform_b)
```

**競合**: 2 つのスレッドが `store` を共有キューに追加し、双方が while ループを継続して `store` を複数回実行してしまう。

### 1.2 提案: session_id によるキュー分離

```
┌────────────────────────────────────────┐
│ ExecutionContext (Main)                │
│  ├─ session_id = "123456"              │
│  ├─ graph (shared ✓)                   │
│  ├─ channel (per-branch ✗)             │  ← 各ブランチでコピー＋マージ
│  └─ task_queue (main queue)            │
└────────────────────────────────────────┘
         │                    │
         ├─ Thread 1          ├─ Thread 2
         │  session_id        │  session_id
         │  = "123456_br1"    │  = "123456_br2"
         ↓                    ↓
    ┌──────────┐         ┌──────────┐
    │ Queue_1  │         │ Queue_2  │  ← キューは分離
    │ Channel1 │         │ Channel2 │  ← チャネルも独立
    └──────────┘         └──────────┘     （Redis: session_id ベースでキー分離）
```

**ポイント**: 各ブランチが固有の `session_id` を持つことで、キュー（および Redis 利用時のチャネルキー）を自動的に分離できる。

- **InMemory**: 新しい ExecutionContext → 新しい TaskQueue インスタンス → 自然に分離
- **Redis**: `session_id` をキー接頭辞に利用 → `graflow:queue:{session_id}` などで隔離

後続タスクは常に **メインキュー** に登録され、全ブランチ完了後にメインスレッドがファンイン処理を行う。

---

## 2. 実装デザイン

### 2.1 既存の session_id メカニズムを活用

Redis の実装ではすでに session_id を用いてキューを分離している。この仕組みをブランチにも利用する。

```python
# graflow/queue/redis.py （既存コード、変更不要）
class RedisTaskQueue(TaskQueue):
    def __init__(self, execution_context, ...):
        self.session_id = execution_context.session_id
        self.queue_key = f"{key_prefix}:queue:{self.session_id}"
        self.specs_key = f"{key_prefix}:specs:{self.session_id}"
```

- **InMemoryTaskQueue**: ExecutionContext ごとに TaskQueue インスタンスが生成されるため自然に分離される。

### 2.2 session_id を持つブランチコンテキスト

```python
# graflow/core/context.py

class ExecutionContext:
    def __init__(
        self,
        graph: TaskGraph,
        start_node: Optional[str] = None,
        parent_context: Optional['ExecutionContext'] = None,  # 追加
        session_id: Optional[str] = None,                     # 追加
        **kwargs
    ):
        if session_id is None:
            session_id = str(uuid.uuid4().int)
        self.session_id = session_id
        self.graph = graph
        self.start_node = start_node
        self.parent_context = parent_context

        # キューは session_id で自動分離（Redis/InMemory 共通）
        self.task_queue = TaskQueueFactory.create(queue_backend, self, **config)

        if parent_context:
            self.channel = ChannelFactory.create_channel(
                backend=channel_backend,
                name=session_id,
                **config,
            )
            # 親のチャネル内容をコピー
            self.channel.merge_from(parent_context.channel)
            # タスクレジストリは共有
            self._task_resolver = parent_context._task_resolver
        else:
            self.channel = ChannelFactory.create_channel(
                backend=channel_backend,
                name=session_id,
                **config
            )
            self._task_resolver = TaskResolver()
```

---

## 3. ブランチコンテキストとファンイン

### 3.1 ブランチコンテキストの API

```python
class ExecutionContext:
    def create_branch_context(self, branch_id: str) -> 'ExecutionContext':
        branch_session_id = f"{self.session_id}_{branch_id}"
        return ExecutionContext(
            graph=self.graph,
            start_node=None,
            parent_context=self,
            session_id=branch_session_id,
            queue_backend=self._queue_backend_type,
            channel_backend=self._channel_backend_type,
            config=self._original_config,
            max_steps=self.max_steps,
            default_max_cycles=self.cycle_controller.default_max_cycles,
            default_max_retries=self.default_max_retries,
        )

    def merge_results(self, sub_context: 'ExecutionContext') -> None:
        if self.channel is not sub_context.channel:
            for key in sub_context.channel.keys():
                self.channel.set(key, sub_context.channel.get(key))
        self.steps += sub_context.steps
        self.cycle_controller.cycle_counts.update(sub_context.cycle_controller.cycle_counts)

    def mark_branch_completed(self, branch_id: str) -> None:
        # 将来的なバリア同期用のフック（現状は no-op）
        return
```

- Channel は branch 作成時にコピーされ、ブランチ終了後にマージされる。
- `TaskFunctionManager` は共有（関数再登録を回避）。
- `group_executor` も親から引き継ぐ。

### 3.2 ThreadingCoordinator の更新

```python
# graflow/coordination/threading.py

class ThreadingCoordinator(TaskCoordinator):
    def execute_group(self, group_id, tasks, execution_context):
        self._ensure_executor()

        def execute_task_with_engine(task, branch_context):
            from graflow.core.engine import WorkflowEngine
            engine = WorkflowEngine()
            engine.execute(branch_context, start_task_id=task.task_id)
            return task.task_id, True, "Success"

        futures = []
        future_context_map = {}
        for task in tasks:
            branch_context = execution_context.create_branch_context(task.task_id)
            future = self._executor.submit(execute_task_with_engine, task, branch_context)
            futures.append(future)
            future_context_map[future] = branch_context

        for future in concurrent.futures.as_completed(futures):
            branch_context = future_context_map[future]
            task_id, success, message = future.result()
            if success:
                execution_context.merge_results(branch_context)
                execution_context.mark_branch_completed(task_id)
            else:
                # TODO: ブランチ失敗時の後続タスク抑止（§5.3）
                ...
```

- 各タスクは独立したブランチコンテキストで実行。
- 成功時に親へ結果をマージし、完了フックを呼び出す。
- 失敗時は今後 `_enqueue_successors_after_parallel_group()` 相当のガードを導入予定。

---

## 4. 動作フロー

### 4.1 ダイヤモンドパターンの実行

```
fetch -> (transform_a | transform_b) -> store
```

| 時刻 | メインスレッド | スレッド1 (transform_a) | スレッド2 (transform_b) | メインキュー | Queue_1 | Queue_2 |
|------|----------------|-------------------------|-------------------------|--------------|---------|---------|
| T0   | fetch 実行     | —                       | —                       | `[fetch]`    | `[]`    | `[]`    |
| T1   | fetch 完了 → PG | —                       | —                       | `[]`         | `[]`    | `[]`    |
| T2   | ブランチ作成     | —                       | —                       | `[]`         | `[transform_a]` | `[transform_b]` |
| T3   | スレッド起動     | transform_a 実行        | transform_b 実行        | `[]`         | `[]`    | `[]`    |
| T4   | 待機            | transform_a 継続       | transform_b 継続        | `[]`         | `[]`    | `[]`    |
| T5   | 待機            | transform_a 完了        | transform_b 完了        | `[]`         | `[]`    | `[]`    |
| T6   | スレッド終了     | —                       | —                       | `[]`         | `[]`    | `[]`    |
| T7   | store をメインへ enqueue | —           | —                       | `[store]`    | `[]`    | `[]`    |
| T8   | store 実行      | —                       | —                       | `[]`         | `[]`    | `[]`    |

結果: 実行順序 `["fetch", "transform_a", "transform_b", "store"]` を達成。

### 4.2 現存する課題

- ブランチ失敗時にメインキューが後続タスクを enqueue しない仕組み（§5.3）を追加する必要がある。
- シナリオテスト（Redis / インメモリ混在環境）での最終確認が必要。

---

## 5. 実装ステータス

### Phase 1: Core Branch Context — ✅ 完了

- `graflow/core/context.py` が `parent_context` / `session_id` のサポート、`create_branch_context()`、`merge_results()`、`mark_branch_completed()` を実装。
- `graflow/core/engine.py` は `execute(start_task_id=...)` でブランチ実行をサポート（既存の後続フィルタリングで重複 enqueue を抑止）。
- 実行済テスト: `pytest tests/test_graph.py tests/core/test_parallel_group.py tests/core/test_sequential_task.py tests/utils/test_graph_mermaid.py`
  - 最終的にはシナリオテスト (`tests/scenario/test_parallel_diamond.py`) も推奨。

### Phase 2: ThreadingCoordinator Integration — ⚙️ 進行中

- `graflow/coordination/threading.py` はブランチコンテキストを生成し、 isolated queue でタスクを実行後 merge する実装に更新済み。
- TODO: `_enqueue_successors_after_parallel_group()` に相当する失敗時ガードの実装（§5.3）。
- TODO: シナリオテスト (`tests/scenario/test_parallel_diamond.py`, `tests/scenario/test_successor_handling.py`) の再実行。

### Phase 3: Edge Cases & Polish — ⏳ 未着手

- シリアライズや weak reference（`parent_context`）等の整理が未実装。
- シナリオテストの拡充項目:
  - ネストした並列グループ
  - ブランチ内での動的タスク生成
  - ブランチ失敗時の挙動
- 完了後に `pytest tests/scenario/test_dynamic_tasks.py` やフルテストの実行を推奨。

---

## 6. コード抜粋

上記の `ExecutionContext` / `ThreadingCoordinator` の抜粋を参照（§3）。

---

## 7. テストポリシー

### 7.1 ユニットテスト

- ブランチコンテキストでキューが分離されているか（session_id が異なるオブジェクトに反映されるか）を確認。
- チャネルの shallow copy / merge が期待通りに機能するかを検証。

### 7.2 シナリオテスト（要実施）

- `pytest tests/scenario/test_parallel_diamond.py`
- `pytest tests/scenario/test_successor_handling.py`
- `pytest tests/scenario/test_dynamic_tasks.py`

### 7.3 回帰テスト

- 上記に加えて完全スイート（`pytest tests/`）での最終確認。

---

## 8. リスクとフォローアップ課題

- **部分失敗時の挙動**: 現状、ブランチが失敗してもメインキューが後続タスクを enqueue してしまうため、ガードを実装する必要がある。
- **Redis 連携**: 実運用での namespace 衝突（key prefix の設定ミス）に注意。
- **メトリクス / ロギング**: session_id をログに含め、ブランチ実行をトレースできるようにする。

---

## 9. まとめ

- session_id を用いたキュー名前空間分離により、並列ブランチが互いのキューを奪い合う問題を解消できる。
- 既存の Redis 実装との整合性が高く、最小限の変更で効果が大きい。
- ブランチ失敗時の後続タスク抑止など、Phase 2 の TODO を完遂すればダイヤモンド問題に対する恒久的な解法となる見込み。
