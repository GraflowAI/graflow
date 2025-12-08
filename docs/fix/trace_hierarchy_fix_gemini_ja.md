# 実行時グラフのトレース階層修正計画

## 問題分析

ユーザーから、`graflow/trace/base.py` の `_runtime_graph` がタスクの親子関係（呼び出し階層）を正しくキャプチャできていないという報告がありました。具体的には、`Tracer.on_task_start` における `parent_task_id` が不正確であったり、欠落していたりすることが頻繁にあります。

### 原因

1.  **ローカルスタックへの依存:** `Tracer.on_task_start` は `context.current_task_id` を確認することによってのみ `parent_task_id` を決定しています。
    ```python
    # graflow/trace/base.py
    parent_task_id = None
    if hasattr(context, 'current_task_id') and context.current_task_id:
        parent_task_id = context.current_task_id
    ```
2.  **スタック状態のタイミング:** `context.current_task_id` は `self._task_execution_stack` のトップを反映します。
    `ExecutionContext.executing_task(task)` において、トレーサーフック `on_task_start` は、新しいタスクコンテキストがスタックにプッシュされる *前* に呼び出されます。
    - **ローカルでのネスト実行:** これは正しく動作します。タスクAが実行中（スタック上）で、タスクBを生成した場合、`on_task_start(B)` はスタック上のAを確認します。結果、`Parent = A` となります。
    - **分散/ワーカー実行:** これは失敗します。タスク（例: `ParallelGroup` 内）を処理するワーカーが実行を開始すると、スタックが空の状態で新しい `ExecutionContext` を初期化します。このとき `current_task_id` は `None` です。トレーサーは親を見つけられず、`TaskSpec` が `parent_span_id` を保持しているにもかかわらず、実行時グラフ内で切断されたノードとなってしまいます。

3.  **リンクの欠落:** `Worker` は `tracer.attach_to_trace(..., parent_span_id=...)` を呼び出しますが、基底クラスである `Tracer` は、この `parent_span_id` を `on_task_start` がフォールバックとしてアクセスできる形で保存していません。

## 提案ソリューション

`graflow/trace/base.py` を修正し、`Tracer` が外部/分散コンテキストの親を認識できるようにします。

### 1. `Tracer` クラスの更新
外部親スパンIDを保存するための属性を追加します。

```python
class Tracer(ABC):
    def __init__(self, enable_runtime_graph: bool = True):
        # ... existing init ...
        self._external_parent_span_id: Optional[str] = None
```

### 2. `attach_to_trace` の更新
特定のトレースコンテキストにアタッチする際に `parent_span_id` をキャプチャします。

```python
    def attach_to_trace(
        self,
        trace_id: str,
        parent_span_id: Optional[str] = None
    ) -> None:
        self._current_trace_id = trace_id
        self._external_parent_span_id = parent_span_id  # <--- これを保存
        self._output_attach_to_trace(trace_id, parent_span_id)
```

### 3. `on_task_start` の更新
ローカルスタックが空の場合、フォールバックとして `_external_parent_span_id` を使用します。

```python
    def on_task_start(self, task: Executable, context: ExecutionContext) -> None:
        parent_task_id = None
        
        # 1. ローカル実行スタックを確認（このプロセス内のネストされたタスク用）
        if hasattr(context, 'current_task_id') and context.current_task_id:
            parent_task_id = context.current_task_id
        
        # 2. ローカルの親がない場合、外部の親を確認（分散ワーカー用）
        elif self._external_parent_span_id:
            # この実行のルートタスクに対してのみ外部の親を使用
            # (このワーカー内の後続のネストされたタスクはローカルの親を持つことになる)
            parent_task_id = self._external_parent_span_id

        self.span_start(
            task.task_id,
            parent_name=parent_task_id,
            # ...
        )
```

### 4. エッジケースの処理: 外部親のクリア
ワーカー内の最初のタスクが開始されると、厳密にはそのチェーンの外部親を「消費」したことになります。しかし、後続のタスクはスタック上のローカルな `current_task_id`（最初のタスク）を参照し、それが `if/elif` ロジックで優先されるため、保持し続けても問題はありません。

## 検証
- **ローカル:** ルートタスクの `executing_task` -> 親は None。ネスト時 -> 親は A。（動作変更なし）
- **ワーカー:** `attach_to_trace(parent=P)`。`executing_task(Child)` -> スタックは空 -> フォールバック P。エッジ `P -> Child` が作成される。

これにより、`_runtime_graph` が分散タスクをその発生元に正しく接続することを保証します。
