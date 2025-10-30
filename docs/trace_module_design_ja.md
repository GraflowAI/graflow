# Graflow Tracing モジュール設計書

## 1. 概要

Graflowワークフローの実行トレース、状態遷移の記録、および外部トレーシングシステム（LangFuse等）との統合を実現するトレーシングシステム。

### 目的

- ワークフロー実行の可視化とデバッグ
- イベントベースの状態遷移記録
- Runtime graph（実行時の動的タスク依存グラフ）の管理
- LangFuseなどの外部トレーシングシステムとの統合
- 将来的なLLM生成トレース（LiteLLM等）のサポート

### 主要機能

1. **統一されたトレーシングAPI** - ワークフロー実行とLLM生成（将来）の両方をサポート
2. **ゼロオーバーヘッド設計** - デフォルトはno-op実装、トレース無効時のパフォーマンス影響なし
3. **拡張可能なアーキテクチャ** - `Tracer`基底クラスを継承してカスタムトレーサーを実装可能
4. **Runtime Graph管理** - 実行時のタスク依存関係、実行順序、タイミング情報を記録

## 2. アーキテクチャ

### 2.1 全体構成

```
ExecutionContext
    ├─ TaskGraph (既存: 静的なワークフロー定義グラフ)
    └─ Tracer (新規: トレース + Runtime Graph統合)
           ├─ _runtime_graph: nx.DiGraph (networkx直接利用)
           ├─ Tracer (抽象基底クラス - ABC)
           ├─ NoopTracer (デフォルト: no-op実装 + runtime graph tracking)
           ├─ ConsoleTracer (コンソール出力 + runtime graph)
           └─ LangFuseTracer (LangFuse統合 + runtime graph + dotenv設定)

WorkflowEngine
    └─ execute() → ExecutionContext.tracer経由でイベント送信
```

**設計の重要な決定事項:**

1. **Tracerは抽象基底クラス（ABC）**
   - すべてのトレーサーは`Tracer`を継承
   - 抽象メソッドで必須APIを定義
   - `NoopTracer`がデフォルト実装（runtime graph trackingあり）

2. **Runtime GraphはnetworkxのDiGraphを直接使用**
   - `TaskGraph`は「ワークフロー定義」（静的、Executableオブジェクトを含む）
   - `nx.DiGraph`は「実行履歴」（動的、実行時情報のみ）
   - 明確な責任分離でよりシンプルな設計

3. **Runtime Graphのノード属性**
   - Executableオブジェクトは不要（実行後の記録のみ）
   - 実行時情報のみ記録（status, start_time, end_time, output, error, metadata）

4. **可視化はユーティリティで対応**
   - `export_runtime_graph()`で取得したデータを既存の`draw_ascii`や外部ツールで加工
   - networkxの分析機能（shortest_path, centrality等）を直接利用可能

5. **LangFuseTracerはdotenvから設定を読み込む**
   - `python-dotenv`を使用して`.env`ファイルから環境変数を読み込む
   - `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`

### 2.2 イベントフロー

```
WorkflowEngine.execute()
    ├─ tracer.trace_start("workflow_id")
    │
    ├─ [タスク実行ループ]
    │   ├─ ExecutionContext.executing_task(task)
    │   │   ├─ tracer.span_start("task_id", metadata={"task_type": "Task"})
    │   │   ├─ [タスク実行]
    │   │   └─ tracer.span_end("task_id", status=COMPLETED)
    │   │
    │   ├─ [後続タスクのキューイング]
    │   │   └─ tracer.event("task_queued", parent_span="task_id")
    │   │
    │   └─ [動的タスク生成時]
    │       └─ tracer.event("dynamic_task_added", parent_span="task_id")
    │
    └─ tracer.trace_end("workflow_id", status=COMPLETED)
```

## 3. ディレクトリ構造

```
graflow/
├── trace/
│   ├── __init__.py              # Public API ドキュメント
│   ├── base.py                  # Tracer (ABC) + runtime graph サポート
│   ├── noop.py                  # NoopTracer (デフォルト実装)
│   ├── console.py               # ConsoleTracer (コンソール出力)
│   └── langfuse.py              # LangFuseTracer (LangFuse統合 + dotenv)
├── core/
│   ├── graph.py                 # TaskGraph (ワークフロー定義グラフ)
│   ├── context.py               # ExecutionContext (tracer統合)
│   └── engine.py                # WorkflowEngine (イベント送信)
├── utils/
│   └── graph.py                 # draw_ascii (可視化ユーティリティ)
```

## 4. コアコンポーネント

### 4.1 `graflow/trace/base.py`

#### 4.1.1 `Tracer` 抽象基底クラス

すべてのトレーサーの抽象基底クラス（ABC）。

**基本構造:**

```python
from abc import ABC, abstractmethod
import networkx as nx
from typing import Optional, List, Dict, Any
from datetime import datetime

class Tracer(ABC):
    """Abstract base class for all tracers.

    All tracers inherit from this class and can optionally track
    runtime graph execution using networkx DiGraph.
    """

    def __init__(self, enable_runtime_graph: bool = True):
        """Initialize tracer with optional runtime graph tracking.

        Args:
            enable_runtime_graph: If True, track execution in a networkx DiGraph
        """
        self.enable_runtime_graph = enable_runtime_graph
        self._runtime_graph: Optional[nx.DiGraph] = (
            nx.DiGraph() if enable_runtime_graph else None
        )
        self._execution_order: List[str] = []  # タスク実行順序
        self._trace_start_time: Optional[datetime] = None
        self._trace_name: Optional[str] = None
```

**抽象メソッド（必須実装）:**

```python
class Tracer(ABC):
    # トレース（最上位レベル）
    @abstractmethod
    def trace_start(self, name: str, trace_id: Optional[str] = None,
                   metadata: Optional[Dict[str, Any]] = None) -> None:
        """Start a trace."""
        pass

    @abstractmethod
    def trace_end(self, name: str, output: Any = None,
                 metadata: Optional[Dict[str, Any]] = None) -> None:
        """End a trace."""
        pass

    # スパン（タスク、LLM生成など）
    @abstractmethod
    def span_start(self, name: str, parent_name: Optional[str] = None,
                  metadata: Optional[Dict[str, Any]] = None) -> None:
        """Start a span."""
        pass

    @abstractmethod
    def span_end(self, name: str, output: Any = None,
                metadata: Optional[Dict[str, Any]] = None) -> None:
        """End a span."""
        pass

    # イベント（キューイング、チェックポイントなど）
    @abstractmethod
    def event(self, name: str, parent_span: Optional[str] = None,
             metadata: Optional[Dict[str, Any]] = None) -> None:
        """Record an event."""
        pass

    # ユーティリティ
    @abstractmethod
    def flush(self) -> None:
        """Flush pending traces."""
        pass
```

**共通メソッド（基底クラスで実装）:**

```python
class Tracer(ABC):
    # Runtime Graph管理（すべてのトレーサーで利用可能）
    def get_execution_order(self) -> List[str]:
        """Get task execution order."""
        return self._execution_order.copy()

    def get_runtime_graph(self) -> Optional[nx.DiGraph]:
        """Get the runtime execution graph."""
        return self._runtime_graph

    def export_runtime_graph(self, format: str = "dict") -> Optional[Dict[str, Any]]:
        """Export runtime graph data (dict/json/graphml をサポート)。"""
        # 実装は base.Tracer.export_runtime_graph を参照
```

**Runtime Graph (networkx DiGraph) のノード属性:**

```python
{
    "status": str,                # 実行ステータス ("running", "completed", "failed")
    "start_time": datetime,       # 開始時刻
    "end_time": Optional[datetime],  # 終了時刻
    "output": Any,                # タスク出力（シリアライズ可能な場合のみ）
    "error": Optional[str],       # エラー情報
    "metadata": Dict[str, Any],   # タスクタイプ、パラメータ等
}
```

**Runtime Graphのエッジ属性:**

```python
{
    "relation": str,  # "parent-child", "depends-on"
}
```

**便利メソッド（後方互換性）:**

```python
# ワークフロー
def on_workflow_start(workflow_name: str, context: ExecutionContext) -> None
def on_workflow_end(workflow_name: str, context: ExecutionContext, result: Any | None = None) -> None

# タスク
def on_task_start(task: Executable, context: ExecutionContext) -> None
def on_task_end(task: Executable, context: ExecutionContext, result: Any | None = None, error: Exception | None = None) -> None
def on_task_queued(task: Executable, context: ExecutionContext) -> None
def on_dynamic_task_added(task_id: str, parent_task_id: str | None = None, is_iteration: bool = False, metadata: Optional[dict] = None) -> None

# パラレルグループ
def on_parallel_group_start(group_id: str, member_ids: list[str], context: ExecutionContext) -> None
def on_parallel_group_end(group_id: str, member_ids: list[str], context: ExecutionContext, results: Optional[Dict[str, Any]] = None) -> None

# LLM生成（将来）
def generation_start(name: str, model: str, parent_span: Optional[str] = None, metadata: Optional[dict] = None) -> None
def generation_end(name: str, output: Any = None, usage: Optional[dict] = None, error: Optional[Exception] = None, metadata: Optional[dict] = None) -> None
```

### 4.2 `graflow/trace/noop.py` - NoopTracer

デフォルトのno-op実装。すべてのメソッドは何もしないが、runtime graphはトラッキングする。

```python
"""No-op tracer implementation (default)."""

from typing import Optional
from .base import Tracer


class NoopTracer(Tracer):
    """No-operation tracer (default).

    具体的な出力処理を行わず、基底クラスが提供する runtime graph
    トラッキングと実行順序記録のみ利用する最小実装です。

    実装では `_output_trace_start` などのテンプレートメソッドを
    すべて no-op でオーバーライドしています。
    """

    def _output_trace_start(self, name: str, trace_id: Optional[str], metadata: Optional[dict]) -> None:
        pass

    def _output_trace_end(self, name: str, output: Optional[object], metadata: Optional[dict]) -> None:
        pass

    def _output_span_start(self, name: str, parent_name: Optional[str], metadata: Optional[dict]) -> None:
        pass

    def _output_span_end(self, name: str, output: Optional[object], metadata: Optional[dict]) -> None:
        pass

    def _output_event(self, name: str, parent_span: Optional[str], metadata: Optional[dict]) -> None:
        pass

    def _output_attach_to_trace(self, trace_id: str, parent_span_id: Optional[str]) -> None:
        pass
```

### 4.3 `graflow/trace/console.py` - ConsoleTracer

コンソールにトレース情報を出力するシンプルなトレーサー。

```python
class ConsoleTracer(Tracer):
    """Console output tracer for debugging and development.

    Prints workflow execution events to stdout with indentation
    to show nesting structure.

    Example:
        >>> tracer = ConsoleTracer()
        >>> tracer.trace_start("my_workflow")
        ▶ TRACE START: my_workflow
        >>> tracer.span_start("task_1", metadata={"task_type": "Task"})
          ▶ task_1 [Task]
        >>> tracer.span_end("task_1")
          ✓ task_1 [completed]
        >>> tracer.trace_end("my_workflow")
        ✓ TRACE END: my_workflow
    """

    def __init__(self, enable_runtime_graph: bool = True, verbose: bool = False):
        """Initialize console tracer.

        Args:
            enable_runtime_graph: Enable runtime graph tracking
            verbose: Print verbose output (events, metadata)
        """
        super().__init__(enable_runtime_graph=enable_runtime_graph)
        self.verbose = verbose
        self._indent_level = 0

    # 実装は `_output_*` フックをオーバーライドしており、
    # span 開始時にインデントを増やし、終了時に戻す。
    # エラーが渡された場合は赤色と "✗" アイコンで表示し、
    # metadata 表示は verbose モードでのみ出力される。
```

### 4.4 `graflow/trace/langfuse.py`

#### 4.4.1 `LangFuseTracer` クラス

LangFuse manual observations APIを使った実装。設定はdotenvから読み込む。

**必要なパッケージ:**

```bash
pip install langfuse python-dotenv
# または
uv add langfuse python-dotenv
```

**`.env`ファイル:**

```.env
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

**初期化:**

```python
from graflow.trace.langfuse import LangFuseTracer

# .envファイルから自動的に設定を読み込む
tracer = LangFuseTracer()

# または明示的に指定（環境変数より優先）
tracer = LangFuseTracer(
    public_key="pk-...",
    secret_key="sk-...",
    host="https://...",
    enabled=True
)
```

**実装の特徴:**

- **dotenv統合**: `.env`ファイルから`LANGFUSE_*`環境変数を自動読み込み
- `trace_start()` → LangFuseルートspanを作成
- `span_start()` → 親spanの子spanとして作成（名前ベース管理） + runtime graph tracking
- `span_end()` → spanを更新してend()を呼び出し + runtime graph更新
- `event()` → 親spanのmetadataとして記録
- `flush()` → LangFuseクライアントのflush()を呼び出し

**内部管理:**

- `_trace_client: Optional[StatefulTraceClient]` - 現在のトレースコンテキスト
- `_span_stack: list[StatefulSpanClient]` - ネストしたspanを管理するスタック
- `attach_to_trace()` は `parent_span_id` を指定すると LangFuse API から既存spanを再取得し、スタックへ積んで親子関係を復元

実装は Template Method パターンに従い、`_output_*` フック内で LangFuse SDK の
`trace()`, `span()`, `event()` を呼び出します。エラー発生時は span の `level`
を `ERROR` に設定し、`shutdown()` で `client.flush()` を確実に実行します。

## 5. 統合ポイント

### 5.1 `ExecutionContext` への統合

#### 5.1.1 初期化

```python
class ExecutionContext:
    def __init__(
        self,
        ...
        tracer: Tracer = NoopTracer(),
    ):
        # デフォルトはNoopTracerインスタンス（runtime graph有効）
        self.tracer = tracer

# Note: runtime graphはtracer経由でアクセス
# context.tracer.get_runtime_graph() -> nx.DiGraph
# context.tracer.get_execution_order() -> List[str]
# context.tracer.export_runtime_graph("dict") -> Dict[str, Any]
```

#### 5.1.2 `executing_task()` コンテキストマネージャー

タスク実行の開始/終了時にトレーサーを呼び出す。

```python
@contextmanager
def executing_task(self, task: Executable):
    """タスク実行の前後で tracer フックを呼び出す。"""

    task_ctx = self.create_task_context(task.task_id)
    self.push_task_context(task_ctx)

    error: Optional[Exception] = None

    try:
        self.tracer.on_task_start(task, self)
        task.set_execution_context(self)
        yield task_ctx
    except Exception as exc:
        error = exc
        raise
    finally:
        self.tracer.on_task_end(task, self, result=None, error=error)
        self.pop_task_context()

# 実装では push 前に親タスクIDを退避しておき、
# `Tracer.on_task_start()` から `context.current_task_id`
# を参照したときに正しい親子関係が復元される。
```

#### 5.1.3 動的タスク生成

`next_task()`と`next_iteration()`でトレーサーを呼び出す。

```python
def next_task(self, task: Executable, ...) -> None:
    """動的タスク追加"""
    ...
    # 🔹 Tracer: 動的タスク追加イベント
    if self.tracer:
        self.tracer.on_dynamic_task_added(
            task_id=task.task_id,
            parent_task_id=current_task_id,
            is_iteration=False,
            metadata={"task_type": type(task).__name__}
        )

def next_iteration(self, task: Executable, ...) -> None:
    """タスク再実行（イテレーション）"""
    ...
    # 🔹 Tracer: イテレーション追加イベント
    if self.tracer:
        self.tracer.on_dynamic_task_added(
            task_id=new_task_id,
            parent_task_id=current_task_id,
            is_iteration=True,
            metadata={"original_task_id": task.task_id}
        )
```

### 5.2 `WorkflowEngine` への統合

#### 5.2.1 `execute()` メソッド

ワークフロー実行の開始/終了時にトレーサーを呼び出す。

```python
def execute(
    self,
    context: ExecutionContext,
    start_task_id: Optional[str] = None
) -> Any:
    assert context.graph is not None, "Graph must be set before execution"

    workflow_name = getattr(context.graph, "name", None) or f"workflow_{context.session_id[:8]}"
    context.tracer.on_workflow_start(workflow_name, context)

    print(f"Starting execution from: {start_task_id or context.start_node}")

    task_id = start_task_id or context.get_next_task()
    last_result: Any = None

    while task_id is not None and context.steps < context.max_steps:
        context.reset_goto_flag()

        graph = context.graph
        if task_id not in graph.nodes:
            print(f"Error: Node {task_id} not found in graph")
            break

        task = graph.get_node(task_id)

        try:
            with context.executing_task(task):
                last_result = self._execute_task(task, context)
        except Exception as exc:
            raise exceptions.as_runtime_error(exc)

        if context.goto_called:
            print(f"🚫 Goto called in {task_id}, skipping successors")
        else:
            successors = list(graph.successors(task_id))

            from graflow.core.task import ParallelGroup
            if isinstance(task, ParallelGroup):
                member_ids = {member.task_id for member in task.tasks}
                successors = [succ for succ in successors if succ not in member_ids]

            for succ in successors:
                succ_task = graph.get_node(succ)
                context.add_to_queue(succ_task)

        context.mark_task_completed(task_id)
        context.increment_step()

        if context.checkpoint_requested:
            from graflow.core.checkpoint import CheckpointManager

            checkpoint_path, checkpoint_metadata = CheckpointManager.create_checkpoint(
                context,
                path=context.checkpoint_request_path,
                metadata=context.checkpoint_request_metadata,
            )
            print(f"Checkpoint created: {checkpoint_path}")
            context.checkpoint_metadata = checkpoint_metadata.to_dict()
            context.last_checkpoint_path = checkpoint_path
            context.clear_checkpoint_request()

        task_id = context.get_next_task()

    print(f"Execution completed after {context.steps} steps")
    context.tracer.on_workflow_end(workflow_name, context, result=last_result)
    return last_result
```

## 6. 使用例

### 6.1 基本的な使用（NoopTracer - デフォルト）

```python
from graflow.core.workflow import workflow
from graflow.core.decorators import task
from graflow.core.context import create_execution_context

@task
def process_data(x: int) -> int:
    return x * 2

# デフォルト: NoopTracer（出力なし、runtime graphのみトラッキング）
context = create_execution_context()

with workflow("simple_workflow", context=context) as wf:
    result = process_data.with_params(x=10)
    wf.execute()

# Runtime graphは取得可能
print(context.tracer.get_execution_order())
print(context.tracer.export_runtime_graph("dict"))
```

### 6.2 LangFuseトレース（dotenv設定）

**Step 1: `.env`ファイルを作成**

```.env
LANGFUSE_PUBLIC_KEY=pk-lf-1234567890abcdef
LANGFUSE_SECRET_KEY=sk-lf-abcdef1234567890
LANGFUSE_HOST=https://cloud.langfuse.com
```

**Step 2: LangFuseTracerを使用**

```python
from graflow.trace.langfuse import LangFuseTracer
from graflow.core.context import create_execution_context
from graflow.core.workflow import workflow

# .envから自動的に設定を読み込む
tracer = LangFuseTracer()

# Execution contextにトレーサーを設定
context = create_execution_context(tracer=tracer)

with workflow("traced_workflow", context=context) as wf:
    task_a = fetch_data.with_params(url="https://api.example.com")
    task_b = process_data.with_params(data=task_a)
    save_results.with_params(data=task_b)

    wf.execute()

# 短命なアプリケーションの場合はflush
tracer.flush()

# Runtime graphのエクスポート例
runtime_graph = context.tracer.get_runtime_graph()
if runtime_graph:
    export = context.tracer.export_runtime_graph("dict")
    if export:
        print(f"Recorded tasks: {len(export['nodes'])}")
        print(f"Execution path: {context.tracer.get_execution_order()}")

    # networkx DiGraphとして直接分析
    import networkx as nx
    print(f"Graph density: {nx.density(runtime_graph)}")
    print(f"Longest path: {nx.dag_longest_path(runtime_graph)}")
```

### 6.3 カスタムトレーサー（コンソール出力）

```python
from graflow.trace.base import Tracer

class SimpleConsoleTracer(Tracer):
    """Template Methodに従った簡易コンソールトレーサー"""

    def __init__(self, enable_runtime_graph: bool = True):
        super().__init__(enable_runtime_graph=enable_runtime_graph)
        self._indent = 0

    def _print(self, icon: str, message: str) -> None:
        indent = "  " * self._indent
        print(f"{indent}{icon} {message}")

    def _output_trace_start(self, name, trace_id, metadata):
        self._print("▶", f"TRACE START {name}")
        self._indent += 1

    def _output_trace_end(self, name, output, metadata):
        self._indent = max(0, self._indent - 1)
        self._print("✓", f"TRACE END {name}")

    def _output_span_start(self, name, parent_name, metadata):
        self._print("▶", f"SPAN {name}")
        self._indent += 1

    def _output_span_end(self, name, output, metadata):
        self._indent = max(0, self._indent - 1)
        self._print("✓", f"SPAN {name}")

    def _output_event(self, name, parent_span, metadata):
        self._print("•", f"EVENT {name}")

    def _output_attach_to_trace(self, trace_id, parent_span_id):
        self._print("↺", f"ATTACH {trace_id} (parent={parent_span_id})")

# 使用例
tracer = SimpleConsoleTracer()
context = create_execution_context(tracer=tracer)

with workflow("console_workflow", context=context) as wf:
    task_a >> task_b >> task_c
    wf.execute()
```

### 6.4 将来：LLM生成トレース

```python
from graflow.core.decorators import task
from graflow.trace import Tracer
import litellm

@task
def generate_summary(text: str, ctx: ExecutionContext) -> str:
    """LLMを使ってサマリーを生成"""
    tracer = ctx.tracer

    # LLM生成のトレース
    generation_id = "gpt4_summary_gen"
    tracer.generation_start(
        name=generation_id,
        model="gpt-4",
        parent_span=ctx.current_task_id,
        metadata={"prompt_preview": text[:100]}
    )

    try:
        response = litellm.completion(
            model="gpt-4",
            messages=[{"role": "user", "content": f"Summarize: {text}"}]
        )

        summary = response.choices[0].message.content

        tracer.generation_end(
            name=generation_id,
            output=summary,
            metadata={"status": "completed"},
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
            }
        )

        return summary

    except Exception as e:
        tracer.generation_end(
            name=generation_id,
            error=e,
            metadata={"status": "failed"}
        )
        raise
```

## 7. Runtime Graph実装の詳細

### 7.1 Tracerクラスのruntime graph管理

```python
class Tracer:
    def span_start(self, name, parent_name=None, metadata=None):
        """Start a span and track in runtime graph."""
        # Runtime graph tracking
        if self._runtime_graph is not None:
            from datetime import datetime

            # ノードを追加
            self._runtime_graph.add_node(
                name,
                status="running",
                start_time=datetime.now(),
                end_time=None,
                output=None,
                error=None,
                metadata=metadata or {}
            )

            # 親子関係を記録
            if parent_name and parent_name in self._runtime_graph:
                self._runtime_graph.add_edge(
                    parent_name,
                    name,
                    relation="parent-child"
                )

            # 実行順序を記録
            self._execution_order.append(name)

    def span_end(self, name, status, output=None, error=None, metadata=None):
        """End a span and update runtime graph."""
        # Runtime graph tracking
        if self._runtime_graph is not None and name in self._runtime_graph:
            from datetime import datetime

            # ノード属性を更新
            self._runtime_graph.nodes[name].update({
                "status": status.value,
                "end_time": datetime.now(),
                "output": output,
                "error": str(error) if error else None,
            })

            # メタデータをマージ
            if metadata:
                self._runtime_graph.nodes[name]["metadata"].update(metadata)

    def get_execution_order(self) -> List[str]:
        """Get task execution order."""
        return self._execution_order.copy()

    def get_runtime_graph(self) -> Optional[nx.DiGraph]:
        """Get the runtime execution graph."""
        return self._runtime_graph

```

> Note: 現行実装では集計用ヘルパー（`get_execution_stats` や `visualize_runtime_graph`）は提供されていないため、
> 必要に応じて `export_runtime_graph()` と networkx の API を組み合わせて可視化・分析する。

### 7.2 実行時のタスク情報取得

```python
# 特定タスクの実行情報を取得
runtime_graph = tracer.get_runtime_graph()
if runtime_graph and "task_1" in runtime_graph:
    task_info = runtime_graph.nodes["task_1"]
    print(f"Status: {task_info['status']}")
    print(f"Duration: {(task_info['end_time'] - task_info['start_time']).total_seconds()}s")
    print(f"Output: {task_info['output']}")
    print(f"Metadata: {task_info['metadata']}")

# タスクの依存関係を取得
children = list(runtime_graph.successors("task_1"))
parents = list(runtime_graph.predecessors("task_1"))
```

## 9. 実装の優先順位

### Phase 1: 基盤実装

1. **`graflow/core/context.py`の修正**
   - ⚠️ **重要**: `session_id`生成をW3C TraceContext準拠に変更
   - 変更: `str(uuid.uuid4().int)` → `uuid.uuid4().hex`

2. **`graflow/trace/base.py`**
   - `Tracer` 抽象基底クラス（ABC）
   - Runtime graph管理メソッド実装（共通機能）

3. **`graflow/trace/noop.py`**
   - `NoopTracer` クラス（デフォルト実装）
   - Runtime graph tracking実装

4. **`graflow/trace/__init__.py`**
   - Public API ドキュメント整備（`__all__` は空のまま、直接 import を想定）

5. **`ExecutionContext`への統合**
   - `tracer`フィールド追加
   - デフォルト値設定（`NoopTracer()` をそのまま使用）

### Phase 2: ConsoleTracer実装

1. **`graflow/trace/console.py`**
   - `ConsoleTracer` クラス実装
   - インデント付きコンソール出力
   - verboseモード

2. **基本的な統合テスト**
   - 単純なワークフローでConsoleTracerを使用

### Phase 3: LangFuse統合

1. **依存関係の追加**
   - `python-dotenv`をpyproject.tomlに追加
   - `langfuse`をoptional dependencyとして追加

2. **`graflow/trace/langfuse.py`**
   - `LangFuseTracer` クラス実装
   - dotenv統合（`.env`から設定読み込み）
   - LangFuse manual observations API統合
   - Runtime graph tracking実装

3. **`WorkflowEngine.execute()`への統合**
   - ワークフロー開始/終了イベント
   - タスクキューイングイベント

4. **動的タスク生成への統合**
   - `ExecutionContext.next_task()`
   - `ExecutionContext.next_iteration()`

### Phase 4: テストと文書化

1. **単体テスト**
   - `Tracer`基底クラスとruntime graph管理
   - `ConsoleTracer`
   - `LangFuseTracer`

2. **統合テスト**
   - 単純なワークフロー
   - パラレルグループ
   - 動的タスク生成
   - Runtime graph分析

3. **使用例とドキュメント**
   - `examples/12_tracing/` ディレクトリ作成
   - 基本的な使用例
   - ConsoleTracer例
   - LangFuseTracer例
   - Runtime graph分析例
   - README更新

## 10. 設計上の考慮事項

### 10.1 パフォーマンス

- **ゼロオーバーヘッド**: デフォルトの`Tracer`クラスはすべてno-op実装
- **条件チェック**: `if context.tracer:` でトレーサー呼び出しをガード
- **非同期flush**: LangFuseのflush()は非同期で実行

### 10.2 拡張性

- **基底クラス設計**: `Tracer`を継承してカスタムトレーサーを実装可能
- **メタデータ活用**: タスクタイプ、モデル名などは`metadata`辞書で柔軟に指定
- **将来のLLMサポート**: `generation_start/end`メソッドで準備済み

### 10.3 後方互換性

- **便利メソッド**: `on_workflow_start`、`on_task_start`などの既存API維持
- **オプトイン**: トレース機能は明示的に有効化（デフォルトはno-op）

### 10.4 分散実行との互換性

- **スレッドセーフ**: `ExecutionContext`のtracer呼び出しはスレッドセーフ
- **ワーカー対応**: 各ワーカーが独自のトレーサーインスタンスを持つ
- **span識別**: タスクIDベースのspan名で分散環境でも追跡可能

## 11. LangFuse統合の詳細

### 11.1 Span階層のマッピング

```
LangFuse Trace
└─ Root Span (workflow_id)
    ├─ Task Span (task_a)
    │   ├─ Event: task_queued (task_b)
    │   └─ Event: dynamic_task_added (task_x)
    ├─ Task Span (task_b)
    └─ Parallel Group Span (parallel_group_1)
        ├─ Task Span (task_c)
        └─ Task Span (task_d)
```

### 11.2 Metadata構造

**ワークフロー:**
```json
{
  "start_node": "task_a",
  "max_steps": 100,
  "total_steps": 5,
  "status": "completed"
}
```

**タスク:**
```json
{
  "task_type": "Task",
  "handler": "direct",
  "status": "completed"
}
```

**イベント:**
```json
{
  "events": [
    {
      "name": "task_queued",
      "task_id": "task_b",
      "task_type": "Task"
    },
    {
      "name": "dynamic_task_added",
      "task_id": "task_x",
      "is_iteration": false,
      "task_type": "Task"
    }
  ]
}
```

## 12. 分散実行（TaskWorker）とのトレース統合

### 12.1 課題

TaskWorkerは別プロセスで動作するため、親プロセスのトレースコンテキストが失われる。
分散実行時も統合的にトレースを見るために、トレースIDとトレーサー設定を持ち回る必要がある。

### 12.2 設計アプローチ

**基本方針:**
1. `TaskSpec`にトレースコンテキスト情報を追加
2. タスクキューイング時にトレースID（`session_id`）と親spanIDを記録
3. TaskWorkerでタスク実行時に親トレースに接続

**トレースIDとして`session_id`を使用（W3C TraceContext準拠）:**
- `ExecutionContext.session_id`は既にワークフロー実行ごとにユニークなID
- **重要**: W3C TraceContext準拠のため、**32桁のhex形式**に変更が必要
  - 現在: `str(uuid.uuid4().int)` → 10進数の長い文字列（非準拠）
  - 変更後: `uuid.uuid4().hex` → 32桁のhex文字列（準拠）
  - 例: `"0af7651916cd43dd8448eb211c80319c"`
- これをトレースIDとして流用することで、追加の管理が不要
- メインプロセスとWorkerプロセスで同じ`session_id`を共有することで、統合トレースを実現

### 12.3 session_idのW3C TraceContext準拠化

**現在の実装（問題）:**
```python
# graflow/core/context.py
self.session_id = session_id or str(uuid.uuid4().int)
# 例: "123456789012345678901234567890" (10進数の長い文字列)
```

**変更後（W3C TraceContext準拠）:**
```python
# graflow/core/context.py
self.session_id = session_id or uuid.uuid4().hex
# 例: "0af7651916cd43dd8448eb211c80319c" (32桁のhex文字列)
```

**W3C TraceContext仕様:**
- trace-id: 32桁のhex（16バイト）
- span-id: 16桁のhex（8バイト）

### 12.4 TaskSpecの拡張

`graflow/queue/base.py`の`TaskSpec`にトレース関連フィールドを追加：

```python
@dataclass
class TaskSpec:
    """Task specification with trace context support."""
    executable: 'Executable'
    execution_context: 'ExecutionContext'
    strategy: str = "reference"
    status: TaskStatus = TaskStatus.READY
    created_at: float = field(default_factory=time.time)

    # Existing fields
    retry_count: int = 0
    max_retries: int = 3
    last_error: Optional[str] = None
    group_id: Optional[str] = None

    # Trace context (新規)
    trace_id: Optional[str] = None           # トレースID (= session_id, W3C準拠32桁hex)
    parent_span_id: Optional[str] = None     # 親spanID（キューイング元タスク）
```

**設計の重要な決定:**
- TaskSpecには**トレース接続情報のみ**を含める（`trace_id`, `parent_span_id`）
- `tracer_type`と`tracer_config`は**含めない**
- Workerは自身の設定ファイルから共通のtracer設定を読み込む
- 全タスクで同じトレーサー設定を使用する前提

### 12.5 トレースコンテキストの伝播

#### 12.5.1 ExecutionContext.add_to_queue()の拡張

タスクキューイング時にトレース接続情報のみを設定：

```python
class ExecutionContext:
    def add_to_queue(self, task: Executable) -> None:
        """Add task to queue with trace context."""
        # トレース接続情報を取得
        trace_id = None
        parent_span_id = None

        if self.tracer:
            # トレースID（ワークフロー全体のID = session_id）
            # session_idは既に32桁hex形式でW3C準拠
            trace_id = self.session_id

            # 親spanID（現在実行中のタスクID）
            parent_span_id = self.current_task_id

        # TaskSpecを作成（trace_idとparent_span_idのみ）
        task_spec = TaskSpec(
            executable=task,
            execution_context=self,
            trace_id=trace_id,
            parent_span_id=parent_span_id,
        )

        self.task_queue.enqueue(task_spec)
```

#### 12.5.2 TaskWorkerのtracer設定

Workerは初期化時に共通のtracer設定を受け取る：

```python
class TaskWorker:
    def __init__(
        self,
        queue: RedisTaskQueue,
        worker_id: str,
        max_concurrent_tasks: int = 4,
        tracer_config: Optional[Dict[str, Any]] = None,  # 設定ファイルから読み込む
    ):
        """Initialize TaskWorker.

        Args:
            queue: RedisTaskQueue instance
            worker_id: Unique worker identifier
            max_concurrent_tasks: Maximum concurrent task count
            tracer_config: Tracer configuration dict with "type" key
                          Example: {"type": "langfuse", "enable_runtime_graph": False}
        """
        self.queue = queue
        self.worker_id = worker_id
        self.tracer_config = tracer_config or {}
```

**tracer_config形式:**
```python
# LangFuse tracer
tracer_config = {
    "type": "langfuse",              # Tracer type: "noop", "console", "langfuse"
    "enable_runtime_graph": False,   # Workerではruntime graph不要
}

# Console tracer
tracer_config = {
    "type": "console",
    "enable_runtime_graph": False,
    "verbose": True,
}

# Noop tracer (no tracing)
tracer_config = {
    "type": "noop",
}
```

**重要な設計決定:**
- `tracer_type`は`tracer_config["type"]`から取得（パラメータ削減）
- **デフォルトはNoopTracer**（tracer_configが空の場合や"type"未指定の場合）
- Workerでは**runtime graphのtrackingは不要**（`enable_runtime_graph=False`推奨）
- LangFuseTracerの場合、API keyは`.env`ファイルから読み込む

### 12.6 TaskWorkerでのトレース初期化

TaskWorkerがタスクを実行する際に親トレースに接続：

```python
class TaskWorker:
    def _process_task_wrapper(self, task_spec: TaskSpec) -> Dict[str, Any]:
        """Execute task with trace context."""
        # Get execution context from task spec
        execution_context = task_spec.execution_context

        # Tracer initialization from worker configuration
        tracer = self._create_tracer()
        if tracer:
            # Set tracer on ExecutionContext
            execution_context.tracer = tracer

            # Attach to parent trace for distributed tracing
            if task_spec.trace_id:
                tracer.attach_to_trace(
                    trace_id=task_spec.trace_id,
                    parent_span_id=task_spec.parent_span_id
                )

        # Execute task...
        # (task execution logic)

        # Flush tracer to ensure data is sent
        if tracer:
            tracer.shutdown()

    def _create_tracer(self) -> Tracer:
        """Create tracer from worker configuration.

        Returns:
            Tracer instance (defaults to NoopTracer)
        """
        # Default to noop tracer if type not specified
        tracer_type = self.tracer_config.get("type", "noop")
        tracer_type = tracer_type.lower()

        # Extract config without "type" key
        config = {k: v for k, v in self.tracer_config.items() if k != "type"}

        if tracer_type == "noop":
            from graflow.trace.noop import NoopTracer
            return NoopTracer(**config)

        elif tracer_type == "console":
            from graflow.trace.console import ConsoleTracer
            return ConsoleTracer(**config)

        elif tracer_type == "langfuse":
            from graflow.trace.langfuse import LangFuseTracer
            # LangFuseは.envからAPI keyを自動読み込み
            return LangFuseTracer(**config)

        else:
            logger.warning(f"Unknown tracer type: {tracer_type}, using NoopTracer")
            from graflow.trace.noop import NoopTracer
            return NoopTracer()
```

**注意:** Workerでは`enable_runtime_graph=False`をtracer_configに含めることを推奨
```

### 12.7 LangFuseでの親トレース接続

LangFuseTracer（セクション 4.4 の実装を参照）に `attach_to_trace()` を追加して、既存トレースへワーカーが合流できるようにする。

```python
def attach_to_trace(self, trace_id: str) -> None:
    """既存のトレースへ合流する（TaskWorker から呼び出す）。"""
    if not self.enabled:
        return

    # session_id (= trace_id) を LangFuse のトレース名として採用
    self._trace_name = trace_id

    # LangFuse API は同じ trace_id の span をグルーピングするため
    # メインプロセスと Worker のトレースが統合される
```

### 12.8 使用例：分散実行でのトレース

```python
from graflow.core.workflow import workflow
from graflow.core.decorators import task
from graflow.core.context import create_execution_context
from graflow.trace.langfuse import LangFuseTracer
from graflow.queue.factory import QueueBackend

# タスク定義
@task
def heavy_task(x: int) -> int:
    import time
    time.sleep(5)  # 重い処理
    return x * 2

# LangFuseトレーサーを作成
tracer = LangFuseTracer()

# Redis queueを使った分散実行
context = create_execution_context(
    queue_backend=QueueBackend.REDIS,
    channel_backend="redis",
    tracer=tracer,
)

with workflow("distributed_workflow", context=context) as wf:
    # タスクをRedis queueにキューイング
    # TaskSpecにtrace_id、parent_span_id、tracer_configが設定される
    result = heavy_task.with_params(x=10)
    wf.execute()

tracer.flush()
```

**TaskWorkerプロセスでの実行:**

```bash
# .envファイルが必要（LangFuseキーを含む）
# LANGFUSE_PUBLIC_KEY=pk-...
# LANGFUSE_SECRET_KEY=sk-...

# TaskWorkerを起動
python -m graflow.worker.main --worker-id worker-1

# TaskWorkerは：
# 1. TaskSpecからtrace_id、parent_span_id、tracer_configを読み取る
# 2. LangFuseTracerを初期化（.envから設定読み込み）
# 3. attach_to_trace()で親トレースに接続
# 4. タスク実行（親トレースのspanとして記録される）
# 5. flush()
```

### 12.9 LangFuseでの表示

分散実行されたタスクも、同一トレースID（`session_id`）でグループ化されて表示される：

```
Trace: distributed_workflow (session_id: wf_1234567890abcdef)
├─ Main Process (localhost)
│   ├─ workflow_start
│   ├─ task_queued (heavy_task)  # メインプロセス
│   └─ workflow_end
│
└─ Worker Process (worker-1)
    └─ heavy_task                 # ワーカープロセス
        ├─ span_start
        ├─ [5秒の実行]
        └─ span_end
```

**重要:**
- トレースID = `ExecutionContext.session_id` (W3C TraceContext準拠の32桁hex)
- 例: `"0af7651916cd43dd8448eb211c80319c"`
- すべてのプロセス（メイン + ワーカー）で同じ`session_id`を共有
- LangFuseでは自動的に同一トレースとしてグループ化される

### 12.10 設計上の考慮事項

#### 12.10.1 W3C TraceContext準拠

- **trace-id**: 32桁のhex（16バイト） - `session_id`として使用
- **span-id**: 16桁のhex（8バイト） - タスクIDから生成する場合は`hashlib`で16桁に変換
  ```python
  import hashlib
  span_id = hashlib.md5(task_id.encode()).hexdigest()[:16]
  ```
- LangFuseが内部でW3C TraceContext形式を使用している場合、この準拠が重要

#### 12.10.2 セキュリティ

- **LangFuseキーの扱い**: TaskSpecには含めず、Workerプロセスの`.env`から読み込む
- **環境変数の統一**: メインプロセスとWorkerプロセスで同じ`.env`を使用

#### 12.10.3 パフォーマンス

- **トレース情報のオーバーヘッド**: TaskSpecに追加するフィールドは最小限
- **シリアライズ**: trace_idとparent_span_idは文字列なので軽量

#### 12.10.4 エラーハンドリング

- **Workerでのtracer初期化失敗**: NoopTracerにフォールバック
- **親トレース接続失敗**: ログに警告を出力し、新規トレースとして記録

## 13. まとめ

本設計により、Graflowは以下を実現する：

1. **統一されたトレーシングインターフェース** - ワークフロー実行とLLM生成（将来）の両方をサポート
2. **分散実行での統合トレース** - TaskWorkerプロセスでのタスク実行も同一トレースに統合
3. **柔軟な実装** - LangFuse、OpenTelemetry、カスタムロギングなど様々なバックエンドに対応可能
4. **パフォーマンス重視** - トレース無効時のオーバーヘッドゼロ
5. **Runtime Graph** - 実行時の動的グラフ管理で詳細な分析が可能
6. **将来の拡張性** - LLM生成トレースなど新機能への対応準備済み

この設計は、既存のGraflowアーキテクチャと自然に統合され、ユーザーに強力なデバッグおよび可視化機能を提供する。特に分散実行環境でも、すべてのタスク実行を統合的に追跡できることが大きな特徴である。

---

## 14. 実装状況 (Implementation Status)

**最終更新日**: 2025年10月26日

### 14.1 完了した実装 (Completed)

#### Phase 1: 基盤実装 ✅ 完了

1. **`graflow/core/context.py` の修正** ✅
   - W3C TraceContext準拠に修正: `str(uuid.uuid4().int)` → `uuid.uuid4().hex`
   - `tracer`パラメータの追加（デフォルト: `NoopTracer()`）
   - `session_id`が32桁hex形式でトレースIDとして使用可能に

2. **`graflow/trace/base.py`** ✅
   - `TraceEvent` dataclass実装
   - `Tracer` 抽象基底クラス（ABC）実装
   - **Template Method パターン適用** (設計時から変更)
     - 基底クラスで具象的なライフサイクルメソッド（`trace_start`, `span_start`等）を実装
     - サブクラスは抽象メソッド（`_output_trace_start`, `_output_span_start`等）のみ実装
     - Runtime graph tracking は基底クラスで自動処理
   - Runtime graph管理メソッド実装
     - `get_runtime_graph()`, `get_execution_order()`, `export_runtime_graph()`

3. **`graflow/trace/noop.py`** ✅
   - `NoopTracer` クラス実装（デフォルトトレーサー）
   - **大幅なコード削減**: ~230行 → ~90行（約60%削減）
   - すべての`_output_*`メソッドは`pass`のみ
   - フックロジックは基底クラスから継承

4. **`graflow/trace/__init__.py`** ✅
   - Public API exports実装
   - `Tracer`, `NoopTracer`, `ConsoleTracer`, `LangFuseTracer`をエクスポート

5. **`ExecutionContext`への統合** ✅
   - `tracer`フィールド追加
   - デフォルト値設定（`NoopTracer(enable_runtime_graph=True)`）
   - `executing_task()`コンテキストマネージャーでトレーサーフック呼び出し
     - `on_task_start()` / `on_task_end()` 統合

#### Phase 2: ConsoleTracer実装 ✅ 完了

1. **`graflow/trace/console.py`** ✅
   - `ConsoleTracer` クラス実装（~270行）
   - フォーマット機能
     - ANSIカラー対応（有効/無効切り替え可能）
     - タイムスタンプ表示
     - インデント付き階層表示
     - メタデータ表示（オプション）
   - **コード削減**: Template Methodパターンにより約20%削減
   - イベントログのための特定フックのオーバーライド

2. **基本的な統合テスト** ✅
   - `examples/01_basics/hello_world.py` で動作確認
   - `examples/02_workflows/simple_pipeline.py` で動作確認
   - ConsoleTracerの出力フォーマット確認

#### Phase 3: LangFuse統合 ✅ 完了（一部）

1. **依存関係の追加** ✅
   - `python-dotenv` をpyproject.tomlに追加済み
   - `langfuse` はオプショナル依存として追加（インポート時にエラーハンドリング）

2. **`graflow/trace/langfuse.py`** ✅
   - `LangFuseTracer` クラス実装（~320行）
   - dotenv統合（`.env`から設定読み込み）
   - LangFuse manual observations API統合
     - `trace()`, `span()`, `event()`メソッドの使用
     - span stackによる階層管理
   - Runtime graph tracking実装（基底クラスから継承）
   - オプショナルインポートの適切な処理
   - `enabled`フラグによるno-opモード（テスト用）

3. **`WorkflowEngine.execute()`への統合** ✅
   - ワークフロー開始イベント（`on_workflow_start()`）: Line 94
   - ワークフロー終了イベント（`on_workflow_end()`）: Line 172
   - ワークフロー名の決定ロジック（graph.nameまたはsession_id prefix）

4. **動的タスク生成への統合** ✅ 完了
   - `ExecutionContext.next_task()` - 完了（タスクIDパターンから`is_iteration`を自動判別）
   - `ExecutionContext.next_iteration()` - 完了（`next_task()`経由で自動的にトレース）
   - **実装の改善**: `is_iteration`パラメータを削除し、タスクID（`_cycle_\d+_[0-9a-f]+$`）から自動判別

### 14.2 設計時からの重要な変更

#### 14.2.1 Template Method パターンの導入

**変更内容**:
- 当初の設計では、各サブクラス（NoopTracer, ConsoleTracer, LangFuseTracer）が完全な実装を持つ想定
- 実装時に、コードの重複を避けるためTemplate Method パターンを採用

**実装の詳細**:
- `Tracer` 基底クラス側で runtime graph への記録や共通前処理／後処理を実装し、最後に `_output_*` 系フックを呼び出す
- サブクラスは `_output_*` フックのみを実装すればよく、詳細コードはセクション 4.2（NoopTracer）、4.3（ConsoleTracer）、4.4（LangFuseTracer）を参照

**効果**:
- NoopTracer: ~230行 → ~90行（約60%削減）
- ConsoleTracer: 約20%のコード削減
- Runtime graph trackingロジックが一箇所に集約
- サブクラスは出力ロジックのみに集中

#### 14.2.2 フックメソッドの配置

**変更内容**:
- 当初は各トレーサーがフックメソッド（`on_workflow_start`, `on_task_start`等）を実装する想定
- 実装時に、フックメソッドも基底クラスに移動し、自動的にライフサイクルメソッドを呼び出すように変更

**実装の詳細**:
```python
# 基底クラスに実装
class Tracer(ABC):
    def on_task_start(self, task, context):
        """Hook called when task starts."""
        parent_task_id = context.current_task_id if hasattr(context, 'current_task_id') else None
        self.span_start(
            task.task_id,
            parent_name=parent_task_id,
            metadata={"task_type": type(task).__name__}
        )
```

**効果**:
- フックロジックの一元管理
- サブクラスはフックをオーバーライドして追加の処理を実装可能（例: ConsoleTracerのイベントログ）

### 14.3 未実装の機能 (Pending)

#### Phase 3 残り: 動的タスク生成への統合 ✅ 完了

- ✅ `ExecutionContext.next_task()` でのトレーサーフック呼び出し
- ✅ `ExecutionContext.next_iteration()` でのトレーサーフック呼び出し
- ✅ `on_dynamic_task_added()` イベントの統合
- ✅ `ExecutionContext.add_to_queue()` でのトレーサーフック呼び出し

**実装完了箇所**:
- `graflow/core/context.py` (next_task, next_iteration, add_to_queue)
- `graflow/trace/console.py` (ConsoleTracer.on_dynamic_task_added, on_task_queued)
- `graflow/trace/langfuse.py` (LangFuseTracer.on_dynamic_task_added, on_task_queued)

**実装の詳細**:
- `next_task()`: 全ケース（新規タスク、既存タスク、goto）で`on_dynamic_task_added()`を呼び出し
  - 内部パラメータ`_is_iteration`を受け取り、トレーサーフックに渡す
- `next_iteration()`: `next_task(iteration_task, _is_iteration=True)`を呼び出し
  - コード重複を避け、共通ロジックを`next_task()`に集約
- `add_to_queue()`: キューイング後に`on_task_queued(task, context)`を呼び出し
- `on_dynamic_task_added()`: 基底クラスでno-op実装、サブクラスでオーバーライド可能
- `on_task_queued()`: 基底クラスでno-op実装、サブクラスでオーバーライド可能

#### Phase 3 残り: 分散実行（TaskWorker）との統合 ✅ 完了

- ✅ `TaskSpec` へのトレース接続情報フィールド追加
  - `trace_id`, `parent_span_id` のみ（tracer_typeとtracer_configは含めない）
- ✅ `ExecutionContext.add_to_queue()` でのトレース接続情報設定
- ✅ `TaskWorker.__init__()` にtracer設定パラメータ追加
- ✅ `TaskWorker._process_task_wrapper()` でのトレーサー初期化
- ✅ `TaskWorker._create_tracer()` 実装（worker configから生成）
- ✅ `Tracer.attach_to_trace()` 実装（base.pyで抽象メソッド定義済み）

**実装完了箇所**:
- `graflow/queue/base.py` (TaskSpec: trace_id, parent_span_id)
- `graflow/core/context.py` (add_to_queue: トレース接続情報設定)
- `graflow/trace/base.py` (attach_to_trace抽象メソッド)
- `graflow/trace/langfuse.py` (_output_attach_to_trace実装)
- `graflow/worker/worker.py` (TaskWorker: tracer_type/tracer_config, _create_tracer, tracer initialization)

**重要な設計変更**:
- TaskSpecには**トレース接続情報のみ**（trace_id, parent_span_id）
- Workerは**自身の設定から**tracer_configを読み込む
  - `tracer_type`は`tracer_config["type"]`に統合（パラメータ削減）
  - **デフォルトはNoopTracer**（tracer_config空の場合）
- 全タスクで同じtracer設定を共有する前提
- Workerでは**runtime graph tracking無効**（enable_runtime_graph=False推奨）
- `Tracer.shutdown()`メソッド追加（デフォルトno-op実装）

#### Phase 4: テストと文書化

- ❌ 単体テスト
  - `Tracer`基底クラスとruntime graph管理
  - `ConsoleTracer`
  - `LangFuseTracer`
- ❌ 統合テスト
  - 単純なワークフロー
  - パラレルグループ
  - 動的タスク生成
  - Runtime graph分析
- ❌ 使用例とドキュメント
  - `examples/12_tracing/` ディレクトリ作成
  - 基本的な使用例
  - ConsoleTracer例
  - LangFuseTracer例
  - Runtime graph分析例
- ❌ README更新

### 14.4 検証済みの動作

1. **基本的なワークフロー実行** ✅
   - `hello_world.py`で動作確認
   - `simple_pipeline.py`で動作確認
   - NoopTracer, ConsoleTracer, LangFuseTracerすべて動作

2. **Runtime Graph Tracking** ✅
   - ノードとエッジの正しい記録
   - 実行順序の記録
   - タイムスタンプと実行時間の記録
   - 統計情報の取得

3. **W3C TraceContext準拠** ✅
   - `session_id`が32桁hex形式で生成される
   - トレースIDとして使用可能

4. **LangFuse統合** ✅
   - dotenvからの設定読み込み
   - トレース、span、イベントの送信
   - flush()による確実なデータ送信
   - エラーステータスの正しい記録

### 14.5 発見された問題と修正

#### 14.5.1 ExecutionContextでのresults属性アクセスエラー

**問題**: `executing_task()`で存在しない`self.results`属性にアクセスしようとしてAttributeError

**修正**: 結果取得を削除し、`result=None`をトレーサーに渡すように変更（結果はハンドラーがコンテキストに保存）

**影響**: Line 573 in `graflow/core/context.py`

#### 14.5.2 Tracerインスタンスの共有問題（Mutable Default Argument）

**問題**: `ExecutionContext.__init__()`と`create()`で`tracer: Tracer = NoopTracer()`をデフォルト引数として使用していたため、全てのコンテキストで同一のNoopTracerインスタンスが共有されていた。Tracerは`_runtime_graph`, `_execution_order`, `_span_stack`などのmutable stateを持つため、独立したワークフローや並行ワーカー間でトレース情報が混在・破損する可能性があった。

**修正**:
1. パラメータを`tracer: Optional[Tracer] = None`に変更
2. コンストラクタ内で`self.tracer = tracer if tracer is not None else NoopTracer()`として、Noneの場合に新しいインスタンスを生成

**影響**:
- `graflow/core/context.py` Line 194-200 (`__init__`)
- `graflow/core/context.py` Line 295, 307 (`create()`)

**検証**: `test_tracer_isolation.py`で各コンテキストが独立したtracerインスタンスを持つことを確認

#### 14.5.3 並列グループのエッジ作成タイミング

**問題**: `on_parallel_group_start`時にメンバータスクノードがまだ存在しないため、エッジが作成されない

**修正**: 基底Tracerで両方のノードが存在する場合のみエッジを追加するようにチェック追加

**影響**: `graflow/trace/base.py` の `on_parallel_group_start()`

#### 14.5.4 Regex Pattern Performance Optimization

**問題**: `next_iteration()`が呼ばれるたびに正規表現パターンをコンパイルしていたため、パフォーマンスに影響

**修正**:
1. モジュールレベルで正規表現パターンをコンパイル: `_ITERATION_PATTERN = re.compile(r'(_cycle_\d+_[0-9a-f]+)+$')`
2. コンパイル済みパターンを使用: `_ITERATION_PATTERN.sub('', task_id)`

**影響**:
- `graflow/core/context.py` Line 5 (import re追加)
- `graflow/core/context.py` Line 31 (パターンコンパイル)
- `graflow/core/context.py` Line 538 (`next_iteration()`での使用)

**効果**: イテレーションタスクの処理パフォーマンス向上（特に`next_iteration()`を頻繁に使用するワークフローで効果的）

### 14.6 次のステップ

**Phase 3 完了状態:**
- ✅ LangFuse統合 - 完了
- ✅ 動的タスク生成への統合 - 完了（`is_iteration`自動判別機能追加）
- ✅ 分散実行（TaskWorker）との統合 - 完了

**優先順位順:**

1. **Phase 4のテスト実装** (重要度: 高)
   - 既存の実装を安定化させるため
   - 各トレーサーの単体テスト（NoopTracer, ConsoleTracer, LangFuseTracer）
   - Runtime graph機能の統合テスト
   - 動的タスク生成のトレーステスト
   - 分散実行（TaskWorker）のトレーステスト

2. **例とドキュメント** (重要度: 中)
   - `examples/12_tracing/` の作成
   - 各トレーサーの使用例
   - Runtime graph分析例
   - 分散トレーシングの使用例

### 14.7 実装の品質指標

- **コード削減率**:
  - NoopTracer: 60%削減
  - ConsoleTracer: 20%削減
- **テストカバレッジ**: 未測定（Phase 4で実施予定）
- **パフォーマンスオーバーヘッド**:
  - NoopTracer: ほぼゼロ（runtime graph無効時）
  - ConsoleTracer: print出力のみ（許容範囲）
  - LangFuseTracer: ネットワークI/O（非同期flush使用）

### 14.8 技術的負債

なし（現時点）

### 14.9 参考情報

- **実装期間**: 2025年10月（Phase 1-3前半）
- **関連ブランチ**: `langfuse`
- **主要な議論**: Template Methodパターンの採用、session_idのW3C準拠化
