# TaskFlow API 設計ドキュメント

## 概要

AirflowのTaskFlow APIにインスパイアされた、graflowでの簡潔なデータフロー定義APIの設計オプションを提示します。既存のChannelとTypedChannelシステムを活用し、タスク間のデータ交換を実現します。

## 設計目標

1. **簡潔性**: Pythonの自然な関数呼び出しでワークフローを定義
2. **型安全性**: TypedChannelを活用した型安全なデータ交換
3. **既存アーキテクチャとの統合**: graflowの既存システムとの親和性
4. **自動依存関係解析**: 関数呼び出しから自動的にタスク依存関係を構築
5. **後方互換性**: 既存のデコレータ連動型（`>>`オペレータ）のサポート継続
6. **段階的移行**: 新旧両方のシンタックスの共存と選択的使用

## 設計オプション

### オプション 1: ChannelReference ベース（推奨）

```python
@workflow
def data_pipeline():
    @task
    def extract() -> dict:
        return {"data": [1, 2, 3], "source": "database"}
    
    @task
    def transform(data: dict) -> dict:
        return {"processed": [x * 2 for x in data["data"]]}
    
    @task
    def load(data: dict) -> str:
        print(f"Loading: {data}")
        return "complete"
    
    # 自然な関数呼び出し - データはChannelを通じて流れる
    raw_data = extract()          # ChannelReference を返す
    processed = transform(raw_data)  # 依存関係を自動検出
    result = load(processed)
    
    return result
```

**特徴:**
- ✅ Airflow TaskFlow APIと類似のシンタックス
- ✅ 既存のChannelシステムを活用
- ✅ 自動依存関係検出
- ✅ 遅延実行（実際のデータ交換は実行時）

**実装詳細:**

#### ディレクトリ構造
```
graflow/
├── core/
│   ├── decorators.py        # @workflow, @taskデコレータ
│   ├── workflow.py          # 既存のWorkflowContext（変更なし）
│   ├── taskflow.py          # 新規: TaskFlow機能
│   └── channel_ref.py       # 新規: ChannelReference実装
```

#### 1. ChannelReferenceクラス（`graflow/core/channel_ref.py`）
```python
from typing import Any, Optional, TYPE_CHECKING
from dataclasses import dataclass

if TYPE_CHECKING:
    from graflow.core.context import TaskExecutionContext

@dataclass
class ChannelReference:
    """Channelに保存されたデータへの参照"""
    task_id: str
    key: str = "return_value"
    
    @property
    def channel_key(self) -> str:
        """Channel内でのキー名を生成"""
        return f"{self.task_id}.{self.key}"
    
    def resolve(self, context: 'TaskExecutionContext') -> Any:
        """Channelからデータを解決"""
        channel = context.get_channel()
        if not channel.has(self.channel_key):
            raise ValueError(f"Channel key '{self.channel_key}' not found")
        return channel.get(self.channel_key)
    
    def __repr__(self) -> str:
        return f"ChannelReference(task_id='{self.task_id}', key='{self.key}')"
```

#### 2. TaskFlowWrapper（`graflow/core/taskflow.py`）
```python
from typing import Any, Callable, List, Dict
from functools import wraps
from graflow.core.task import Task, TaskWrapper
from graflow.core.channel_ref import ChannelReference
from graflow.core.context import TaskExecutionContext

class TaskFlowWrapper(TaskWrapper):
    """TaskFlow機能を持つTaskWrapper"""
    
    def __init__(self, func: Callable, task_id: str):
        super().__init__(func, task_id)
        self._stored_args: List[Any] = []
        self._stored_kwargs: Dict[str, Any] = {}
        self._is_callable = True
    
    def __call__(self, *args, **kwargs) -> Any:
        """コンテキストに応じて実行方式を変更する関数呼び出し"""
        # Workflowコンテキストをチェック
        from graflow.core.workflow import get_current_workflow_context
        current_workflow_context = get_current_workflow_context(create_if_not_exist=False)
        
        if current_workflow_context is not None:
            # Workflowコンテキスト内 - TaskFlow実行モード
            if not self._is_callable:
                raise RuntimeError(f"Task '{self.task_id}' has already been called")
            
            # 引数を保存
            self._stored_args = list(args)
            self._stored_kwargs = dict(kwargs)
            self._is_callable = False
            
            # ChannelReferenceを返す（遅延実行）
            return ChannelReference(task_id=self.task_id)
        else:
            # Workflowコンテキスト外 - 直接実行モード（デバッグ用）
            return self.func(*args, **kwargs)
    
    def execute_with_context(self, context: TaskExecutionContext) -> Any:
        """実際のタスク実行（Channel解決付き）"""
        # 引数のChannelReferenceを実際の値に解決
        resolved_args = []
        for arg in self._stored_args:
            if isinstance(arg, ChannelReference):
                resolved_value = arg.resolve(context)
                resolved_args.append(resolved_value)
            else:
                resolved_args.append(arg)
        
        resolved_kwargs = {}
        for key, value in self._stored_kwargs.items():
            if isinstance(value, ChannelReference):
                resolved_value = value.resolve(context)
                resolved_kwargs[key] = resolved_value
            else:
                resolved_kwargs[key] = value
        
        # 元の関数を実行
        result = self.func(*resolved_args, **resolved_kwargs)
        
        # 結果をChannelに保存
        channel = context.get_channel()
        channel_key = f"{self.task_id}.return_value"
        channel.set(channel_key, result)
        
        return result
    
    def get_dependencies(self) -> List[str]:
        """依存関係を自動解析"""
        dependencies = []
        for arg in self._stored_args + list(self._stored_kwargs.values()):
            if isinstance(arg, ChannelReference):
                dependencies.append(arg.task_id)
        return dependencies

class WorkflowBuilder:
    """ワークフロー構築器"""
    
    def __init__(self, name: str):
        self.name = name
        self.tasks: Dict[str, TaskFlowWrapper] = {}
        self.execution_order: List[str] = []
    
    def add_task(self, task: TaskFlowWrapper):
        """タスクを追加"""
        self.tasks[task.task_id] = task
    
    def build_execution_plan(self) -> List[str]:
        """依存関係に基づいて実行順序を決定"""
        # トポロジカルソート実装
        visited = set()
        temp_visited = set()
        execution_order = []
        
        def dfs(task_id: str):
            if task_id in temp_visited:
                raise ValueError(f"Circular dependency detected involving task '{task_id}'")
            if task_id in visited:
                return
            
            temp_visited.add(task_id)
            
            # 依存関係を処理
            task = self.tasks[task_id]
            for dep_task_id in task.get_dependencies():
                if dep_task_id in self.tasks:
                    dfs(dep_task_id)
            
            temp_visited.remove(task_id)
            visited.add(task_id)
            execution_order.append(task_id)
        
        # 全てのタスクを処理
        for task_id in self.tasks:
            if task_id not in visited:
                dfs(task_id)
        
        return execution_order
    
    def execute(self) -> Any:
        """ワークフローを実行"""
        from graflow.core.workflow import WorkflowContext
        
        # 既存のWorkflowContextを利用
        with WorkflowContext(self.name) as ctx:
            execution_order = self.build_execution_plan()
            
            # タスクを実行順序に従って実行
            last_result = None
            for task_id in execution_order:
                task = self.tasks[task_id]
                result = task.execute_with_context(ctx)
                last_result = result
            
            return last_result
```

#### 3. @workflowデコレータ（`graflow/core/decorators.py`への追加）
```python
def workflow(func_or_name=None):
    """TaskFlow API用のworkflowデコレータ"""
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> WorkflowBuilder:
            # ワークフロー名を決定
            workflow_name = func.__name__
            if isinstance(func_or_name, str):
                workflow_name = func_or_name
            
            builder = WorkflowBuilder(workflow_name)
            
            # 関数内のローカル変数をキャプチャするためのフレームハック
            import inspect
            frame = inspect.currentframe()
            
            # taskデコレータをオーバーライド
            original_task = globals().get('task')
            
            def taskflow_task(func_or_name=None):
                def task_decorator(task_func: Callable) -> TaskFlowWrapper:
                    task_id = task_func.__name__
                    if isinstance(func_or_name, str):
                        task_id = func_or_name
                    
                    wrapper = TaskFlowWrapper(task_func, task_id)
                    builder.add_task(wrapper)
                    return wrapper
                
                if callable(func_or_name):
                    return task_decorator(func_or_name)
                return task_decorator
            
            # 一時的にtaskデコレータを置き換え
            globals()['task'] = taskflow_task
            
            try:
                # ワークフロー関数を実行
                result = func(*args, **kwargs)
                
                # 最終的な出力を設定
                if isinstance(result, ChannelReference):
                    builder.final_output = result
                
                return builder
                
            finally:
                # 元のtaskデコレータを復元
                globals()['task'] = original_task
        
        return wrapper
    
    if callable(func_or_name):
        return decorator(func_or_name)
    return decorator
```

#### 4. 使用例の実行フロー
```python
@workflow
def data_pipeline():
    @task
    def extract() -> dict:
        return {"data": [1, 2, 3], "source": "database"}
    
    @task  
    def transform(data: dict) -> dict:
        return {"processed": [x * 2 for x in data["data"]]}
    
    @task
    def load(data: dict) -> str:
        print(f"Loading: {data}")
        return "complete"
    
    # 実行時の処理：
    raw_data = extract()        # ChannelReference(task_id="extract")を返す
    processed = transform(raw_data)  # 依存関係：extract → transform
    result = load(processed)    # 依存関係：transform → load
    
    return result

# ワークフロー実行
workflow_builder = data_pipeline()
final_result = workflow_builder.execute()
```

#### 5. 実行時の内部処理フロー
1. **ワークフロー定義時**：
   - `@workflow`で関数をラップ
   - 内部の`@task`でTaskFlowWrapperを作成
   - 関数呼び出しでChannelReferenceを生成・依存関係を記録

2. **実行時**：
   - `build_execution_plan()`で依存関係をトポロジカルソート
   - 順序通りにタスクを実行
   - 各タスクでChannelReferenceを実際の値に解決
   - 結果をChannelに保存

3. **エラーハンドリング**：
   - 循環依存の検出
   - 未定義タスクへの参照チェック
   - Channel内データの存在確認



### オプション 4: コンテキストマネージャー型

```python
with taskflow_context("my_workflow") as ctx:
    @task
    def extract():
        return {"data": [1, 2, 3]}
    
    @task  
    def transform(data):
        return {"processed": data["data"]}
    
    # 関数呼び出しでデータフローを定義
    data = extract()
    result = transform(data)
    
    ctx.set_output(result)

# 実行
ctx.execute()
```

**特徴:**
- ✅ 明示的なコンテキスト管理
- ✅ スコープの明確化
- ⚠️ 追加のボイラープレート

**実装詳細:**

#### ディレクトリ構造
```
graflow/
├── core/
│   ├── decorators.py        # 既存の@taskデコレータ（変更なし）
│   ├── workflow.py          # 既存のWorkflowContext（変更なし）
│   ├── taskflow_context.py  # 新規: taskflow_context実装
│   └── channel_ref.py       # 新規: ChannelReference実装（オプション1と共通）
```

#### 1. TaskFlowContextクラス（`graflow/core/taskflow_context.py`）
```python
from typing import Any, Dict, List, Optional, Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from graflow.core.workflow import WorkflowContext
from graflow.core.channel_ref import ChannelReference
from graflow.core.task import TaskWrapper

@dataclass
class TaskFlowContext:
    """TaskFlow実行コンテキスト"""
    name: str
    tasks: Dict[str, 'TaskFlowTask'] = field(default_factory=dict)
    call_order: List[str] = field(default_factory=list)
    final_output: Optional[ChannelReference] = None
    _workflow_context: Optional[WorkflowContext] = None
    
    def __enter__(self):
        """コンテキストマネージャーの開始"""
        self._setup_task_decorator()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """コンテキストマネージャーの終了"""
        self._restore_task_decorator()
        return False
    
    def _setup_task_decorator(self):
        """taskデコレータを一時的に置き換え"""
        import graflow.core.decorators as decorators
        
        # 元のtaskデコレータを保存
        self._original_task = decorators.task
        
        # 新しいtaskデコレータを設定
        def context_task(func_or_name=None):
            def task_decorator(task_func: Callable) -> 'TaskFlowTask':
                task_id = task_func.__name__
                if isinstance(func_or_name, str):
                    task_id = func_or_name
                
                task_flow_task = TaskFlowTask(task_func, task_id, self)
                self.tasks[task_id] = task_flow_task
                return task_flow_task
            
            if callable(func_or_name):
                return task_decorator(func_or_name)
            return task_decorator
        
        decorators.task = context_task
    
    def _restore_task_decorator(self):
        """元のtaskデコレータを復元"""
        import graflow.core.decorators as decorators
        decorators.task = self._original_task
    
    def set_output(self, output: ChannelReference):
        """最終出力を設定"""
        self.final_output = output
    
    def execute(self) -> Any:
        """ワークフローを実行"""
        # 実行順序を決定（トポロジカルソート）
        execution_order = self._build_execution_plan()
        
        # 既存のWorkflowContextを利用して実行
        with WorkflowContext(self.name) as ctx:
            self._workflow_context = ctx
            
            # タスクを順次実行
            results = {}
            for task_id in execution_order:
                task = self.tasks[task_id]
                result = task.execute_with_context(ctx)
                results[task_id] = result
            
            # 最終出力を返す
            if self.final_output:
                return self.final_output.resolve(ctx)
            
            # 最後に実行されたタスクの結果を返す
            if execution_order:
                return results[execution_order[-1]]
            
            return None
    
    def _build_execution_plan(self) -> List[str]:
        """依存関係に基づく実行順序を構築"""
        # トポロジカルソート実装
        visited = set()
        temp_visited = set()
        execution_order = []
        
        def dfs(task_id: str):
            if task_id in temp_visited:
                raise ValueError(f"Circular dependency detected: {task_id}")
            if task_id in visited:
                return
            
            temp_visited.add(task_id)
            task = self.tasks[task_id]
            
            # 依存タスクを先に処理
            for dep_task_id in task.get_dependencies():
                if dep_task_id in self.tasks:
                    dfs(dep_task_id)
            
            temp_visited.remove(task_id)
            visited.add(task_id)
            execution_order.append(task_id)
        
        # 全タスクを処理
        for task_id in self.tasks:
            if task_id not in visited:
                dfs(task_id)
        
        return execution_order

class TaskFlowTask:
    """TaskFlowContext内でのタスク実装"""
    
    def __init__(self, func: Callable, task_id: str, context: TaskFlowContext):
        self.func = func
        self.task_id = task_id
        self.context = context
        self._stored_args: List[Any] = []
        self._stored_kwargs: Dict[str, Any] = {}
        self._called = False
    
    def __call__(self, *args, **kwargs) -> ChannelReference:
        """タスク呼び出し時の処理"""
        if self._called:
            raise RuntimeError(f"Task '{self.task_id}' has already been called")
        
        # 引数を保存
        self._stored_args = list(args)
        self._stored_kwargs = dict(kwargs)
        self._called = True
        
        # 呼び出し順序を記録
        self.context.call_order.append(self.task_id)
        
        # ChannelReferenceを返す
        return ChannelReference(task_id=self.task_id)
    
    def get_dependencies(self) -> List[str]:
        """依存関係を解析"""
        dependencies = []
        for arg in self._stored_args + list(self._stored_kwargs.values()):
            if isinstance(arg, ChannelReference):
                dependencies.append(arg.task_id)
        return dependencies
    
    def execute_with_context(self, workflow_ctx: WorkflowContext) -> Any:
        """実際のタスク実行"""
        # 引数のChannelReferenceを解決
        resolved_args = []
        for arg in self._stored_args:
            if isinstance(arg, ChannelReference):
                resolved_value = arg.resolve(workflow_ctx)
                resolved_args.append(resolved_value)
            else:
                resolved_args.append(arg)
        
        resolved_kwargs = {}
        for key, value in self._stored_kwargs.items():
            if isinstance(value, ChannelReference):
                resolved_value = value.resolve(workflow_ctx)
                resolved_kwargs[key] = resolved_value
            else:
                resolved_kwargs[key] = value
        
        # 関数実行
        result = self.func(*resolved_args, **resolved_kwargs)
        
        # 結果をChannelに保存
        channel = workflow_ctx.get_channel()
        channel_key = f"{self.task_id}.return_value"
        channel.set(channel_key, result)
        
        return result

@contextmanager
def taskflow_context(name: str):
    """TaskFlowコンテキストマネージャー"""
    context = TaskFlowContext(name)
    yield context
```

#### 2. 使用例の実行フロー
```python
# 基本的な使用例
with taskflow_context("data_pipeline") as ctx:
    @task
    def extract():
        print("Extracting data...")
        return {"data": [1, 2, 3], "source": "database"}
    
    @task
    def transform(data):
        print(f"Transforming: {data}")
        return {"processed": [x * 2 for x in data["data"]]}
    
    @task
    def load(data):
        print(f"Loading: {data}")
        return "complete"
    
    # データフローの定義
    raw_data = extract()         # ChannelReference(task_id="extract")
    processed = transform(raw_data)  # 依存関係: extract → transform
    result = load(processed)     # 依存関係: transform → load
    
    # 最終出力を設定
    ctx.set_output(result)

# 実行
final_result = ctx.execute()
print(f"Pipeline result: {final_result}")
```

#### 3. 複雑な依存関係の例
```python
with taskflow_context("complex_pipeline") as ctx:
    @task
    def fetch_users():
        return [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]
    
    @task
    def fetch_orders():
        return [{"user_id": 1, "amount": 100}, {"user_id": 2, "amount": 200}]
    
    @task
    def join_data(users, orders):
        # ユーザーと注文データを結合
        result = []
        for user in users:
            user_orders = [o for o in orders if o["user_id"] == user["id"]]
            result.append({
                "user": user,
                "orders": user_orders,
                "total": sum(o["amount"] for o in user_orders)
            })
        return result
    
    @task
    def generate_report(joined_data):
        return {
            "total_users": len(joined_data),
            "total_revenue": sum(item["total"] for item in joined_data),
            "data": joined_data
        }
    
    # 並列実行される独立タスク
    users = fetch_users()
    orders = fetch_orders()
    
    # 両方の結果に依存するタスク
    joined = join_data(users, orders)
    report = generate_report(joined)
    
    ctx.set_output(report)

# 実行（fetch_users と fetch_orders は並列実行される）
result = ctx.execute()
```

#### 4. エラーハンドリングと検証
```python
with taskflow_context("validated_pipeline") as ctx:
    @task
    def risky_operation():
        # 失敗する可能性のある処理
        import random
        if random.random() < 0.3:
            raise ValueError("Random failure occurred")
        return {"success": True, "data": "important_result"}
    
    @task
    def validate_result(data):
        if not data.get("success"):
            raise ValueError("Validation failed")
        return data
    
    @task
    def safe_fallback():
        return {"success": True, "data": "fallback_result"}
    
    try:
        result = risky_operation()
        validated = validate_result(result)
        ctx.set_output(validated)
    except Exception as e:
        print(f"Error occurred: {e}, using fallback")
        fallback = safe_fallback()
        ctx.set_output(fallback)

# エラーハンドリング付きで実行
try:
    result = ctx.execute()
    print(f"Success: {result}")
except Exception as e:
    print(f"Pipeline failed: {e}")
```

#### 5. 実行時の内部処理フロー
1. **コンテキスト開始**：
   - `taskflow_context()`でTaskFlowContextを作成
   - `__enter__()`で`@task`デコレータを一時的に置き換え

2. **タスク定義**：
   - `@task`でTaskFlowTaskインスタンスを作成
   - コンテキストのtasksディクショナリに登録

3. **データフロー定義**：
   - タスク呼び出しで引数を保存、ChannelReferenceを返す
   - 依存関係が自動的に記録される

4. **実行**：
   - `ctx.execute()`で依存関係をトポロジカルソート
   - 既存のWorkflowContextを利用してタスクを順次実行
   - ChannelReferenceを実際の値に解決してタスクを実行

5. **コンテキスト終了**：
   - `__exit__()`で元の`@task`デコレータを復元

#### 6. オプション1との比較

| 特徴 | オプション1（@workflow） | オプション4（context manager） |
|------|------------------------|--------------------------------|
| シンタックス | `@workflow` + 関数 | `with taskflow_context() as ctx:` |
| スコープ管理 | 関数スコープ | with文によるスコープ |
| エラーハンドリング | try/except外で処理 | with文内でtry/except可能 |
| 動的制御 | 制限的 | 条件分岐・ループが自然 |
| ボイラープレート | 少ない | やや多い |
| 実行制御 | builder.execute() | ctx.execute() |
| 直感性 | より関数的 | より手続き的 |

## 技術実装詳細

### ChannelReference システム

```python
class ChannelReference:
    """Channelに保存されたデータへの参照"""
    
    def __init__(self, task_id: str, key: str = "return_value"):
        self.task_id = task_id
        self.key = key
    
    @property
    def channel_key(self) -> str:
        return f"{self.task_id}.{self.key}"
    
    def resolve(self, context: TaskExecutionContext) -> Any:
        """Channelからデータを解決"""
        channel = context.get_channel()
        return channel.get(self.channel_key)
```

### タスク拡張システム

```python
def _create_enhanced_task(self, original_task: TaskWrapper) -> TaskWrapper:
    """Channel解決機能付きタスクを作成"""
    
    def enhanced_func(context: TaskExecutionContext = None):
        # 引数のChannelReferenceを実際の値に解決
        resolved_args = []
        for arg in stored_args:
            if isinstance(arg, ChannelReference):
                resolved_value = arg.resolve(context)
                resolved_args.append(resolved_value)
            else:
                resolved_args.append(arg)
        
        # 元の関数を実行
        result = original_task.func(*resolved_args)
        
        # 結果をChannelに保存
        channel = context.get_channel()
        channel_key = f"{original_task.task_id}.return_value"
        channel.set(channel_key, result)
        
        return result
```

### TypedChannel 統合

```python
class TypedChannelReference(ChannelReference, Generic[T]):
    """型安全なChannel参照"""
    
    def __init__(self, task_id: str, message_type: Type[T], key: str = "return_value"):
        super().__init__(task_id, key)
        self.message_type = message_type
    
    def resolve(self, context: TaskExecutionContext) -> T:
        """型安全なデータ解決"""
        typed_channel = context.get_typed_channel(self.message_type)
        return typed_channel.receive(self.channel_key)
```

## 使用例比較

### 従来のgraflow（継続サポート）
```python
with workflow("data_pipeline") as wf:
    @task
    def extract():
        return {"data": [1, 2, 3]}
    
    @task
    def transform(data):
        return {"processed": data["data"]}
    
    # 明示的な依存関係定義（従来通り）
    extract >> transform
    
    wf.execute()
```

### 新しいTaskFlow API
```python
@workflow
def data_pipeline():
    @task
    def extract():
        return {"data": [1, 2, 3]}
    
    @task
    def transform(data):
        return {"processed": data["data"]}
    
    # 自然な関数呼び出し
    data = extract()
    result = transform(data)
    return result

# 実行
workflow = data_pipeline()
workflow.execute()
```

### ハイブリッド利用（新機能）
```python
@workflow
def hybrid_pipeline():
    @task
    def extract():
        return {"data": [1, 2, 3]}
    
    @task
    def transform(data):
        return {"processed": data["data"]}
    
    @task
    def validate(data):
        return {"valid": True, **data}
    
    @task
    def save(data):
        print(f"Saving: {data}")
        return "saved"
    
    # TaskFlow風の関数呼び出し
    data = extract()
    processed = transform(data)
    
    # 従来の>>オペレータも併用可能
    validated = validate(processed)
    validate >> save  # 明示的な依存関係も使える
    
    return validated

# 実行
workflow = hybrid_pipeline()
workflow.execute()
```

## 推奨実装

**オプション1（ChannelReference ベース）** を推奨します。

**理由:**
1. **自然性**: Pythonの関数呼び出しと同じシンタックス
2. **互換性**: 既存のChannelシステムを最大限活用
3. **拡張性**: TypedChannel統合も容易
4. **学習コスト**: Airflow経験者には馴染みやすい

## 実装フェーズ

### フェーズ1: 基本実装
- ChannelReferenceクラス
- `@workflow`デコレータを`graflow/core/decorators.py`に実装
- TaskFlowWrapper
- 基本的な依存関係解析

### フェーズ2: 型安全性
- TypedChannelReference
- TypedChannel統合
- 型検証機能

### フェーズ3: 高度な機能
- 条件分岐
- 並列実行最適化
- エラーハンドリング強化

## パフォーマンス考慮事項

### メモリ使用量
- Channelに保存されるデータサイズ
- 大きなデータの場合はファイルベースChannel検討

### 実行効率
- Channel解決オーバーヘッド
- 依存関係解析の最適化

### スケーラビリティ
- 多数のタスクでの依存関係管理
- Channel容量制限

## セキュリティ考慮事項

- Channel内データの暗号化
- タスク間データアクセス制御
- 機密データの自動削除

## 互換性

### 既存システムとの互換性
- **完全後方互換性**: 既存の`workflow`コンテキストと`>>`オペレータは変更なし
- **段階的移行**: 新しいTaskFlow APIと従来APIの共存
- **ハイブリッド利用**: 同一ワークフロー内での両方のシンタックス使用可能

### 移行戦略
1. **Phase 1**: 新しいTaskFlow APIの導入（既存APIは無変更）
2. **Phase 2**: ドキュメントと例での新API推奨
3. **Phase 3**: 新機能は新APIで優先的に実装
4. **Long-term**: 既存APIは保守モード（削除せず、新機能追加なし）

### 将来拡張性
- 分散実行への対応
- 外部システム連携
- より高度な型安全性

## ハイブリッドアプローチの利点

### 1. 学習曲線の緩和
```python
# 既存ユーザーは従来の方法を継続使用可能
with workflow("my_workflow") as wf:
    @task
    def task_a():
        return "data"
    
    @task  
    def task_b(data):
        return f"processed_{data}"
    
    task_a >> task_b  # 慣れ親しんだシンタックス
    wf.execute()

# 新規ユーザーは直感的なTaskFlow APIを使用
@workflow
def my_workflow():
    @task
    def task_a():
        return "data"
    
    @task
    def task_b(data):
        return f"processed_{data}"
    
    data = task_a()       # より自然
    result = task_b(data)
    return result
```

### 2. 機能の段階的導入
```python
@workflow
def gradual_migration():
    # 新しいTaskFlow機能
    @task
    def extract():
        return {"data": [1, 2, 3]}
    
    @task
    def transform(data):
        return {"processed": data["data"]}
    
    # 複雑な部分は従来の明示的方法
    @task
    def complex_validation(data):
        return {"validated": True, **data}
    
    @task
    def save_to_db(data):
        return "saved"
    
    # ハイブリッド使用
    data = extract()
    processed = transform(data)
    
    # 複雑な依存関係は明示的に
    validated = complex_validation(processed)
    complex_validation >> save_to_db
    
    return validated
```

### 3. エコシステムの分断回避
- 既存のワークフローは無変更で動作
- 新機能は段階的に採用可能
- ライブラリとの互換性維持

## 実装考慮事項

### TaskFlowとworkflowコンテキストの統合
```python
# graflow/core/workflow.py の既存のWorkflowContextをそのまま利用
# TaskFlow機能は以下の方式で統合:

class TaskFlowMixin:
    """TaskFlow機能をWorkflowContextに追加するミックスイン"""
    
    def __init__(self):
        self.channel_references = {}
        self.auto_dependencies = True
    
    def enable_taskflow_mode(self):
        """TaskFlow機能を有効化"""
        self.auto_dependencies = True
    
    def disable_taskflow_mode(self):
        """従来の明示的依存関係モードに戻す"""
        self.auto_dependencies = False

# 既存のWorkflowContextを拡張せず、デコレータレベルで機能を追加
```

### デコレータの統合
```python
# graflow/core/decorators.py で既存の@taskデコレータをそのまま利用
# @workflowデコレータを新規追加

def task(*args, **kwargs):
    """既存のtaskデコレータ（変更なし）"""
    # 現在の実装をそのまま維持
    pass

def workflow(func_or_name=None):
    """新しいworkflowデコレータ（TaskFlow API用）"""
    # TaskFlow機能を提供する新しいデコレータ
    # 既存のWorkflowContextを内部で利用
    pass
```

## コンテキスト検出によるデバッグサポート

### 設計指針

TaskFlowWrapper.__call__メソッドでは、実行時にworkflowコンテキストの有無を検出し、実行方式を動的に切り替えます。

#### 実行方式の切り替え

1. **Workflowコンテキスト内での実行**：
   - `@workflow` デコレータ内で実行されている場合
   - ChannelReferenceを返し、遅延実行される
   - 依存関係が自動的に構築される

2. **Workflowコンテキスト外での直接実行**：
   - 通常のPython関数として直接呼び出された場合
   - 即座に関数を実行し、結果を直接返す
   - デバッグやテストに最適

### 実装例と動作確認

```python
# graflow/core/workflow.py のコンテキスト変数を利用
from graflow.core.workflow import _current_context

@task
def my_task(data: str) -> str:
    return f"Processed: {data}"

# ケース1: Workflowコンテキスト外（デバッグモード）
result = my_task("test_data")
print(result)  # "Processed: test_data" - 直接実行される

# ケース2: Workflowコンテキスト内（TaskFlowモード）
@workflow
def my_workflow():
    @task
    def extract():
        return "raw_data"
    
    @task
    def transform(data):
        return f"Processed: {data}"
    
    # ここではChannelReferenceが返される
    data = extract()  # ChannelReference(task_id="extract")
    result = transform(data)  # ChannelReference(task_id="transform")
    return result

# ワークフロー実行
workflow_builder = my_workflow()
final_result = workflow_builder.execute()  # 実際のデータ処理が実行される
```

### デバッグフレンドリーな設計

#### 1. 個別タスクのテスト
```python
@task
def complex_calculation(x: int, y: int) -> int:
    # 複雑な計算ロジック
    result = x * y + x ** 2 - y
    return result

# Workflowコンテキスト外で直接テスト可能
test_result = complex_calculation(5, 3)
assert test_result == 37  # 5*3 + 5^2 - 3 = 15 + 25 - 3 = 37
```

#### 2. 段階的デバッグ
```python
@workflow
def debug_friendly_workflow():
    @task
    def step1():
        return [1, 2, 3, 4, 5]
    
    @task
    def step2(numbers):
        # この部分で問題が発生していると仮定
        return [x * 2 for x in numbers if x % 2 == 0]
    
    @task
    def step3(filtered_numbers):
        return sum(filtered_numbers)
    
    data = step1()
    filtered = step2(data)
    result = step3(filtered)
    return result

# デバッグ: 個別ステップの動作確認
test_data = [1, 2, 3, 4, 5]
debug_result = step2(test_data)  # 直接実行してデバッグ
print(debug_result)  # [4, 8] - 期待通りの結果

# 本番: ワークフロー全体の実行
workflow = debug_friendly_workflow()
final_result = workflow.execute()
```

### コンテキスト検出の技術詳細

#### contextvars の活用
```python
# graflow/core/workflow.py 内
import contextvars

_current_context: contextvars.ContextVar[Optional[WorkflowContext]] = contextvars.ContextVar(
    'current_workflow', default=None
)

class WorkflowContext:
    def __enter__(self):
        # 現在のコンテキストを設定
        self._previous_context = _current_context.get()
        _current_context.set(self)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # 前のコンテキストを復元
        _current_context.set(self._previous_context)
```

#### TaskFlowWrapper の実装詳細
```python
class TaskFlowWrapper(TaskWrapper):
    def __call__(self, *args, **kwargs) -> Any:
        """コンテキストを検出して実行方式を決定"""
        from graflow.core.workflow import _current_context
        current_workflow_context = _current_context.get()
        
        if current_workflow_context is not None:
            # Workflowコンテキスト内: TaskFlow実行モード
            return self._handle_taskflow_execution(*args, **kwargs)
        else:
            # Workflowコンテキスト外: 直接実行モード
            return self._handle_direct_execution(*args, **kwargs)
    
    def _handle_taskflow_execution(self, *args, **kwargs) -> ChannelReference:
        """TaskFlow実行モード: 引数を保存してChannelReferenceを返す"""
        if not self._is_callable:
            raise RuntimeError(f"Task '{self.task_id}' has already been called")
        
        self._stored_args = list(args)
        self._stored_kwargs = dict(kwargs)
        self._is_callable = False
        
        return ChannelReference(task_id=self.task_id)
    
    def _handle_direct_execution(self, *args, **kwargs) -> Any:
        """直接実行モード: 即座に関数を実行"""
        # 通常の関数呼び出しとして実行
        return self.func(*args, **kwargs)
```

### 利点

1. **開発効率の向上**: 個別タスクを独立してテスト・デバッグ可能
2. **学習コストの削減**: 通常のPython関数として動作するため直感的
3. **段階的デバッグ**: ワークフロー全体ではなく問題のある部分のみフォーカス可能
4. **テスタビリティ**: unit testが書きやすい

### 注意事項

- Workflowコンテキスト外での直接実行では、Channelによるデータ交換は行われない
- 依存関係は構築されないため、タスク間の連携は期待できない
- コンテキスト検出のオーバーヘッドは最小限に抑える必要がある

## 結論

**ハイブリッドアプローチによるChannel/TypedChannelベースのTaskFlow API**により、graflowユーザーは：

### 即座に享受できる利益
1. **より簡潔で直感的な**ワークフロー定義が可能（新規ユーザー向け）
2. **既存投資の保護**（既存ワークフローは無変更で継続動作）
3. **柔軟な移行戦略**（チームのペースで新APIを導入可能）

### 長期的な価値
4. **型安全な**データ交換を享受
5. **既存の豊富な機能**を活用継続
6. **段階的な移行**でリスクを最小化
7. **エコシステムの分断回避**

この設計により、graflowは：
- **既存ユーザーの満足度を維持**しながら
- **新規ユーザーの獲得を促進**し
- **モダンなワークフローオーケストレーションツール**としての地位を確立できます

### 推奨実装順序
1. **Phase 1**: ChannelReferenceベースのTaskFlow API実装
   - `@workflow`デコレータを`graflow/core/decorators.py`に追加
2. **Phase 2**: 既存workflowコンテキストとの統合
3. **Phase 3**: ハイブリッド利用のサポート強化
4. **Phase 4**: TypedChannelReference による型安全性向上

この段階的アプローチにより、リスクを最小化しながら価値を最大化できます。

## 実装場所

### デコレータ実装
- **場所**: `graflow/core/decorators.py`
- **追加内容**: 
  - `@workflow`デコレータ（TaskFlow API用）
  - 既存の`@task`デコレータはそのまま維持
  - ChannelReferenceクラス
  - TaskFlowWrapper

### WorkflowContext利用方針
- **既存コード**: `graflow/core/workflow.py`の`WorkflowContext`をそのまま利用
- **変更方針**: 既存のWorkflowContextクラスには変更を加えない
- **統合方法**: `@workflow`デコレータ内で既存のWorkflowContextを利用