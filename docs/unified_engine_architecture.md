# Graflow統一Engine実行アーキテクチャ設計書

## 概要

Graflowは**統一WorkflowEngine実行モデル**により、ローカル/分散環境を問わず一貫した実行体験を提供する。全ての実行パス（Main、TaskWorker、GroupExecutor）が同一の`WorkflowEngine.execute()`を使用し、**起点ノード指定のみで実行範囲を制御**する。

**設計思想**: Master/Slave関係を排除し、各実行ノードが対等な立場でWorkflowEngineを実行する分散対等アーキテクチャ。

## 基本設計原則

### 1. **単一実行エンジン**
- 全ての実行パスが`WorkflowEngine.execute(context, start_task_id)`を使用
- 実行ロジックの完全統一により、一貫した動作保証

### 2. **起点ノード制御**
- `start_task_id: Optional[str] = None`パラメータで実行開始点を指定
- 同一エンジン、異なる起点による柔軟な実行制御

### 3. **対等ノード実行**
- Master/Slave関係なし、各ノードが独立してエンジン実行
- Channel基盤による透過的状態共有で協調動作

### 4. **自動依存関係処理**
- Successor（依存先タスク）の自動キューイングと実行継続
- グラフ構造に基づく自然なワークフロー進行

## アーキテクチャ図

```
【統一Engine対等ノードアーキテクチャ】

 ┌─────────────────────────────────────────────────────────────┐
 │                  WorkflowEngine.execute()                   │
 │              (context, start_task_id=?)                     │
 │  ┌─────────────────────────────────────────────────────┐    │
 │  │  統一実行ロジック                                   │    │
 │  │  • Task実行 + Context管理                          │    │
 │  │  • Successor自動処理                                │    │
 │  │  • Step increment                                  │    │
 │  │  • Result storage via Channel                      │    │
 │  └─────────────────────────────────────────────────────┘    │
 └─────────────────────────────────────────────────────────────┘
                               ▲
                    ┌──────────┼──────────┐
              ┌───────────┐ ┌───────────┐ ┌───────────┐
              │   Node1   │ │   Node2   │ │   Node3   │
              │   Main    │ │TaskWorker │ │TaskWorker │
              │ 全体制御  │ │task_A実行 │ │task_B実行 │
              └───────────┘ └───────────┘ └───────────┘
                     │           │           │
                     └───────────┼───────────┘
                                 │
                    ┌─────────────────────────┐
                    │      Channel基盤        │
                    │  ┌─────────┐ ┌─────────┐ │
                    │  │ Memory  │ │  Redis  │ │
                    │  │Channel  │ │Channel  │ │
                    │  │(ローカル)│ │(分散)   │ │
                    │  └─────────┘ └─────────┘ │
                    │     透過的状態共有        │
                    └─────────────────────────┘
```

## 実行パターンと動作例

### **パターン1: Main Node全体制御**
```python
# ワークフロー全体を起点から実行
context = ExecutionContext.create(graph, start_node="start")
engine = WorkflowEngine()
engine.execute(context)  # start_task_id=None → context.start_node使用
```

### **パターン2: TaskWorker特定タスク実行**
```python
# 特定タスクから実行開始、successorも自動処理
def _process_task_wrapper(self, task_spec: TaskSpec):
    engine = WorkflowEngine()
    engine.execute(task_spec.execution_context, start_task_id=task_spec.task_id)
    # → task_spec.task_id実行 → その後successorを自動キューイング・実行
```

### **パターン3: GroupExecutor並列グループ実行**
```python
# 並列グループの各タスクを統一Engine経由で実行
def direct_execute(self, group_id: str, tasks: List['Executable'], context: ExecutionContext):
    engine = WorkflowEngine()
    for task in tasks:
        engine.execute(context, start_task_id=task.task_id)
        # → 各task実行 → successorも自動処理
```

## 統一実行ロジック詳細

### **engine.execute()の動作**

```python
def execute(self, context: ExecutionContext, start_task_id: Optional[str] = None) -> None:
    # 1. 起点ノード決定
    if start_task_id is not None:
        task_id = start_task_id          # 指定タスクから開始
    else:
        task_id = context.get_next_task() # キューから取得

    # 2. 実行ループ（successor自動処理含む）
    while task_id is not None and not context.is_completed():
        # タスク実行
        task = context.graph.get_node(task_id)
        with context.executing_task(task) as _:
            result = task.run()
            context.set_result(task_id, result)  # Channel経由で状態共有

        context.increment_step()

        # 3. Successor自動処理
        if start_task_id is not None:
            break  # 単一タスクモード: 1タスクのみ実行
        else:
            # ワークフローモード: successorを自動キューイング
            for succ in graph.successors(task_id):
                context.add_to_queue(graph.get_node(succ))
            task_id = context.get_next_task()
```

### **Channel基盤状態共有**

```python
# 実行結果の透過的共有
context.set_result(task_id, result)
# ↓
# ローカル: MemoryChannel → 即座に同一オブジェクト更新
# 分散: RedisChannel → Redis経由で他ノードにも自動同期

# 他ノードからの結果取得
result = context.get_result("some_task_id")
# ↓ 実行環境に関係なく透過的にアクセス
```

## 分散実行での動作シナリオ

### **シナリオ: 3ノード分散並列実行**

```python
# === Node1: Main Coordinator ===
graph = TaskGraph()
# task_A | task_B | task_C → task_D (A,B,C並列後、Dが実行)

context = ExecutionContext.create(
    graph, "start",
    channel_backend="redis",  # 分散用RedisChannel
    queue_backend="redis"
)
engine.execute(context)
# → start実行 → A,B,Cを並列キューに投入

# === Node2: TaskWorker ===
# Redis queueからtask_A取得
engine.execute(context, start_task_id="task_A")
# → task_A実行 → 結果をRedisChannel保存
# → A完了、Dの依存条件チェック（B,C待ち）

# === Node3: TaskWorker ===
# Redis queueからtask_B取得
engine.execute(context, start_task_id="task_B")
# → task_B実行 → 結果をRedisChannel保存
# → B完了、Dの依存条件チェック（C待ち）

# === Node1 or Node2/3: ===
# Redis queueからtask_C取得
engine.execute(context, start_task_id="task_C")
# → task_C実行 → 結果をRedisChannel保存
# → C完了、A,B,C全完了でDがキューに自動投入

# === 任意のNode: ===
# Redis queueからtask_D取得
engine.execute(context, start_task_id="task_D")
# → task_D実行 → ワークフロー完了
```

**重要**: どのノードも同じ`engine.execute()`を使用、RedisChannel経由で状態共有、起点ノードが違うのみ。

## 薄いラッパー実装

### **TaskWorker: Engine委譲**
```python
class TaskWorker:
    def _process_task_wrapper(self, task_spec: TaskSpec) -> Dict[str, Any]:
        """統一Engine実行への薄いラッパー"""
        engine = WorkflowEngine()
        engine.execute(task_spec.execution_context, start_task_id=task_spec.task_id)
        # Engine内でcontext.set_result()済み、Channel経由で状態共有完了
        return {"success": True, "task_id": task_spec.task_id}
```

### **GroupExecutor: Engine委譲**
```python
class GroupExecutor:
    def direct_execute(self, group_id: str, tasks: List['Executable'], context: ExecutionContext):
        """統一Engine実行への薄いラッパー"""
        engine = WorkflowEngine()
        for task in tasks:
            engine.execute(context, start_task_id=task.task_id)
            # 各taskとそのsuccessorを統一Engine経由で実行
```

## 設計効果・メリット

### 1. **実行パス統一**
- 3つの分岐パス（Engine/GroupExecutor/TaskWorker） → 1つの統一パス
- 一元的な実行ロジック改善で全パス同時向上

### 2. **Master/Slave排除**
- 対等ノード実行による負荷分散と冗長性向上
- 単一障害点（SPOF）の除去

### 3. **透過的分散実行**
- Channel基盤により、ローカル/分散の違いをアプリケーションレベルで意識不要
- 同一コードでスケーラブルな分散実行

### 4. **依存関係自動処理**
- グラフ構造に基づく自動successor処理
- 明示的な依存関係管理が不要

### 5. **開発・保守性向上**
- 実行ロジック一元化によるデバッグ・テスト簡素化
- 機能追加時の影響範囲限定

## 実行制御パターン

### **制御1: 全ワークフロー実行**
```python
engine.execute(context)  # start_task_id=None
# → context.start_nodeから開始、全依存関係を自動実行
```

### **制御2: 特定タスクのみ実行**
```python
engine.execute(context, start_task_id="task_X")  # 単一タスクモード
# → task_X実行のみ、successorは実行しない
```

### **制御3: 特定タスクから継続実行**
```python
# context.start_node = "task_X"に設定後
engine.execute(context)  # start_task_id=None
# → task_Xから開始、その後successorも自動実行
```

## 実装ガイドライン

### **新機能開発時**
1. WorkflowEngineの実行ロジック拡張を検討
2. Channel基盤での状態共有を活用
3. 薄いラッパーとしての委譲実装

### **デバッグ・トラブルシューティング**
1. WorkflowEngine.execute()の動作ログ確認
2. Channel基盤での状態共有状況確認
3. グラフ依存関係の妥当性検証

### **パフォーマンス最適化**
1. Channel基盤の効率的利用
2. 不要なタスク実行の回避
3. 並列実行度の調整

## まとめ

**Graflow統一Engine実行アーキテクチャ**は、Channel基盤による透過的状態共有と単一WorkflowEngineによる統一実行ロジックにより、Master/Slave関係のない対等ノード分散実行を実現する。

起点ノード指定による柔軟な実行制御と自動依存関係処理により、シンプルかつ強力なワークフロー実行環境を提供し、ローカル開発から大規模分散実行まで一貫したエクスペリエンスを実現する。

---

*本設計書は、実装完了後の最終アーキテクチャを記述したものである。Channel基盤の既存機能を最大限活用し、実行システムの統一と簡素化を達成した。*