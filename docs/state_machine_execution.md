# Graflowステートマシン実行モデル

## 概要

Graflowは**ステートマシンに基づくタスク実行モデル**を採用し、タスクグラフの各ノードを状態遷移させながら実行する。WorkflowEngineが統一実行エンジンとして、ExecutionContextと協調してタスクの状態管理と実行制御を行う。

**設計思想**:
- **単一ステートマシン**: 全タスクがExecutionContext内で一元管理される状態を持つ
- **明示的状態遷移**: タスクの状態（pending → executing → completed）が明確に追跡される
- **統一実行ループ**: WorkflowEngine.execute()による単一の実行フロー

## ステートマシン状態遷移図

```
【タスクライフサイクル】

    ┌─────────────┐
    │   PENDING   │  ← 初期状態（キューに追加済み）
    │  (待機中)   │
    └──────┬──────┘
           │ get_next_task()
           │ context.executing_task(task)
           ▼
    ┌─────────────┐
    │  EXECUTING  │  ← 実行中
    │  (実行中)   │     • handler.execute_task()
    └──────┬──────┘     • task.run()
           │            • set_result()
           │
           ▼
    ┌─────────────┐
    │  COMPLETED  │  ← 完了状態
    │  (完了)     │     • mark_task_completed()
    └──────┬──────┘     • increment_step()
           │
           │ タスクノードのsuccessorをキューイング
           ▼
    ┌─────────────┐
    │   PENDING   │  ← 次のタスクへ
    │  (次タスク) │
    └─────────────┘


【並列グループの状態遷移（BSPモデル）】

    ┌──────────────────┐
    │  ParallelGroup   │
    │   (待機中)       │
    └────────┬─────────┘
             │
             │ run() 実行開始
             ▼
    ┌──────────────────────────────────────────┐
    │         SUPERSTEP EXECUTION              │
    │  ┌──────────┐  ┌──────────┐  ┌──────────┐│
    │  │ Task A   │  │ Task B   │  │ Task C   ││ ← 並列実行
    │  │EXECUTING │  │EXECUTING │  │EXECUTING ││   （Superstep）
    │  └─────┬────┘  └─────┬────┘  └─────┬────┘│
    │        │             │             │     │
    │        ▼             ▼             ▼     │
    │  ┌─────────┐   ┌─────────┐   ┌─────────┐│
    │  │Complete │   │Complete │   │Complete ││
    │  └─────────┘   └─────────┘   └─────────┘│
    └───────────────────┬──────────────────────┘
                        │
                        │ Barrier同期
                        │ policy.on_group_finished()
                        ▼
              ┌──────────────────┐
              │ ParallelGroup    │
              │   COMPLETED      │
              └────────┬─────────┘
                       │
                       │ successor実行
                       ▼
              ┌──────────────────┐
              │   Next Tasks     │
              └──────────────────┘
```

## 統一実行エンジン: WorkflowEngine.execute()

### 実行ループ疑似コード

```python
def execute(context: ExecutionContext, start_task_id: Optional[str] = None) -> Any:
    """統一ワークフロー実行エンジン

    状態遷移:
    1. Task取得（PENDING → EXECUTING準備）
    2. Task実行（EXECUTING）
    3. 結果保存（COMPLETED）
    4. Successor処理（次のPENDING生成）
    """

    # ========================================
    # フェーズ1: 初期化
    # ========================================
    if start_task_id is not None:
        task_id = start_task_id          # 指定タスクから開始
    else:
        task_id = context.get_next_task() # キューから取得（PENDING→取得）

    last_result = None

    # ========================================
    # フェーズ2: 実行ループ（ステートマシン駆動）
    # ========================================
    while task_id is not None and context.steps < context.max_steps:
        # ----------------------------------
        # ステップ1: タスク取得と検証
        # ----------------------------------
        task = context.graph.get_node(task_id)

        # ----------------------------------
        # ステップ2: 状態遷移 PENDING → EXECUTING
        # ----------------------------------
        try:
            with context.executing_task(task):
                # タスク実行（ハンドラ経由）
                # Handler: DirectTaskHandler, DockerTaskHandler, etc.
                handler = self._get_handler(task)
                last_result = handler.execute_task(task, context)

                # 結果保存（Channel経由で共有）
                # context.set_result(task_id, result)

        except FeedbackTimeoutError as e:
            # HITL対応: Checkpoint作成して一時停止
            self._handle_feedback_timeout(e, task_id, context)
            return None  # ワークフロー一時停止

        except Exception as e:
            # エラー発生時も状態を記録
            raise exceptions.as_runtime_error(e)

        # ----------------------------------
        # ステップ3: Successor処理（依存関係解決）
        # ----------------------------------
        if context.goto_called:
            # Goto制御: 通常のSuccessor処理をスキップ
            pass
        else:
            # 通常フロー: Successorを自動キューイング
            successors = context.graph.successors(task_id)
            for succ in successors:
                succ_task = context.graph.get_node(succ)
                context.add_to_queue(succ_task)  # 新しいPENDING状態生成

        # ----------------------------------
        # ステップ4: 状態遷移 EXECUTING → COMPLETED
        # ----------------------------------
        context.mark_task_completed(task_id)
        context.increment_step()

        # ----------------------------------
        # ステップ5: Checkpoint処理（オプション）
        # ----------------------------------
        if context.checkpoint_requested:
            self._handle_deferred_checkpoint(context)

        # ----------------------------------
        # ステップ6: 次タスク取得
        # ----------------------------------
        task_id = context.get_next_task()  # キューから次のPENDINGタスク取得

    return last_result
```

## ExecutionContext: ステートマシン管理

ExecutionContextはタスクの状態を一元管理する**ステートコンテナ**として機能する。

```python
class ExecutionContext:
    """ワークフロー実行状態管理

    状態管理要素:
    - task_queue: PENDING状態のタスクキュー
    - completed_tasks: COMPLETED状態のタスク集合
    - results: タスク実行結果（Channel経由で共有）
    - current_task_context: 現在EXECUTING中のタスクコンテキスト
    - _goto_called_in_current_task: goto制御フラグ
    """

    # ========================================
    # キュー管理（ステート遷移制御）
    # ========================================

    def get_next_task(self) -> Optional[str]:
        """PENDING → EXECUTING遷移の開始

        Returns:
            次に実行するタスクID（PENDINGキューから取得）
        """
        return self.task_queue.get_next_task()

    def add_to_queue(self, task: Executable):
        """→ PENDING遷移

        新しいタスクをPENDINGキューに追加
        トレース情報（trace_id, parent_span_id）も自動設定
        """
        task_spec = TaskSpec(
            executable=task,
            execution_context=self,
            trace_id=self.trace_id,
            parent_span_id=self.current_task_id
        )
        self.task_queue.enqueue(task_spec)

    # ========================================
    # タスク実行状態管理
    # ========================================

    def executing_task(self, task: Executable):
        """コンテキストマネージャー: EXECUTING状態管理

        with context.executing_task(task):
            # タスク実行中の状態
            # current_task_context が設定される
        """
        # EXECUTING状態のタスクコンテキスト作成
        task_context = TaskExecutionContext(...)
        self.current_task_context = task_context

        try:
            yield task_context
        finally:
            # EXECUTING → COMPLETED遷移準備
            self.current_task_context = None

    def mark_task_completed(self, task_id: str):
        """EXECUTING → COMPLETED遷移

        完了タスクを記録し、依存関係チェックに使用
        """
        self.completed_tasks.add(task_id)

    # ========================================
    # 動的タスク生成とフロー制御
    # ========================================

    def next_task(
        self,
        executable: Executable,
        goto: bool = False,
        _is_iteration: bool = False
    ) -> str:
        """動的タスク生成またはタスクジャンプ

        Args:
            executable: 実行するExecutableオブジェクト
            goto: Trueの場合、現タスクのsuccessor処理をスキップ
            _is_iteration: イテレーションタスクの内部フラグ

        動作:
        1. 既存タスク指定 → ジャンプ（goto自動有効化）
        2. 新規タスク指定 + goto=False → 動的タスク追加（successor正常処理）
        3. 新規タスク指定 + goto=True → 動的タスク追加（successorスキップ）

        Returns:
            タスクID
        """
        task_id = executable.task_id
        is_new_task = task_id not in self.graph.nodes

        if goto:
            # 明示的goto: successorスキップ（新規/既存問わず）
            if is_new_task:
                self.graph.add_node(executable, task_id)
            self.add_to_queue(executable)
            self._goto_called_in_current_task = True

        elif is_new_task:
            # 新規タスク: 動的タスク追加（通常のsuccessor処理）
            self.graph.add_node(executable, task_id)
            self.add_to_queue(executable)
            # goto_called は False のまま → successor正常処理

        else:
            # 既存タスク: 自動ジャンプ（successorスキップ）
            self.add_to_queue(executable)
            self._goto_called_in_current_task = True

        return task_id

    def next_iteration(self, data: Any = None, task_id: Optional[str] = None) -> str:
        """現在のタスクを再実行（サイクル・イテレーション）

        Args:
            data: 次イテレーションに渡すデータ
            task_id: 対象タスクID（Noneの場合は現在のタスク）

        Returns:
            生成されたイテレーションタスクID

        動作:
        1. サイクルカウント確認（max_cycles超過チェック）
        2. イテレーション用タスクID生成（例: task_cycle_1_abc123）
        3. データを引き継ぐラッパー関数作成
        4. next_task()経由で新規タスクとして追加

        Raises:
            CycleLimitExceededError: サイクル上限超過
        """
        # 現在のタスクID取得
        if task_id is None:
            task_id = self.current_task_context.task_id

        # サイクルカウント確認
        task_ctx = self._task_contexts.get(task_id)
        if not task_ctx.can_iterate():
            raise CycleLimitExceededError(...)

        # イテレーションタスクID生成
        cycle_count = task_ctx.register_cycle()
        iteration_id = f"{task_id}_cycle_{cycle_count}_{uuid.uuid4().hex[:8]}"

        # データ引き継ぎラッパー作成
        current_task = self.graph.get_node(task_id)
        def iteration_func():
            return current_task(task_ctx, data)

        iteration_task = TaskWrapper(iteration_id, iteration_func)

        # 動的タスクとして追加（_is_iteration=True）
        return self.next_task(iteration_task, _is_iteration=True)

    # ========================================
    # Goto制御
    # ========================================

    @property
    def goto_called(self) -> bool:
        """現在のタスク実行中にgotoが呼ばれたか"""
        return self._goto_called_in_current_task

    def reset_goto_flag(self) -> None:
        """次タスク実行のためgotoフラグをリセット"""
        self._goto_called_in_current_task = False
```

## ParallelGroupのBSP実行モデル

ParallelGroupは**BSP（Bulk Synchronous Parallel）モデル**に基づくsuperstep実行を行う。

### BSPモデルの3フェーズ

```
【BSP Superstep構造】

┌─────────────────────────────────────────────────────────────┐
│                    Superstep N                               │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Phase 1    │  │   Phase 2    │  │   Phase 3    │      │
│  │ Computation  │→ │ Communication│→ │ Barrier Sync │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│                                                              │
│  Phase 1: 並列タスク実行（Task A, B, C同時実行）            │
│  Phase 2: 結果共有（Channel経由でset_result()）             │
│  Phase 3: Barrier同期（全タスク完了待ち）                   │
└─────────────────────────────────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                    Superstep N+1                             │
│           （全タスク完了後、次のステップへ）                │
└─────────────────────────────────────────────────────────────┘
```

### ParallelGroup実行疑似コード

```python
class ParallelGroup(Executable):
    """BSPモデルに基づく並列実行グループ"""

    def run(self) -> Any:
        """並列グループ実行（Superstep）

        BSPフェーズ:
        1. Computation: 全タスクを並列実行
        2. Communication: 結果をChannel経由で共有
        3. Barrier: 全タスク完了まで待機
        """
        context = self.get_execution_context()

        # Phase 1: Computation（並列計算フェーズ）
        # GroupExecutorに委譲
        GroupExecutor.execute_parallel_group(
            self.task_id,
            self.tasks,
            context,
            backend=self._execution_config.get("backend"),
            policy=self._execution_config.get("policy")
        )
        # Phase 2 & 3は内部で自動実行される


class GroupExecutor:
    """BSP並列実行オーケストレーター"""

    @staticmethod
    def execute_parallel_group(
        group_id: str,
        tasks: List[Executable],
        exec_context: ExecutionContext,
        backend: CoordinationBackend,
        policy: GroupExecutionPolicy
    ):
        """Superstep実行の統一インターフェース

        BSP実装:
        - Computation: coordinatorによる並列実行
        - Communication: 各タスクがset_result()でChannel更新
        - Barrier: coordinator.execute_group()内で同期
        """

        coordinator = create_coordinator(backend)

        # ========================================
        # BSP Superstep実行
        # ========================================

        # Phase 1: Computation（並列計算）
        # coordinator内で全タスクを並列実行
        # 各タスク: engine.execute(context, start_task_id=task.task_id)

        # Phase 2: Communication（結果共有）
        # 各タスク実行内で: context.set_result(task_id, result)
        # → Channel経由で全ノードに結果共有

        # Phase 3: Barrier（同期待機）
        # coordinator.execute_group()内で全タスク完了待ち
        coordinator.execute_group(group_id, tasks, exec_context, policy)

        # Barrierクリア後、policyによる成功/失敗判定
        # policy.on_group_finished(group_id, tasks, results, exec_context)


# 具体的なCoordinator実装例
class ThreadingCoordinator(TaskCoordinator):
    """スレッドベースBSP実装"""

    def execute_group(self, group_id, tasks, context, policy):
        """BSP Superstep with threading

        Phase 1: ThreadPoolExecutorで並列実行
        Phase 2: 各タスクがcontext.set_result()
        Phase 3: futures.wait()でBarrier同期
        """
        with ThreadPoolExecutor(max_workers=self.thread_count) as executor:
            # Phase 1: Computation
            futures = [
                executor.submit(self._execute_task, task, context)
                for task in tasks
            ]

            # Phase 3: Barrier（wait()で全futures完了待ち）
            done, not_done = futures.wait(futures, return_when=ALL_COMPLETED)

            # 結果集約とPolicy判定
            results = [f.result() for f in done]
            policy.on_group_finished(group_id, tasks, results, context)
```

## 動的フロー制御パターン

### パターン1: 動的タスク生成（next_task）

```python
@task(inject_context=True)
def conditional_task(ctx: TaskExecutionContext):
    """条件に基づいて動的にタスクを生成"""
    result = some_computation()

    if result > 100:
        # 新規タスクを動的に追加（グラフに存在しない）
        new_task = Task("high_value_handler")
        ctx.next_task(new_task)  # successor正常処理
    else:
        # 既存タスクへジャンプ
        existing_task = ctx.execution_context.graph.get_node("default_handler")
        ctx.next_task(existing_task)  # goto自動有効化（successorスキップ）

# 実行フロー:
# conditional_task → (動的判定) → high_value_handler or default_handler
```

### パターン2: イテレーション（next_iteration）

```python
@task(inject_context=True)
def retry_until_success(ctx: TaskExecutionContext, data: int = 0):
    """成功するまでリトライ（サイクル実行）"""
    result = attempt_operation(data)

    if not result.success:
        # 現在のタスクを再実行（max_cycles上限まで）
        ctx.next_iteration(data + 1)  # データを次イテレーションに引き継ぎ
    else:
        return result

# 実行フロー:
# retry_until_success → retry_until_success_cycle_1 → retry_until_success_cycle_2 → ...
```

### パターン3: 明示的Goto制御

```python
@task(inject_context=True)
def branching_task(ctx: TaskExecutionContext):
    """明示的gotoによるフロー制御"""
    mode = get_execution_mode()

    if mode == "fast":
        # 新規タスクを追加してジャンプ（successorスキップ）
        fast_task = Task("fast_path")
        ctx.next_task(fast_task, goto=True)
    else:
        # 通常フロー（successorが実行される）
        return normal_processing()

# グラフ構造:
# branching_task → successor_A
#               → successor_B
#
# goto=True の場合: branching_task → fast_path （successor_A, Bはスキップ）
# goto=False の場合: branching_task → successor_A, successor_B （通常フロー）
```

### パターン4: Goto制御フラグの動作

```
【Goto制御のステートマシン影響】

通常フロー（goto=False）:
┌─────────────┐
│   Task A    │
│ EXECUTING   │
└──────┬──────┘
       │ 完了
       ├─→ add_to_queue(successor_1)
       ├─→ add_to_queue(successor_2)
       └─→ mark_task_completed("A")

task_queue: [successor_1, successor_2]  ← 通常のsuccessor処理


Gotoフロー（goto=True）:
┌─────────────┐
│   Task A    │
│ EXECUTING   │
│ goto=True   │
└──────┬──────┘
       │ 完了
       │ _goto_called_in_current_task = True
       │ successorスキップ！
       └─→ mark_task_completed("A")

task_queue: [jump_target]  ← gotoで指定したタスクのみ
```

## 実行シナリオ例

### シナリオ1: 逐次実行（A → B → C）

```
初期状態:
  task_queue: [A]
  completed_tasks: {}

--- ステップ1: Task A実行 ---
1. get_next_task() → "A"  (PENDINGキューから取得)
2. executing_task(A)       (PENDING → EXECUTING)
   - handler.execute_task(A)
   - set_result("A", result_a)
3. mark_task_completed("A") (EXECUTING → COMPLETED)
4. add_to_queue(B)         (Bを新規PENDING追加)

状態:
  task_queue: [B]
  completed_tasks: {A}

--- ステップ2: Task B実行 ---
1. get_next_task() → "B"
2. executing_task(B)
   - handler.execute_task(B)
   - set_result("B", result_b)
3. mark_task_completed("B")
4. add_to_queue(C)

状態:
  task_queue: [C]
  completed_tasks: {A, B}

--- ステップ3: Task C実行 ---
1. get_next_task() → "C"
2. executing_task(C)
   - handler.execute_task(C)
   - set_result("C", result_c)
3. mark_task_completed("C")
4. add_to_queue() → successorなし

状態:
  task_queue: []
  completed_tasks: {A, B, C}

→ ワークフロー完了
```

### シナリオ2: 並列実行（A → (B | C | D) → E）

```
【ASCII実行フロー】

Start → A → ParallelGroup{B,C,D} → E → End

初期状態:
  task_queue: [A]
  completed_tasks: {}

--- ステップ1: Task A実行 ---
状態遷移: A (PENDING → EXECUTING → COMPLETED)
  task_queue: [parallel_group_1]
  completed_tasks: {A}

--- ステップ2: ParallelGroup実行（BSP Superstep） ---
状態遷移: parallel_group_1 (PENDING → EXECUTING)

┌─────────────────────────────────────────────────────┐
│            BSP Superstep Execution                  │
│                                                     │
│  Phase 1: Computation（並列計算フェーズ）           │
│    ┌────────┐  ┌────────┐  ┌────────┐             │
│    │Task B  │  │Task C  │  │Task D  │             │
│    │EXEC    │  │EXEC    │  │EXEC    │  ← 同時実行 │
│    └───┬────┘  └───┬────┘  └───┬────┘             │
│        │           │           │                   │
│  Phase 2: Communication（結果共有）                │
│        ├───→ set_result("B", result_b)             │
│                   ├───→ set_result("C", result_c)  │
│                               ├→ set_result("D", ..)│
│        │           │           │                   │
│  Phase 3: Barrier Sync（同期待機）                 │
│        └───────────┴───────────┘                   │
│                    │                               │
│         全タスク完了（Barrier）                     │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
    policy.on_group_finished()  ← 成功/失敗判定
                      │
                      ▼
状態遷移: parallel_group_1 (EXECUTING → COMPLETED)

  task_queue: [E]  ← Successor自動追加
  completed_tasks: {A, parallel_group_1, B, C, D}

--- ステップ3: Task E実行 ---
状態遷移: E (PENDING → EXECUTING → COMPLETED)
  task_queue: []
  completed_tasks: {A, parallel_group_1, B, C, D, E}

→ ワークフロー完了
```

### シナリオ3: ダイヤモンド依存（A → B,C → D）

```
【グラフ構造】
     A
    / \
   B   C
    \ /
     D

--- ステップ1: A完了 ---
  task_queue: [B, C]  ← A完了後、B,C同時にPENDING追加
  completed_tasks: {A}

--- ステップ2: B実行 ---
  task_queue: [C]
  completed_tasks: {A, B}

Dの依存チェック:
  dependencies(D) = {B, C}
  completed = {A, B}
  B ∈ completed ✓
  C ∉ completed ✗ → Dはまだキューに追加されない

--- ステップ3: C実行 ---
  task_queue: []
  completed_tasks: {A, B, C}

Dの依存チェック:
  dependencies(D) = {B, C}
  completed = {A, B, C}
  B ∈ completed ✓
  C ∈ completed ✓ → 全依存完了！Dをキューに追加

  task_queue: [D]

--- ステップ4: D実行 ---
  task_queue: []
  completed_tasks: {A, B, C, D}

→ ワークフロー完了
```

### シナリオ4: 動的タスク生成（next_task使用）

```
【動的タスク追加のフロー】

初期グラフ:
  A → B → C

タスクA内でnext_task()を呼び出し:
  ctx.next_task(Task("dynamic_X"))

--- ステップ1: Task A実行 ---
状態遷移: A (PENDING → EXECUTING)
  実行中に next_task(Task("dynamic_X")) 呼び出し
    ↓
  1. graph.add_node("dynamic_X", executable)  ← グラフに新規ノード追加
  2. add_to_queue(Task("dynamic_X"))          ← PENDINGキューに追加
  3. goto_called = False                      ← successor正常処理

状態遷移: A (EXECUTING → COMPLETED)
  successor処理:
    add_to_queue(B)  ← 通常のsuccessor

task_queue: [dynamic_X, B]  ← 動的タスクとsuccessorの両方がキューに
completed_tasks: {A}

現在のグラフ:
  A → B → C
  └→ dynamic_X  ← 動的に追加

--- ステップ2: dynamic_X実行 ---
状態遷移: dynamic_X (PENDING → EXECUTING → COMPLETED)
  task_queue: [B]  ← dynamic_Xにsuccessorなし、Bのみ残る
  completed_tasks: {A, dynamic_X}

--- ステップ3以降: B → C と続く ---
```

### シナリオ5: イテレーション実行（next_iteration使用）

```
【サイクル実行のフロー】

タスク定義:
  @task(inject_context=True)
  def polling_task(ctx: TaskExecutionContext, count: int = 0):
      if count < 3:
          ctx.next_iteration(count + 1)
      return count

初期状態:
  task_queue: [polling_task]
  completed_tasks: {}

--- ステップ1: polling_task実行（初回） ---
状態遷移: polling_task (PENDING → EXECUTING)
  実行: polling_task(ctx, count=0)
  count < 3 → True
  ↓
  ctx.next_iteration(count + 1) 呼び出し:
    1. cycle_count = task_ctx.register_cycle() → 1
    2. iteration_id = "polling_task_cycle_1_abc12345"
    3. iteration_func = lambda: polling_task(ctx, count=1)
    4. iteration_task = TaskWrapper(iteration_id, iteration_func)
    5. next_task(iteration_task, _is_iteration=True)
       ↓
       graph.add_node(iteration_task)
       add_to_queue(iteration_task)

状態遷移: polling_task (EXECUTING → COMPLETED)
  task_queue: [polling_task_cycle_1_abc12345]
  completed_tasks: {polling_task}

--- ステップ2: polling_task_cycle_1実行 ---
状態遷移: polling_task_cycle_1 (PENDING → EXECUTING)
  実行: polling_task(ctx, count=1)
  count < 3 → True
  ↓
  ctx.next_iteration(count + 2)
    iteration_id = "polling_task_cycle_2_def67890"

状態遷移: polling_task_cycle_1 (EXECUTING → COMPLETED)
  task_queue: [polling_task_cycle_2_def67890]
  completed_tasks: {polling_task, polling_task_cycle_1}

--- ステップ3: polling_task_cycle_2実行 ---
状態遷移: polling_task_cycle_2 (PENDING → EXECUTING)
  実行: polling_task(ctx, count=2)
  count < 3 → True
  ↓
  ctx.next_iteration(count + 3)
    iteration_id = "polling_task_cycle_3_ghi01234"

状態遷移: polling_task_cycle_2 (EXECUTING → COMPLETED)
  task_queue: [polling_task_cycle_3_ghi01234]
  completed_tasks: {polling_task, polling_task_cycle_1, polling_task_cycle_2}

--- ステップ4: polling_task_cycle_3実行 ---
状態遷移: polling_task_cycle_3 (PENDING → EXECUTING)
  実行: polling_task(ctx, count=3)
  count < 3 → False  ← イテレーション終了条件
  return 3

状態遷移: polling_task_cycle_3 (EXECUTING → COMPLETED)
  task_queue: []  ← イテレーション完了、successorなし
  completed_tasks: {polling_task, ..., polling_task_cycle_3}

→ ワークフロー完了

【サイクル上限制御】
task_ctx.max_cycles = 10  ← デフォルト上限
task_ctx.cycle_count = 4  ← 現在のサイクル数

cycle_count > max_cycles の場合:
  CycleLimitExceededError 発生
```

### シナリオ6: Goto制御による分岐

```
【Gotoによる条件分岐フロー】

グラフ構造:
  start → decision → branch_A
                  → branch_B
                  → branch_C

decision タスク内でgoto制御:
  if condition == "A":
      ctx.next_task(branch_A_task, goto=True)  # goto明示
  elif condition == "B":
      existing_B = graph.get_node("branch_B")
      ctx.next_task(existing_B)  # 既存タスクへジャンプ（goto自動）
  else:
      # 何もしない → 通常のsuccessor処理

--- ケース1: condition="A" (goto=True) ---
ステップ1: decision実行
  ctx.next_task(branch_A_task, goto=True)
  ↓
  _goto_called_in_current_task = True

完了時:
  successor処理がスキップされる（goto_called=True）
  task_queue: [branch_A_task]  ← gotoで指定したタスクのみ
  branch_B, branch_C は実行されない

--- ケース2: condition="B" (既存タスクジャンプ) ---
ステップ1: decision実行
  existing_B = graph.get_node("branch_B")
  ctx.next_task(existing_B)
  ↓
  is_new_task = False
  ↓ 自動でgoto有効化
  _goto_called_in_current_task = True

完了時:
  task_queue: [branch_B]  ← ジャンプ先のみ
  branch_A, branch_C は実行されない

--- ケース3: condition="C" (通常フロー) ---
ステップ1: decision実行
  何も呼び出さない
  ↓
  _goto_called_in_current_task = False

完了時:
  successor処理が実行される
  task_queue: [branch_A, branch_B, branch_C]  ← 全successor実行
```

## 分散実行でのステートマシン

分散環境では、**ExecutionContextの状態がChannel経由で共有**される。

```
【分散ノードでの状態共有】

┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│   Node 1    │      │   Node 2    │      │   Node 3    │
│             │      │             │      │             │
│ engine.     │      │ engine.     │      │ engine.     │
│ execute()   │      │ execute()   │      │ execute()   │
│             │      │             │      │             │
│ task: A     │      │ task: B     │      │ task: C     │
│ EXECUTING   │      │ EXECUTING   │      │ EXECUTING   │
└──────┬──────┘      └──────┬──────┘      └──────┬──────┘
       │                    │                    │
       │ set_result("A")    │ set_result("B")    │ set_result("C")
       └────────────────────┼────────────────────┘
                            ▼
              ┌──────────────────────────┐
              │      RedisChannel        │
              │  ┌────────────────────┐  │
              │  │ results["A"] = ... │  │
              │  │ results["B"] = ... │  │
              │  │ results["C"] = ... │  │
              │  └────────────────────┘  │
              │    共有状態ストア         │
              └──────────────────────────┘
                            ▲
       ┌────────────────────┼────────────────────┐
       │ get_result("A")    │ get_result("B")    │ get_result("C")
       ▼                    ▼                    ▼
┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│   Node 1    │      │   Node 2    │      │   Node 3    │
│             │      │             │      │             │
│  全ノードで状態を参照可能                              │
│  completed_tasks, results, queueはRedis経由で同期      │
└─────────────┘      └─────────────┘      └─────────────┘

各ノードのステートマシンは独立しているが、
ExecutionContextの共有状態（Channel）により協調動作
```

## まとめ

Graflowのステートマシン実行モデルは、以下の特徴を持つ：

### 1. **統一ステートマシン**
- WorkflowEngine.execute()が全実行パスで使用される
- ExecutionContextが状態を一元管理（PENDING/EXECUTING/COMPLETED）
- task_queueがPENDINGタスクのバッファとして機能

### 2. **明示的状態遷移**
- タスクライフサイクル: PENDING → EXECUTING → COMPLETED
- 状態遷移がコンテキストメソッドで明確に定義される
  - `get_next_task()`: PENDING → EXECUTING遷移開始
  - `executing_task()`: EXECUTING状態管理
  - `mark_task_completed()`: EXECUTING → COMPLETED遷移
  - `add_to_queue()`: → PENDING遷移（新規タスク追加）

### 3. **動的フロー制御**
- **next_task()**: 実行時の動的タスク生成とタスクジャンプ
  - 新規タスク: グラフに追加してPENDINGキューへ
  - 既存タスク: ジャンプ（goto自動有効化）
- **next_iteration()**: サイクル・イテレーション実行
  - サイクルカウント管理による上限制御
  - データ引き継ぎ機能
- **goto制御**: Successor処理の動的スキップ
  - `goto=True`: 明示的なsuccessorスキップ
  - 既存タスクへのジャンプ: 自動goto有効化

### 4. **BSP並列実行モデル**
- ParallelGroupはSuperstep実行（Computation / Communication / Barrier）
- 同期ポイントでの一貫性保証
- GroupExecutorによる統一的な並列制御

### 5. **透過的分散状態管理**
- Channel基盤により、ローカル/分散で状態共有方法が異なるだけ
- ステートマシンのロジックは環境に依存しない
- 分散環境でも同一のnext_task/next_iteration/goto制御が使用可能

### 6. **依存関係自動解決**
- Successor自動キューイングによる自然なワークフロー進行
- Completed状態の追跡により依存関係チェック
- goto制御による柔軟なフロー変更

### 7. **Queue駆動実行**
- task_queueが実行順序を制御
- 動的タスク生成も同じqueueメカニズムで統一的に処理
- ローカル（MemoryQueue）/分散（RedisQueue）の透過的切り替え

このアーキテクチャにより、シンプルな状態遷移モデルで複雑な動的ワークフロー実行を実現している。

---

*本文書は、graflow/core/engine.py および docs/unified_engine_architecture.md の設計に基づく*
