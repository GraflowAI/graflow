# ランタイムグラフ親子リンク（Codex/JA）

## 発生している問題
- 通常のワークフロー実行でランタイムグラフのノードが分断され、親子エッジがほぼ作られない。
- `Tracer.on_task_start` はスタックに積む前の `context.current_task_id` から `parent_task_id` を取る（`graflow/trace/base.py:299-323`, `graflow/core/context.py:1193-1226`）。多くのケースでスタックが空のため `parent_name=None` となり、エッジが張られない。
- 後続タスクの enqueue は `context.executing_task` を抜けてスタックを pop した後に行われる。`add_to_queue` が読む `current_task_id` はその時点で `None`（`graflow/core/context.py:693-711`）。
- デキュー時に親メタデータを捨てている: `TaskQueue.get_next_task` は `TaskSpec` を返さず task_id だけを返すため `parent_span_id` が渡らない（`graflow/queue/base.py:123-126`）。親付きで enqueue した（動的タスクなど）場合でも `on_task_start` からは見えない。
- 結果として `_runtime_graph` は通常のエンジン実行で親情報を受け取れない。

## 推奨アクション
1) **後続タスク enqueue 時に親を保持する**  
   - 後続をキューに積む間は現在のタスクをスタックに残す（または task_id を一時保存）。`with context.executing_task(task)` ブロック内に後続スケジューリングを移す、あるいは pop 前に `current_parent_task_id` をキャッシュし、スタックが空ならそれを `add_to_queue` が使う。

2) **タスク開始時に親メタデータをトレーサへ渡す**  
   - `TaskSpec` を捨てない。デキューで `TaskSpec`（または task_id と parent_span_id の両方）を返し、`current_task_id` が無い場合は `parent_span_id` を `executing_task`/`on_task_start` に渡す。消費後はキャッシュをクリアして漏れを防ぐ。

3) **分散実行との整合**  
   - ワーカー側 `attach_to_trace(parent_span_id=...)` を使う場合、トレーサにその親を保持させ、プロセス最初のタスクのフォールバックにする。これはキュー経由の親伝搬を補完するもの。

4) **リグレッションカバレッジ**  
   - `WorkflowEngine.execute` で 2 ノードの簡単なグラフを実行し、ランタイムグラフに parent→child エッジがあることを確認するテストを追加。  
   - ハンドラ内で enqueue する動的タスクケース、およびワーカー attach ケースを加え、`TaskSpec` や `attach_to_trace` 由来の親データでエッジが張られることを確認する。

## 修正の進め方（案）
- デキュー時に `TaskSpec` を扱うよう `WorkflowEngine/ExecutionContext` を調整し、`parent_span_id` をトレーサフックに渡す。  
- 親 ID が残っている（スタック or キャッシュ）間に後続 enqueue を行う。  
- トレーサ開始ロジックを「スタック親優先 → キュー由来 `parent_span_id` → ワーカーフォールバック」の順に更新する。  
- ランタイムグラフ系テストを再実行し、エッジ確認のアサーションを追加する。  
