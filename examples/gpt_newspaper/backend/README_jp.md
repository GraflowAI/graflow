# GPT Newspaper - Graflow Workflow Examples

## 概要

このディレクトリには、**AI新聞記事を自動生成するシステム**の2つのバージョンが含まれています。Graflowの動的タスク生成機能を段階的に理解できるように設計されています。

## 2つのワークフロー

### 1. `newspaper_workflow.py` - **基本版**

シンプルで理解しやすい基本的なワークフロー。Graflowの核となる機能を学ぶのに最適です。

**処理フロー:**
```
検索 → キュレーション → 執筆 → 批評 → デザイン
```

**主な特徴:**
- ✅ `goto=True`を使った執筆-批評ループ
- ✅ チャンネルベースの状態管理
- ✅ ThreadPoolExecutorによる複数記事の並列処理

### 2. `newspaper_dynamic_workflow.py` - **高度版**

実践的な本番環境を想定した高度なワークフロー。Graflowの全機能を活用します。

**処理フロー:**
```
トピック分析 → 複数角度での検索 → 情報キュレーション →
複数スタイルでの執筆 → 批評・改善ループ → 品質チェック → デザイン
```

**主な特徴:**
- ✅ ランタイムでの動的タスク展開（`context.next_task()`）
- ✅ 動的なギャップ補填（`context.next_iteration()`）
- ✅ 並列ライターペルソナ（BestEffortGroupPolicy）
- ✅ 品質ゲート（AtLeastNGroupPolicy）

---

## 実行方法

```bash
# 環境変数の設定
export TAVILY_API_KEY="your_tavily_api_key"
export OPENAI_API_KEY="your_openai_api_key"

# 基本版の実行
PYTHONPATH=. uv run python examples/gpt_newspaper/backend/newspaper_workflow.py

# 高度版の実行
PYTHONPATH=. uv run python examples/gpt_newspaper/backend/newspaper_dynamic_workflow.py
```

---

## 学習パス

**推奨学習順序:**

1. **`newspaper_workflow.py`から始める** - 基本パターンを理解
2. **`newspaper_dynamic_workflow.py`に進む** - 高度な機能を学ぶ

---

# Part 1: 基本版 (`newspaper_workflow.py`)

## ワークフローの全体像

```
search → curate → write ⟲ critique → design
                    ↑      ↓
                    └──────┘ (フィードバックループ)
```

## 核となる機能: goto=Trueによるループバック

`newspaper_workflow.py:123-170`

```python
@task(id=f"critique_{article_id}", inject_context=True)
def critique_task(context: TaskExecutionContext) -> Dict:
    """記事を批評し、フィードバックがあれば執筆タスクへループバック"""
    channel = context.get_channel()
    article = channel.get("article")
    iteration = channel.get("iteration", default=0)

    result = critique_agent.run(article)
    channel.set("article", result)

    if result.get("critique") is not None:
        # フィードバックがある場合
        if iteration >= 5:
            print(f"⚠️ Max iterations reached")
            result["critique"] = None  # 強制承認
        else:
            channel.set("iteration", iteration + 1)
            # ★既存のwrite_taskへループバック
            context.next_task(write_task, goto=True)
            return result

    # 承認された場合、自然にdesign_taskへ進む
    return result
```

### チャンネルによる状態管理

```python
# 状態の保存
channel.set("article", result)
channel.set("iteration", iteration + 1)

# 状態の取得
article = channel.get("article")
iteration = channel.get("iteration", default=0)
```

---

# Part 2: 高度版 (`newspaper_dynamic_workflow.py`)

## ワークフローの全体像

```
topic_intake
    ↓
search_router (動的に複数の検索タスクを生成)
    ↓ ↓ ↓ ↓
  search_angle_1 | search_angle_2 | search_angle_3 | ...
    ↓ ↓ ↓ ↓
curate ⟲ (不足時に補足調査を追加)
    ↓
writer_personas (並列実行)
    ↓ ↓ ↓
  write_feature | write_brief | write_data_digest
    ↓ ↓ ↓
select_draft (最良のドラフトを選択)
    ↓
write ⟲←──┐
    ↓      │
critique ──┘ (フィードバックがあればループバック)
    ↓
quality_gate (並列実行、2/3成功で通過)
    ↓ ↓ ↓
  fact_check | compliance_check | risk_check
    ↓ ↓ ↓
quality_gate_summary
    ↓
design
```

## Graflowの高度な機能を活かした3つのポイント

### 1. **動的な検索タスク展開** 📡

`newspaper_dynamic_workflow.py:149-175`

```python
@task(id=f"search_router_{article_id}", inject_context=True)
def search_router_task(context: TaskExecutionContext) -> Dict:
    """トピックに応じた複数の検索角度を動的に決定"""
    channel = context.get_channel()
    angles: List[str] = channel.get("angles", default=["overview"])

    print(f"[{article_id}] 🔍 Launching {len(angles)} targeted searches...")

    for angle in angles:
        angle_id = _slugify(angle)

        def run_angle(angle_label=angle, angle_slug=angle_id):
            task_query = f"{query} - focus on {angle_label}"
            result = search_agent.run({"query": task_query, "angle": angle_label})
            # 結果をチャンネルに集約
            aggregated = channel.get("search_results", default=[])
            aggregated.append(result)
            channel.set("search_results", aggregated)
            return result

        # ★ランタイムでタスクを動的追加
        angle_task = TaskWrapper(f"search_{article_id}_{angle_id}", run_angle)
        context.next_task(angle_task)

    return {"scheduled": len(angles)}
```

#### 何をやっているか

- トピック分析の結果に基づいて、「政策」「市場動向」「気候影響」「技術」などの複数の検索角度を決定
- 各角度ごとに**ランタイムで検索タスクを動的生成**（`context.next_task()`）
- 並列で複数の検索を実行し、結果をチャンネルに集約

#### 従来のワークフローエンジンとの違い

- ✅ 事前に検索タスク数が決まっていなくてもOK
- ✅ トピックの内容に応じて柔軟に検索範囲を拡張できる
- ✅ 実行時の状態に基づいてワークフローが自己適応

---

### 2. **動的なギャップ補填** 🔄

`newspaper_dynamic_workflow.py:205-229`

```python
@task(id=f"curate_{article_id}", inject_context=True)
def curate_task(context: TaskExecutionContext) -> Dict:
    """キュレーション時に情報源が不足していたら補足調査を追加"""
    channel = context.get_channel()
    expected = channel.get("expected_search_tasks", default=0)
    completed = channel.get("completed_search_tasks", default=0)

    # 検索タスクの完了を待つ
    if expected and completed < expected:
        print(f"[{article_id}] ⏳ Waiting for {expected - completed} search tasks...")
        time.sleep(0.05)
        context.next_iteration()  # ★自分自身を再実行
        return {"status": "waiting"}

    # ソースをキュレーション
    result = curator_agent.run(article_data)

    # ソースが不足している場合
    min_sources = 3
    if len(result.get("sources", [])) < min_sources and not channel.get("gap_fill_requested"):
        channel.set("gap_fill_requested", True)

        def supplemental_research():
            # 補足調査を実行
            supplemental_result = search_agent.run({
                "query": f"{query} statistics and data",
                "angle": "data supplement"
            })
            # 結果を集約
            aggregated = channel.get("search_results", default=[])
            aggregated.append(supplemental_result)
            channel.set("search_results", aggregated)
            return supplemental_result

        # ★補足調査タスクを動的に追加
        context.next_task(TaskWrapper(f"search_{article_id}_supplemental", supplemental_research))
        print(f"[{article_id}] 🔄 Not enough sources, scheduling supplemental research...")
        context.next_iteration()  # 再度キュレーションを実行
        return {"status": "gap_filling"}

    return result
```

#### 何をやっているか

- キュレーション時に情報源が不足していることを検出（最低3ソース必要）
- 補足調査タスクを**動的に追加**（`context.next_task()`）
- キュレーションタスク自身を**再実行**（`context.next_iteration()`）して補足情報を待つ

#### ポイント

- ✅ 品質基準を満たすまで自動的に情報収集を拡張
- ✅ 実行時の状態に応じた柔軟な対応
- ✅ 無限ループを防ぐための `gap_fill_requested` フラグ

---

### 3. **執筆-批評の反復ループ** 🔁

`newspaper_dynamic_workflow.py:329-359`

```python
@task(id=f"critique_{article_id}", inject_context=True)
def critique_task(context: TaskExecutionContext) -> Dict:
    """記事を批評し、フィードバックがあれば執筆タスクへループバック"""
    print(f"[{article_id}] 🔎 Critiquing article...")

    channel = context.get_channel()
    article = _get_article_from_channel(context)
    iteration = channel.get("iteration", default=0)

    # 批評エージェントが記事をレビュー
    result = critique_agent.run(article)
    channel.set("article", result)

    # フィードバックがある場合
    if result.get("critique") is not None:
        print(f"[{article_id}] 🔄 Critique feedback received, looping back to writer...")

        if iteration >= MAX_REVISION_ITERATIONS:
            print(f"[{article_id}] ⚠️  Max iterations ({MAX_REVISION_ITERATIONS}) reached")
            result["critique"] = None  # 強制的に承認
        else:
            channel.set("iteration", iteration + 1)
            # ★既存のwrite_taskへループバック
            context.next_task(write_task, goto=True)
            return result

    print(f"[{article_id}] ✅ Article approved by critique!")
    return result
```

#### 何をやっているか

1. 批評エージェントが記事をレビュー
2. フィードバックがある場合、**既存のwrite_taskへループバック**（`goto=True`）
3. 最大3回まで執筆-批評サイクルを繰り返す
4. 批評がOKなら品質チェックフェーズへ進む

#### 従来のワークフローとの違い

- ✅ 静的なDAG（有向非巡回グラフ）では表現できない循環フローを実現
- ✅ 品質が満たされるまで動的に反復
- ✅ 無限ループ防止のための反復回数制限（`MAX_REVISION_ITERATIONS = 3`）

---

## その他のGraflow機能活用

### 4. **並列ライターペルソナ** ✍️

`newspaper_dynamic_workflow.py:263-268`

```python
writer_personas = (
    write_feature_task | write_brief_task | write_data_digest_task
).with_execution(
    backend=CoordinationBackend.THREADING,
    policy=BestEffortGroupPolicy(),
)
```

- 3種類の執筆スタイル（feature、brief、data_digest）を**並列実行**
- `BestEffortGroupPolicy()`: ベストエフォートで実行し、最も良いものを選択
- スレッドベースの並列実行で効率化

### 5. **品質ゲート with 部分的成功許容** 🛡️

`newspaper_dynamic_workflow.py:385-390`

```python
quality_gate = (
    fact_check_task | compliance_check_task | risk_check_task
).with_execution(
    backend=CoordinationBackend.THREADING,
    policy=AtLeastNGroupPolicy(min_success=2),  # ★最低2つ成功が必要
)
```

- ファクトチェック、コンプライアンス、リスク評価を並列実行
- **最低2つが成功すれば**次のフェーズへ進む（部分的な失敗を許容）
- 本番環境での柔軟な品質管理を実現

---

## ワークフローのタスクグラフ

```python
# newspaper_dynamic_workflow.py:428
topic_intake_task >> search_router_task >> curate_task >> writer_personas >> select_draft_task >> write_task >> critique_task >> quality_gate >> quality_gate_summary_task >> design_task
```

### 視覚化

```
topic_intake
    ↓
search_router (動的に複数の検索タスクを生成)
    ↓ ↓ ↓ ↓
  search_angle_1 | search_angle_2 | search_angle_3 | ...
    ↓ ↓ ↓ ↓
curate ⟲ (不足時に補足調査を追加)
    ↓
writer_personas (並列実行)
    ↓ ↓ ↓
  write_feature | write_brief | write_data_digest
    ↓ ↓ ↓
select_draft (最良のドラフトを選択)
    ↓
write ⟲←──┐
    ↓      │
critique ──┘ (フィードバックがあればループバック)
    ↓
quality_gate (並列実行、2/3成功で通過)
    ↓ ↓ ↓
  fact_check | compliance_check | risk_check
    ↓ ↓ ↓
quality_gate_summary
    ↓
design
```

---

## 2つのワークフローの比較

| 機能 | 基本版 (`newspaper_workflow.py`) | 高度版 (`newspaper_dynamic_workflow.py`) |
|------|----------------------------------|------------------------------------------|
| **タスク数** | 5タスク | 15+タスク（動的に増加） |
| **検索戦略** | 単一クエリ | トピック分析に基づく複数角度検索 |
| **動的タスク生成** | なし（goto=Trueのみ） | 検索展開、ギャップ補填、補足調査 |
| **執筆アプローチ** | 単一スタイル | 3つのペルソナ並列実行 → 最良選択 |
| **品質保証** | 批評ループのみ | 批評 + 3段階品質ゲート |
| **並列実行** | 記事レベル（ThreadPoolExecutor） | タスクレベル + 記事レベル |
| **エラーハンドリング** | 反復回数制限のみ | 部分的成功許容（AtLeastNGroupPolicy） |
| **コード行数** | ~325行 | ~604行 |
| **学習曲線** | 🟢 初心者向け | 🟡 中級〜上級者向け |

---

## チャンネルを使った状態管理

両方のワークフローで、タスク間の状態共有に**チャンネル**を活用しています：

```python
channel = context.get_channel()

# 検索結果の集約
channel.set("search_results", aggregated)
channel.get("search_results", default=[])

# イテレーション管理
channel.set("iteration", iteration + 1)
channel.get("iteration", default=0)

# 記事データの共有
channel.set("article", result)
channel.get("article")

# フラグ管理
channel.set("gap_fill_requested", True)
channel.get("gap_fill_requested", default=False)
```

### チャンネルの利点

- ✅ タスク間でデータを共有できる
- ✅ 反復実行時に状態を維持できる
- ✅ 動的タスクからも同じチャンネルにアクセス可能

---

## まとめ

これら2つのワークフローは、**Graflowの動的タスク生成機能**を段階的に理解できるように設計されています。

### 基本版で学べること

- ✅ `goto=True`による既存タスクへのループバック
- ✅ チャンネルを使った状態管理
- ✅ 反復実行の制御（無限ループ防止）
- ✅ 複数ワークフローの並列実行

### 高度版で学べること

- ✅ ランタイムでの動的タスク生成（`context.next_task()`）
- ✅ 自己反復による待機・再試行（`context.next_iteration()`）
- ✅ 並列グループと実行ポリシー（BestEffortGroupPolicy、AtLeastNGroupPolicy）
- ✅ 複雑な状態管理とタスク協調

### 従来のワークフローエンジンとの比較

| 機能 | 従来のワークフロー | Graflow（基本版） | Graflow（高度版） |
|------|-------------------|------------------|------------------|
| タスク定義 | 事前に全て定義 | 事前定義 + goto | 実行時に動的生成 |
| 循環フロー | サポートなし（DAGのみ） | `goto=True`でループバック | 複数パターンのループ |
| 条件分岐 | 事前定義が必要 | 実行時の状態で判断 | 動的タスク追加で分岐 |
| 並列実行 | 固定数の並列タスク | ワークフローレベル | タスクレベル + ワークフローレベル |
| エラーハンドリング | 固定的なリトライ | 反復回数制限 | 動的な補完タスク追加 + 部分的成功許容 |

### 実用例

この設計パターンは以下のような実世界のユースケースに適用できます：

- 📰 **コンテンツ生成パイプライン**: 品質チェックを伴う反復的な生成
- 🤖 **マルチエージェントシステム**: フィードバックループを持つエージェント協調
- 🔄 **反復的改善ワークフロー**: 品質基準を満たすまでの自動改善
- 📊 **データ分析パイプライン**: データ品質に応じた動的な処理拡張
- ⚙️ **ETLパイプライン**: データソース不足時の自動補完

---

## 参考リンク

- [Graflow Documentation](https://github.com/myui/graflow)
- [Dynamic Tasks Example](../../07_dynamic_tasks/)
- [Parallel Execution Example](../../08_workflow_composition/)
