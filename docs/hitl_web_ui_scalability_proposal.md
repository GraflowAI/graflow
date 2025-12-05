# Web UI拡張性改善提案

**課題**: 現在の`graflow/api/templates/`構成はHITL専用で、将来的な拡張（管理画面等）を考慮していない

**作成日**: 2025-12-05

---

## 現在の構成の問題点

### Current Structure (HITL only)

```
graflow/api/
├── templates/              # フラット構造
│   ├── base.html          # 共通ベース
│   ├── feedback_form.html # HITL専用
│   ├── success.html       # HITL専用
│   ├── expired.html       # HITL専用
│   └── error.html         # HITL専用？共通？
└── endpoints/
    ├── feedback.py        # REST API
    └── web_ui.py          # HITL Web UI
```

**問題点**:
1. ✗ 全テンプレートがフラットに配置 → スケールしない
2. ✗ HITL専用と共通が区別されていない
3. ✗ 将来的に`/ui/admin/`等を追加すると混在する
4. ✗ テンプレート名が衝突するリスク（例: `admin/error.html` vs `feedback/error.html`）

---

## 改善提案

### 提案1: UI機能別ディレクトリ構造（推奨）

```
graflow/api/
├── templates/
│   ├── common/                # 共通テンプレート
│   │   ├── base.html          # ベーステンプレート
│   │   ├── components/        # 共通コンポーネント（オプション）
│   │   │   ├── header.html
│   │   │   ├── footer.html
│   │   │   └── nav.html
│   │   └── error.html         # 汎用エラーページ
│   │
│   ├── feedback/              # HITL フィードバック UI
│   │   ├── form.html          # feedback_form.html → form.html
│   │   ├── success.html
│   │   └── expired.html
│   │
│   └── admin/                 # 管理画面（将来）
│       ├── dashboard.html
│       ├── feedback_list.html
│       ├── workflow_status.html
│       └── settings.html
│
└── endpoints/
    ├── feedback.py            # REST API（既存）
    ├── web_ui.py              # HITL Web UI（既存） → feedback_ui.py にリネーム推奨
    └── admin_ui.py            # 管理画面 UI（将来）
```

**URL対応**:
- `/ui/feedback/{id}` → `templates/feedback/form.html`
- `/ui/admin/dashboard` → `templates/admin/dashboard.html`

**利点**:
- ✅ 機能ごとに明確に分離
- ✅ テンプレート名の衝突を回避
- ✅ 共通部品と機能固有部品の区別が明確
- ✅ 新しいUI機能追加が容易

**Jinja2テンプレート継承**:
```jinja2
{# templates/feedback/form.html #}
{% extends "shared/base.html" %}

{# templates/admin/dashboard.html #}
{% extends "shared/base.html" %}
```

---

### 提案2: エンドポイント構造準拠（より厳密）

```
graflow/api/
├── templates/
│   ├── ui/                    # /ui/* に対応
│   │   ├── feedback/
│   │   │   ├── form.html
│   │   │   ├── success.html
│   │   │   └── expired.html
│   │   └── admin/
│   │       └── ...
│   │
│   └── shared/
│       ├── base.html
│       └── components/
│
└── endpoints/
    ├── ui/
    │   ├── __init__.py
    │   ├── feedback.py        # /ui/feedback/*
    │   └── admin.py           # /ui/admin/*
    └── api/
        └── feedback.py        # /api/feedback/*
```

**URL対応**:
- `/ui/feedback/{id}` → `templates/ui/feedback/form.html`
- `/ui/admin/dashboard` → `templates/ui/admin/dashboard.html`

**利点**:
- ✅ URLパスとディレクトリ構造が完全一致
- ✅ エンドポイントコードとテンプレートの対応が直感的

**欠点**:
- ✗ ネストが深い（`templates/ui/feedback/`）
- ✗ やや冗長

---

### 提案3: モジュール化（大規模向け）

```
graflow/
├── api/
│   ├── __init__.py
│   ├── app.py
│   └── main.py
│
├── ui/                        # Web UI専用モジュール
│   ├── __init__.py
│   │
│   ├── feedback/              # HITL フィードバックモジュール
│   │   ├── __init__.py
│   │   ├── routes.py          # FastAPI router
│   │   ├── schemas.py         # Pydantic models
│   │   └── templates/
│   │       ├── form.html
│   │       ├── success.html
│   │       └── expired.html
│   │
│   ├── admin/                 # 管理画面モジュール
│   │   ├── __init__.py
│   │   ├── routes.py
│   │   ├── schemas.py
│   │   └── templates/
│   │       └── ...
│   │
│   └── shared/
│       ├── static/            # CSS, JS
│       └── templates/
│           ├── base.html
│           └── components/
│
└── hitl/
    └── ...
```

**利点**:
- ✅ 完全なモジュール分離
- ✅ 各UI機能が独立したパッケージ
- ✅ テンプレート、ルーター、スキーマが同じ場所に配置
- ✅ 大規模プロジェクトに最適

**欠点**:
- ✗ 大きな構造変更が必要
- ✗ 小〜中規模プロジェクトには過剰

---

## 推奨: 提案1（UI機能別ディレクトリ構造）

### 理由

1. **シンプル**: 現在の構造からの移行が容易
2. **明確**: 機能ごとの境界が明確
3. **拡張性**: 新機能追加が簡単
4. **保守性**: テンプレートの場所が予測しやすい

### 移行手順

#### Step 1: ディレクトリ構造変更

```bash
# 新しいディレクトリ作成
mkdir -p graflow/api/templates/shared
mkdir -p graflow/api/templates/feedback

# ファイル移動
mv graflow/api/templates/base.html graflow/api/templates/shared/
mv graflow/api/templates/feedback_form.html graflow/api/templates/feedback/form.html
mv graflow/api/templates/success.html graflow/api/templates/feedback/success.html
mv graflow/api/templates/expired.html graflow/api/templates/feedback/expired.html

# error.htmlの扱い
# Option A: 共通エラーページとして shared/ に移動
mv graflow/api/templates/error.html graflow/api/templates/shared/

# Option B: feedback専用として feedback/ に移動
# mv graflow/api/templates/error.html graflow/api/templates/feedback/
```

#### Step 2: テンプレートパス更新

**`graflow/api/endpoints/web_ui.py`** (または `feedback_ui.py` にリネーム):

```python
# Before
return templates.TemplateResponse(
    "feedback_form.html",
    {"request": request, "feedback_request": feedback_request}
)

# After
return templates.TemplateResponse(
    "feedback/form.html",  # パス変更
    {"request": request, "feedback_request": feedback_request}
)
```

全てのテンプレート参照を更新:
- `"feedback_form.html"` → `"feedback/form.html"`
- `"success.html"` → `"feedback/success.html"`
- `"expired.html"` → `"feedback/expired.html"`
- `"error.html"` → `"shared/error.html"` (または `"feedback/error.html"`)

#### Step 3: テンプレート継承パス更新

**`templates/feedback/form.html`**:

```jinja2
{# Before #}
{% extends "base.html" %}

{# After #}
{% extends "shared/base.html" %}
```

全てのテンプレートで `extends` を更新。

#### Step 4: テスト更新

`tests/hitl/test_web_ui.py` で、テンプレートパスの変更を確認するテストは不要（内部実装）。
ただし、レスポンスが正常にレンダリングされることを確認。

---

## 将来的な拡張例

### 管理画面追加（/ui/admin/）

```
graflow/api/
├── templates/
│   ├── shared/
│   │   └── base.html
│   ├── feedback/              # HITL
│   │   └── ...
│   └── admin/                 # 🆕 管理画面
│       ├── dashboard.html     # メインダッシュボード
│       ├── feedback_list.html # フィードバック一覧
│       ├── workflow_list.html # ワークフロー一覧
│       └── settings.html      # 設定
│
└── endpoints/
    ├── feedback.py            # REST API
    ├── feedback_ui.py         # HITL Web UI
    └── admin_ui.py            # 🆕 管理画面 UI
```

**`endpoints/admin_ui.py`**:
```python
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

router = APIRouter(tags=["admin-ui"])

@router.get("/ui/admin/dashboard", response_class=HTMLResponse)
async def admin_dashboard(request: Request):
    templates = request.app.state.templates

    # 全フィードバックリクエスト取得
    feedback_manager = request.app.state.feedback_manager
    all_requests = feedback_manager.list_pending_requests()

    return templates.TemplateResponse(
        "admin/dashboard.html",  # templates/admin/dashboard.html
        {
            "request": request,
            "pending_count": len(all_requests),
            "requests": all_requests
        }
    )
```

### ワークフロー可視化 UI（/ui/workflow/）

```
templates/
├── shared/
├── feedback/
├── admin/
└── workflow/              # 🆕 ワークフロー可視化
    ├── list.html          # ワークフロー一覧
    ├── detail.html        # ワークフロー詳細
    └── visualize.html     # グラフ可視化
```

---

## エンドポイント命名規則の提案

現在の `web_ui.py` は汎用的すぎるため、リネームを推奨：

### Before
```
endpoints/
├── feedback.py        # REST API
└── web_ui.py          # HITL Web UI（名前が汎用的）
```

### After
```
endpoints/
├── feedback.py        # REST API for HITL
├── feedback_ui.py     # Web UI for HITL feedback (旧 web_ui.py)
├── admin_ui.py        # Web UI for admin dashboard (将来)
└── workflow_ui.py     # Web UI for workflow visualization (将来)
```

または、`ui/` サブディレクトリにまとめる:

```
endpoints/
├── api/
│   └── feedback.py    # REST API
└── ui/
    ├── feedback.py    # HITL Web UI
    ├── admin.py       # Admin dashboard
    └── workflow.py    # Workflow visualization
```

---

## まとめ

### 推奨構成

```
graflow/api/
├── templates/
│   ├── shared/                # 共通テンプレート
│   │   ├── base.html
│   │   └── error.html
│   ├── feedback/              # HITL feedback
│   │   ├── form.html
│   │   ├── success.html
│   │   └── expired.html
│   └── admin/                 # 管理画面（将来）
│       └── ...
│
└── endpoints/
    ├── feedback.py            # REST API
    ├── feedback_ui.py         # HITL Web UI (renamed from web_ui.py)
    └── admin_ui.py            # Admin UI (将来)
```

### 移行タスク ✅ **完了**

- [x] ディレクトリ構造変更
  - [x] `templates/common/` 作成
  - [x] `templates/feedback/` 作成
  - [x] ファイル移動
    - [x] `base.html` → `common/base.html`
    - [x] `error.html` → `common/error.html`
    - [x] `feedback_form.html` → `feedback/form.html`
    - [x] `success.html` → `feedback/success.html`
    - [x] `expired.html` → `feedback/expired.html`
- [x] コード更新
  - [x] テンプレートパス更新（3箇所: form, success, expired）
  - [x] `extends` パス更新（全テンプレート: 3ファイル）
- [x] コード更新（追加）
  - [x] `endpoints/web_ui.py` → `feedback_ui.py` リネーム
  - [x] `app.py` router import 更新
- [x] テスト
  - [x] 動作確認（app作成成功）
  - [x] ルート登録確認（全エンドポイント正常）
- [ ] オプショナル（後で実施可能）
  - [ ] テストケース更新（必要に応じて）
  - [ ] ドキュメント更新
    - [ ] `docs/hitl_web_ui_design.md`
    - [ ] `graflow/api/README.md`

### 影響範囲

- **低リスク**: テンプレートパス変更のみ
- **後方互換性**: エンドポイントURL (`/ui/feedback/{id}`) は変更なし
- **所要時間**: 30分程度

---

**提案ステータス**: ✅ **実装完了**
**優先度**: Medium（将来の拡張を見据えた場合は High）
**作成日**: 2025-12-05
**実装完了日**: 2025-12-05

---

## 実装完了サマリー

### 新しいディレクトリ構造

```
graflow/api/
├── templates/
│   ├── common/                # 共通テンプレート
│   │   ├── base.html
│   │   └── error.html
│   └── feedback/              # HITL フィードバック
│       ├── form.html          # 旧 feedback_form.html
│       ├── success.html
│       └── expired.html
└── endpoints/
    ├── feedback.py            # REST API
    └── feedback_ui.py         # 旧 web_ui.py
```

### 変更内容

1. **テンプレート再編成**:
   - `common/` - 共通テンプレート（base.html, error.html）
   - `feedback/` - HITL専用テンプレート
   - `feedback_form.html` → `form.html` にリネーム

2. **エンドポイントリネーム**:
   - `web_ui.py` → `feedback_ui.py` （より明確な命名）

3. **パス更新**:
   - テンプレート参照: `"feedback/form.html"` 等
   - テンプレート継承: `{% extends "common/base.html" %}`

### 動作確認済み

- ✅ FastAPI app作成成功
- ✅ 全ルート正常登録（8エンドポイント）
- ✅ テンプレート読み込み成功

### 将来の拡張例

```
templates/
├── common/
├── feedback/      # HITL（実装済み）
└── admin/         # 管理画面（将来追加可能）
    ├── dashboard.html
    ├── feedback_list.html
    └── settings.html
```

この構造により、新しいUI機能の追加が容易になりました。
