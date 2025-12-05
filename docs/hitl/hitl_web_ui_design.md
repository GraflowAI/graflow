# HITL Web UI Design

**目的**: Jinja2テンプレートとFastAPIを使用した、HITL feedbackを入力するためのWebアプリケーション

**更新日**: 2025-12-05

---

## 設計方針

### 選択されたアーキテクチャ

| 項目 | 選択 | 理由 |
|------|------|------|
| **アーキテクチャ** | 既存API拡張 (graflow/api/) | 単一アプリケーションで管理しやすい、既存のFeedbackManagerを再利用 |
| **リアルタイム更新** | ポーリング (シンプル) | 実装が簡単、Jinja2テンプレートのみで完結 |
| **CSSフレームワーク** | シンプルCSS (Pico CSS) | classなしで動作、依存関係最小、CDN経由で読み込み |
| **認証方式** | feedback_id as token | UUIDが十分にランダム、presigned URL的なアプローチ |

---

## URL構造

### エンドポイント設計

```
Web UI (HTML):
  GET  /ui/feedback/{feedback_id}           # フィードバックフォーム表示
  POST /ui/feedback/{feedback_id}/submit    # フォーム送信処理
  GET  /ui/feedback/{feedback_id}/success   # 送信完了ページ
  GET  /ui/feedback/{feedback_id}/expired   # 期限切れページ

REST API (既存):
  GET  /api/feedback                        # 一覧取得 (管理者用)
  GET  /api/feedback/{feedback_id}          # 詳細取得
  POST /api/feedback/{feedback_id}/respond  # レスポンス送信
  DELETE /api/feedback/{feedback_id}        # キャンセル
```

**URL設計の利点**:
- `/ui/` プレフィックスで Web UI であることが明確
- `/api/` との明確な区別、混乱なし
- 将来的な拡張性（例: `/ui/admin/` ダッシュボード等）

### セキュリティモデル

- **認証**: `feedback_id` (UUID) がトークンとして機能
- **アクセス制御**: URLを知っている人のみがアクセス可能 (presigned URL的)
- **有効期限**: `FeedbackRequest.expires_at` で管理
- **CSRF対策**: FastAPIの `CSRFProtect` または hidden token フィールド

**利点**:
- 追加の認証インフラ不要
- URLをメール/Slack等で安全に共有可能
- UUIDの推測困難性により十分な安全性

**注意点**:
- URLを共有する際はHTTPSを使用すること
- ログにfeedback_idを記録する際は注意が必要

---

## モジュール構造

```
graflow/
└── api/
    ├── __init__.py
    ├── __main__.py
    ├── main.py
    ├── app.py                 # FastAPI app factory (既存)
    ├── endpoints/
    │   ├── __init__.py
    │   ├── feedback.py        # REST API endpoints (既存)
    │   └── web_ui.py          # 🆕 Web UI endpoints (HTML)
    ├── schemas/
    │   ├── __init__.py
    │   ├── feedback.py        # API schemas (既存)
    │   └── web_ui.py          # 🆕 Web forms schemas
    ├── templates/             # 🆕 Jinja2 templates
    │   ├── base.html          # ベーステンプレート
    │   ├── feedback_form.html # フィードバックフォーム
    │   ├── success.html       # 送信完了ページ
    │   ├── expired.html       # 期限切れページ
    │   └── error.html         # エラーページ
    └── static/                # 🆕 静的ファイル (オプション)
        └── style.css          # カスタムCSS (オプション)
```

---

## テンプレート設計

### 1. ベーステンプレート (base.html)

```html
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}Graflow Feedback{% endblock %}</title>

    <!-- Pico CSS (Classless CSS framework) -->
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@picocss/pico@2/css/pico.min.css">

    <!-- Auto-refresh for pending status (optional) -->
    {% if auto_refresh %}
    <meta http-equiv="refresh" content="{{ refresh_interval|default(30) }}">
    {% endif %}

    {% block extra_head %}{% endblock %}
</head>
<body>
    <main class="container">
        <header>
            <h1>Graflow Feedback</h1>
        </header>

        {% block content %}{% endblock %}

        <footer>
            <small>Powered by Graflow HITL</small>
        </footer>
    </main>
</body>
</html>
```

### 2. フィードバックフォーム (feedback_form.html)

```html
{% extends "base.html" %}

{% block title %}Feedback Request - {{ request.prompt }}{% endblock %}

{% block content %}
<article>
    <header>
        <h2>Feedback Request</h2>
    </header>

    <section>
        <p><strong>Prompt:</strong></p>
        <p>{{ request.prompt }}</p>

        {% if request.metadata %}
        <details>
            <summary>Additional Information</summary>
            <pre>{{ request.metadata | tojson(indent=2) }}</pre>
        </details>
        {% endif %}
    </section>

    <section>
        <form method="POST" action="/feedback/{{ request.feedback_id }}/submit">
            {% if request.feedback_type == "approval" %}
                <!-- Approval Form -->
                <fieldset>
                    <legend>Your Decision</legend>
                    <label>
                        <input type="radio" name="approved" value="true" required>
                        Approve
                    </label>
                    <label>
                        <input type="radio" name="approved" value="false" required>
                        Reject
                    </label>
                </fieldset>

                <label for="reason">Reason (optional):</label>
                <textarea id="reason" name="reason" rows="3" placeholder="Enter your reason..."></textarea>

            {% elif request.feedback_type == "text" %}
                <!-- Text Input Form -->
                <label for="text">Your Response:</label>
                <textarea id="text" name="text" rows="5" required placeholder="Enter your response..."></textarea>

            {% elif request.feedback_type == "selection" %}
                <!-- Selection Form -->
                <fieldset>
                    <legend>Select an Option</legend>
                    {% for option in request.options %}
                    <label>
                        <input type="radio" name="selected" value="{{ option }}" required>
                        {{ option }}
                    </label>
                    {% endfor %}
                </fieldset>

            {% elif request.feedback_type == "multi_selection" %}
                <!-- Multi-Selection Form -->
                <fieldset>
                    <legend>Select Options (multiple allowed)</legend>
                    {% for option in request.options %}
                    <label>
                        <input type="checkbox" name="selected_multiple" value="{{ option }}">
                        {{ option }}
                    </label>
                    {% endfor %}
                </fieldset>

            {% elif request.feedback_type == "custom" %}
                <!-- Custom Form -->
                <label for="custom_data">Custom Data (JSON):</label>
                <textarea id="custom_data" name="custom_data" rows="5" required placeholder='{"key": "value"}'></textarea>

            {% endif %}

            <!-- Common Fields -->
            <label for="responded_by">Your Name/Email (optional):</label>
            <input type="text" id="responded_by" name="responded_by" placeholder="john@example.com">

            <!-- CSRF Token (if implemented) -->
            <input type="hidden" name="csrf_token" value="{{ csrf_token }}">

            <button type="submit">Submit Feedback</button>
        </form>
    </section>

    <footer>
        <small>Request ID: {{ request.feedback_id }}</small>
        {% if request.expires_at %}
        <small>Expires at: {{ request.expires_at }}</small>
        {% endif %}
    </footer>
</article>
{% endblock %}
```

### 3. 送信完了ページ (success.html)

```html
{% extends "base.html" %}

{% block title %}Feedback Submitted{% endblock %}

{% block content %}
<article>
    <header>
        <h2>✓ Feedback Submitted</h2>
    </header>

    <section>
        <p>Thank you! Your feedback has been successfully submitted.</p>

        {% if response %}
        <details>
            <summary>Submitted Response</summary>
            <dl>
                {% if response.approved is not none %}
                <dt>Decision:</dt>
                <dd>{{ "Approved" if response.approved else "Rejected" }}</dd>
                {% endif %}

                {% if response.text %}
                <dt>Text:</dt>
                <dd>{{ response.text }}</dd>
                {% endif %}

                {% if response.selected %}
                <dt>Selected:</dt>
                <dd>{{ response.selected }}</dd>
                {% endif %}

                {% if response.selected_multiple %}
                <dt>Selected (multiple):</dt>
                <dd>{{ response.selected_multiple | join(", ") }}</dd>
                {% endif %}

                {% if response.reason %}
                <dt>Reason:</dt>
                <dd>{{ response.reason }}</dd>
                {% endif %}

                {% if response.responded_by %}
                <dt>Responded by:</dt>
                <dd>{{ response.responded_by }}</dd>
                {% endif %}
            </dl>
        </details>
        {% endif %}
    </section>

    <footer>
        <p><small>You can safely close this window.</small></p>
    </footer>
</article>
{% endblock %}
```

### 4. 期限切れページ (expired.html)

```html
{% extends "base.html" %}

{% block title %}Request Expired{% endblock %}

{% block content %}
<article>
    <header>
        <h2>⚠ Request Expired</h2>
    </header>

    <section>
        <p>This feedback request has expired or has already been responded to.</p>

        {% if request %}
        <details>
            <summary>Request Details</summary>
            <dl>
                <dt>Status:</dt>
                <dd>{{ request.status }}</dd>

                {% if request.expires_at %}
                <dt>Expired at:</dt>
                <dd>{{ request.expires_at }}</dd>
                {% endif %}
            </dl>
        </details>
        {% endif %}
    </section>

    <footer>
        <p><small>If you believe this is an error, please contact the workflow administrator.</small></p>
    </footer>
</article>
{% endblock %}
```

### 5. エラーページ (error.html)

```html
{% extends "base.html" %}

{% block title %}Error{% endblock %}

{% block content %}
<article>
    <header>
        <h2>❌ Error</h2>
    </header>

    <section>
        <p>{{ error_message | default("An unexpected error occurred.") }}</p>

        {% if error_detail %}
        <details>
            <summary>Error Details</summary>
            <pre>{{ error_detail }}</pre>
        </details>
        {% endif %}
    </section>

    <footer>
        <a href="javascript:history.back()">Go Back</a>
    </footer>
</article>
{% endblock %}
```

---

## エンドポイント実装

### graflow/api/endpoints/web_ui.py (新規作成)

```python
"""Web UI endpoints for HITL feedback."""

from typing import Optional

from fastapi import APIRouter, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from datetime import datetime

from graflow.hitl.types import FeedbackResponse

router = APIRouter(tags=["web"])

# Jinja2テンプレート設定は app.py で行う
# templates = Jinja2Templates(directory="graflow/api/templates")


@router.get("/feedback/{feedback_id}", response_class=HTMLResponse)
async def show_feedback_form(
    request: Request,
    feedback_id: str,
):
    """Display feedback form for the given feedback_id.

    Args:
        request: FastAPI request object
        feedback_id: Feedback request ID (acts as authentication token)

    Returns:
        HTML response with feedback form
    """
    # Get feedback manager from app state
    feedback_manager = request.app.state.feedback_manager

    # Get feedback request
    feedback_request = feedback_manager.get_request(feedback_id)

    if not feedback_request:
        raise HTTPException(status_code=404, detail="Feedback request not found")

    # Check if already responded
    if feedback_request.status != "pending":
        return RedirectResponse(url=f"/feedback/{feedback_id}/expired")

    # Check if expired
    if feedback_request.expires_at:
        expires_dt = datetime.fromisoformat(feedback_request.expires_at)
        if datetime.now() > expires_dt:
            return RedirectResponse(url=f"/feedback/{feedback_id}/expired")

    # Render form
    templates = request.app.state.templates
    return templates.TemplateResponse(
        "feedback_form.html",
        {
            "request": request,
            "feedback_request": feedback_request,
            "csrf_token": "TODO",  # TODO: Implement CSRF protection
        },
    )


@router.post("/feedback/{feedback_id}/submit")
async def submit_feedback(
    request: Request,
    feedback_id: str,
    approved: Optional[str] = Form(None),
    reason: Optional[str] = Form(None),
    text: Optional[str] = Form(None),
    selected: Optional[str] = Form(None),
    selected_multiple: Optional[list[str]] = Form(None),
    custom_data: Optional[str] = Form(None),
    responded_by: Optional[str] = Form(None),
):
    """Process submitted feedback form.

    Args:
        request: FastAPI request object
        feedback_id: Feedback request ID
        approved: Approval decision (for approval type)
        reason: Reason for decision
        text: Text input (for text type)
        selected: Selected option (for selection type)
        selected_multiple: Selected options (for multi_selection type)
        custom_data: Custom JSON data (for custom type)
        responded_by: User identifier

    Returns:
        Redirect to success page
    """
    # Get feedback manager from app state
    feedback_manager = request.app.state.feedback_manager

    # Get feedback request
    feedback_request = feedback_manager.get_request(feedback_id)

    if not feedback_request:
        raise HTTPException(status_code=404, detail="Feedback request not found")

    # Check if already responded
    if feedback_request.status != "pending":
        return RedirectResponse(url=f"/feedback/{feedback_id}/expired")

    # Build response based on feedback type
    response_data = {
        "feedback_id": feedback_id,
        "response_type": feedback_request.feedback_type,
        "responded_at": datetime.now().isoformat(),
        "responded_by": responded_by,
    }

    # Add type-specific fields
    if feedback_request.feedback_type == "approval":
        response_data["approved"] = approved == "true" if approved else None
        response_data["reason"] = reason

    elif feedback_request.feedback_type == "text":
        response_data["text"] = text

    elif feedback_request.feedback_type == "selection":
        response_data["selected"] = selected

    elif feedback_request.feedback_type == "multi_selection":
        response_data["selected_multiple"] = selected_multiple or []

    elif feedback_request.feedback_type == "custom":
        # Parse JSON
        import json
        try:
            response_data["custom_data"] = json.loads(custom_data) if custom_data else {}
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON in custom_data")

    # Create response object
    feedback_response = FeedbackResponse(**response_data)

    # Submit response via FeedbackManager
    success = feedback_manager.provide_feedback(feedback_id, feedback_response)

    if not success:
        raise HTTPException(status_code=500, detail="Failed to submit feedback")

    # Redirect to success page
    return RedirectResponse(
        url=f"/feedback/{feedback_id}/success",
        status_code=303,  # See Other (POST -> GET redirect)
    )


@router.get("/feedback/{feedback_id}/success", response_class=HTMLResponse)
async def show_success_page(
    request: Request,
    feedback_id: str,
):
    """Display success page after feedback submission.

    Args:
        request: FastAPI request object
        feedback_id: Feedback request ID

    Returns:
        HTML response with success message
    """
    # Get feedback manager from app state
    feedback_manager = request.app.state.feedback_manager

    # Get response
    feedback_response = feedback_manager.get_response(feedback_id)

    # Render success page
    templates = request.app.state.templates
    return templates.TemplateResponse(
        "success.html",
        {
            "request": request,
            "response": feedback_response,
        },
    )


@router.get("/feedback/{feedback_id}/expired", response_class=HTMLResponse)
async def show_expired_page(
    request: Request,
    feedback_id: str,
):
    """Display expired/already responded page.

    Args:
        request: FastAPI request object
        feedback_id: Feedback request ID

    Returns:
        HTML response with expired message
    """
    # Get feedback manager from app state
    feedback_manager = request.app.state.feedback_manager

    # Get request (may be None)
    feedback_request = feedback_manager.get_request(feedback_id)

    # Render expired page
    templates = request.app.state.templates
    return templates.TemplateResponse(
        "expired.html",
        {
            "request": request,
            "feedback_request": feedback_request,
        },
    )
```

---

## app.py の更新

`graflow/api/app.py` に以下を追加：

```python
from pathlib import Path
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

def create_feedback_api(
    feedback_backend: str | FeedbackBackend = "filesystem",
    feedback_config: Optional[dict] = None,
    title: str = "Graflow Feedback API",
    enable_cors: bool = True,
    cors_origins: Optional[list[str]] = None,
    enable_web_ui: bool = True,  # 🆕 Web UI有効化フラグ
) -> FastAPI:
    """Create FastAPI application for feedback management.

    Args:
        feedback_backend: Backend type ("filesystem" or "redis")
        feedback_config: Backend-specific configuration
        title: API title
        enable_cors: Enable CORS middleware
        cors_origins: Allowed CORS origins
        enable_web_ui: Enable Web UI endpoints (default: True)

    Returns:
        FastAPI application instance
    """
    app = FastAPI(
        title=title,
        description="Graflow Human-in-the-Loop Feedback API with Web UI",
        version="1.0.0",
    )

    # ... existing code ...

    # 🆕 Web UI setup
    if enable_web_ui:
        # Setup Jinja2 templates
        template_dir = Path(__file__).parent / "templates"
        templates = Jinja2Templates(directory=str(template_dir))
        app.state.templates = templates

        # Setup static files (optional)
        static_dir = Path(__file__).parent / "static"
        if static_dir.exists():
            app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

        # Include web UI router
        from graflow.api.endpoints import web
        app.include_router(web.router)

    # Include API router
    app.include_router(feedback.router, prefix="/api")

    return app
```

---

## 使用例

### 1. サーバー起動

```bash
# Web UI有効
python -m graflow.api --backend redis --redis-host localhost
```

### 2. ワークフローからの利用

```python
from graflow.core.decorators import task

@task(inject_context=True)
def request_approval(context):
    response = context.request_feedback(
        feedback_type="approval",
        prompt="Approve deployment to production?",
        timeout=300.0,
    )

    feedback_id = response.feedback_id  # または exception.feedback_id

    # Web UI URL
    web_url = f"http://localhost:8000/ui/feedback/{feedback_id}"
    print(f"Please provide feedback at: {web_url}")

    # URLをメール、Slack等で送信
    # send_notification(web_url)

    return response.approved
```

### 3. ユーザーのフロー

1. ユーザーがURLを受け取る（メール/Slack等）
2. ブラウザでURLを開く: `http://localhost:8000/ui/feedback/{feedback_id}`
3. フォームが表示される（認証済み状態）
4. フィードバックを入力して送信
5. 成功ページが表示される
6. ワークフローが再開される

---

## セキュリティ考慮事項

### 1. feedback_id as Token

**強度**:
- UUID v4: 122ビットのランダム性
- 総数: 2^122 ≈ 5.3 × 10^36
- ブルートフォース攻撃は実質不可能

**推奨事項**:
- HTTPSを使用（URLの傍受防止）
- ログに `feedback_id` を記録する際は注意
- 有効期限を適切に設定（`expires_at`）
- レート制限の実装（オプション）

### 2. CSRF対策

**Option A: SameSite Cookie (推奨)**
```python
# FastAPI session middleware
from starlette.middleware.sessions import SessionMiddleware

app.add_middleware(SessionMiddleware, secret_key="your-secret-key")
```

**Option B: Double Submit Cookie**
```python
# Hidden token in form
<input type="hidden" name="csrf_token" value="{{ csrf_token }}">
```

**Option C: なし**
- feedback_id自体がトークンなので、簡易ケースでは不要とも言える
- ただし、XSS脆弱性がある場合のリスクあり

### 3. Input Validation

- Pydantic でバリデーション
- XSS対策: Jinja2の自動エスケープ
- SQL Injection: 該当なし（NoSQLバックエンド）

---

## ポーリング実装

### Option A: メタタグ Auto-Refresh

```html
<!-- 30秒ごとに自動リフレッシュ（ステータスが pending の場合のみ） -->
{% if feedback_request.status == "pending" %}
<meta http-equiv="refresh" content="30">
{% endif %}
```

**利点**: JavaScript不要、シンプル
**欠点**: ページ全体リロード、UX劣る

### Option B: JavaScript Polling

```html
<script>
// 10秒ごとにステータスチェック
setInterval(async () => {
    const response = await fetch('/api/feedback/{{ feedback_id }}');
    const data = await response.json();

    if (data.status !== 'pending') {
        location.reload();
    }
}, 10000);
</script>
```

**利点**: 柔軟、UX良い
**欠点**: JavaScript必要

**推奨**: Option A（シンプルさ優先）、必要に応じてOption Bに移行

---

## テスト計画

### 1. 単体テスト

```python
# tests/hitl/test_web_ui.py
from fastapi.testclient import TestClient

def test_show_feedback_form(client, feedback_manager):
    """Test feedback form display."""
    # Create pending request
    request = create_test_request()
    feedback_manager.store_request(request)

    # GET form
    response = client.get(f"/feedback/{request.feedback_id}")
    assert response.status_code == 200
    assert "Feedback Request" in response.text

def test_submit_approval(client, feedback_manager):
    """Test approval submission."""
    request = create_test_request(feedback_type="approval")
    feedback_manager.store_request(request)

    # POST submission
    response = client.post(
        f"/feedback/{request.feedback_id}/submit",
        data={"approved": "true", "reason": "LGTM"},
    )
    assert response.status_code == 303  # Redirect

    # Check response stored
    feedback_response = feedback_manager.get_response(request.feedback_id)
    assert feedback_response.approved is True
```

### 2. E2Eテスト

```python
def test_full_workflow_with_web_ui(tmp_path):
    """Test complete workflow with Web UI feedback."""
    # Start workflow in background thread
    # Wait for feedback request
    # Simulate browser interaction via TestClient
    # Verify workflow completion
```

---

## 実装チェックリスト

### Phase 1: 基本実装 ✅ **完了**

- [x] `graflow/api/templates/` ディレクトリ作成
- [x] `graflow/api/templates/base.html` 作成
- [x] `graflow/api/templates/feedback_form.html` 作成
- [x] `graflow/api/templates/success.html` 作成
- [x] `graflow/api/templates/expired.html` 作成
- [x] `graflow/api/templates/error.html` 作成
- [x] `graflow/api/endpoints/web_ui.py` 作成
  - [x] `GET /ui/feedback/{feedback_id}` エンドポイント
  - [x] `POST /ui/feedback/{feedback_id}/submit` エンドポイント
  - [x] `GET /ui/feedback/{feedback_id}/success` エンドポイント
  - [x] `GET /ui/feedback/{feedback_id}/expired` エンドポイント
- [x] `graflow/api/app.py` 更新
  - [x] Jinja2Templates設定
  - [x] Web UIルーター統合
  - [x] `enable_web_ui` パラメータ追加
- [x] `graflow/api/main.py` 更新
  - [x] `--disable-web-ui` CLIオプション追加

**実装内容**:
- 全フィードバックタイプ対応（approval, text, selection, multi_selection, custom）
- Pico CSS使用（CDN経由、依存なし）
- レスポンシブデザイン
- 期限切れ・既回答の処理
- エラーハンドリング

### Phase 2: セキュリティ ⏳ **部分的**

- [ ] CSRF対策実装 - **TODO**（現在はなし）
- [x] Input validation強化（Pydantic自動バリデーション）
- [x] XSS対策確認（Jinja2のauto-escape有効）
- [ ] レート制限（オプション） - **TODO**

**現状**:
- XSS: Jinja2の自動エスケープで保護済み
- Input Validation: Pydantic modelsで型チェック済み
- CSRF: 未実装（feedback_idがトークンとして機能するため低リスク）

### Phase 3: テスト ⏳ **部分的**

- [x] `tests/hitl/test_web_ui.py` 作成（23テストケース）
- [x] 各エンドポイントの単体テスト
  - [x] フォーム表示（全フィードバックタイプ）
  - [x] フォーム送信（全フィードバックタイプ）
  - [x] 成功ページ表示
  - [x] 期限切れページ表示
  - [x] エラーケース（404, invalid JSON等）
  - [x] Web UI有効/無効切り替え
- [ ] E2Eテスト - **TODO**
- [ ] セキュリティテスト - **TODO**

**テストカバレッジ**: 主要機能は全てカバー済み

### Phase 4: ドキュメント ✅ **完了**

- [x] `graflow/api/README.md` 更新（Web UI使用法）
  - [x] Web UIセクション追加
  - [x] 使用方法説明
  - [x] エンドポイント一覧
  - [x] サンプルコード
- [x] 設計ドキュメント（`docs/hitl_web_ui_design.md`）
- [ ] `examples/11_hitl/` に Web UI使用例追加 - **TODO**（オプション）
- [ ] スクリーンショット追加 - **TODO**（オプション）

---

## 将来の拡張

### Phase 5: 高度な機能（オプション）

- [ ] WebSocket/SSE によるリアルタイム更新
- [ ] 管理ダッシュボード（全リクエスト一覧）
- [ ] フィードバック履歴表示
- [ ] 複数言語対応（i18n）
- [ ] ダークモード対応
- [ ] アクセシビリティ改善（ARIA属性）
- [ ] PWA対応（オフライン動作）

---

## まとめ

### 設計の特徴

✅ **シンプル**: Jinja2テンプレート + Pico CSS、依存最小
✅ **セキュア**: feedback_id をトークンとして利用、HTTPS推奨
✅ **拡張性**: 既存API拡張、将来的にWebSocket等追加可能
✅ **実用的**: ポーリングでリアルタイム性確保、複雑さ回避

### 実装状況

✅ **Phase 1 (基本実装)**: 完了
✅ **Phase 4 (ドキュメント)**: 完了
⏳ **Phase 2 (セキュリティ)**: 部分的（XSS対策済み、CSRF未実装）
⏳ **Phase 3 (テスト)**: 部分的（単体テスト完了、E2E未実装）

### 次のステップ（オプション）

現在の実装で基本機能は全て動作します。以下は必要に応じて追加可能：

1. **Phase 2完了**: CSRF対策、レート制限
2. **Phase 3完了**: E2Eテスト、セキュリティテスト
3. **実例追加**: `examples/11_hitl/` に Web UI使用例
4. **Phase 5**: WebSocket、管理ダッシュボード等の高度な機能

---

**設計ステータス**: ✅ **Phase 1実装完了 - 稼働中**
**実装バージョン**: 1.0
**作成日**: 2025-12-05
**実装完了日**: 2025-12-05

**実装済み機能**:
- ✅ 全フィードバックタイプ対応（5種類）
- ✅ Jinja2テンプレートベースのUI
- ✅ Pico CSS によるレスポンシブデザイン
- ✅ feedback_id認証
- ✅ CLI統合（--disable-web-ui）
- ✅ 包括的な単体テスト（23ケース）
- ✅ ドキュメント完備

**動作確認済み**:
- FastAPI app作成: ✅
- テンプレート読み込み: ✅
- エンドポイント登録: ✅
- 基本機能テスト: ✅

**実装ファイル**:
```
graflow/api/
├── templates/              # 🆕 Jinja2テンプレート
│   ├── base.html          # ベーステンプレート
│   ├── feedback_form.html # フィードバックフォーム（5種類対応）
│   ├── success.html       # 送信完了ページ
│   ├── expired.html       # 期限切れページ
│   └── error.html         # エラーページ
├── endpoints/
│   ├── feedback.py        # REST API（既存）
│   └── web_ui.py          # 🆕 Web UIエンドポイント（4エンドポイント）
├── app.py                 # 🔄 Jinja2/Web UI統合追加
├── main.py                # 🔄 --disable-web-ui オプション追加
└── README.md              # 🔄 Web UIセクション追加

tests/hitl/
└── test_web_ui.py         # 🆕 Web UIテスト（23ケース）

docs/
├── hitl_design.md         # 既存（REST API設計）
└── hitl_web_ui_design.md  # 🆕 本ドキュメント
```
