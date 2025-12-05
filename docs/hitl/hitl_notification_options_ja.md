# HITL通知設計 - 詳細分析

**文書**: HITLフィードバック用通知オプション  
**日付**: 2025-01-28  
**関連**: hitl_design.md Option 6

---

## 概要

このドキュメントは、HITL（Human-in-the-Loop）フィードバック応答の通知メカニズムを詳しく分析する。外部API経由でフィードバックが提供された場合、ワークフローが再開できるよう通知が必要となる。

---

## 問題設定

**課題**: 外部からフィードバックが提供されたとき、実行中（またはチェックポイントで停止中）のワークフローへどのように通知するか？

**シナリオ**:
1. **アクティブポーリング**: ワークフローがタイムアウト内で積極的にポーリングしている
2. **チェックポイント待機**: ワークフローがチェックポイントを作成して終了（タイムアウト発生）
3. **分散ワーカー**: 複数ワーカーがワークフロー再開を行う可能性

**要件**:
- アクティブポーリング時の低レイテンシ
- 確実な配送（通知ロストの防止）
- 分散環境でのサポート
- 追加インフラ依存の最小化
- 通知失敗時の優雅なデグレード

---

## Option 6A: Redis Pub/Sub のみ

### 実装

```python
class FeedbackManager:
    def provide_feedback(self, feedback_id, response):
        # Store response
        self._store_response(response)

        # Publish notification via Redis Pub/Sub
        if self.backend == "redis":
            self._redis_client.publish(
                f"feedback:{feedback_id}",
                "completed"
            )
```

```python
# Worker polling loop
def _poll_for_response(self, feedback_id, timeout):
    # Subscribe to Pub/Sub channel
    pubsub = self._redis_client.pubsub()
    pubsub.subscribe(f"feedback:{feedback_id}")

    elapsed = 0
    while elapsed < timeout:
        # Wait for notification with timeout
        message = pubsub.get_message(timeout=0.5)
        if message and message["type"] == "message":
            # Notification received, fetch response
            response = self._get_response(feedback_id)
            if response:
                return response

        elapsed += 0.5

    return None  # Timeout
```

### Pros ✅

1. **低レイテンシ**: ほぼ即時通知（通常 < 100ms）
2. **組み込み**: Redis利用中なら追加インフラ不要
3. **スケーラブル**: 複数サブスクライバ（複数ワーカー）をサポート
4. **シンプル**: 標準的なPub/Subパターン

### Cons ❌

1. **Redis依存**: Redisバックエンドが必須
2. **永続化なし**: サブスクライバがいないとメッセージが失われる
3. **Fire-and-forget**: 配送保証なし
4. **メモリバックエンド非対応**: Redis専用

### ユースケース

- ✅ Redisバックエンドの分散ワークフロー
- ✅ アクティブポーリング（1分未満タイムアウト）
- ❌ 長期チェックポイント（数時間/数日）
- ❌ メモリバックエンド

### リスク分析

**リスク**: サブスクライバ不在時に通知が失われる

**シナリオ**:
```
1. ワーカーがフィードバックを30秒ポーリング
2. タイムアウト → チェックポイント作成 → ワーカー終了
3. 5分後に人がフィードバック提供
4. Pub/Subメッセージ送信 → サブスクライバなし → ロスト
5. ワーカー2がチェックポイントから再開
6. ストレージをポーリング → レスポンスを発見 ✅
```

**緩和策**: レスポンスはバックエンドに保存されるため、再開時のポーリングで取得可能。Pub/Subはアクティブポーリングの最適化用途。

---

## Option 6B: Webhook のみ

### 実装

```python
@dataclass
class FeedbackRequest:
    webhook_url: Optional[str] = None  # Callback URL
    webhook_headers: Optional[dict] = None  # Custom headers
    webhook_retry_count: int = 3  # Retry attempts

class FeedbackManager:
    def provide_feedback(self, feedback_id, response):
        # Store response
        self._store_response(response)

        # Call webhook if configured
        request = self._get_request(feedback_id)
        if request.webhook_url:
            self._call_webhook(request, response)

    def _call_webhook(self, request, response):
        import requests

        url = request.webhook_url
        headers = request.webhook_headers or {}
        headers["Content-Type"] = "application/json"

        payload = {
            "feedback_id": request.feedback_id,
            "task_id": request.task_id,
            "session_id": request.session_id,
            "response": response.to_dict()
        }

        # Retry logic
        for attempt in range(request.webhook_retry_count):
            try:
                resp = requests.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=5
                )
                if resp.status_code < 300:
                    print(f"[FeedbackManager] Webhook delivered: {url}")
                    return
                else:
                    print(f"[FeedbackManager] Webhook failed: {resp.status_code}")
            except Exception as e:
                print(f"[FeedbackManager] Webhook error (attempt {attempt+1}): {e}")
                time.sleep(2 ** attempt)  # Exponential backoff

        print(f"[FeedbackManager] Webhook failed after {request.webhook_retry_count} attempts")
```

### Pros ✅

1. **Redis非依存**: メモリバックエンドでも利用可
2. **外部連携**: 外部システム（Slack, email等）へ通知
3. **柔軟**: 任意のアクションをトリガー可能
4. **永続リトライ**: エクスポネンシャルバックオフ付きリトライ

### Cons ❌

1. **ネットワーク依存**: ネットワーク接続が必須
2. **高レイテンシ**: HTTPオーバーヘッド（100〜1000ms+）
3. **複雑性**: エラーハンドリング、リトライ、セキュリティ
4. **ブロードキャスト不可**: 単一Webhookエンドポイント（複数購読者不可）

### ユースケース

- ✅ 外部システム連携（Slack, PagerDutyなど）
- ✅ Email/SMS通知
- ✅ メモリバックエンド
- ❌ 低レイテンシ要求（< 100ms）

### セキュリティ考慮

**認証**:
```python
@dataclass
class FeedbackRequest:
    webhook_secret: Optional[str] = None  # HMAC secret

# Generate HMAC signature
import hmac
import hashlib

signature = hmac.new(
    webhook_secret.encode(),
    json.dumps(payload).encode(),
    hashlib.sha256
).hexdigest()

headers["X-Graflow-Signature"] = signature
```

**Webhook受信側の検証**:
```python
@app.post("/webhook/feedback")
def receive_feedback_webhook(request: Request):
    # Verify signature
    signature = request.headers.get("X-Graflow-Signature")
    expected = hmac.new(secret, request.body, hashlib.sha256).hexdigest()

    if not hmac.compare_digest(signature, expected):
        raise HTTPException(401, "Invalid signature")

    # Process webhook
    ...
```

---

## Option 6C: Pub/Sub と Webhook の併用

### 実装

```python
class FeedbackManager:
    def provide_feedback(self, feedback_id, response):
        # Store response
        self._store_response(response)

        request = self._get_request(feedback_id)

        # Publish to Redis Pub/Sub (if Redis backend)
        if self.backend == "redis":
            self._redis_client.publish(
                f"feedback:{feedback_id}",
                "completed"
            )

        # Call webhook (if configured)
        if request.webhook_url:
            # Call webhook in background thread
            threading.Thread(
                target=self._call_webhook,
                args=(request, response),
                daemon=True
            ).start()
```

### Pros ✅

1. **いいとこ取り**: 低レイテンシ＋外部連携
2. **冗長性**: 複数チャネルで通知
3. **柔軟性**: ユースケースに応じて使い分け
4. **後方互換**: 既存のRedis利用者に影響なし

### Cons ❌

1. **複雑性**: コードパスと失敗パターンが増える
2. **テスト負荷**: 両方のメカニズムをテスト必要
3. **重複通知の可能性**: 同じ通知が複数経路で送られる

### ユースケース

- ✅ 複数統合ポイントを持つエンタープライズ環境
- ✅ 低レイテンシと外部通知の両方が必要なワークフロー
- ❌ シンプルなデプロイ（オーバーキル）

---

## Option 6D: Server-Sent Events (SSE)

### 実装

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import asyncio

app = FastAPI()

# Global event queue
feedback_events: dict[str, asyncio.Queue] = {}

@app.get("/api/feedback/{feedback_id}/events")
async def feedback_events_stream(feedback_id: str):
    """Server-Sent Events stream for feedback updates."""

    # Create queue for this feedback_id
    queue = asyncio.Queue()
    feedback_events[feedback_id] = queue

    async def event_generator():
        try:
            while True:
                # Wait for event
                event = await queue.get()
                yield f"data: {event}\n\n"
        finally:
            # Cleanup
            del feedback_events[feedback_id]

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )

# In provide_feedback()
async def provide_feedback(feedback_id, response):
    # Store response
    self._store_response(response)

    # Send SSE event if subscriber exists
    if feedback_id in feedback_events:
        await feedback_events[feedback_id].put("completed")
```

### Pros ✅

1. **ブラウザ親和性**: ネイティブEventSource API
2. **永続接続**: 自動再接続
3. **HTTPベース**: プロキシ/ファイアウォールを通りやすい
4. **Redis非依存**: 純粋なHTTPで完結

### Cons ❌

1. **接続オーバーヘッド**: サブスクライバごとに1接続
2. **サーバリソース**: Keep-alive接続の維持が必要
3. **片方向**: サーバ→クライアントのみ
4. **ワーカープロセスには不向き**: Webクライアント向け設計

### ユースケース

- ✅ Web UIダッシュボード
- ✅ リアルタイムなフィードバック状況表示
- ❌ バックエンドワーカー
- ❌ 分散ワークフロー

---

## Option 6E: ポーリングのみ（通知なし）

### 実装

```python
def _poll_for_response(self, feedback_id, timeout):
    """Simple polling without notification."""
    poll_interval = 0.5
    elapsed = 0

    while elapsed < timeout:
        # Check for response
        response = self._get_response(feedback_id)
        if response:
            return response

        # Sleep
        time.sleep(poll_interval)
        elapsed += poll_interval

    return None  # Timeout
```

### Pros ✅

1. **依存なし**: どこでも動作
2. **シンプル**: 最小限のコード
3. **信頼性**: ストレージ直接確認
4. **一貫性**: すべてのバックエンドで同じ振る舞い

### Cons ❌

1. **高レイテンシ**: 最大でpoll_interval（デフォルト500ms）
2. **リソース消費**: 継続的なポーリング
3. **最適化なし**: レイテンシを縮められない

### ユースケース

- ✅ シンプルなデプロイ
- ✅ 開発/テスト
- ❌ 本番環境（非効率）

---

## 比較マトリクス

| 機能 | Pub/Sub | Webhook | 両方 | SSE | Polling |
|------|---------|---------|------|-----|---------|
| **レイテンシ** | < 100ms | 100-1000ms | < 100ms | < 100ms | < 500ms |
| **Redis要件** | ✅ Yes | ❌ No | ✅ Yes | ❌ No | ❌ No |
| **ネットワーク要件** | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes | ❌ No |
| **マルチサブスクライバ** | ✅ Yes | ❌ No | ✅ Yes | ✅ Yes | ✅ Yes |
| **配送保証** | ❌ No | ⚠️ Retry | ⚠️ Retry | ❌ No | ✅ Yes |
| **外部連携** | ❌ No | ✅ Yes | ✅ Yes | ⚠️ Limited | ❌ No |
| **複雑性** | Low | Medium | High | Medium | Low |
| **バックエンド対応** | Redis | All | All | All | All |
| **本番適性** | ✅ Yes | ✅ Yes | ✅ Yes | ⚠️ Web only | ⚠️ Dev only |

---

## ハイブリッドアプローチ: 通知 + ポーリングフォールバック

### 推奨アーキテクチャ

```python
class FeedbackManager:
    def _poll_for_response(self, feedback_id, timeout):
        """Polling with notification optimization."""

        # Setup notification listener (non-blocking)
        notification_received = threading.Event()

        if self.backend == "redis":
            # Subscribe to Pub/Sub in background thread
            def pubsub_listener():
                pubsub = self._redis_client.pubsub()
                pubsub.subscribe(f"feedback:{feedback_id}")

                for message in pubsub.listen():
                    if message["type"] == "message":
                        notification_received.set()
                        break

            thread = threading.Thread(target=pubsub_listener, daemon=True)
            thread.start()

        # Polling loop with notification optimization
        poll_interval = 0.5
        elapsed = 0

        while elapsed < timeout:
            # Check for response
            response = self._get_response(feedback_id)
            if response:
                return response

            # Wait for notification or poll interval
            if notification_received.wait(timeout=poll_interval):
                # Notification received, check immediately
                response = self._get_response(feedback_id)
                if response:
                    return response

            elapsed += poll_interval

        return None  # Timeout
```

### 利点

- ✅ **最低レイテンシ**: Redis利用時はPub/Sub、他はポーリング
- ✅ **信頼性**: ストレージを唯一のソース・オブ・トゥルースとして常に確認
- ✅ **バックエンド非依存**: メモリ/Redisどちらでも動作
- ✅ **通知ロストなし**: ポーリングで最終的に必ず取得

---

## ユースケース別推奨

### シンプルなデプロイ（単一プロセス、メモリバックエンド）
**推奨**: Option 6E（ポーリングのみ）
- 追加依存なし
- シンプルで信頼性あり
- 開発用途でレイテンシ許容

### 分散ワークフロー（Redisバックエンド）
**推奨**: Option 6A（Redis Pub/Sub）＋ポーリングフォールバック（ハイブリッド）
- アクティブポーリング時に低レイテンシ
- ポーリングで信頼性確保
- 追加インフラ不要

### 外部システム連携
**推奨**: Option 6B（Webhookのみ）
- Slack, email, PagerDutyなどへ通知
- Redis不要
- リトライによる信頼性

### エンタープライズ環境
**推奨**: Option 6C（Pub/Sub＋Webhook併用）
- ワーカー向け低レイテンシ（Pub/Sub）
- 外部通知（Webhook）
- 最大の柔軟性

### Webダッシュボード
**推奨**: Option 6D（UI向けSSE）＋ Option 6A（ワーカー向けPub/Sub）
- リアルタイムUI更新
- 低レイテンシなワーカー通知

---

## 実装フェーズ

### Phase 1: コア（ポーリング＋Pub/Sub）
```python
class FeedbackManager:
    def __init__(self, backend, notification_mode="auto"):
        """
        notification_mode:
            "auto" - Pub/Sub if Redis, else polling only
            "polling" - Polling only (no notifications)
            "pubsub" - Redis Pub/Sub only (requires Redis)
        """
```

**優先度**: High  
**タイムライン**: Week 1-2  
**成果物**:
- ポーリングループ実装
- Redis Pub/Sub統合
- ハイブリッド（Pub/Sub＋ポーリング）実装

### Phase 2: Webhook対応
```python
@dataclass
class FeedbackRequest:
    webhook_url: Optional[str] = None
    webhook_secret: Optional[str] = None
    webhook_retry_count: int = 3
```

**優先度**: Medium  
**タイムライン**: Week 3-4  
**成果物**:
- Webhook呼び出しロジック
- エクスポネンシャルバックオフ付きリトライ
- HMAC署名検証
- エラーハンドリングとログ

### Phase 3: Web UI向けSSE
```python
@app.get("/api/feedback/{feedback_id}/events")
async def feedback_events_stream(feedback_id: str):
    # SSE implementation
```

**優先度**: Low（オプション）  
**タイムライン**: Week 5+  
**成果物**:
- SSEエンドポイント
- ブラウザEventSource統合
- 自動再接続

---

## 推奨判断

Option 6としての推奨は以下。

**Phase 1実装**: **ハイブリッド（Pub/Sub＋ポーリングフォールバック）**
```python
notification_mode = "auto"  # Default
# - Uses Pub/Sub if Redis backend
# - Falls back to polling for Memory backend
# - Always polls storage as source of truth
```

**構成例**:
```python
context = ExecutionContext.create(
    feedback_backend="redis",
    feedback_config={
        "notification_mode": "auto",  # auto, polling, pubsub, webhook
        "poll_interval": 0.5,
        "webhook_url": None,  # Optional
    }
)
```

**理由**:
1. ✅ すべてのバックエンド（Memory, Redis）で動作
2. ✅ Redis利用時に低レイテンシ
3. ✅ ポーリングで信頼性確保
4. ✅ 実装がシンプル（Phase 1で完結）
5. ✅ 拡張性あり（Phase 2でWebhook追加可能）

---

## 議論すべきオープンクエスチョン

1. **Webhookのリトライ戦略**: エクスポネンシャルバックオフか固定間隔か？
2. **Webhookタイムアウト**: レスポンスを待つ時間（デフォルト5秒）が妥当か？
3. **通知失敗時の扱い**: ログのみか、例外送出か？
4. **マルチWebhook**: 複数Webhook URLをサポートするか？
5. **通知の永続化**: 通知配送ステータスを保存するか？

---

**ステータス**: Discussion Draft  
**次のステップ**: 通知モードの選択を最終決定する
