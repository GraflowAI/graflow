# Company Intelligence MCP Server

企業インテリジェンスレポートを生成するMCPサーバー。FastMCPを使用してREST API経由でアクセス可能。

## クイックスタート

```bash
# 1. ディレクトリに移動
cd examples/mcp_server

# 2. 依存関係のインストール
uv add fastmcp tavily-python

# 3. 環境変数設定
cp .env.example .env
# .envにTAVILY_API_KEY, OPENAI_API_KEYを設定

# 4. サーバー起動
uv run python server.py

# 5. ワークフローのみ実行（サーバーなし）
uv run python workflow.py "トレジャーデータ"

# 6. クライアント例
uv run python client_example.py "トレジャーデータ"
```

## 概要

このMCPサーバーは、企業名をキーとして以下のワークフローを実行し、商談準備用のインテリジェンスレポートを生成します：

```
Search → Curate → Write → Critique → Output
       ↑                    ↓
       └──── Feedback Loop ─┘
```

### ワークフロー詳細

1. **Search**: Tavily APIを使用して複数ソースから情報収集
   - 企業ニュース
   - 業界動向
   - 企業プロフィール
   - 競合情報

2. **Curate**: LLMで検索結果をフィルタリング・整理
   - 関連性スコアリング
   - 重複排除
   - カテゴリ分類

3. **Write**: LLMでインテリジェンスレポートを生成
   - エグゼクティブサマリー
   - セクション別詳細
   - 商談ポイント

4. **Critique**: LLMでレポートを評価
   - 正確性、完全性、関連性、構成、ソース品質
   - 基準未達の場合はWriteに差し戻し（最大3回）

5. **Output**: 最終レポートをMarkdown形式で出力

## 環境設定

### 1. .envファイルの作成

```bash
cd examples/mcp_server
cp .env.example .env
# .envファイルを編集してAPIキーを設定
```

### 必須の環境変数

```bash
TAVILY_API_KEY=your-tavily-api-key
OPENAI_API_KEY=your-openai-api-key
```

### オプション（Langfuseトレーシング）

```bash
LANGFUSE_PUBLIC_KEY=your-langfuse-public-key
LANGFUSE_SECRET_KEY=your-langfuse-secret-key
LANGFUSE_HOST=http://localhost:3000
ENABLE_TRACING=true
```

### その他の設定

```bash
GRAFLOW_LLM_MODEL=gpt-5-mini     # デフォルトモデル
WRITER_MODEL=gpt-5-mini          # Writer用モデル
CRITIQUE_MODEL=gpt-5-mini        # Critique用モデル
MAX_CRITIQUE_ITERATIONS=3         # 最大リビジョン回数
MAX_SEARCH_RESULTS=10             # 検索結果の最大数
MCP_SERVER_HOST=0.0.0.0           # サーバーホスト
MCP_SERVER_PORT=9100              # サーバーポート
```

## 起動方法

### サーバー起動

```bash
# ディレクトリに移動
cd examples/mcp_server

# サーバー起動
uv run python server.py

# または uvicorn で起動
uv run uvicorn server:app --host 0.0.0.0 --port 9100
```

### ワークフローのみ実行（サーバーなし）

```bash
cd examples/mcp_server
uv run python workflow.py "トレジャーデータ"
```

## API エンドポイント

### REST API

| エンドポイント | メソッド | 説明 |
|--------------|---------|------|
| `/health` | GET | ヘルスチェック |
| `/config` | GET | 現在の設定を取得 |
| `/mcp` | POST | MCP SSEエンドポイント |

### MCP Tools

#### generate_company_intelligence

企業の包括的なインテリジェンスレポートを生成。

**Input:**
```json
{
  "company_name": "トレジャーデータ",
  "enable_tracing": true
}
```

**Output:**
```json
{
  "company_name": "トレジャーデータ",
  "report_markdown": "# トレジャーデータ インテリジェンスレポート\n...",
  "executive_summary": "...",
  "key_takeaways": ["ポイント1", "ポイント2"],
  "sections": [...],
  "sources_count": 15,
  "iterations": 2,
  "generated_at": "2024-01-15T10:30:00",
  "critique_score": 0.85
}
```

#### search_company_news

企業の最新ニュースを検索（軽量版）。

**Input:**
```json
{
  "company_name": "Salesforce",
  "max_results": 10
}
```

#### search_industry_trends

業界動向を検索。

**Input:**
```json
{
  "company_name": "Sony",
  "industry": "エレクトロニクス",
  "max_results": 10
}
```

## クライアントからの利用

### Python クライアント例

```bash
cd examples/mcp_server

# サーバー起動後、別ターミナルで
uv run python client_example.py "トレジャーデータ"

# ニュース検索のみ
uv run python client_example.py "Salesforce" --action news

# 業界トレンド検索
uv run python client_example.py "Sony" --action trends
```

### curl での呼び出し

```bash
# ヘルスチェック
curl http://localhost:9100/health

# レポート生成（SSE経由）
curl http://localhost:9100/mcp
```

### httpx での呼び出し

```python
import httpx

# ヘルスチェック
response = httpx.get("http://localhost:9100/health")
print(response.json())

# 設定確認
response = httpx.get("http://localhost:9100/config")
print(response.json())
```

## Claude Code MCP セットアップ

Claude CodeからこのMCPサーバーを利用できます。

### MCPインストールスコープ

MCPサーバーは3つのスコープで構成できます：

| スコープ | 保存場所 | 用途 |
|---------|---------|------|
| `local` | `~/.claude.json`（プロジェクトパス下） | 個人用、現在のプロジェクトのみ（デフォルト） |
| `project` | プロジェクトルートの`.mcp.json` | チーム共有用（バージョン管理にコミット） |
| `user` | `~/.claude.json` | 全プロジェクトで利用可能 |

### 方法1: CLIでインストール（推奨）

まずサーバーを起動：
```bash
cd examples/mcp_server && uv run python server.py
```

別ターミナルでMCPを登録：
```bash
# ローカルスコープ（デフォルト）- 現在のプロジェクトのみ
claude mcp add --transport http company-intel http://localhost:9100/mcp

# プロジェクトスコープ - チームで共有（.mcp.jsonに保存）
claude mcp add --transport http --scope project company-intel http://localhost:9100/mcp

# ユーザースコープ - 全プロジェクトで利用
claude mcp add --transport http --scope user company-intel http://localhost:9100/mcp
```

### 方法2: プロジェクト設定ファイル（チーム共有用）

プロジェクトルートに `.mcp.json` を作成（バージョン管理にコミット）：

```json
{
  "mcpServers": {
    "company-intelligence": {
      "command": "uv",
      "args": ["run", "python", "server.py"],
      "cwd": "./examples/mcp_server",
      "env": {
        "TAVILY_API_KEY": "${TAVILY_API_KEY}",
        "OPENAI_API_KEY": "${OPENAI_API_KEY}"
      }
    }
  }
}
```

環境変数は `${VAR}` 形式で参照できます（実際の値はユーザーの環境から取得）。

### MCPサーバーの管理

```bash
# 登録済みサーバーの一覧
claude mcp list

# サーバーの詳細を確認
claude mcp get company-intel

# サーバーを削除
claude mcp remove company-intel
```

### 確認

Claude Codeを再起動し、`/mcp` コマンドで接続状態を確認できます。

## Langfuse トレーシング

ワークフローの実行はLangfuseでトレースできます：

1. [Langfuse Cloud](https://cloud.langfuse.com) でアカウント作成
2. プロジェクトを作成し、APIキーを取得
3. `.env`ファイルに環境変数を設定
4. サーバー起動後、Langfuseダッシュボードでトレースを確認

トレースには以下が含まれます：
- ワークフロー全体の実行時間
- 各タスク（search, curate, write, critique）の実行時間
- LLM呼び出しの詳細（プロンプト、レスポンス、トークン数）
- エラーとスタックトレース

## プロジェクト構成

```
examples/mcp_server/
├── __init__.py              # モジュール初期化
├── config.py                # 設定管理（dotenv対応）
├── workflow.py              # Graflowワークフロー定義
├── server.py                # FastMCP サーバー
├── client_example.py        # クライアント例
├── .env.example             # 環境変数テンプレート
├── claude_mcp_config.json   # Claude Code MCP設定例
├── README.md                # English README
├── README_ja.md             # このファイル
└── agents/
    ├── __init__.py
    ├── search.py        # 検索エージェント（Tavily）
    ├── curator.py       # キュレーションエージェント（LLM）
    ├── writer.py        # ライターエージェント（LLM）
    └── critique.py      # クリティークエージェント（LLM）
```

## 依存関係

```
graflow
fastmcp
tavily-python
litellm
python-dotenv
httpx
uvicorn
pydantic
```

インストール:
```bash
uv add fastmcp tavily-python python-dotenv
```

## ライセンス

Apache License 2.0
