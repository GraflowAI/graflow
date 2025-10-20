# GPT Newspaper - Docker Setup

Docker Composeを使用してGPT Newspaperアプリケーションを実行するためのガイドです。

## 📋 前提条件

- Docker Engine 20.10以上
- Docker Compose 2.0以上
- 必要なAPIキー:
  - Tavily API Key (https://tavily.com/)
  - OpenAI API Key (https://platform.openai.com/) または他のLLMプロバイダー

## 🚀 クイックスタート

### 1. 環境変数の設定

`.env`ファイルを作成して、必要なAPIキーを設定します:

```bash
# backend/.env.exampleをコピー
cp backend/.env.example .env

# エディタで.envを開いて、APIキーを設定
# TAVILY_API_KEY=your_actual_tavily_key
# OPENAI_API_KEY=your_actual_openai_key
```

### 2. アプリケーションの起動

```bash
# すべてのサービスをビルドして起動
docker-compose up --build

# またはバックグラウンドで起動
docker-compose up -d --build
```

### 3. アプリケーションにアクセス

- **フロントエンド**: http://localhost:3000
- **バックエンドAPI**: http://localhost:8000
- **API ドキュメント**: http://localhost:8000/docs

## 📁 サービス構成

### Backend (FastAPI)

- **ポート**: 8000
- **技術スタック**: Python 3.11 + FastAPI + Graflow
- **コンテナ名**: `gpt-newspaper-backend`

### Frontend (React + Nginx)

- **ポート**: 3000 (ホスト) → 80 (コンテナ)
- **技術スタック**: React 19 + TypeScript + Vite + Material-UI
- **コンテナ名**: `gpt-newspaper-frontend`

## 📂 プロジェクト構造

```
gpt_newspaper/
├── backend/                      # バックエンドソースコード
│   ├── agents/                   # AIエージェント実装
│   │   ├── search.py            # Web検索エージェント
│   │   ├── curator.py           # ソース選定エージェント
│   │   ├── writer.py            # 記事執筆エージェント
│   │   ├── critique.py          # 記事レビューエージェント
│   │   ├── designer.py          # HTMLデザインエージェント
│   │   ├── editor.py            # 記事編集エージェント
│   │   └── publisher.py         # 出版エージェント
│   ├── templates/                # HTMLテンプレート
│   │   ├── article/             # 記事テンプレート
│   │   └── newspaper/           # 新聞レイアウトテンプレート
│   ├── utils/                    # ユーティリティ
│   │   └── litellm.py           # LLMクライアント
│   ├── api.py                    # FastAPI エントリーポイント
│   ├── config.py                 # 設定管理
│   ├── newspaper_workflow.py    # Graflowワークフロー定義
│   ├── requirements.txt          # Python依存関係
│   └── .env.example              # 環境変数サンプル
├── frontend/                     # フロントエンドソースコード
│   ├── src/                      # Reactソースコード
│   │   ├── components/          # UIコンポーネント
│   │   └── services/            # APIクライアント
│   ├── public/                   # 静的ファイル
│   ├── package.json              # Node.js依存関係
│   ├── nginx.conf                # Nginx設定
│   ├── vite.config.ts            # Vite設定
│   └── tsconfig.json             # TypeScript設定
├── outputs/                      # 生成されたニュースペーパー (永続化)
├── Dockerfile.backend            # バックエンドDockerfile
├── Dockerfile.frontend           # フロントエンドDockerfile
├── docker-compose.yml            # Docker Compose設定
├── .dockerignore                 # Docker除外ファイル
├── DOCKER.md                     # このファイル
├── README.md                     # プロジェクト概要
└── WEB_APP.md                    # Webアプリドキュメント
```

## 🔧 よく使うコマンド

### サービスの起動・停止

```bash
# 起動 (フォアグラウンド)
docker-compose up

# 起動 (バックグラウンド)
docker-compose up -d

# 停止
docker-compose down

# 停止 + ボリューム削除
docker-compose down -v
```

### ログの確認

```bash
# すべてのサービスのログ
docker-compose logs -f

# バックエンドのみ
docker-compose logs -f backend

# フロントエンドのみ
docker-compose logs -f frontend
```

### 再ビルド

```bash
# すべてのサービスを再ビルド
docker-compose build

# 特定のサービスのみ
docker-compose build backend
docker-compose build frontend

# キャッシュを使わずに再ビルド
docker-compose build --no-cache
```

### コンテナの状態確認

```bash
# 実行中のコンテナを表示
docker-compose ps

# ヘルスチェック状態を確認
docker inspect gpt-newspaper-backend | grep -A 10 Health
docker inspect gpt-newspaper-frontend | grep -A 10 Health
```

## 🛠️ 開発モード

ソースコードの変更をリアルタイムで反映させたい場合は、`docker-compose.yml`のコメントアウトされたボリュームマウントを有効にします:

```yaml
services:
  backend:
    volumes:
      # これらのコメントを解除
      - ./backend:/app
      - ../../graflow:/workspace/graflow
```

その後、再起動します:

```bash
docker-compose down
docker-compose up -d
```

## 🔍 トラブルシューティング

### 問題: ポート3000が既に使用されている

別のポートを使用する場合は、`docker-compose.yml`を編集:

```yaml
services:
  frontend:
    ports:
      - "8080:80"  # 3000を8080に変更
```

### 問題: バックエンドが起動しない

1. APIキーが正しく設定されているか確認:
```bash
docker-compose exec backend env | grep API_KEY
```

2. ログを確認:
```bash
docker-compose logs backend
```

### 問題: フロントエンドがバックエンドに接続できない

1. バックエンドが起動しているか確認:
```bash
docker-compose ps
curl http://localhost:8000/
```

2. ネットワーク接続を確認:
```bash
docker network inspect gpt_newspaper_gpt-newspaper-network
```

### 問題: ビルドエラーが発生する

キャッシュをクリアして再ビルド:

```bash
# Docker キャッシュをクリア
docker builder prune -a

# 再ビルド
docker-compose build --no-cache
docker-compose up
```

## 📊 リソース使用量

### メモリ使用量の確認

```bash
docker stats
```

### ディスク使用量の確認

```bash
# すべてのDocker リソース
docker system df

# 未使用リソースのクリーンアップ
docker system prune -a
```

## 🔐 セキュリティ

### 本番環境での推奨事項

1. **環境変数の管理**:
   - `.env`ファイルをgitにコミットしない
   - Docker Secretsまたは環境変数管理ツールを使用

2. **CORS設定**:
   - `api.py`のCORS設定を本番ドメインに限定:
   ```python
   allow_origins=["https://your-domain.com"]
   ```

3. **HTTPS**:
   - リバースプロキシ (nginx/Traefik) でHTTPSを設定
   - Let's Encryptで証明書を取得

4. **レート制限**:
   - APIにレート制限を追加 (FastAPI-Limiter等)

## 📦 データの永続化

生成されたニュースペーパーは`outputs/`ディレクトリに保存されます。このディレクトリはホストにマウントされているため、コンテナを削除してもデータは保持されます。

```bash
# outputsディレクトリの確認
ls -la outputs/

# 特定の実行結果を確認
ls -la outputs/run_*/
```

## 🔄 アップデート

新しいバージョンに更新する場合:

```bash
# 現在のコンテナを停止
docker-compose down

# コードを更新 (git pull等)
git pull origin main

# 再ビルドして起動
docker-compose up --build -d
```

## 📚 関連ドキュメント

- [README.md](README.md) - アプリケーション概要
- [WEB_APP.md](WEB_APP.md) - Webアプリケーション詳細
- [frontend/README.md](frontend/README.md) - フロントエンド詳細

## 💡 Tips

### ローカル開発との切り替え

Docker環境とローカル開発環境を併用する場合:

```bash
# Dockerで起動
docker-compose up -d

# ローカルで開発 (別のポートで)
# Terminal 1 - Backend
cd examples/gpt_newspaper/backend
uvicorn api:app --reload --port 8001

# Terminal 2 - Frontend
cd examples/gpt_newspaper/frontend
npm run dev  # Port 5173
```

### コンテナ内でシェルを起動

デバッグやメンテナンス用:

```bash
# バックエンドコンテナ
docker-compose exec backend /bin/bash

# フロントエンドコンテナ
docker-compose exec frontend /bin/sh
```

### 特定のサービスのみ起動

```bash
# バックエンドのみ
docker-compose up backend

# フロントエンドのみ (バックエンドが起動している必要あり)
docker-compose up frontend
```

## ❓ サポート

問題が発生した場合は、以下を確認してください:

1. Dockerバージョン: `docker --version`
2. Docker Composeバージョン: `docker-compose --version`
3. ログ: `docker-compose logs -f`
4. コンテナ状態: `docker-compose ps`

それでも解決しない場合は、GitHubのIssueを作成してください。
