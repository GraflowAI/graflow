# Graflow への貢献ガイド

Graflow に興味を持っていただきありがとうございます。皆さんの参加がプロジェクトをより良いものにします。このガイドでは、貢献の方法について説明します。

## オープンな実力主義 (Meritocracy)

Graflow は [Apache 流の実力主義 (Meritocracy)](https://www.apache.org/foundation/how-it-works/#meritocracy) の原則に従っています。役割と責任は、肩書きや所属ではなく、実際の貢献を通じて得られるものです。

コントリビューターがプロジェクトメンバーへと成長していくことを歓迎します:

| ロール | 説明 |
|--------|------|
| **Contributor** | パッチ、バグ報告、ドキュメント改善、ディスカッションへの参加など、何らかの形で貢献する人。 |
| **Committer** | 継続的で質の高い貢献を通じて、リポジトリへの書き込み権限を得たコントリビューター。 |
| **Maintainer** | PR レビュー、ロードマップの策定、新規参加者のメンタリングなど、プロジェクト全体の運営を担うコミッター。 |

これらのロールに「応募」するための固定プロセスはありません。継続的かつ建設的に貢献していれば、ピアの評価によりより大きな責任を担うよう招待されます。1行の typo 修正でも大規模な新機能でも、すべての貢献に価値があります。

## 貢献の方法

- **バグ報告** - 問題を発見した場合は [Issue を作成](#バグ報告)してください。
- **機能提案** - アイデアがあれば[ディスカッションを開始](#機能提案)してください。
- **ドキュメント改善** - typo の修正、サンプルの追加、説明の改善。
- **コード貢献** - バグ修正、新機能、パフォーマンス改善。
- **インテグレーション追加** - 新しい Queue/Channel バックエンド、LLM プロバイダー、トレーシング連携。

[`good first issue`](https://github.com/GraflowAI/graflow/labels/good%20first%20issue) や [`help wanted`](https://github.com/GraflowAI/graflow/labels/help%20wanted) ラベルの付いた Issue は、最初の貢献に適しています。

## はじめに

### 前提条件

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) パッケージマネージャー
- Git

### セットアップ

```bash
# リポジトリを Fork してクローン
git clone https://github.com/<your-username>/graflow.git
cd graflow

# 開発用依存関係のインストール
uv sync --dev

# (任意) graphviz を含む全 extras のインストール (macOS)
make install-all
```

### 動作確認

```bash
make check-all  # format + lint + test を実行
```

## 開発ワークフロー

### 1. Issue の確認・作成

作業を始める前に、[既存の Issue](https://github.com/GraflowAI/graflow/issues) を確認して重複を避けてください。作業する Issue にはコメントを残して、他の人に知らせましょう。

### 2. Fork してブランチを作成

```bash
git checkout -b your-feature-name
```

わかりやすいブランチ名を使ってください（例: `fix/parallel-group-merge`, `feat/kafka-backend`）。

### 3. 変更を実装

プロジェクトの[コードスタイル](#コードスタイル)に従ってください。PR は一つの関心事に集中させましょう。

### 4. 品質チェックの実行

PR を提出する前に、すべてのチェックをパスさせてください:

```bash
make format     # ruff による自動フォーマット
make lint       # ruff リンター
make test       # pytest テストスイート
```

まとめて実行する場合:

```bash
make check-all
```

### 5. コミット

「何をしたか」よりも「なぜそうしたか」を伝えるコミットメッセージを書いてください。

```bash
# Good
git commit -m "fix: Prevent duplicate task execution in ParallelGroup"

# Bad
git commit -m "fixed stuff"
```

### 6. Pull Request の作成

`main` ブランチに対して PR を作成してください。PR の説明には以下を含めてください:

- 変更内容とその理由
- 関連する Issue への参照（例: `Fixes #42`）
- テスト方法の説明

## Pull Request ガイドライン

### 全般

- **スコープを小さく**: 1つの PR では1つの関心事のみを扱ってください。バグ修正とリファクタリングや新機能を混ぜないでください。
- **後方互換性**: 重大なバグやセキュリティ修正を除き、既存の動作を壊す変更は避けてください。
- **不要な依存関係の追加禁止**: 新しいハード依存関係の追加は事前に議論が必要です。オプション依存は `[project.optional-dependencies]` に追加してください。
- **テスト必須**: バグ修正にはバグを再現するテストを含めてください。新機能にはユニットテスト、必要に応じてインテグレーションテストが必要です。
- **ドキュメント**: 新機能には docstring を含め、重要な機能にはサンプルの更新も行ってください。

### レビューで重視する点

- コードが明確でプロジェクトの規約に従っている
- テストが重要なケースをカバーしている
- 不要な複雑さや過剰な設計がない
- すべての公開 API に型ヒントがある
- 変更が後方互換性を保っている

### バグ修正

バグの明確な説明と、それを再現するテストを含めてください。修正は最小限かつ焦点を絞ったものにしてください。

### 新機能

新機能の受け入れ基準はより高くなります。大きな時間を費やす前に、Issue を作成してアプローチを議論してください。新機能には以下が必要です:

- ユニットテスト（必要に応じてインテグレーションテスト）
- 公開 API の docstring
- 新しい使用パターンを導入する場合は `examples/` にサンプルを追加

### AI ツールを利用した貢献

AI ツールの活用は歓迎しますが、意味のある人間のレビューと理解が伴う必要があります。PR の作成にかかった労力がレビューにかかる労力を下回る場合、その貢献は提出すべきではありません。人間の判断を伴わない低品質な AI 生成 PR はレビューなしでクローズされる場合があります。

## バグ報告

まず[既存の Issue](https://github.com/GraflowAI/graflow/issues) を検索してください。報告されていない場合は、以下の情報を含めて新しい Issue を作成してください:

- 明確でわかりやすいタイトル
- **最小限の再現コード**
- 再現手順
- 期待される動作と実際の動作
- Python バージョン、OS、Graflow バージョン (`python -c "import graflow; print(graflow.__version__)"`)
- 関連するログやエラーメッセージ

1つの Issue には1つのバグのみ記載してください。複数の問題がある場合は、別々のチケットを作成してください。

## 機能提案

まず[既存の Issue](https://github.com/GraflowAI/graflow/issues) と[ディスカッション](https://github.com/GraflowAI/graflow/discussions)を検索してください。機能を提案する際は以下を含めてください:

- ユースケースと動機
- 使用方法の具体例
- 既存の機能では対応できない理由

## コードスタイル

### フォーマットとリンティング

- **フォーマッター / リンター**: [ruff](https://docs.astral.sh/ruff/)
- **型チェッカー**: [mypy](https://mypy-lang.org/)
- **行の長さ**: 120 文字
- **インポート順序**: ruff 経由の isort（`graflow` をファーストパーティとして認識）

### 型ヒント

- すべての関数シグネチャに必須 (`disallow_untyped_defs = true`)
- 暗黙の Optional 禁止 (`no_implicit_optional = true`)
- `Any` の使用は可能な限り避ける

### 命名規則

| 要素 | スタイル | 例 |
|------|----------|-----|
| クラス | `PascalCase` | `TaskQueue`, `ExecutionContext` |
| 関数 / メソッド | `snake_case` | `get_next_task`, `add_to_queue` |
| 定数 | `UPPER_SNAKE_CASE` | `MAX_RETRIES`, `DEFAULT_TIMEOUT` |
| プライベート | `_leading_underscore` | `_internal_method` |

### Docstring

公開クラスおよびメソッドには、必要に応じて `Args`、`Returns`、`Raises` セクションを含む docstring を記述してください。

## テスト

テストは `tests/` ディレクトリにモジュール別で配置されています:

```
tests/
    core/             # コア機能
    channels/         # Channel 実装
    coordination/     # 協調処理
    queue/            # Queue 実装
    worker/           # ワーカー
    integration/      # インテグレーションテスト
    scenario/         # エンドツーエンドシナリオ
    unit/             # ユニットテスト
```

### テストの実行

```bash
# 全テスト
uv run pytest tests/ -v

# 特定のディレクトリ
uv run pytest tests/core/ -v

# 特定のテストファイル
uv run pytest tests/core/test_task.py -v

# 特定のテスト
uv run pytest tests/core/test_task.py::test_name -v
```

### テストの書き方

- `pytest` フィクスチャを使用（`conftest.py` を参照）
- 外部依存（Redis、Docker）はモックを使用
- Queue/Channel は memory と Redis の両バックエンドをテスト
- 分散シナリオにはインテグレーションテストを追加

## プロジェクト構成

```
graflow/
    core/           # エンジン、タスク、ワークフロー、コンテキスト、グラフ
    queue/          # タスクキューバックエンド (memory, Redis)
    channels/       # タスク間通信 (memory, Redis)
    worker/         # 分散タスクワーカー
    coordination/   # 並列実行の協調
    hitl/           # Human-in-the-Loop フィードバック
    llm/            # LLM 連携
    trace/          # トレーシングとオブザーバビリティ
    api/            # REST API エンドポイント
    serialization/  # タスクのシリアライゼーション (cloudpickle)
    debug/          # 可視化とデバッグ
```

## 主な貢献分野

[プロジェクトロードマップ](https://github.com/orgs/GraflowAI/projects/1)で計画中の機能と優先度を確認できます。Kanban ボード上のアイテムは貢献の良い候補です。取り組みたいものがあれば Issue にコメントしてください。

### エージェントフレームワーク連携の追加

Graflow は現在 `graflow/llm/` で [ADK (Google Agent Development Kit)](https://google.github.io/adk-docs/) と [PydanticAI](https://ai.pydantic.dev/) をサポートしています。他のエージェントフレームワークへの対応追加を歓迎します。例:

- [Strands Agents](https://strandsagents.com/latest/) (AWS)
- [Microsoft Agent Framework](https://github.com/microsoft/agent-framework)
- その他のエージェント / スーパーエージェントフレームワーク

新しい連携を追加するには、既存の ADK および PydanticAI の実装パターンに従い `graflow/llm/` にアダプターを実装してください。

### ドキュメントの改善

ドキュメントへの貢献は常に歓迎します:

- typo の修正、説明の改善、読みやすさの向上
- ソースコード内の docstring の追加・改善
- `examples/` への新しいサンプルの追加
- 構成の変更や新しいドキュメントページの提案

## ヘルプ

- 質問は [GitHub Discussion](https://github.com/GraflowAI/graflow/discussions) で受け付けています
- 質問する前に既存の Issue やディスカッションを確認してください
- PR で行き詰まった場合はメンテナーにタグ付けしてください

## ライセンス

貢献することにより、あなたの貢献が [Apache License 2.0](./LICENSE) のもとでライセンスされることに同意したものとみなされます。
