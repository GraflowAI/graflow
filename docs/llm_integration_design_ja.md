# Graflow LLM統合設計ドキュメント

## 概要

Graflowに LiteLLM を使用した LLM 統合機能を追加する設計ドキュメント。

### 目的

- タスク内で LLM を簡単に利用できるようにする
- LiteLLM を通じて複数の LLM プロバイダー（OpenAI, Anthropic, Google など）をサポート
- **タスクごとに異なるモデルを使用可能**（コスト・性能の最適化）
- **LiteLLM の `langfuse_otel` と `ExecutionContext.trace_id` で統一的にトレーシング**
- Google ADK の LlmAgent を活用した Supervisor/ReAct パターンのサポート

### 設計方針

1. **疎結合**: LLM機能はGraflowの実行エンジンと疎結合で、あくまでタスク内で利用するユーティリティ
2. **Fat単一ノード**: Supervisor/ReActエージェントは単一のタスクノード内で完結し、Graflowのタスクグラフとは切り離す
3. **独立性**: ADKのtoolsとGraflowのtasksは完全に独立
4. **依存性**: LiteLLM, Google ADK は optional dependency として扱う
5. **トレーシング**: LiteLLM の `langfuse_otel` コールバックを使用し、`trace_id` のみを引き継ぐ（Graflow tracer と並行動作）
6. **柔軟性**: タスクごとに異なる LLMClient インスタンスを作成し、最適なモデルを選択可能

---

## アーキテクチャ

### モジュール構成

```
graflow/llm/
├── __init__.py
├── client.py          # LLMClient - LiteLLMラッパー
├── config.py          # LLMConfig - 設定管理
├── serialization.py   # Agent シリアライゼーション（YAML）
└── agents/
    ├── __init__.py
    ├── base.py        # LLMAgent - ReAct/Supervisor用基底クラス
    └── adk_agent.py   # AdkLLMAgent - Google ADK LlmAgentラッパー
```

**注**: LiteLLM の `langfuse` コールバックと OpenTelemetry を使用するため、独自の tracing モジュールは不要。

### アーキテクチャ図

```
┌────────────────────────────────────────────────────────────────┐
│ Graflow Execution Layer                                       │
│                                                                │
│  ExecutionContext                                             │
│    ├─ trace_id (W3C TraceContext 32-digit hex) ←──┐          │
│    ├─ tracer (LangFuseTracer/NoopTracer)          │          │
│    └─ llm_client (LLMClient instance)              │          │
│                                                     │          │
│  Task Graph: [Task A] ──> [Supervisor Task] ──> [Task C]     │
│                                │                               │
│                                └─ Fat単一ノード                │
│                                   (内部でReAct/Sub-agents実行) │
└────────────────────────────────────────────────────────────────┘
                        │                            │
                        │ DI                         │ trace_id 引き継ぎ
                        ▼                            ▼
┌────────────────────────────────────────────────────────────────┐
│ LLM Layer (Graflow実行エンジンと疎結合)                        │
│                                                                │
│  LLMClient (completion + agent registry)                      │
│    ├─ completion(messages, **params)                          │
│    │   └─ metadata["trace_id"] に ExecutionContext.trace_id   │
│    │      を設定して litellm.completion() を呼び出し           │
│    ├─ register_agent(name, agent)                             │
│    └─ get_agent(name) -> LLMAgent                             │
│                                                                │
│  LLMAgent (base class for ReAct/Supervisor)                   │
│    └─ AdkLLMAgent (wraps ADK LlmAgent)                        │
│         ├─ sub_agents support                                 │
│         ├─ tools (独立、graflow tasksではない)                 │
│         └─ run() 内で LLMClient.completion() を呼び出し        │
└────────────────────────────────────────────────────────────────┘
                        │
                        │ LiteLLM langfuse_otel callback
                        ▼
┌────────────────────────────────────────────────────────────────┐
│ Langfuse (via OpenTelemetry)                                  │
│                                                                │
│  同じ trace_id で Graflow タスクと LLM 呼び出しを関連付け      │
│  - Graflow tracer: ワークフロー・タスクのトレース              │
│  - LiteLLM langfuse_otel: LLM 呼び出しのトレース               │
│  → Langfuse UI で統一的に可視化                                │
└────────────────────────────────────────────────────────────────┘
```

---

## コアコンポーネント

### 1. LLMClient (`graflow/llm/client.py`)

LiteLLM のラッパーで、以下の機能を提供：

1. **Completion API**: LiteLLM の `completion()` への簡易アクセス
2. **自動トレーシング**: OpenTelemetry context から trace_id/span_id を自動検出

#### 主要メソッド

```python
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from litellm import ModelResponse
else:
    ModelResponse = Any

class LLMClient:
    # Completion API
    def completion(
        messages,
        model=None,
        generation_name=None,
        tags=None,
        **params
    ) -> ModelResponse:
        """LiteLLM の ModelResponse をそのまま返す"""

    def completion_text(messages, model=None, **params) -> str:
        """便利メソッド: response.choices[0].message.content を返す"""
```

#### 実装例

```python
from typing import TYPE_CHECKING, Any, Dict, List, Optional
import importlib

if TYPE_CHECKING:
    from litellm import ModelResponse
else:
    ModelResponse = Any

class LLMClient:
    def __init__(
        self,
        model: str,
        **default_params: Any
    ):
        self.model = model
        self.default_params = default_params

        try:
            self._litellm = importlib.import_module("litellm")
        except ImportError:
            raise RuntimeError("liteLLM is not installed.")

    def completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        generation_name: Optional[str] = None,
        tags: Optional[List[str]] = None,
        **params: Any
    ) -> ModelResponse:
        """統一されたcompletion API

        Args:
            messages: メッセージリスト
            model: モデルオーバーライド（オプション）
            generation_name: Langfuseのgeneration名（オプション）
            tags: Langfuseのタグ（オプション）
            **params: LiteLLMに渡すその他のパラメータ

        Returns:
            LiteLLM の ModelResponse オブジェクト
            （response.choices[0].message.content でテキスト取得）
        """
        kwargs = {**self.default_params, **params}
        actual_model = model or self.model

        # Langfuseメタデータ設定
        if generation_name or tags:
            metadata = kwargs.get('metadata', {})
            if generation_name:
                metadata['generation_name'] = generation_name
            if tags:
                metadata['tags'] = tags
            kwargs['metadata'] = metadata

        # OpenTelemetry context から自動的に parent span を検出
        return self._litellm.completion(
            model=actual_model,
            messages=messages,
            **kwargs
        )

    def completion_text(self, messages: List[Dict[str, str]], **params: Any) -> str:
        """便利メソッド: response.choices[0].message.content を返す"""
        response = self.completion(messages, **params)
        return extract_text(response)


def extract_text(response: ModelResponse) -> str:
    """LiteLLM の ModelResponse からテキストを抽出"""
    choices = getattr(response, "choices", None)
    if not choices:
        return ""

    choice = choices[0]
    message = getattr(choice, "message", None)
    if message is None:
        return ""

    content = getattr(message, "content", None)
    return content or ""
```

#### 設計ポイント

- **統一API**: `completion()` で全ての機能を提供（model, generation_name, tags）
- **LiteLLM Response を返す**: 薄いラッパーとして、LiteLLM の `ModelResponse` をそのまま返す
  - 余計な抽象化を避け、LiteLLM の全機能にアクセス可能
  - `response.choices[0].message.content` で標準的にアクセス
  - `completion_text()` でテキストのみを簡単に取得
- **モデルオーバーライド**: `completion(model="gpt-4")` で一時的にモデルを変更可能
- **Langfuseメタデータ**: `generation_name` と `tags` で Langfuse でのトレース整理
- **自動トレーシング**: OpenTelemetry context から trace_id/span_id を自動検出

---

### 2. LLMAgent (`graflow/llm/agents/base.py`)

ReAct/Supervisor パターン用の基底クラス。

#### 主要メソッド

```python
class LLMAgent(ABC):
    @abstractmethod
    def run(query: str, **kwargs) -> Any:
        """エージェントのメインロジック（ReActループ、サブエージェント調整など）"""
        pass

    def get_state() -> Dict[str, Any]:
        """状態のシリアライズ"""
        pass

    def set_state(state: Dict[str, Any]) -> None:
        """状態の復元"""
        pass
```

#### 設計ポイント

- **Fat単一ノード**: エージェントは単一タスクノード内で完結
- **独自の制御フロー**: ReActループやサブエージェント調整は `run()` 内で実装
- **Graflowグラフと分離**: 動的タスク生成（`ctx.next_task()`）は使用しない

---

### 3. AdkLLMAgent (`graflow/llm/agents/adk_agent.py`)

Google ADK の `LlmAgent` をラップし、Graflow の `LLMAgent` インターフェースに適合させる。

#### 主要機能

- **Sub-agents サポート**: ADK の階層的エージェント構造を活用
- **Tools 統合**: ADK の tool calling 機能（Graflow tasks とは独立）
- **LiteLLM 統合**: ADK で LiteLLM を使用

#### 実装例

```python
class AdkLLMAgent(LLMAgent):
    def __init__(
        self,
        name: str,
        llm_client: LLMClient,
        model: Optional[str] = None,
        description: Optional[str] = None,
        instruction: Optional[str] = None,
        tools: Optional[List[Callable]] = None,
        sub_agents: Optional[List[LLMAgent]] = None,
        **agent_kwargs
    ):
        # ADK LiteLlm モデルを作成
        lite_model = LiteLlm(model=model or llm_client.model)

        # ADK sub_agents に変換
        adk_sub_agents = [
            sub.\_adk_agent for sub in sub_agents
            if isinstance(sub, AdkLLMAgent)
        ]

        # ADK LlmAgent を作成
        self._adk_agent = LlmAgent(
            name=name,
            model=lite_model,
            description=description,
            instruction=instruction,
            tools=tools or [],
            sub_agents=adk_sub_agents,
            **agent_kwargs
        )

    def run(self, query: str, **kwargs) -> Any:
        return self._adk_agent.run(query, **kwargs)
```

---

### 4. LLMConfig (`graflow/llm/config.py`)

LLM機能の設定管理。

```python
@dataclass
class LLMConfig:
    # LiteLLM 設定
    model: str = "gpt-4o-mini"
    default_params: Dict[str, Any] = field(default_factory=dict)

    # トレーシング設定
    enable_tracing: bool = True
    langfuse_public_key: Optional[str] = None
    langfuse_secret_key: Optional[str] = None
    langfuse_host: Optional[str] = None  # セルフホスト用

    # メタデータ
    default_tags: List[str] = field(default_factory=list)

    @classmethod
    def from_env(cls) -> "LLMConfig":
        """環境変数から設定を読み込む"""
        return cls(
            model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            langfuse_public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            langfuse_secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            langfuse_host=os.getenv("LANGFUSE_OTEL_HOST"),
        )
```

---

## ExecutionContext 統合

### ExecutionContext 拡張

```python
class ExecutionContext:
    def __init__(
        self,
        # ... 既存のパラメータ ...
        llm_config: Optional[LLMConfig] = None,
        llm_client: Optional[LLMClient] = None,
    ):
        self._llm_config = llm_config
        self._llm_client = llm_client
        self._llm_agents: Dict[str, LLMAgent] = {}  # Agent Registry

    @property
    def llm_client(self) -> Optional[LLMClient]:
        """LLMクライアントを取得（遅延初期化）"""
        if self._llm_client is None and self._llm_config is not None:
            # トレーシングのセットアップ
            if self._llm_config.enable_tracing:
                setup_langfuse_for_litellm(self._llm_config)

            # クライアント作成
            self._llm_client = LLMClient(
                model=self._llm_config.model,
                **self._llm_config.default_params
            )

        return self._llm_client

    def register_llm_agent(self, name: str, agent: LLMAgent) -> None:
        """LLMAgent を登録"""
        self._llm_agents[name] = agent

    def get_llm_agent(self, name: str) -> LLMAgent:
        """LLMAgent を取得"""
        if name not in self._llm_agents:
            raise KeyError(f"LLMAgent '{name}' not found in registry")
        return self._llm_agents[name]
```

### ファクトリー関数

```python
def create_execution_context(
    # ... 既存のパラメータ ...
    llm_config: Optional[LLMConfig] = None,
    llm_model: Optional[str] = None,  # ショートカット
) -> ExecutionContext:
    """実行コンテキストを作成（LLMサポート付き）"""

    if llm_model and not llm_config:
        llm_config = LLMConfig(model=llm_model)
    elif llm_config is None:
        # 環境変数から読み込み
        if os.getenv("LLM_MODEL"):
            llm_config = LLMConfig.from_env()

    return ExecutionContext(
        # ... 既存の引数 ...
        llm_config=llm_config,
    )
```

---

## タスクデコレーター拡張

### @task デコレーター

```python
def task(
    id_or_func: Optional[F] | str | None = None,
    *,
    id: Optional[str] = None,
    inject_context: bool = False,
    inject_llm_client: bool = False,  # 新規
    inject_llm_agent: bool = False,   # 新規
    model: Optional[str] = None,      # 新規
    agent_name: Optional[str] = None, # 新規
    handler: Optional[str] = None
) -> TaskWrapper | Callable[[F], TaskWrapper]:
    """
    タスクデコレーター

    Args:
        inject_context: ExecutionContext を第一引数に注入
        inject_llm_client: LLMClient を第一引数に注入
        inject_llm_agent: LLMAgent を第一引数に注入
        model: inject_llm_client 使用時のモデルオーバーライド
        agent_name: inject_llm_agent 使用時のエージェント名
    """
```

### 使用例

```python
# シンプルな LLM タスク（ExecutionContext のデフォルトモデルを使用）
@task(inject_llm_client=True)
def summarize(llm: LLMClient, text: str) -> str:
    """デフォルトモデル（例: gpt-4o-mini）でタスクを実行"""
    messages = [
        {"role": "system", "content": "You are a summarization assistant."},
        {"role": "user", "content": f"Summarize: {text}"}
    ]
    return llm.completion_text(messages)

# タスク固有のモデル指定（別の LLMClient インスタンスを使用）
@task(inject_llm_client=True, model="gpt-4o")
def complex_analysis(llm: LLMClient, data: str) -> str:
    """
    タスクごとに異なるモデルを指定可能。

    - ExecutionContext のデフォルトが gpt-4o-mini でも、このタスクは gpt-4o を使用
    - 別の LLMClient インスタンスが作成される（default_params は継承）
    - タスクごとに最適なモデルを選択可能
    """
    return llm.completion_text([{"role": "user", "content": data}])

# 高性能モデルを必要とするタスク
@task(inject_llm_client=True, model="claude-3-5-sonnet-20241022")
def advanced_reasoning(llm: LLMClient, problem: str) -> str:
    """複雑な推論タスクには Claude Sonnet を使用"""
    return llm.completion_text([{"role": "user", "content": problem}])

# コスト効率重視のタスク
@task(inject_llm_client=True, model="gpt-4o-mini")
def simple_formatting(llm: LLMClient, text: str) -> str:
    """簡単なタスクには低コストモデルを使用"""
    return llm.completion_text([{"role": "user", "content": text}])

# LLM Agent を使用
@task(inject_llm_agent=True, agent_name="supervisor")
def run_supervisor(agent: LLMAgent, query: str) -> str:
    """Agent Registry から "supervisor" を取得して実行"""
    return agent.run(query)
```

**重要**: `model` パラメータを指定した場合:
- タスク実行時に**新しい LLMClient インスタンス**が作成される
- ExecutionContext の `default_params`（temperature, max_tokens など）は継承される
- モデルのみが上書きされる
- タスクごとに最適なモデルを選択可能（コスト・性能のバランス調整）

### TaskWrapper の実装

```python
class TaskWrapper(Executable):
    def __init__(
        self,
        task_id: str,
        func: Callable,
        inject_context: bool = False,
        inject_llm_client: bool = False,
        inject_llm_agent: bool = False,
        llm_model_override: Optional[str] = None,
        agent_name: Optional[str] = None,
        handler_type: Optional[str] = None,
    ):
        self.inject_llm_client = inject_llm_client
        self.inject_llm_agent = inject_llm_agent
        self.llm_model_override = llm_model_override
        self.agent_name = agent_name
        # ...

    def __call__(self, *args, **kwargs) -> Any:
        exec_context = self.get_execution_context()

        # LLMClient injection
        if self.inject_llm_client:
            llm_client = exec_context.llm_client

            if llm_client is None:
                raise RuntimeError(
                    f"Task {self.task_id} requires inject_llm_client=True "
                    "but no LLM client configured"
                )

            # タスク固有のモデル指定がある場合、別の LLMClient インスタンスを作成
            # これにより、タスクごとに異なるモデルを使用可能
            if self.llm_model_override:
                # 新しい LLMClient インスタンスを作成
                # - model: タスク固有のモデル（例: "gpt-4o", "claude-3-5-sonnet-20241022"）
                # - default_params: ExecutionContext の設定を継承（temperature, max_tokens など）
                llm_client = LLMClient(
                    model=self.llm_model_override,
                    **llm_client.default_params
                )

            return self.func(llm_client, *args, **kwargs)

        # LLMAgent injection
        if self.inject_llm_agent:
            if self.agent_name is None:
                raise RuntimeError(
                    f"Task {self.task_id} requires agent_name parameter "
                    "when inject_llm_agent=True"
                )

            try:
                agent = exec_context.get_llm_agent(self.agent_name)
            except KeyError:
                raise RuntimeError(
                    f"Task {self.task_id} requires LLMAgent '{self.agent_name}' "
                    "but not found in registry. Use ctx.register_llm_agent() first."
                )

            return self.func(agent, *args, **kwargs)

        # ... 他の injection ロジック ...
```

---

## トレーシング統合

### OpenTelemetry による trace_id/span_id 自動伝搬

Graflow ワークフローと LLM 呼び出しを **OpenTelemetry の current context** を通じて自動的に関連付けます。

#### トレーシングアーキテクチャ

```
LangFuseTracer (Graflow)
    │
    ├─ span_start("task_a")
    │   └─ OpenTelemetry context に SpanContext を設定
    │       ↓
    │   [Current Context に trace_id + span_id]
    │
    ├─ Task 実行
    │   ├─ LiteLLM completion()
    │   │   └─ Current Context から自動取得 ✅
    │   │       → Langfuse に child span として送信
    │   │
    │   └─ ADK agent.run()
    │       └─ Current Context から自動取得 ✅
    │           → Langfuse に child span として送信
    │           ├─ sub_agent calls
    │           └─ tool calls
    │
    └─ span_end("task_a")
        └─ OpenTelemetry context をクリア
```

#### トレーシング経路

1. **Graflow タスク**: `graflow/trace/langfuse.py` の `LangFuseTracer`
   - ワークフロー開始/終了、タスク実行（span として記録）
   - **OpenTelemetry context を設定** ← 新規追加
   - 並列グループ、動的タスク生成など

2. **LiteLLM**: LiteLLM の組み込み Langfuse integration
   - `litellm.callbacks = ["langfuse"]` で有効化
   - **OpenTelemetry context から自動的に parent span を検出**
   - モデル名、プロンプト、レスポンス、トークン数など

3. **Google ADK** (Optional): ADK の組み込み Langfuse integration
   - `GoogleADKInstrumentor().instrument()` で有効化
   - **OpenTelemetry context から自動的に parent span を検出**
   - ReAct ループ、sub-agents、tool 呼び出しなど

#### LangFuseTracer の OpenTelemetry 統合（新規実装）

**Key Point**: Langfuse Python SDK v3 は内部で OpenTelemetry を使用しています。Langfuse span の `trace_id` と `id` (span_id) を OpenTelemetry context に設定することで、LiteLLM や ADK が自動的に parent span を検出できます。

Graflow の `LangFuseTracer` に OpenTelemetry context 設定を追加：

```python
# graflow/trace/langfuse.py

from opentelemetry import trace, context as otel_context
from opentelemetry.trace import NonRecordingSpan, SpanContext, TraceFlags

class LangFuseTracer(Tracer):
    def __init__(self, ...):
        super().__init__(...)
        # OpenTelemetry context cleanup 用
        self._otel_context_tokens: List[Any] = []

    def _output_span_start(self, name, parent_name, metadata):
        """Langfuse span 作成 + OpenTelemetry context 設定"""
        # 既存: Langfuse span 作成
        if self._span_stack:
            span = self._span_stack[-1].start_span(name=name, ...)
        else:
            span = self._root_span.start_span(name=name, ...)
        self._span_stack.append(span)

        # 新規: OpenTelemetry context を設定
        if hasattr(span, 'trace_id') and hasattr(span, 'id'):
            trace_id_int = int(span.trace_id, 16)  # hex → int
            span_id_int = int(span.id, 16)         # hex → int

            span_context = SpanContext(
                trace_id=trace_id_int,
                span_id=span_id_int,
                is_remote=False,
                trace_flags=TraceFlags(0x01)
            )

            # Current context に設定
            ctx = trace.set_span_in_context(NonRecordingSpan(span_context))
            token = otel_context.attach(ctx)
            self._otel_context_tokens.append(token)

    def _output_span_end(self, name, output, metadata):
        """Langfuse span 終了 + OpenTelemetry context クリア"""
        span = self._span_stack.pop()
        span.update(output=output, metadata=metadata)
        span.end()

        # OpenTelemetry context をクリア
        if self._otel_context_tokens:
            token = self._otel_context_tokens.pop()
            otel_context.detach(token)
```

#### LiteLLM Langfuse セットアップ

```python
# graflow/llm/config.py

def setup_langfuse_for_litellm(config: LLMConfig) -> None:
    """LiteLLM の Langfuse integration を有効化

    Note:
        metadata や trace_id の手動設定は不要。
        LiteLLM の langfuse コールバックが自動的にトレースを作成します。
    """
    if not config.enable_tracing:
        return

    import os
    import litellm

    os.environ["LANGFUSE_PUBLIC_KEY"] = config.langfuse_public_key
    os.environ["LANGFUSE_SECRET_KEY"] = config.langfuse_secret_key
    if config.langfuse_host:
        os.environ["LANGFUSE_HOST"] = config.langfuse_host

    # Langfuse コールバックを有効化
    litellm.callbacks = ["langfuse"]
```

**LLM 呼び出しのトレーシング:**

```python
# 通常の completion 呼び出し
response = llm.completion(messages)
# ↑ 自動的に Langfuse に記録される（generation_name="completion"）

# メタデータ付き
response = llm.completion(
    messages,
    generation_name="sentiment_analysis",
    tags=["production", "v1.0"]
)
# ↑ Langfuse UI で "sentiment_analysis" として表示
```

参考: [LiteLLM Langfuse Integration](https://docs.litellm.ai/docs/observability/langfuse_integration)

#### ADK Langfuse Integration

Google ADK も OpenTelemetry 経由で Langfuse に自動的にトレースされます。

```python
# ADK instrumentation 有効化（アプリケーション起動時に一度だけ実行）
from openinference.instrumentation.google_adk import GoogleADKInstrumentor

GoogleADKInstrumentor().instrument()

# Agent 作成（通常通り）
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm

agent = LlmAgent(
    name="my_agent",
    model=LiteLlm(model="gpt-4o-mini"),
    # OpenTelemetry context から自動的に parent span を検出
)

# タスク内で Agent 実行
@task(inject_llm_client=True)
def run_agent_task(llm: LLMClient, query: str) -> str:
    agent = llm.get_agent("my_agent")
    result = agent.run(query)
    # ↑ Graflow span の child span として Langfuse に記録される
    return result
```

参考: [Langfuse Google ADK Integration](https://langfuse.com/integrations/frameworks/google-adk)

---

## 使用例

### 1. シンプルな LLM タスク

```python
from graflow.core.decorators import task
from graflow.core.context import create_execution_context
from graflow.llm import LLMClient, LLMConfig

@task(inject_llm_client=True)
def summarize(llm: LLMClient, text: str) -> str:
    messages = [
        {"role": "system", "content": "You are a summarization assistant."},
        {"role": "user", "content": f"Summarize: {text}"}
    ]
    return llm.completion_text(messages)

# 実行
context = create_execution_context(
    llm_config=LLMConfig(model="gpt-4o-mini", enable_tracing=True)
)
with context:
    result = summarize.run(text="Long article...")
```

### 2. ワークフローでの LLM タスク

```python
from graflow.core.workflow import workflow

@task(inject_llm_client=True)
def analyze_sentiment(llm: LLMClient, text: str) -> str:
    messages = [{"role": "user", "content": f"Sentiment of: {text}"}]
    return llm.completion_text(messages)

@task(inject_llm_client=True)
def extract_entities(llm: LLMClient, text: str) -> List[str]:
    messages = [{"role": "user", "content": f"Extract entities: {text}"}]
    result = llm.completion_text(messages)
    return result.split(", ")

@task
def combine_results(sentiment: str, entities: List[str]) -> Dict:
    return {"sentiment": sentiment, "entities": entities}

# ワークフロー
with workflow("nlp_pipeline", llm_config=LLMConfig.from_env()) as wf:
    text = "Apple Inc. announced great earnings today."

    sentiment = analyze_sentiment(text=text)
    entities = extract_entities(text=text)
    result = combine_results(sentiment, entities)

    sentiment >> result
    entities >> result

    wf.execute()
```

### 3. ADK Supervisor パターン

```python
from graflow.llm.agents import AdkLLMAgent
from graflow.core.context import ExecutionContext

def web_search(query: str) -> str:
    """Web検索ツール（ADKツール、Graflowタスクではない）"""
    return search_api(query)

def extract_facts(text: str) -> List[str]:
    """事実抽出ツール（ADKツール、Graflowタスクではない）"""
    return fact_extraction_api(text)

@task(inject_context=True, inject_llm_client=True)
def setup_supervisor(ctx: ExecutionContext, llm: LLMClient) -> str:
    """Supervisorエージェントのセットアップ"""

    # サブエージェント定義
    researcher = AdkLLMAgent(
        name="researcher",
        llm_client=llm,
        model="gemini-2.0-flash",
        description="Research specialist",
        instruction="You research topics thoroughly",
        tools=[web_search, extract_facts]  # 独立した関数
    )

    writer = AdkLLMAgent(
        name="writer",
        llm_client=llm,
        model="gemini-2.0-flash",
        description="Content writer",
        instruction="You write engaging content"
    )

    # Supervisor（階層構造）
    supervisor = AdkLLMAgent(
        name="supervisor",
        llm_client=llm,
        model="gemini-2.0-flash",
        description="Coordinates research and writing",
        instruction="You coordinate between researcher and writer",
        sub_agents=[researcher, writer]  # サブエージェント
    )

    # ExecutionContext のレジストリに登録
    ctx.register_llm_agent("supervisor", supervisor)
    return "Setup complete"

@task(inject_llm_agent=True, agent_name="supervisor")
def run_supervisor(agent: LLMAgent, query: str) -> str:
    """Supervisorエージェントの実行（Fat単一ノード）"""

    # Agent Registry から "supervisor" を自動取得
    # Supervisor実行（内部でReActループとサブエージェント調整）
    result = agent.run(query)

    return result

# ワークフロー
with workflow("content_pipeline", llm_config=LLMConfig.from_env()) as wf:
    setup = setup_supervisor()
    result = run_supervisor(query="Write an article about AI trends")

    setup >> result  # Graflowグラフは粗粒度
    wf.execute()
```

### 4. モデルオーバーライドとメタデータ

```python
# デフォルト: gpt-4o-mini
@task(inject_llm_client=True)
def simple_task(llm: LLMClient, text: str) -> str:
    return llm.completion_text([...])

# このタスクだけ gpt-4 を使用
@task(inject_llm_client=True, model="gpt-4")
def complex_task(llm: LLMClient, text: str) -> str:
    return llm.completion_text([...])

# 実行時にモデルを指定
@task(inject_llm_client=True)
def flexible_task(llm: LLMClient, text: str, use_gpt4: bool) -> str:
    model = "gpt-4" if use_gpt4 else "gpt-4o-mini"
    return llm.completion_text([...], model=model)

# Langfuseメタデータ付き（generation_name, tags）
@task(inject_llm_client=True)
def tracked_analysis(llm: LLMClient, text: str) -> str:
    messages = [{"role": "user", "content": f"Analyze: {text}"}]
    response = llm.completion(
        messages,
        generation_name="sentiment_analysis",
        tags=["production", "v1.0"],
        temperature=0.7
    )
    return extract_text(response)

# すべてを組み合わせ
@task(inject_llm_client=True)
def full_featured_task(llm: LLMClient, text: str) -> str:
    return llm.completion_text(
        messages=[...],
        model="gpt-4",
        generation_name="content_generation",
        tags=["research", "high-priority"],
        temperature=0.9,
        max_tokens=1000
    )
```

---

## 設計の重要ポイント

### 1. 疎結合アーキテクチャ

- **LLM機能は Graflow 実行エンジンと疎結合**
  - LLMClient/LLMAgent は単なる DI されるユーティリティ
  - Graflow のタスク実行ロジックとは独立
  - ExecutionContext に組み込まれるが、オプショナル

### 2. Fat単一ノード設計

- **Supervisor/ReAct は単一タスクノード内で完結**
  - 動的タスク生成（`ctx.next_task()`）は使用しない
  - ADK の sub_agents やツール呼び出しはタスク内部で処理
  - Graflow グラフは粗粒度のワークフロー管理に専念

### 3. Tools と Tasks の分離

- **ADK tools ≠ Graflow tasks**
  - ADK の tools は独立した Python 関数
  - Graflow tasks はワークフローの構成要素
  - 両者を統合しない（混乱を避ける）

### 4. モデル選択の柔軟性

- **3段階のモデル指定**
  1. ExecutionContext レベル: `LLMConfig(model="...")`
  2. タスクレベル: `@task(inject_llm_client=True, model="...")`
  3. 実行時レベル: `llm.completion(..., model="...")`

### 5. トレーシング統合

- **自動的に trace_id/session_id を伝搬**
  - ExecutionContext の trace_id を LLMClient が継承
  - Langfuse で Graflow ワークフローと LLM 呼び出しを関連付け
  - 環境変数で簡単にトレーシングを有効化

---

## 実装計画

### Phase 1: コアインフラ

- [ ] `graflow/llm/client.py` - LLMClient 実装（completion API のみ）
- [ ] `graflow/llm/config.py` - LLMConfig 実装（Langfuse 設定含む）
- [ ] テスト: `tests/llm/test_client.py`

### Phase 2: Context 統合

- [ ] `graflow/core/context.py` - ExecutionContext 拡張
  - Agent Registry 追加（`_llm_agents`, `_llm_agents_yaml`）
  - `register_llm_agent()`, `get_llm_agent()` メソッド
- [ ] `graflow/core/decorators.py` - @task デコレーター拡張
  - `inject_llm_client`, `inject_llm_agent`, `agent_name` パラメータ
- [ ] `graflow/core/task.py` - TaskWrapper 拡張
  - LLMClient injection
  - LLMAgent injection
- [ ] テスト: `tests/llm/test_injection.py`

### Phase 3: OpenTelemetry 統合

- [ ] `graflow/trace/langfuse.py` - OpenTelemetry context 設定を追加
  - `_output_span_start()` で SpanContext を設定
  - `_output_span_end()` で context をクリア
  - `_otel_context_tokens` でトークン管理
- [ ] テスト: `tests/trace/test_otel_integration.py`

### Phase 4: Agent 統合

- [ ] `graflow/llm/agents/base.py` - LLMAgent 基底クラス
- [ ] `graflow/llm/agents/adk_agent.py` - AdkLLMAgent 実装
  - Google ADK LlmAgent のラッパー
  - `_from_adk_agent()` クラスメソッド（復元用）
- [ ] テスト: `tests/llm/test_adk_agent.py`

### Phase 5: 分散実行サポート

- [ ] `graflow/llm/serialization.py` - Agent YAML シリアライゼーション
  - `agent_to_yaml()` - BaseAgent → YAML 文字列
  - `yaml_to_agent()` - YAML 文字列 → BaseAgent
- [ ] `graflow/core/context.py` - シリアライゼーション対応
  - `__getstate__()` - Agent インスタンスを除外
  - `__setstate__()` - _llm_agents を初期化
- [ ] Worker での Agent 復元テスト
- [ ] Redis バックエンドでのテスト
- [ ] 統合テスト: `tests/llm/integration/test_distributed_agent.py`

### Phase 6: ドキュメントとサンプル

- [ ] README 更新
- [ ] `examples/llm/` - サンプルコード
  - [ ] `01_simple_completion.py`
  - [ ] `02_workflow_with_llm.py`
  - [ ] `03_adk_supervisor.py`
  - [ ] `04_model_override.py`
  - [ ] `05_distributed_agent.py`
- [ ] 統合テスト: `tests/llm/integration/`

---

## テスト戦略

### Unit Tests

```
tests/llm/
├── test_client.py              # LLMClient のテスト（LiteLLM モック）
├── test_config.py              # LLMConfig のテスト
├── test_tracing.py             # トレーシング統合のテスト
├── test_injection.py           # inject_llm_client のテスト
└── test_adk_agent.py           # AdkLLMAgent のテスト（ADK モック）
```

### Integration Tests

```
tests/llm/integration/
├── test_workflow_with_llm.py   # ワークフローでの LLM 使用
├── test_distributed_llm.py     # 分散実行でのテスト
└── test_adk_supervisor.py      # ADK Supervisor パターン
```

### モック戦略

- LiteLLM: `unittest.mock` でモック化
- ADK: Google ADK のインポートをモック化
- Langfuse: 環境変数で無効化

---

## 依存関係

### Required

- `python >= 3.11`
- `graflow` (core)

### Optional

- `litellm >= 1.0.0` - LLM プロバイダー統合（OpenAI, Anthropic, Google など）
  - Langfuse integration を含む
- `google-adk >= 0.1.0` - Google ADK (Supervisor/ReAct パターン)
  - LiteLLM サポートを含む

**注**: OpenTelemetry は不要。LiteLLM と ADK が組み込みの Langfuse integration を提供。

### pyproject.toml

```toml
[project.optional-dependencies]
llm = [
    "litellm>=1.0.0",
]
adk = [
    "google-adk>=0.1.0",
]
# すべての LLM 機能を有効化
all-llm = [
    "litellm>=1.0.0",
    "google-adk>=0.1.0",
]
```

### インストール

```bash
# LiteLLM のみ
uv pip install graflow[llm]

# ADK も含む
uv pip install graflow[all-llm]

# または
pip install "graflow[all-llm]"
```

---

## エラーハンドリング

### Optional 依存のエラーハンドリング方針

LLM 機能は optional dependency として扱い、依存が不足している場合は明確なエラーメッセージを表示します。

#### 1. LiteLLM が未インストール

```python
class LLMClient:
    def __init__(self, model: str, **default_params: Any):
        try:
            self._litellm = importlib.import_module("litellm")
        except ImportError:
            raise RuntimeError(
                "LiteLLM is not installed. "
                "Install with: pip install 'graflow[llm]'"
            )
```

**動作**: 即時例外を発生させる（LLMClient は liteLLM 必須）

#### 2. Langfuse トレーシングが無効

```python
def setup_langfuse_for_litellm(config: LLMConfig) -> None:
    """LiteLLM の Langfuse integration を有効化"""
    if not config.enable_tracing:
        return

    import os
    import litellm

    # 環境変数チェック
    if not config.langfuse_public_key or not config.langfuse_secret_key:
        logger.warning(
            "Langfuse credentials not found. "
            "LLM calls will not be traced. "
            "Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY to enable tracing."
        )
        return

    # Langfuse コールバックを有効化
    os.environ["LANGFUSE_PUBLIC_KEY"] = config.langfuse_public_key
    os.environ["LANGFUSE_SECRET_KEY"] = config.langfuse_secret_key
    if config.langfuse_host:
        os.environ["LANGFUSE_HOST"] = config.langfuse_host

    litellm.callbacks = ["langfuse"]
```

**動作**: 警告を出してトレース無しで続行（トレーシングはオプショナル）

#### 3. ADK が未インストール

```python
class AdkLLMAgent(LLMAgent):
    def __init__(self, name: str, llm_client: LLMClient, **kwargs):
        try:
            from google.adk.agents import LlmAgent
            from google.adk.models.lite_llm import LiteLlm
        except ImportError:
            raise ImportError(
                "Google ADK is not installed. "
                "Install with: pip install 'graflow[adk]'"
            )
```

**動作**: Agent 作成時に例外を発生させる（ADK 使用時のみ必要）

#### 4. LLMClient が ExecutionContext に未設定

```python
class TaskWrapper(Executable):
    def __call__(self, *args, **kwargs) -> Any:
        if self.inject_llm_client:
            exec_context = self.get_execution_context()
            llm_client = exec_context.llm_client

            if llm_client is None:
                raise RuntimeError(
                    f"Task {self.task_id} requires inject_llm_client=True "
                    "but no LLM client configured. "
                    "Set llm_config in create_execution_context() or workflow()."
                )
```

**動作**: タスク実行時に例外を発生させる（設定ミスを明確に指摘）

#### 5. LLMAgent が Agent Registry に未登録

```python
class TaskWrapper(Executable):
    def __call__(self, *args, **kwargs) -> Any:
        if self.inject_llm_agent:
            if self.agent_name is None:
                raise RuntimeError(
                    f"Task {self.task_id} requires agent_name parameter "
                    "when inject_llm_agent=True"
                )

            try:
                agent = exec_context.get_llm_agent(self.agent_name)
            except KeyError:
                raise RuntimeError(
                    f"Task {self.task_id} requires LLMAgent '{self.agent_name}' "
                    "but not found in registry. "
                    "Use ctx.register_llm_agent('{self.agent_name}', agent) first."
                )
```

**動作**: タスク実行時に例外を発生させる（Agent 未登録を明確に指摘）

### エラーメッセージの一貫性

すべてのエラーメッセージに以下を含める：

1. **問題の説明**: 何が不足しているか
2. **解決方法**: インストールコマンドまたは設定方法
3. **コンテキスト**: どのタイミングでエラーが発生したか

---

## FAQ

### Q1: ADK tools と Graflow tasks を統合できないか？

**A**: 意図的に分離しています。理由：

- ADK tools は細粒度の関数（web検索、計算など）
- Graflow tasks は粗粒度のワークフロー構成要素
- 統合すると責務が曖昧になり、複雑性が増す

### Q2: Supervisor が動的にタスクを生成できないか？

**A**: できますが、推奨しません。理由：

- Supervisor は Fat単一ノードとして設計
- 動的タスク生成は Graflow グラフを複雑化
- ADK の sub_agents で十分な表現力がある

もし必要なら、`inject_context=True` も併用：

```python
@task(inject_context=True, inject_llm_client=True)
def dynamic_supervisor(ctx: ExecutionContext, llm: LLMClient, query: str):
    supervisor = llm.get_agent("supervisor")
    result = supervisor.run(query)

    # 必要なら動的にタスク追加
    if result.needs_follow_up:
        ctx.next_task(follow_up_task)

    return result
```

### Q3: タスクごとに異なるモデルを使いたい

**A**: `model` パラメータを使って、タスクごとに異なる LLMClient インスタンスを作成できます：

```python
# ExecutionContext のデフォルトモデルを gpt-4o-mini に設定
llm_config = LLMConfig(model="gpt-4o-mini", default_params={"temperature": 0.7})
context = ExecutionContext.create(graph, start_node, llm_config=llm_config)

# タスクA: デフォルトモデル（gpt-4o-mini）を使用
@task(inject_llm_client=True)
def summarize(llm: LLMClient, text: str) -> str:
    # llm.model == "gpt-4o-mini"
    return llm.completion_text([...])

# タスクB: 高性能モデル（gpt-4o）を使用
@task(inject_llm_client=True, model="gpt-4o")
def analyze(llm: LLMClient, data: str) -> str:
    # llm.model == "gpt-4o"
    # 新しい LLMClient インスタンスが作成される
    # default_params（temperature=0.7）は継承される
    return llm.completion_text([...])

# タスクC: 別のプロバイダー（Claude）を使用
@task(inject_llm_client=True, model="claude-3-5-sonnet-20241022")
def reason(llm: LLMClient, problem: str) -> str:
    # llm.model == "claude-3-5-sonnet-20241022"
    # 別の LLMClient インスタンスが作成される
    return llm.completion_text([...])
```

**ポイント**:
- 各タスクで別の LLMClient インスタンスが作成される
- `default_params`（temperature, max_tokens など）は ExecutionContext の設定を継承
- モデルのみが上書きされる
- コスト効率と性能のバランスを最適化可能

**ユースケース**:
- 簡単なタスク: `gpt-4o-mini`（低コスト）
- 複雑な分析: `gpt-4o` or `claude-3-5-sonnet-20241022`（高性能）
- コード生成: `claude-3-5-sonnet-20241022`（コード特化）

### Q4: `inject_context` と `inject_llm_client` を同時に使える？

**A**: 可能です：

```python
# 両方使う場合（Agent Registry にアクセスする場合など）
@task(inject_context=True, inject_llm_client=True)
def my_task(ctx: ExecutionContext, llm: LLMClient, data: str):
    # LLM completion
    result = llm.completion_text([...])

    # Agent を登録
    agent = create_agent()
    ctx.register_llm_agent("my_agent", agent)

    return result

# または ctx 経由でアクセス
@task(inject_context=True)
def my_task(ctx: ExecutionContext, data: str):
    llm = ctx.llm_client
    llm.completion_text([...])
```

### Q5: 分散実行で LLMClient はどうなる？

**A**: LLMConfig がシリアライズされ、Worker で再構築：

1. ExecutionContext が Worker にシリアライズ
2. Worker で ExecutionContext を復元
3. `llm_client` プロパティアクセス時に遅延初期化
4. Langfuse トレーシングは Worker でも自動的に有効化される

### Q6: 分散実行で LLMAgent はどうなる？

**A**: Agent を YAML でシリアライズして Worker に送信：

```python
# Agent のシリアライゼーション
from google.adk.utils.yaml_utils import dump_pydantic_to_yaml
import io

def agent_to_yaml(agent: BaseAgent) -> str:
    """BaseAgent を YAML 文字列に変換"""
    buf = io.StringIO()
    dump_pydantic_to_yaml(agent, buf)
    return buf.getvalue()

# Agent の復元
import yaml
from google.adk.agents import BaseAgent

def yaml_to_agent(yaml_str: str) -> BaseAgent:
    """YAML 文字列から BaseAgent を復元"""
    cfg = yaml.safe_load(yaml_str)
    agent = BaseAgent.from_config(cfg)
    return agent
```

**ExecutionContext での処理:**

1. `register_llm_agent()` 時に Agent を YAML に変換して保存
2. Worker にシリアライズ時に YAML 文字列を含める
3. Worker で `get_llm_agent()` 時に YAML から復元

詳細は「分散実行サポート」セクションを参照

---

## 分散実行サポート

### Agent のシリアライゼーション

Google ADK の BaseAgent は YAML ベースの設定をサポートしており、これを活用して分散実行を実現します。

#### シリアライゼーションヘルパー

```python
# graflow/llm/serialization.py

import io
import yaml
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from google.adk.agents import BaseAgent

def agent_to_yaml(agent: "BaseAgent") -> str:
    """BaseAgent を YAML 文字列に変換

    Args:
        agent: Google ADK の BaseAgent インスタンス

    Returns:
        YAML 文字列

    Raises:
        ImportError: Google ADK が未インストール
    """
    try:
        from google.adk.utils.yaml_utils import dump_pydantic_to_yaml
    except ImportError:
        raise ImportError(
            "Google ADK is not installed. "
            "Install with: pip install 'graflow[adk]'"
        )

    buf = io.StringIO()
    dump_pydantic_to_yaml(agent, buf)
    return buf.getvalue()


def yaml_to_agent(yaml_str: str) -> "BaseAgent":
    """YAML 文字列から BaseAgent を復元

    Args:
        yaml_str: YAML 文字列

    Returns:
        BaseAgent インスタンス

    Raises:
        ImportError: Google ADK が未インストール
    """
    try:
        from google.adk.agents import BaseAgent
    except ImportError:
        raise ImportError(
            "Google ADK is not installed. "
            "Install with: pip install 'graflow[adk]'"
        )

    cfg = yaml.safe_load(yaml_str)
    agent = BaseAgent.from_config(cfg)
    return agent
```

#### ExecutionContext の拡張

```python
class ExecutionContext:
    def __init__(self, ...):
        self._llm_agents: Dict[str, LLMAgent] = {}
        self._llm_agents_yaml: Dict[str, str] = {}  # YAML 文字列キャッシュ

    def register_llm_agent(self, name: str, agent: LLMAgent) -> None:
        """LLMAgent を登録（YAML でシリアライズ）"""
        self._llm_agents[name] = agent

        # Agent を YAML に変換してキャッシュ（分散実行用）
        if isinstance(agent, AdkLLMAgent):
            from graflow.llm.serialization import agent_to_yaml
            self._llm_agents_yaml[name] = agent_to_yaml(agent._adk_agent)

    def get_llm_agent(self, name: str) -> LLMAgent:
        """LLMAgent を取得（必要なら YAML から復元）"""
        # キャッシュにあればそれを返す
        if name in self._llm_agents:
            return self._llm_agents[name]

        # YAML から復元（Worker での実行時）
        if name in self._llm_agents_yaml:
            from graflow.llm.serialization import yaml_to_agent
            from graflow.llm.agents import AdkLLMAgent

            adk_agent = yaml_to_agent(self._llm_agents_yaml[name])
            agent = AdkLLMAgent._from_adk_agent(adk_agent)
            self._llm_agents[name] = agent  # キャッシュ
            return agent

        raise KeyError(f"LLMAgent '{name}' not found in registry")

    def __getstate__(self):
        """シリアライゼーション時の処理"""
        state = self.__dict__.copy()
        # Agent インスタンスはシリアライズしない（YAML のみ送信）
        state['_llm_agents'] = {}
        return state

    def __setstate__(self, state):
        """デシリアライゼーション時の処理"""
        self.__dict__.update(state)
        # _llm_agents は空で初期化（必要時に YAML から復元）
        if '_llm_agents' not in self.__dict__:
            self._llm_agents = {}
```

### 分散実行フロー

```
┌─────────────────────────────────────────────────────────┐
│ Main Process                                            │
│                                                         │
│  1. Agent 作成                                          │
│     supervisor = AdkLLMAgent(...)                      │
│                                                         │
│  2. Agent 登録                                          │
│     ctx.register_llm_agent("supervisor", supervisor)   │
│     └─ agent_to_yaml() で YAML に変換                  │
│                                                         │
│  3. ExecutionContext をシリアライズ                     │
│     └─ _llm_agents_yaml (YAML 文字列) のみ送信        │
│                                                         │
└─────────────────────────────────────────────────────────┘
                        │
                        │ Redis Queue
                        ▼
┌─────────────────────────────────────────────────────────┐
│ Worker Process                                          │
│                                                         │
│  1. ExecutionContext を復元                            │
│     └─ _llm_agents は空                                │
│     └─ _llm_agents_yaml に YAML 文字列                 │
│                                                         │
│  2. Agent を取得                                        │
│     agent = ctx.get_llm_agent("supervisor")            │
│     └─ yaml_to_agent() で YAML から復元                │
│                                                         │
│  3. Agent を実行                                        │
│     result = agent.run(query)                          │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 注意点

1. **Agent の状態**: Agent のランタイム状態（実行履歴など）はシリアライズされない
2. **Tools のシリアライゼーション**: ADK tools も YAML に含まれる（関数参照は文字列として保存）
3. **Sub-agents**: 階層的な sub-agents も再帰的にシリアライズされる
4. **LLMClient の参照**: Agent 内の LLMClient 参照は復元時に再構築が必要

### 実装優先度

- **Phase 1-4**: ローカル実行での動作確認
- **Phase 5**: 分散実行サポート（YAML シリアライゼーション）

---

## 参考資料

- [LiteLLM Documentation](https://docs.litellm.ai/)
- [LiteLLM Langfuse Integration](https://docs.litellm.ai/docs/observability/langfuse_integration)
- [Langfuse Documentation](https://langfuse.com/docs)
- [Google ADK Documentation](https://developers.google.com/adk)
- Graflow 既存設計ドキュメント:
  - `docs/architecture_ja.md`
  - `docs/trace_module_design_ja.md`

---

## まとめ

### 設計の核心

この設計は **OpenTelemetry の自動伝搬** を活用することで、シンプルかつ強力な LLM 統合を実現します。

#### 3つの統合レイヤー

1. **Graflow Layer**
   - `@task(inject_llm_client=True)` で LLMClient を DI
   - `ExecutionContext` が LLMClient のライフサイクルを管理
   - `LangFuseTracer` が OpenTelemetry context を設定

2. **LLM Layer**
   - `LLMClient`: LiteLLM のシンプルなラッパー（completion API）
   - `LLMAgent`: Google ADK のラッパー（Supervisor/ReAct パターン）
   - Agent Registry: ExecutionContext で管理（simple dict）

3. **Tracing Layer**
   - LangFuseTracer が OpenTelemetry context に trace_id/span_id を設定
   - LiteLLM が current context から自動的に parent span を検出
   - Langfuse UI で統一的に可視化（手動での trace_id 受け渡し不要）

#### 自動伝搬のメリット

**手動管理が不要:**
```python
# ✅ シンプルで堅牢
llm = LLMClient(model="gpt-4o-mini")
llm.completion(messages)  # trace_id/span_id は自動検出！
```

**実装のシンプルさ:**
- **LLMClient**: trace_id/span_id パラメータ不要
- **LangFuseTracer**: `_output_span_start()` と `_output_span_end()` に各10行程度の追加のみ
- **LiteLLM**: 既存の Langfuse コールバックを使用（変更なし）

#### トレーシングの可視化

Langfuse UI での表示例：

```
Trace: workflow_execution (trace_id: abc123...)
  └─ Span: supervisor_task
      ├─ Span: litellm.completion (model: gpt-4o-mini)  ← 自動階層化
      │   └─ usage: {total_tokens: 150}
      └─ Span: adk.agent.run (agent: supervisor)        ← 自動階層化
          ├─ Span: sub_agent.researcher
          │   └─ Span: tool.web_search
          └─ Span: sub_agent.writer
```

### 次のステップ

1. **Phase 1**: LLMClient, LLMConfig 実装（基本機能）
2. **Phase 2**: ExecutionContext 統合、@task デコレーター拡張（Agent Registry 含む）
3. **Phase 3**: LangFuseTracer に OpenTelemetry context 設定を追加（自動伝搬実現）
4. **Phase 4**: AdkLLMAgent 実装（Supervisor/ReAct パターン）
5. **Phase 5**: 分散実行サポート（Agent YAML シリアライゼーション）
6. **Phase 6**: サンプルコードとドキュメント

### 設計の成功基準

- ✅ タスク内で `@task(inject_llm_client=True)` で簡単に LLM 利用
- ✅ タスク内で `@task(inject_llm_agent=True, agent_name="...")` で簡単に Agent 利用
- ✅ Agent Registry を ExecutionContext で管理（simple dict）
- ✅ LiteLLM で複数の LLM プロバイダーをサポート（OpenAI, Anthropic, Google など）
- ✅ Langfuse で Graflow タスクと LLM 呼び出しを統一的にトレース
- ✅ Google ADK で Supervisor/ReAct パターンをサポート
- ✅ 疎結合アーキテクチャで Graflow 実行エンジンと独立
- ✅ OpenTelemetry 自動伝搬で trace_id/span_id の手動管理不要

### 将来の拡張

以下は将来のサポート対象として検討中：

1. **複数クライアント管理**: `@task(inject_llm_client=True, llm_alias="supervisor")`
2. **ADK Langfuse トレーシング**: GoogleADKInstrumentor を使った Agent 実行のトレース
3. **LLMManager**: 複数 LLMClient の管理と Agent 登録の集約
