# Remote Task Handler 設計ディスカッション

> **Status**: Draft / Under Discussion
> **Date**: 2026-04-12
> **Scope**: GraflowをGoから利用可能にするための Remote Task Handler 設計

---

## 1. 背景と目的

Graflowは Python 用のワークフローエンジンだが、Go からもタスクを実装・実行できるようにしたい。既存の Python ユーザを壊さずに、言語中立のタスク実行パスを追加する。

### 検討した全体アプローチ

| アプローチ | 概要 | 評価 |
|---|---|---|
| gRPC制御面のみ | GoからPython登録済みタスクを組成・監視 | 最小変更だがGoでタスクを書けない |
| YAML DSL + コンテナタスク (Argo型) | 言語中立だがオーバーヘッド大 | 軽量タスクに不向き |
| Polyglot Worker (Temporal型) | 各言語ネイティブでタスク実装 | 最も柔軟だが設計コスト大 |
| Sidecar (stdio JSON-RPC) | 超シンプルだが分散実行と相性悪い | 小規模向き |
| REST + OpenAPI | 既存API拡張だがストリーミングに弱い | HITL等との相性悪い |

### 採択方針

**Remote Task Handler** として、既存の `TaskHandler` 抽象に gRPC ベースの新ハンドラを追加する方針を採択。段階的に機能を拡張する。

---

## 2. 既存アーキテクチャの確認

### TaskHandler 階層

```
TaskHandler (ABC)  ← graflow/core/handler.py
├── DirectTaskHandler   ← in-process実行 (デフォルト)
├── DockerTaskHandler   ← Docker container内実行 (cloudpickle経由)
└── RemoteTaskHandler   ← 【新規】gRPC経由でGo等のリモートワーカに委譲
```

- `DirectTaskHandler`: `task.run()` を直接呼ぶ。最もシンプル
- `DockerTaskHandler`: cloudpickle でタスクと context を丸ごとシリアライズし、Docker container 内で実行。結果を stdout の `CONTEXT:` 行で回収
- いずれも `execute_task(task, context) -> Any` を実装。engine のトポロジカル実行ループから同期的に呼ばれる

### Channel 抽象

`Channel` (ABC) は K/V + list + atomic + lock の API。gRPC proxy 化に適する（各メソッドが 1 RPC で対応可能）。

```python
class Channel(ABC):
    def set(self, key, value, ttl=None): ...
    def get(self, key, default=None): ...
    def delete(self, key): ...
    def exists(self, key): ...
    def keys(self): ...
    def clear(self): ...
    def append(self, key, value, ttl=None): ...
    def prepend(self, key, value, ttl=None): ...
    def atomic_add(self, key, amount=1): ...
    def lock(self, key, timeout=10.0): ...  # context manager
```

### 既存 TaskWorker (Python, Redis)

- `TaskWorker` は Redis キューからタスクを pull して `WorkflowEngine` で実行
- Go ワーカとは別の仕組みとして共存させる想定

---

## 3. シリアライゼーション戦略

### cloudpickle の限界と Fory (Apache Fury) の導入検討

| 観点 | cloudpickle | Fory | protobuf |
|---|---|---|---|
| 言語対応 | Python only | Python/Go/Java/C++/JS/Rust | ほぼ全言語 |
| 関数/クロージャ | 可 | 不可 | 不可 |
| 動的型 (dict[str, Any]) | 可 | 制約あり (要型登録) | 不可 (Any型はあるが制約あり) |
| スキーマ進化 | なし | 部分的 | フィールド番号ベースで堅牢 |
| 速度 | 遅い | 速い | 速い |

### 使い分け方針 (案)

- **制御構造 (workflow spec, gRPC)**: protobuf (スキーマ進化・codegen が安定)
- **タスク引数・戻り値 (cross-language)**: protobuf または JSON (codec 選択可能)
- **Python内部 (direct/docker)**: cloudpickle 維持 (既存互換)
- **Fory**: Python内部の cloudpickle 置換として段階導入。cross-language 通信の正本は protobuf に寄せる

### タスク引数の制約

`RemoteTaskHandler` 経由のタスクは、引数・戻り値を「protobuf/JSON で表現可能な型」に限定する:

- primitives, dataclass, list/dict, 登録済み type のみ
- クロージャ・関数オブジェクトは引数にできない (明示エラー)
- `task_ref` (名前参照) でタスクを識別。Python のバイトコードは送らない

---

## 4. gRPC サービス設計 (ドラフト)

### 4.1 TaskDispatch — タスクの配送

```proto
service TaskDispatch {
  rpc PollTask(PollRequest) returns (PollResponse);
  rpc DispatchTask(DispatchRequest) returns (DispatchResponse);  // Push型
  rpc Heartbeat(stream HeartbeatRequest) returns (stream HeartbeatResponse);
  rpc CompleteTask(CompleteRequest) returns (CompleteResponse);
}

message PollRequest {
  string worker_id = 1;
  string task_queue = 2;
  repeated string capabilities = 3;  // worker が実装している task_ref 一覧
  int32 poll_timeout_ms = 4;
}

message PollResponse {
  string lease_token = 1;
  string task_id = 2;
  string task_ref = 3;        // 論理タスク名 (例 "etl.transform")
  bytes  input = 4;           // ペイロード
  string codec = 5;           // "json" | "fory" | "protobuf"
  int64  deadline_unix_ms = 6;
  map<string, string> metadata = 7;  // trace_id, workflow_id, attempt
}

message CompleteRequest {
  string lease_token = 1;
  oneof outcome {
    TaskSuccess success = 2;
    TaskFailure failure = 3;
  }
}

message TaskSuccess { bytes output = 1; string codec = 2; }
message TaskFailure { string type = 1; string message = 2; string stacktrace = 3; bool retryable = 4; }
```

### 4.2 ChannelProxy — Channel の gRPC プロキシ

既存 `Channel` ABC のメソッドを 1:1 で RPC 化。全 RPC に `lease_token` を流して認可。

```proto
service ChannelProxy {
  rpc Set(SetRequest) returns (Empty);
  rpc Get(GetRequest) returns (GetResponse);
  rpc Delete(KeyRequest) returns (BoolResponse);
  rpc Exists(KeyRequest) returns (BoolResponse);
  rpc Keys(ChannelRequest) returns (KeysResponse);
  rpc Append(AppendRequest) returns (Int64Response);
  rpc Prepend(AppendRequest) returns (Int64Response);
  rpc AtomicAdd(AtomicAddRequest) returns (NumberResponse);
  rpc Lock(stream LockRequest) returns (stream LockResponse);
}
```

### 4.3 Control — 補助系 (段階的に追加)

```proto
service Control {
  rpc RequestFeedback(FeedbackReq) returns (FeedbackResp);  // HITL
  rpc ScheduleCheckpoint(CheckpointReq) returns (Empty);
  rpc EmitTraceEvent(TraceEvent) returns (Empty);
  rpc NextTask(NextTaskReq) returns (Empty);                // 動的タスク追加
}
```

---

## 5. Python 側の実装スケッチ

### RemoteTaskHandler

```python
class RemoteTaskHandler(TaskHandler):
    """Dispatch tasks to remote workers (Go, etc.) via gRPC."""

    def __init__(self, task_queue: str, codec: str = "json", timeout: float = 300.0,
                 dispatch_server: "TaskDispatchServer" = None):
        self.task_queue = task_queue
        self.codec = codec
        self.timeout = timeout
        self.dispatch = dispatch_server

    def get_name(self) -> str:
        return f"remote:{self.task_queue}"

    def execute_task(self, task: Executable, context: ExecutionContext) -> Any:
        task_ref = getattr(task, "task_ref", None) or task.__class__.__name__
        payload = encode(task.inputs, self.codec)

        lease = self.dispatch.submit(
            task_id=task.task_id, task_ref=task_ref,
            input=payload, codec=self.codec, task_queue=self.task_queue,
            metadata={"workflow_id": context.workflow_id, "trace_id": context.trace_id},
        )

        outcome = lease.wait(timeout=self.timeout)
        if outcome.failure:
            exc = _rebuild_exception(outcome.failure)
            context.set_result(task.task_id, exc)
            raise exc

        result = decode(outcome.output, outcome.codec)
        context.set_result(task.task_id, result)
        return result
```

### Go SDK のイメージ

```go
w := graflow.NewWorker(graflow.WorkerConfig{
    Endpoint:  "engine.internal:50051",
    WorkerID:  "go-worker-1",
    TaskQueue: "go-etl",
    Codec:     graflow.CodecJSON,
})

w.Register("etl.transform", func(ctx graflow.Context, in TransformInput) (TransformOutput, error) {
    ch := ctx.Channel("metrics")
    ch.AtomicAdd("rows_processed", int64(len(in.Rows)))
    return TransformOutput{...}, nil
})

w.Run(context.Background())
```

---

## 6. 未決定の設計論点

### 論点 1: Push vs Pull (最上流の分岐)

| | Push (engine → worker) | Pull (worker → engine) |
|---|---|---|
| モデル | RPC呼び出し (DockerTaskHandler に近い) | Temporal / Celery 型 |
| engine知識 | worker endpoint を知る必要 | 不要 |
| NAT/fw | worker が公開必要 | worker は発信のみ |
| スケール | engine が LB 的に分散 | worker が自律的に取り合い |
| デバッグ | シンプル (同期呼び出し) | lease/timeout/再配布を実装する必要 |
| DockerTaskHandler との類似度 | 高い | 低い |
| TaskWorker (Redis) との類似度 | 低い | 高い |

**検討ポイント**: graflow のターゲットユーザ像で決まる。単一マシン・小規模なら Push がシンプル。大規模・独立デプロイなら Pull。**proto には両方の RPC を定義しておき、初期実装をどちらにするか**が論点。

### 論点 2: Remote Task に露出する ExecutionContext の範囲

| 要素 | 露出? | 備考 |
|---|---|---|
| 上流タスクの結果 | **必須** | 引数として事前注入 or RPC で取得 |
| Channel read/write | **必須** | graflow の中核機能 |
| workflow metadata | **必須** | observability, idempotency |
| HITL feedback request | 望ましい | graflow の差別化要因 |
| next_task / next_iteration | 要議論 | Go から task_ref 名前渡しなら可能 |
| terminate / cancel workflow | 要議論 | 可能だが権限が重い |
| checkpoint trigger | 要議論 | engine 側がやる方が自然かも |
| graph 構造参照 | 不要 | カプセル化を破る |

**検討ポイント**: 「Go タスクは独立した純粋関数」と扱えば inputs/outputs + channels だけで済む (シンプル)。「engine の coroutine」として扱うなら next_task 等まで必要だが Go SDK が肥大化する。

### 論点 3: 型定義の source of truth

| 方式 | 長所 | 短所 |
|---|---|---|
| **(a) Python dataclass 正本** | Python ファースト体験維持 | Go 側の同期が手動 or codegen ツール同梱必要 |
| **(b) Go struct 正本** | Go ネイティブ | Python 側の同期が課題 |
| **(c) protobuf IDL 正本** | 既存 protoc で codegen。スキーマ進化が堅牢 | 型定義が .proto に分散。開発体験が一手間増える |

**検討ポイント**: graflow がコード生成ツールを同梱するかどうかで決まる。同梱しないなら (c) protobuf が最小コスト。

### 論点 4: Dispatcher の場所

| 方式 | 長所 | 短所 |
|---|---|---|
| **(a) engine 内蔵** (in-memory) | シンプル、低レイテンシ | SPOF、再起動で in-flight タスク消失 |
| **(b) Redis queue 上** | HA、再起動耐性 | Go worker も Redis 依存。cloudpickle ≠ 言語中立の壁 |
| **(c) 抽象化して両方** | ユーザ選択 | 実装 2 倍 |

**検討ポイント**: checkpoint 機能でengine crash からの復帰は既に想定済み。dispatcher が in-memory でも checkpoint 復帰時に lease 失効・再試行できれば OK かもしれない。

### 論点 5: At-least-once / At-most-once / Idempotency

Remote task は二重実行の可能性がある (heartbeat 切れ → 再 dispatch → 元は生きていた)。

- engine 側 dedup (lease_token + task_id)
- ユーザ側 idempotent 責任
- ハイブリッド

既存 `@task` の retry 処理に合わせるのが筋。

### 論点 6: 既存 TaskWorker (Python, Redis) との関係

| 方式 | 概要 |
|---|---|
| **A. 完全別物** | Python worker は Redis、Go worker は gRPC。engine が振り分け |
| **B. 統一** | Python worker も gRPC 経由に移行 |
| **C. gRPC → Redis** | gRPC Dispatch が内部的に Redis に載る |

A が互換性的に安全。B は美しいが大手術。

### 論点 7: Task versioning

`task_ref = "etl.transform"` だけではスキーマ不一致時に落ちる。

- ユーザ責任 (ignore)
- `task_ref + version` を必須に
- protobuf フィールド番号でスキーマ進化に任せる

### 論点 8: Security

| 要素 | 初期方針案 |
|---|---|
| worker → engine 認証 | mTLS or static token |
| lease_token | opaque ID、in-memory 照合 |
| Channel 認可 | 自分の lease に紐づく channel のみ操作可能 |
| multi-tenant | スコープ外 |

---

## 7. 実装フェーズ (暫定)

| Phase | 内容 | 依存する論点 |
|---|---|---|
| 1 | proto 定義と生成 (`graflow/proto/`) | 論点 1, 2, 3 |
| 2 | DispatchServer 実装 (engine 同居) | 論点 4 |
| 3 | RemoteTaskHandler 実装と登録 | - |
| 4 | ChannelProxyServer | 論点 2 |
| 5 | Codec 層 (JSON → Fory 段階導入) | 論点 3 |
| 6 | Go SDK 最小実装 | - |
| 7 | HITL (RequestFeedback) 追加 | 論点 2 |
| 8 | 動的タスク (NextTask) と checkpoint | 論点 2 |

---

## 8. 議論の進め方

**論点 1 → 2 → 3 → 4** の順に決めると後続が自動的に絞れる。

- **1 (Push/Pull)** → proto の形が決まる
- **2 (Context 露出範囲)** → ChannelProxy/Control の粒度が決まる
- **3 (型正本)** → Fory/protobuf の役割が決まる
- **4 (Dispatcher 配置)** → HA 設計が決まる
- 5-8 は詳細として後追い
