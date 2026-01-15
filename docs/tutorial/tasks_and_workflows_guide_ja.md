# タスクとワークフローガイド

Graflowでワークフローを構築するための実践ガイド — 最初のタスクから高度なパターンまで。

このガイドでは、実践的な例を通じてタスクの定義とワークフローの構築方法を学びます。

### チートシート

| 概念 | 構文 | 目的 |
|---------|--------|---------|
| タスク定義 | `@task` | 関数をタスクに変換 |
| カスタムタスクID | `@task(task_id="id")` | タスク識別子を明示的に指定 |
| ワークフロー作成 | `with workflow("name") as wf:` | タスクグラフを定義 |
| 直列 | `task_a >> task_b` | タスクを順番に実行 |
| 並列 | `task_a \| task_b` | タスクを同時に実行 |
| タスクの連結 | `chain(task_a, task_b, task_c)` | 直列チェーンを作成 |
| 並列タスク | `parallel(task_a, task_b, task_c)` | 並列グループを作成 |
| タスクインスタンス | `task(task_id="id", param=value)` | パラメータ付きで新しいタスクインスタンスを作成 |
| グループ名設定 | `task_group.set_group_name("name")` | 並列グループの名前を変更 |
| 実行設定 | `task_group.with_execution(policy="...")` | 並列グループの実行ポリシーを設定 |
| コンテキスト注入 | `@task(inject_context=True)` | チャンネル/ワークフロー制御にアクセス |
| LLMクライアント注入 | `@task(inject_llm_client=True)` | LLM APIを直接呼び出し |
| LLMエージェント注入 | `@task(inject_llm_agent="name")` | ツール付きSuperAgentを注入 |
| プロンプトマネージャー作成 | `PromptManagerFactory.create("yaml", ...)` | プロンプトバックエンドのファクトリ |
| プロンプトマネージャー | `ctx.prompt_manager` | プロンプトテンプレートにアクセス |
| テキストプロンプト取得 | `pm.get_text_prompt("name")` | テキストプロンプトテンプレートを取得 |
| チャットプロンプト取得 | `pm.get_chat_prompt("name")` | チャットプロンプトテンプレートを取得 |
| プロンプトをレンダリング | `prompt.render(var=value)` | テンプレート変数を置換 |
| チャンネル取得 | `ctx.get_channel()` | キー・バリューチャンネルにアクセス |
| TTL付き保存 | `channel.set(key, value, ttl=300)` | 有効期限付きで保存(秒) |
| リスト末尾に追加 | `channel.append(key, value)` | リスト末尾に追加 |
| リスト先頭に追加 | `channel.prepend(key, value)` | リスト先頭に追加 |
| 型付きチャンネル取得 | `ctx.get_typed_channel(Schema)` | 型安全なチャンネルにアクセス |
| フィードバック要求 | `ctx.request_feedback(...)` | HITLによる承認/入力 |
| 初期パラメータ | `wf.execute(initial_channel={...})` | ワークフローの初期パラメータを設定 |
| 全結果取得 | `wf.execute(ret_context=True)` | すべてのタスク結果にアクセス |
| タスク結果取得 | `ctx.get_result(task_id)` | 特定タスクの結果を取得 |
| タスクをキューへ追加 | `ctx.next_task(task)` | タスクを追加し通常の後続へ |
| タスクへジャンプ | `ctx.next_task(task, goto=True)` | 既存タスクへ移動し後続をスキップ |
| 自己ループ | `ctx.next_iteration()` | リトライ/収束パターン |
| 正常終了 | `ctx.terminate_workflow()` | 正常に終了 |
| エラー終了 | `ctx.cancel_workflow()` | エラーで終了 |

---

## 目次

**はじめに**
- [レベル1: 最初のタスク](#レベル1-最初のタスク) - @taskデコレータとタスクID
- [レベル2: 最初のワークフロー](#レベル2-最初のワークフロー) - ワークフローコンテキストと実行
- [レベル3: タスク合成](#レベル3-タスク合成) - 直列(>>)と並列(|)演算子
- [レベル4: パラメータの受け渡し](#レベル4-パラメータの受け渡し) - チャンネルとパラメータバインド

**コアコンセプト**
- [レベル5: タスクインスタンス](#レベル5-タスクインスタンス) - 異なるパラメータでの再利用
- [レベル6: チャンネルとコンテキスト](#レベル6-チャンネルとコンテキスト) - タスク間通信、注入、プロンプト管理
- [レベル7: 実行パターン](#レベル7-実行パターン) - 結果取得と実行制御
- [レベル8: 複雑なワークフロー](#レベル8-複雑なワークフロー) - ダイヤモンドパターンと複数インスタンス

**高度なトピック**
- [レベル9: 動的タスク生成](#レベル9-動的タスク生成) - 実行時のタスク追加と制御フロー

**リファレンス**
- [ベストプラクティス](#ベストプラクティス)
- [まとめ](#まとめ)

---

## コアコンセプト

始める前に、主要な概念を確認します:

- **タスク**: 作業単位 ( `@task` デコレータ付きのPython関数 )
- **ワークフロー**: 依存関係を持つタスクの集合
- **タスクグラフ**: タスク実行順序を表す有向グラフ
- **実行コンテキスト**: 実行時の状態 (チャンネル、結果、メタデータ)

---

## レベル1: 最初のタスク

まずは基本中の基本、`@task`デコレータから始めます。

### @taskデコレータ

任意のPython関数をGraflowタスクに変換できます:

```python
from graflow.core.decorators import task

@task
def hello():
    """A simple task."""
    print("Hello, Graflow!")
    return "success"
```

**何が起きたのか?**
- `@task` により通常の関数がGraflowタスクになります
- タスクはワークフロー内で使うことも直接実行することもできます

### カスタムタスクID

デフォルトでは関数名がタスクIDになります。カスタムIDを指定することも可能です:

```python
# Default: task_id is "hello"
@task
def hello():
    print("Hello!")

# Custom: task_id is "greeting_task"
@task(task_id="greeting_task")
def hello():
    print("Hello!")
```

**💡 重要ポイント:**
- `@task` を使ってタスクを作成
- デフォルトの `task_id` は関数名
- `@task(task_id="custom_id")` で明示的に命名

### .run() でタスクをテストする

タスクは `.run()` を使って直接実行し、テストできます:

```python
@task
def calculate(x: int, y: int) -> int:
    """Add two numbers."""
    return x + y

# Test the task directly
result = calculate.run(x=5, y=3)
print(result)  # Output: 8
```

**`.run()` を使うタイミング:**
- ✅ 単体テストでタスクを検証する
- ✅ タスクロジックのクイック検証
- ✅ タスク挙動のデバッグ
- ❌ 本番ワークフローでは使用しない ( `workflow.execute()` を使う )

**例: パラメータ付きでテスト**

```python
@task
def process_data(data: list[int], multiplier: int = 2) -> list[int]:
    """Process data with a multiplier."""
    return [x * multiplier for x in data]

# Test with different parameters
result1 = process_data.run(data=[1, 2, 3])
print(result1)  # Output: [2, 4, 6]

result2 = process_data.run(data=[1, 2, 3], multiplier=3)
print(result2)  # Output: [3, 6, 9]
```

**💡 重要ポイント:** `.run()` を使ってワークフロー投入前にタスク単体で検証しましょう。

---

## レベル2: 最初のワークフロー

次に、複数のタスクをワークフローでつなげます。

### 完全なワークフロー例

```python
from graflow.core.workflow import workflow
from graflow.core.decorators import task

with workflow("simple_pipeline") as wf:
    @task
    def start():
        print("Starting!")

    @task
    def middle():
        print("Middle!")

    @task
    def end():
        print("Ending!")

    # Connect tasks: start → middle → end
    start >> middle >> end

    # Execute the workflow
    wf.execute()
```

**出力:**
```
Starting!
Middle!
Ending!
```

**何が起きているか:**
- `with workflow("name")` がワークフローコンテキストを作成
- 内部で定義したタスクは自動登録されます
- `>>` がタスクを直列に接続 (start → middle → end)
- `wf.execute()` がワークフローを実行

**💡 重要ポイント:**
- `with workflow("name")` でワークフローを作成
- ワークフローコンテキスト内でタスクを定義
- `>>` で直列接続
- `wf.execute()` で実行

---

## レベル3: タスク合成

`>>` (直列) と `|` (並列) 演算子を使ったタスクの組み合わせを学びます。

### 直列と並列の組み合わせ

```python
with workflow("composition") as wf:
    @task
    def start():
        print("Start")

    @task
    def parallel_a():
        print("Parallel A")

    @task
    def parallel_b():
        print("Parallel B")

    @task
    def end():
        print("End")

    # Pattern: start → (parallel_a | parallel_b) → end
    start >> (parallel_a | parallel_b) >> end

    wf.execute()
```

**実行フロー:**
1. `start` が最初に実行される
2. `parallel_a` と `parallel_b` が同時に実行される
3. 並列タスク完了後に `end` が実行される

**出力:**
```
Start
Parallel A
Parallel B
End
```

**演算子:**
- `>>` は直列依存 (順番に実行)
- `|` は並列実行 (同時に実行)
- かっこでグループ化: `(task_a | task_b)`

**💡 重要ポイント:**
- `task_a >> task_b` は「aの後にbを実行」
- `task_a | task_b` は「aとbを同時に実行」
- 混在パターン: `a >> (b | c) >> d`

### ヘルパー関数: chain() と parallel()

複数タスクの直列/並列を作る場合はヘルパー関数が便利です:

```python
from graflow.core.task import chain, parallel

with workflow("helpers") as wf:
    @task
    def task_a():
        print("A")

    @task
    def task_b():
        print("B")

    @task
    def task_c():
        print("C")

    @task
    def task_d():
        print("D")

    # Using chain(*tasks) - equivalent to task_a >> task_b >> task_c
    seq = chain(task_a, task_b, task_c)

    # Using parallel(*tasks) - equivalent to task_a | task_b | task_c
    par = parallel(task_a, task_b, task_c)

    # Combine them
    _pipeline = seq >> par

    wf.execute()
```

**関数シグネチャ:**
- `chain(*tasks)` - 1個以上のタスクを引数として受け取る
- `parallel(*tasks)` - 2個以上のタスクを引数として受け取る

**使い分け:**
- `chain(*tasks)`: 3つ以上の直列接続で読みやすい
- `parallel(*tasks)`: 3つ以上の並列グループで読みやすい
- 演算子 (`>>`, `|`): 2タスクや混在パターンに適する

**例: 動的なタスクリスト**

```python
# If you have tasks in a list, unpack them with *
task_list = [task_a, task_b, task_c, task_d]

# Unpack the list into parallel()
parallel_group = parallel(*task_list)

# Or use operators in a loop
group = task_list[0]
for task in task_list[1:]:
    group = group | task
```

**例: 事前バインドした引数 (タスクインスタンス) の使用**

```python
@task
def fetch_weather(city: str) -> dict:
    return {"city": city, "temp": 20}

# Create task instances with bound parameters
tokyo = fetch_weather(task_id="tokyo", city="Tokyo")
paris = fetch_weather(task_id="paris", city="Paris")
london = fetch_weather(task_id="london", city="London")

with workflow("weather") as wf:
    # Use parallel() with task instances
    all_cities = parallel(tokyo, paris, london)

    wf.execute()
```

**例: chain() と parallel() を使った動的インスタンス**

```python
@task
def process_batch(batch_id: int, data: list) -> dict:
    return {"batch_id": batch_id, "count": len(data)}

# Generate task instances dynamically
cities = ["Tokyo", "Paris", "London", "NYC"]
fetch_tasks = [
    fetch_weather(task_id=f"fetch_{city.lower()}", city=city)
    for city in cities
]

batches = [1, 2, 3]
process_tasks = [
    process_batch(task_id=f"batch_{i}", batch_id=i, data=[])
    for i in batches
]

with workflow("dynamic") as wf:
    # Use parallel() with task instances
    all_fetches = parallel(*fetch_tasks)

    # Use chain() with task instances
    all_batches = chain(*process_tasks)

    # Combine
    all_fetches >> all_batches

    wf.execute()
```

### 並列グループの設定

並列グループは名前や実行ポリシーを設定できます:

```python
with workflow("configured") as wf:
    @task
    def task_a():
        print("A")

    @task
    def task_b():
        print("B")

    @task
    def task_c():
        print("C")

    # Create parallel group with custom name
    group = (task_a | task_b | task_c).set_group_name("my_parallel_tasks")

    # Configure execution policy
    group.with_execution(policy="best_effort")  # Continue even if some tasks fail

    wf.execute()
```

**利用可能な実行ポリシー:**

| ポリシー | 挙動 |
|--------|----------|
| `"strict"` (default) | すべてのタスク成功が必須。失敗すると全体が失敗 |
| `"best_effort"` | 失敗しても継続し、結果を収集 |
| `AtLeastNGroupPolicy(min_success=N)` | N個以上の成功が必要 |
| `CriticalGroupPolicy(critical_task_ids=[...])` | 指定タスクの成功が必須 |

**例: ベストエフォートの並列実行**

```python
# Continue workflow even if some parallel tasks fail
(fetch_api | fetch_db | fetch_cache).with_execution(policy="best_effort")
```

**例: カスタムグループ名**

```python
# Rename group for clarity in logs and visualization
parallel_fetches = (fetch_a | fetch_b | fetch_c).set_group_name("data_fetches")
```

**例: 高度な実行設定**

```python
from graflow.coordination.coordinator import CoordinationBackend

# Use threading backend with custom thread count
(task_a | task_b | task_c | task_d).with_execution(
    backend=CoordinationBackend.THREADING,
    backend_config={"thread_count": 2},
    policy="best_effort"
)

# AtLeastN policy: Require at least 3 out of 4 tasks to succeed
from graflow.core.handlers.group_policy import AtLeastNGroupPolicy

(task_a | task_b | task_c | task_d).with_execution(
    policy=AtLeastNGroupPolicy(min_success=3)
)

# Critical policy: Specific tasks must succeed
from graflow.core.handlers.group_policy import CriticalGroupPolicy

(task_a | task_b | task_c).with_execution(
    policy=CriticalGroupPolicy(critical_task_ids=["task_a", "task_b"])
)
```

**💡 重要ポイント:**
- `chain()` と `parallel()` を使うと多タスク作成が簡潔
- `.set_group_name()` で並列グループに意味のある名前を付ける
- `.with_execution(policy=...)` で失敗時の扱いを制御
- `backend` と `backend_config` で実行バックエンドを設定

---

## レベル4: パラメータの受け渡し

チャンネルとパラメータバインドを使ってタスク間でデータを渡す方法を学びます。

### チャンネルでタスク間通信

タスクは共有チャンネルを読み書きして通信します (詳細は[レベル6](#レベル6-チャンネルとコンテキスト)):

```python
from graflow.core.context import TaskExecutionContext

with workflow("channel_communication") as wf:
    @task(inject_context=True)
    def producer(ctx: TaskExecutionContext):
        channel = ctx.get_channel()
        channel.set("user_id", "user_123")

    @task(inject_context=True)
    def consumer(ctx: TaskExecutionContext):
        channel = ctx.get_channel()
        user_id = channel.get("user_id")
        print(f"User: {user_id}")

    producer >> consumer
    wf.execute()
```

### 部分的なパラメータバインド

一部のパラメータをタスク生成時にバインドし、残りはチャンネルから取得できます:

```python
with workflow("partial_binding") as wf:
    @task
    def calculate(base: int, multiplier: int, offset: int) -> int:
        result = base * multiplier + offset
        print(f"calculate: {base} * {multiplier} + {offset} = {result}")
        return result

    # Bind only 'base', others come from channel
    task_instance = calculate(task_id="calc", base=10)

    # Execute with channel values for multiplier and offset
    _, ctx = wf.execute(
        ret_context=True,
        initial_channel={"multiplier": 3, "offset": 5}
    )

    result = ctx.get_result("calc")
    print(f"Result: {result}")
```

**出力:**
```
calculate: 10 * 3 + 5 = 35
Result: 35
```

**何が起きたか:**
- `base=10` はタスク生成時にバインド (最優先)
- `multiplier=3` と `offset=5` はチャンネルから取得
- バインド済みパラメータはチャンネル値を上書き

**💡 重要ポイント:**
- タスクはチャンネル経由で通信可能 (詳細は[レベル6](#レベル6-チャンネルとコンテキスト))
- 一部をバインドし、残りをチャンネルから取得できる
- バインド済みパラメータが優先される

---

## レベル5: タスクインスタンス

**Graflowの新機能**: 1つのタスク定義から複数インスタンスを作成できます。

### 課題

同じロジックを異なるパラメータで再利用したいケース:

```python
# ❌ Without task instances (repetitive)
@task
def fetch_tokyo():
    return fetch("Tokyo")

@task
def fetch_paris():
    return fetch("Paris")
```

### 解決策

パラメータをバインドしたタスクインスタンスを作成:

```python
# ✅ With task instances (reusable)
@task
def fetch_weather(city: str) -> str:
    return f"Weather for {city}"

# Create instances with different parameters
tokyo = fetch_weather(task_id="tokyo", city="Tokyo")
paris = fetch_weather(task_id="paris", city="Paris")
london = fetch_weather(task_id="london", city="London")

with workflow("weather") as wf:
    # Use instances in workflow
    tokyo >> paris >> london
    wf.execute()
```

**出力:**
```
Weather for Tokyo
Weather for Paris
Weather for London
```

### 自動生成されるタスクID

すべてに `task_id` を付けたくない場合は省略できます:

```python
@task
def process(value: int) -> int:
    return value * 2

# Auto-generated IDs: process_{random_uuid}
task1 = process(value=10)  # task_id: process_a3f2b9c1
task2 = process(value=20)  # task_id: process_b7e8f4d2
task3 = process(value=30)  # task_id: process_c5d9e6f7

with workflow("auto_ids") as wf:
    task1 >> task2 >> task3
    wf.execute()
```

**⚠️ 注意: タスクIDの一意性を確保**

複数インスタンスを作成する場合、タスクIDは必ず一意にします:

```python
# ✅ Good: Unique task_ids
tokyo = fetch_weather(task_id="tokyo", city="Tokyo")
paris = fetch_weather(task_id="paris", city="Paris")
london = fetch_weather(task_id="london", city="London")

# ❌ Bad: Duplicate task_ids cause conflicts
task1 = fetch_weather(task_id="fetch", city="Tokyo")
task2 = fetch_weather(task_id="fetch", city="Paris")  # ERROR: "fetch" already exists!

# ✅ Good: Auto-generated IDs are always unique
task1 = fetch_weather(city="Tokyo")   # Auto: fetch_weather_a3f2b9c1
task2 = fetch_weather(city="Paris")   # Auto: fetch_weather_b7e8f4d2
```

**💡 重要ポイント:**
- タスクインスタンスは同じロジックを別パラメータで再利用
- `task_id` を付ける場合は一意性が必須
- `task_id` を省略すれば自動で一意なIDが生成される
- 各インスタンスは独立している

---

## レベル6: チャンネルとコンテキスト

### チャンネルバックエンド

Graflowはローカル/分散の切り替えをシームレスに行える2種類のバックエンドを提供します:

**1. MemoryChannel (デフォルト)** - ローカル実行用:
- ✅ 高速: インメモリで低遅延
- ✅ シンプル: インフラ不要
- ✅ チェックポイント互換: 自動保存
- ⚠️ 制約: 単一プロセスのみ

**2. RedisChannel** - 分散実行用:
- ✅ 分散: 複数ワーカー/マシンで状態共有
- ✅ 永続: Redis永続化で耐障害性
- ✅ スケーラブル: 多数ワーカーでも一貫性
- ⚠️ 必要: Redisサーバー

**バックエンドの切り替え:**

```python
# Local execution (default) - uses MemoryChannel
with workflow("local") as wf:
    task_a >> task_b
    wf.execute()

# Distributed execution - uses RedisChannel
from graflow.channels.factory import ChannelFactory, ChannelBackend

channel = ChannelFactory.create_channel(
    backend=ChannelBackend.REDIS,
    redis_client=redis_client
)

with workflow("distributed") as wf:
    task_a >> task_b
    wf.execute()
```

### チャンネルの使い方

#### 基本チャンネル: `ctx.get_channel()`

単純なキー・バリューの保存に使います:

```python
@task(inject_context=True)
def producer(ctx: TaskExecutionContext):
    """Write data to channel."""
    channel = ctx.get_channel()

    # Store simple values
    channel.set("user_id", "user_123")
    channel.set("score", 95.5)
    channel.set("active", True)

    # Store complex objects
    channel.set("user_profile", {
        "name": "Alice",
        "email": "alice@example.com",
        "age": 30
    })

@task(inject_context=True)
def consumer(ctx: TaskExecutionContext):
    """Read data from channel."""
    channel = ctx.get_channel()

    # Retrieve values
    user_id = channel.get("user_id")        # "user_123"
    score = channel.get("score")            # 95.5
    active = channel.get("active")          # True
    profile = channel.get("user_profile")   # dict

    # With default value
    setting = channel.get("setting", default="default_value")
```

**チャンネルメソッド:**

| メソッド | 説明 | 例 |
|--------|-------------|---------|
| `set(key, value)` | 値を保存 | `channel.set("count", 42)` |
| `set(key, value, ttl)` | 有効期限付きで保存 (秒) | `channel.set("temp", 100, ttl=300)` |
| `get(key)` | 値を取得 | `value = channel.get("count")` |
| `get(key, default)` | デフォルト値付きで取得 | `value = channel.get("count", default=0)` |
| `append(key, value)` | リスト末尾に追加 | `channel.append("logs", "entry")` |
| `append(key, value, ttl)` | 有効期限付きで末尾追加 | `channel.append("logs", "entry", ttl=60)` |
| `prepend(key, value)` | リスト先頭に追加 | `channel.prepend("queue", "item")` |
| `delete(key)` | キーを削除 | `channel.delete("count")` |
| `exists(key)` | 存在チェック | `if channel.exists("count"):` |

**リスト操作: append() と prepend()**

複数値の収集に便利です:

```python
@task(inject_context=True)
def collect_logs(ctx: TaskExecutionContext):
    channel = ctx.get_channel()

    # Append to end of list (FIFO queue)
    channel.append("logs", "Log entry 1")
    channel.append("logs", "Log entry 2")
    channel.append("logs", "Log entry 3")

    logs = channel.get("logs")
    print(logs)  # ["Log entry 1", "Log entry 2", "Log entry 3"]

@task(inject_context=True)
def use_stack(ctx: TaskExecutionContext):
    channel = ctx.get_channel()

    # Prepend to beginning of list (LIFO stack)
    channel.prepend("stack", "First")
    channel.prepend("stack", "Second")
    channel.prepend("stack", "Third")

    stack = channel.get("stack")
    print(stack)  # ["Third", "Second", "First"]
```

**ユースケース:**
- `append()`: ログの蓄積、並列タスクの結果収集、FIFOキュー
- `prepend()`: LIFOスタック、優先度高いアイテム、逆順収集

**Time-to-Live (TTL): 自動有効期限**

TTLを使って一時データを自動的に削除できます:

```python
@task(inject_context=True)
def cache_data(ctx: TaskExecutionContext):
    channel = ctx.get_channel()

    # Cache for 5 minutes (300 seconds)
    channel.set("api_response", {"data": "..."}, ttl=300)

    # Temporary flag expires in 60 seconds
    channel.set("processing", True, ttl=60)

    # Collect logs that expire after 10 minutes
    channel.append("recent_logs", "Error occurred", ttl=600)

@task(inject_context=True)
def check_cache(ctx: TaskExecutionContext):
    channel = ctx.get_channel()

    # After TTL expires, key is automatically removed
    data = channel.get("api_response", default="expired")
    if data == "expired":
        print("Cache expired, refetching...")
```

**TTLの挙動:**
- TTLは**秒**単位
- TTL経過後にキーは自動削除
- 期限切れキーに `get()` すると `None` (またはデフォルト値)
- `set()` と `append()`/`prepend()` はTTL対応
- 一時キャッシュ、レート制限、セッションデータに有効

**例: TTL付きで並列タスク結果を収集**

```python
@task(inject_context=True)
def fetch_data(ctx: TaskExecutionContext, source: str):
    channel = ctx.get_channel()
    data = f"Data from {source}"

    # Collect results with 1-hour expiration
    channel.append("fetch_results", data, ttl=3600)

    return data

with workflow("collect_results") as wf:
    fetch_a = fetch_data(task_id="fetch_a", source="api")
    fetch_b = fetch_data(task_id="fetch_b", source="db")
    fetch_c = fetch_data(task_id="fetch_c", source="cache")

    parallel(fetch_a, fetch_b, fetch_c)

    wf.execute()
```

#### 型安全なチャンネル: `ctx.get_typed_channel()`

型付きチャンネルで型チェックとIDE補完を活用できます:

```python
from typing import TypedDict

# Define schema
class UserProfile(TypedDict):
    user_id: str
    name: str
    email: str
    age: int
    premium: bool

@task(inject_context=True)
def collect_user_data(ctx: TaskExecutionContext):
    """Store user data with type safety."""

    # Get typed channel
    typed_channel = ctx.get_typed_channel(UserProfile)

    # IDE autocompletes fields!
    user_profile: UserProfile = {
        "user_id": "user_123",
        "name": "Alice",
        "email": "alice@example.com",
        "age": 30,
        "premium": True
    }

    # Type-checked storage
    typed_channel.set("current_user", user_profile)

@task(inject_context=True)
def process_user_data(ctx: TaskExecutionContext):
    """Retrieve user data with type safety."""

    # Get typed channel with same schema
    typed_channel = ctx.get_typed_channel(UserProfile)

    # Retrieve with type hints
    user: UserProfile = typed_channel.get("current_user")

    # IDE knows the structure!
    print(user["name"])    # IDE autocompletes "name"
    print(user["email"])   # IDE autocompletes "email"
```

**型付きチャンネルの利点:**

- ✅ **IDE補完**: フィールド名と型が候補表示
- ✅ **型チェック**: mypy/pyrightが型ミスを検出
- ✅ **自己文書化**: TypedDictがAPI契約になる
- ✅ **リファクタ安全**: IDEで安全にフィールド名変更
- ✅ **チーム開発**: 共有スキーマでミス防止

**使い分けの目安:**

| 用途 | メソッド | 理由 |
|----------|--------|-----|
| シンプルな値 (文字列/数値) | `get_channel()` | オーバーヘッドが少ない |
| その場しのぎのデータ交換 | `get_channel()` | スキーマ不要 |
| 構造化データ | `get_typed_channel()` | 型安全 |
| チーム開発 | `get_typed_channel()` | 共有スキーマ |
| 大規模プロジェクト | `get_typed_channel()` | 保守性向上 |

**例: 併用パターン**

```python
@task(inject_context=True)
def process_order(ctx: TaskExecutionContext):
    # Use typed channel for structured data
    order_channel = ctx.get_typed_channel(OrderData)
    order = order_channel.get("current_order")

    # Use basic channel for simple flags
    basic_channel = ctx.get_channel()
    basic_channel.set("processing_started", True)
    basic_channel.set("timestamp", "2024-01-01T12:00:00")
```

### 依存性注入

Graflowは自動でリソースを提供する3種類の注入を提供します。

#### 1. コンテキスト注入: `inject_context=True`

実行コンテキストを注入し、チャンネル、結果、ワークフローのメタデータにアクセスします:

```python
@task(inject_context=True)
def my_task(ctx: TaskExecutionContext, value: int):
    # Access channel
    channel = ctx.get_channel()
    channel.set("result", value * 2)

    # Access session info
    print(f"Session: {ctx.session_id}")

    # Access other task results
    previous = ctx.get_result("previous_task")

    return value * 2
```

**使いどころ:**
- チャンネル経由のタスク間通信
- 他タスクの結果取得
- ワークフロー制御 (next_task, next_iteration, terminate_workflow)

#### 2. LLMクライアント注入: `inject_llm_client=True`

軽量なLLMクライアントを注入し、直接APIを呼び出します:

```python
from graflow.llm.client import LLMClient

@task(inject_llm_client=True)
def analyze_text(llm: LLMClient, text: str) -> str:
    # Direct LLM API call
    response = llm.completion_text(
        messages=[{"role": "user", "content": f"Analyze: {text}"}],
        model="gpt-4o-mini"
    )
    return response
```

**使いどころ:**
- エージェント不要のシンプルなLLM呼び出し
- 複数モデル利用 (タスクごとにモデルを変える)
- コスト最適化 (簡単な処理は安価モデル)

**対応:** OpenAI ChatGPT、Anthropic Claude、Google Gemini、AWS Bedrock など (LiteLLM経由)

#### 3. LLMエージェント注入: `inject_llm_agent="agent_name"`

ReActループとツールを備えたフル機能のLLMエージェント (SuperAgent) を注入します:

```python
from graflow.llm.agents.base import LLMAgent

# First, register the agent in workflow
context.register_llm_agent("supervisor", my_agent)

# Then inject into task
@task(inject_llm_agent="supervisor")
def supervise_task(agent: LLMAgent, query: str) -> str:
    # Agent handles ReAct loop, tool calls internally
    result = agent.run(query)
    return result["output"]
```

**使いどころ:**
- ツール呼び出しを伴う複雑な推論
- マルチターン対話
- 自律的なエージェント動作が必要なタスク

**互換:** Google ADK、PydanticAI、カスタムエージェント

#### 注入のまとめ

| 注入タイプ | パラメータ | 用途 |
|----------------|-----------|----------|
| `inject_context=True` | `ctx: TaskExecutionContext` | チャンネル、ワークフロー制御、結果取得 |
| `inject_llm_client=True` | `llm: LLMClient` | シンプルなLLM API呼び出し |
| `inject_llm_agent="name"` | `agent: LLMAgent` | ツール付きの複雑なエージェント処理 |

**💡 重要ポイント:**
- 注入はタスク実行時に自動で行われる
- 先頭引数が注入された依存を受け取る
- `inject_context=True, inject_llm_client=True` の併用も可能
- エージェントは事前登録が必要: `context.register_llm_agent(name, agent)`

#### 代替案: コンテキスト経由でLLMクライアント/エージェントにアクセス

`inject_context=True` を使っている場合、コンテキスト経由でLLMにアクセスできます:

```python
@task(inject_context=True)
def task_with_llm(ctx: TaskExecutionContext, query: str):
    # Access LLM client via context
    response = ctx.llm_client.completion_text(
        messages=[{"role": "user", "content": query}],
        model="gpt-4o-mini"
    )

    # Access LLM agent via context
    agent = ctx.get_llm_agent("supervisor")
    result = agent.run(query)

    return {"llm": response, "agent": result}
```

**使い分け:**
- 直接注入 (`inject_llm_client=True`): LLMのみ使う場合にシンプル
- コンテキスト経由 (`ctx.llm_client`): チャンネル/制御も必要な場合

### プロンプト管理

GraflowはLLMプロンプトをバージョン管理・ラベル管理するためのプロンプト管理モジュールを提供します。

#### プロンプトマネージャーの設定

`PromptManagerFactory` を使ってプロンプトマネージャーを作成し、ワークフローに渡します:

```python
from pathlib import Path
from graflow.core.workflow import workflow
from graflow.prompts.factory import PromptManagerFactory

# YAMLベースのプロンプトマネージャー (ローカルファイル)
prompts_dir = Path(__file__).parent / "prompts"
pm = PromptManagerFactory.create("yaml", prompts_dir=str(prompts_dir))

# または Langfuseベースのプロンプトマネージャー (クラウド)
pm = PromptManagerFactory.create(
    "langfuse",
    fetch_timeout_seconds=10,  # 10秒タイムアウト
    max_retries=2,             # 失敗時は最大2回リトライ
)

# ワークフローコンテキストに渡す
with workflow("my_workflow", prompt_manager=pm) as ctx:
    # タスクは context.prompt_manager でアクセス可能
    ...
```

**利用可能なバックエンド:**

| バックエンド | 用途 | 設定 |
|---------|----------|---------------|
| `"yaml"` | ローカル開発、バージョン管理されたプロンプト | `prompts_dir="./prompts"` |
| `"langfuse"` | クラウドベース、チーム協業、A/Bテスト | `fetch_timeout_seconds`, `max_retries` |

**Langfuseセットアップ** (`pip install graflow[tracing]` が必要):
```bash
export LANGFUSE_PUBLIC_KEY=pk-lf-...
export LANGFUSE_SECRET_KEY=sk-lf-...
export LANGFUSE_HOST=https://cloud.langfuse.com  # またはセルフホストURL
```

#### タスク内でプロンプトにアクセス

`context.prompt_manager` を使ってタスク内でプロンプトにアクセスします:

```python
@task(inject_context=True)
def greet(ctx: TaskExecutionContext) -> str:
    pm = ctx.prompt_manager

    # テキストプロンプトを取得し、変数をレンダリング
    prompt = pm.get_text_prompt("greeting")
    return prompt.render(name="Alice", product="Graflow")
    # 出力: "Hello Alice, welcome to Graflow!"
```

#### テキストプロンプト vs チャットプロンプト

**テキストプロンプト** - 単一文字列テンプレート:

```python
@task(inject_context=True)
def generate_greeting(ctx: TaskExecutionContext) -> str:
    pm = ctx.prompt_manager

    # テキストプロンプトを取得
    prompt = pm.get_text_prompt("greeting")

    # render()は文字列を返す
    message: str = prompt.render(name="Alice")
    return message
```

**チャットプロンプト** - LLM API用のメッセージリストテンプレート:

```python
@task(inject_context=True)
def generate_conversation(ctx: TaskExecutionContext) -> list:
    pm = ctx.prompt_manager

    # チャットプロンプトを取得
    prompt = pm.get_chat_prompt("assistant")

    # render()はメッセージ辞書のリストを返す
    messages: list[dict] = prompt.render(domain="Python", task="debugging")
    # [
    #   {"role": "system", "content": "You are an expert in Python."},
    #   {"role": "user", "content": "Help me with debugging."}
    # ]
    return messages
```

#### ラベルとバージョンによるアクセス

特定バージョンのプロンプトにアクセス:

```python
# ラベル指定 (本番/ステージング環境に推奨)
prompt = pm.get_text_prompt("greeting", label="production")
prompt = pm.get_text_prompt("greeting", label="staging")

# バージョン番号指定
prompt = pm.get_text_prompt("greeting", version=1)
prompt = pm.get_text_prompt("greeting", version=2)
```

#### YAMLプロンプト形式

YAMLファイルにプロンプトを保存:

```yaml
# prompts/greeting.yaml
greeting:
  type: text
  labels:
    production:
      content: "Hello {{name}}, welcome to {{product}}!"
      version: 1
      metadata:
        author: "team@example.com"
    staging:
      content: "Hi {{name}}! Testing {{product}}."
      version: 2

# チャットプロンプトの例
assistant:
  type: chat
  labels:
    production:
      content:
        - role: system
          content: "You are a helpful assistant specializing in {{domain}}."
        - role: user
          content: "Help me with {{task}}."
```

**主な機能:**
- `{{variable}}` プレースホルダー (Jinja2構文)
- ラベルベースのアクセス (`production`, `staging` など)
- ファイル変更時の自動リロード
- サブディレクトリ対応 (例: `customer/welcome`)

#### 完全な例

```python
from pathlib import Path
from graflow.core.workflow import workflow
from graflow.core.decorators import task
from graflow.core.context import TaskExecutionContext
from graflow.prompts.factory import PromptManagerFactory

# プロンプトマネージャーを作成
prompts_dir = Path(__file__).parent / "prompts"
pm = PromptManagerFactory.create("yaml", prompts_dir=str(prompts_dir))

with workflow("customer_onboarding", prompt_manager=pm) as ctx:

    @task(inject_context=True)
    def setup(context: TaskExecutionContext):
        channel = context.get_channel()
        channel.set("customer_name", "Alice")
        channel.set("product_name", "Graflow")

    @task(inject_context=True)
    def greet_customer(context: TaskExecutionContext) -> str:
        pm = context.prompt_manager
        channel = context.get_channel()

        name = channel.get("customer_name")
        product = channel.get("product_name")

        # 本番用プロンプトを取得してレンダリング
        prompt = pm.get_text_prompt("greeting", label="production")
        return prompt.render(name=name, product=product)

    @task(inject_context=True)
    def generate_assistant(context: TaskExecutionContext) -> list:
        pm = context.prompt_manager

        # LLM API用のチャットプロンプトを取得
        prompt = pm.get_chat_prompt("assistant", label="production")
        messages = prompt.render(domain="Python", task="onboarding")

        # LLM APIへ送信可能
        return messages

    setup >> greet_customer >> generate_assistant
    ctx.execute("setup")
```

**💡 重要ポイント:**
- `PromptManagerFactory.create()` でプロンプトマネージャーを作成
- ワークフローに渡す: `workflow("name", prompt_manager=pm)`
- タスク内では `context.prompt_manager` でアクセス
- 文字列には `get_text_prompt()`、メッセージリストには `get_chat_prompt()`
- 環境別プロンプトにはラベル (`production`, `staging`) を使用
- 完全な例は `examples/14_prompt_management/` を参照

### Human-in-the-Loop: `ctx.request_feedback()`

`ctx.request_feedback()` で人間のフィードバックをワークフローに組み込みます:

```python
@task(inject_context=True)
def request_approval(ctx: TaskExecutionContext, deployment_plan: dict) -> bool:
    """Request human approval before deployment."""

    response = ctx.request_feedback(
        feedback_type="approval",
        prompt="Approve deployment to production?",
        timeout=300,  # Wait 5 minutes
        notification_config={
            "type": "slack",
            "webhook_url": "https://hooks.slack.com/services/XXX",
            "message": "Deployment approval needed!"
        }
    )

    if not response.approved:
        ctx.cancel_workflow("Deployment rejected by user")

    return response.approved
```

**フィードバック種別:**

1. **承認** - Yes/No 判断
   ```python
   response = ctx.request_feedback(
       feedback_type="approval",
       prompt="Approve this action?"
   )
   # response.approved: bool
   ```

2. **テキスト入力** - 自由入力
   ```python
   response = ctx.request_feedback(
       feedback_type="text",
       prompt="Enter configuration value:"
   )
   # response.text: str
   ```

3. **選択** - 1つ選ぶ
   ```python
   response = ctx.request_feedback(
       feedback_type="selection",
       prompt="Choose deployment environment:",
       options=["staging", "production"]
   )
   # response.selected: str
   ```

4. **複数選択** - 複数選ぶ
   ```python
   response = ctx.request_feedback(
       feedback_type="multi_selection",
       prompt="Select features to enable:",
       options=["feature_a", "feature_b", "feature_c"]
   )
   # response.selected: list[str]
   ```

**タイムアウトとチェックポイントの挙動:**

タイムアウトが発生すると、Graflowはチェックポイントを作成してワークフローを一時停止します:

```python
response = ctx.request_feedback(
    feedback_type="approval",
    prompt="Approve deployment?",
    timeout=300  # 5 minutes
)

# If no response within 5 minutes:
# 1. Checkpoint is automatically created
# 2. Workflow pauses
# 3. User can provide feedback later via API
# 4. Workflow resumes from checkpoint when feedback is received
```

**ユースケース:**
- デプロイ承認
- データ検証レビュー
- ドメイン専門家によるパラメータ調整
- エラー回復時の意思決定

### Request Feedbackの冪等性

**HITLワークフローでは重要**: `ctx.request_feedback()` を使うタスクは冪等でなければなりません。

**なぜ冪等性が重要か:**

タスクがフィードバック待ちでタイムアウトすると:
1. チェックポイントが自動作成される
2. ワークフローが一時停止する
3. 後でユーザーがフィードバックを提供する
4. **チェックポイントから再開し、タスクが再実行される**

つまり同じタスクが複数回実行される可能性があるため、再実行しても安全である必要があります:

```python
# ⚠️ NOT Idempotent - Dangerous with request_feedback
@task(inject_context=True)
def deploy_with_approval(ctx: TaskExecutionContext):
    # Deploy FIRST (wrong order!)
    deployment_id = api.deploy_to_production()

    # Then ask for approval
    response = ctx.request_feedback(
        feedback_type="approval",
        prompt="Approve deployment?"
    )

    # If timeout occurs and task resumes, deploy happens AGAIN!
    # This creates duplicate deployments!
```

```python
# ✅ Idempotent - Safe with request_feedback
@task(inject_context=True)
def deploy_with_approval(ctx: TaskExecutionContext, deployment_plan: dict):
    channel = ctx.get_channel()

    # Check if already deployed
    if not channel.get("deployment_approved"):
        # Ask for approval FIRST
        response = ctx.request_feedback(
            feedback_type="approval",
            prompt="Approve deployment?",
            timeout=300
        )

        if not response.approved:
            ctx.cancel_workflow("Deployment rejected")

        # Mark as approved
        channel.set("deployment_approved", True)

    # Check if already deployed
    if not channel.get("deployment_completed"):
        # Deploy only once
        deployment_id = api.deploy_to_production(deployment_plan)
        channel.set("deployment_completed", True)
        channel.set("deployment_id", deployment_id)

    return channel.get("deployment_id")
```

**ベストプラクティス:**

1. **副作用の前にフィードバックを要求する**
2. **チャンネルのフラグで完了状態を管理する**
3. **フラグ確認後に処理して重複を防ぐ**
4. **外部APIには冪等性キーを使う**

**💡 重要ポイント:** `ctx.request_feedback()` を使うタスクは、チェックポイント再開に備えて常に冪等にしましょう。

### パラメータの優先順位

パラメータ解決の優先順は次の通りです: **注入 > バインド > チャンネル**

```python
@task
def calculate(value: int, multiplier: int) -> int:
    return value * multiplier

# Bind value=10, multiplier from channel
task = calculate(task_id="calc", value=10)

wf.execute(initial_channel={"value": 100, "multiplier": 5})
# Result: 10 × 5 = 50 (bound value beats channel value)
```

---

## レベル7: 実行パターン

### タスク結果の理解

タスクが値を返すと、GraflowはタスクIDを使ってチャンネルに保存します:

```python
# Auto-generated task_id (function name)
@task
def calculate():
    return 42

# Stored as: channel.set("calculate.__result__", 42)
# Access: ctx.get_result("calculate") → 42

# Custom task_id
task1 = calculate(task_id="calc1")
task2 = calculate(task_id="calc2")

# Stored as: channel.set("calc1.__result__", 42)
#            channel.set("calc2.__result__", 42)
# Access: ctx.get_result("calc1"), ctx.get_result("calc2")
```

**結果の保存形式:** `{task_id}.__result__`

```python
# When a task completes:
channel.set(f"{task_id}.__result__", return_value)

# When you call get_result():
def get_result(task_id: str, default=None):
    return channel.get(f"{task_id}.__result__", default)
```

### パターン1: 最終結果を取得

```python
with workflow("simple") as wf:
    @task
    def compute():
        return 42

    result = wf.execute()
    print(result)  # 42 (last task's return value)
```

### パターン2: 全結果を取得

実行コンテキストから全タスクの結果を取得します:

```python
with workflow("all_results") as wf:
    @task
    def task_a():
        return "A"

    @task
    def task_b():
        return "B"

    task_a >> task_b

    # Get execution context to access all results
    _, ctx = wf.execute(ret_context=True)

    # Access individual task results
    print(ctx.get_result("task_a"))  # Output: A
    print(ctx.get_result("task_b"))  # Output: B
```

**重要ポイント:**
- `ret_context=True` は `(final_result, execution_context)` を返す
- `ctx.get_result(task_id)` で任意の結果を取得できる
- タスクが値を返すと自動で保存される

### パターン3: 特定のタスクから開始

**自動検出 (引数なし):**

`wf.execute()` に引数を渡さないと、開始ノードが自動検出されます:

```python
with workflow("auto_start") as wf:
    @task
    def step1():
        print("Step 1")

    @task
    def step2():
        print("Step 2")

    step1 >> step2

    # Auto-detects step1 (node with no predecessors)
    wf.execute()
```

**自動検出の仕組み:**
1. **入次数が0のノード** (先行タスクなし) を探す
2. **1つのみ**ならそれが開始ノード
3. **0個**なら `GraphCompilationError` (空/循環)
4. **複数**なら `GraphCompilationError` (開始点が曖昧)

**例: 複数エントリポイント (エラー)**

```python
with workflow("ambiguous") as wf:
    @task
    def task_a():
        print("A")

    @task
    def task_b():
        print("B")

    @task
    def task_c():
        print("C")

    # Two separate chains - two entry points!
    task_a >> task_c
    task_b >> task_c

    # ERROR: Multiple start nodes found (task_a and task_b)
    # wf.execute()  # Raises GraphCompilationError

    # Solution: Specify start node explicitly
    wf.execute(start_node="task_a")
```

**開始ノードを手動指定:**

前段のタスクをスキップしたい場合は開始ノードを明示します:

```python
with workflow("skip") as wf:
    @task
    def step1():
        print("Step 1")

    @task
    def step2():
        print("Step 2")

    @task
    def step3():
        print("Step 3")

    step1 >> step2 >> step3

    # Start from step2 (skip step1)
    wf.execute(start_node="step2")
```

**出力:**
```
Step 2
Step 3
```

**💡 重要ポイント:**
- `wf.execute()` は開始ノードを自動検出
- 開始ノードが0/複数の場合はエラー
- `wf.execute(start_node="task_id")` で開始点を指定
- `wf.execute(ret_context=True)` は `(result, context)` を返す
- `ctx.get_result(task_id)` で結果取得

---

## レベル8: 複雑なワークフロー

### ダイヤモンドパターン

1つのタスクが分岐し、並列実行後に合流します:

```python
@task(inject_context=True)
def source(ctx: TaskExecutionContext, value: int) -> int:
    ctx.get_channel().set("value", value)
    return value

@task(inject_context=True)
def double(ctx: TaskExecutionContext) -> int:
    value = ctx.get_channel().get("value")
    result = value * 2
    ctx.get_channel().set("doubled", result)
    return result

@task(inject_context=True)
def triple(ctx: TaskExecutionContext) -> int:
    value = ctx.get_channel().get("value")
    result = value * 3
    ctx.get_channel().set("tripled", result)
    return result

@task(inject_context=True)
def combine(ctx: TaskExecutionContext) -> int:
    doubled = ctx.get_channel().get("doubled")
    tripled = ctx.get_channel().get("tripled")
    return doubled + tripled

with workflow("diamond") as wf:
    src = source(task_id="src", value=5)

    # Diamond: src → (double | triple) → combine
    src >> (double | triple) >> combine

    result = wf.execute(start_node="src")
    print(result)  # Output: 25 (5*2 + 5*3)
```

### 複数インスタンスのパイプライン

複数アイテムを並列処理する例:

```python
@task
def fetch(source: str) -> dict:
    return {"source": source, "data": f"data_{source}"}

@task
def process(data: dict) -> str:
    return f"Processed {data['source']}"

with workflow("multi_pipeline") as wf:
    # Create instances
    fetch_a = fetch(task_id="fetch_a", source="api")
    fetch_b = fetch(task_id="fetch_b", source="db")
    fetch_c = fetch(task_id="fetch_c", source="file")

    # Run in parallel
    all_fetches = fetch_a | fetch_b | fetch_c

    _, ctx = wf.execute(
        start_node=all_fetches.task_id,
        ret_context=True
    )

    # Get results
    for task_id in ["fetch_a", "fetch_b", "fetch_c"]:
        print(ctx.get_result(task_id))
```

**💡 重要パターン:** タスクインスタンス作成 → `|` で並列 → 実行

---

## レベル9: 動的タスク生成

**高度な機能**: 実行中にワークフローグラフを変更できます。

### なぜ実行時に動的にするのか?

**コンパイル時グラフの問題:**

多くのワークフローシステムでは、分岐やループを事前定義する必要があります:

```python
# ❌ Compile-time approach (LangGraph style)
def should_retry(state):
    return "retry" if state["score"] < 0.8 else "continue"

graph.add_conditional_edges(
    "process",
    should_retry,  # All paths predefined
    {
        "retry": "retry_node",
        "continue": "finalize_node"
    }
)
app = graph.compile()  # Graph is now fixed
```

**制限事項:**
- すべての分岐を事前に定義する必要がある
- 動的条件 (ファイル数/データサイズ) の扱いが難しい
- ループ回数が定義時に固定される
- 適応的なロジックを表現しづらい

**Graflowの解決策: 実行時の柔軟性**

Graflowでは通常のPython条件分岐を使い、必要に応じてタスクを動的生成できます。

### 実行時にタスクを追加する

`context.next_task()` を使ってタスクを動的に追加したり、既存タスクへジャンプできます:

**`goto` パラメータ:**

- **`ctx.next_task(task, goto=False)`** (デフォルト):
  - タスクを実行キューに追加
  - 現在タスク終了後に通常の後続へ進む
  - 追加作業を行いつつ制御フローは変えない

- **`ctx.next_task(task, goto=True)`**:
  - 指定タスクへ即時ジャンプ
  - 現在タスクの後続をスキップ
  - **既にグラフに存在するタスクへのジャンプに使う**

**例1: 既存タスクへのジャンプ**

`goto=True` を使って、すでに定義済みのタスクにジャンプします:

```python
with workflow("error_handling") as wf:
    @task(inject_context=True)
    def risky_operation(ctx: TaskExecutionContext):
        """Process data with potential errors."""
        try:
            # Risky operation
            if random.random() < 0.3:  # 30% chance of critical error
                raise CriticalError("Critical failure!")
            print("Operation succeeded")
        except CriticalError:
            # Jump to existing emergency handler task
            emergency_task = ctx.graph.get_node("emergency_handler")
            ctx.next_task(emergency_task, goto=True)  # Skip normal successors

    @task
    def emergency_handler():
        """Handle emergency situations."""
        print("Emergency handler activated!")
        # Send alerts, rollback, etc.

    @task
    def normal_continuation():
        """This runs only if risky_operation succeeds."""
        print("Continuing normal flow")

    # Define workflow
    risky_operation >> normal_continuation

    wf.execute()
```

**出力 (エラー時):**
```
Emergency handler activated!
```

**出力 (成功時):**
```
Operation succeeded
Continuing normal flow
```

**例2: 条件分岐で既存タスクへ**

```python
with workflow("conditional") as wf:
    @task(inject_context=True)
    def router(ctx: TaskExecutionContext, user_type: str):
        """Route to different paths based on user type."""
        if user_type == "premium":
            premium_task = ctx.graph.get_node("premium_flow")
            ctx.next_task(premium_task, goto=True)
        elif user_type == "basic":
            basic_task = ctx.graph.get_node("basic_flow")
            ctx.next_task(basic_task, goto=True)

    @task
    def premium_flow():
        print("Premium user processing")

    @task
    def basic_flow():
        print("Basic user processing")

    @task
    def default_continuation():
        print("This is skipped when goto=True")

    router >> default_continuation

    wf.execute(initial_channel={"user_type": "premium"})
```

**例3: 追加作業のエンキュー (goto=False)**

`goto=False` (デフォルト) で制御フローを変えずにタスク追加:

```python
@task(inject_context=True)
def process(ctx: TaskExecutionContext):
    @task
    def extra_logging():
        print("Extra logging task")

    # Enqueue: Add extra_logging, then continue to normal successors
    ctx.next_task(extra_logging)  # goto=False is default

    print("Main processing")

@task
def continuation():
    print("Normal continuation")

with workflow("enqueue_demo") as wf:
    process >> continuation
    wf.execute()
```

**出力:**
```
Main processing
Extra logging task
Normal continuation
```

**💡 重要な違い:**
- **`goto=False`** (デフォルト): 「このタスクを追加し、通常どおり続行」
- **`goto=True`**: 「既存タスクへジャンプし、通常の後続をスキップ」
- 既存タスクの取得は `ctx.graph.get_node(task_id)`

### next_iteration による自己ループ

`context.next_iteration()` を使ってリトライ/収束パターンを実現できます:

```python
@task(inject_context=True)
def optimize(ctx: TaskExecutionContext):
    """Optimize until convergence."""
    channel = ctx.get_channel()
    iteration = channel.get("iteration", default=0)
    accuracy = channel.get("accuracy", default=0.5)

    # Training step
    new_accuracy = train_step(accuracy)
    print(f"Iteration {iteration}: accuracy={new_accuracy:.2f}")

    if new_accuracy >= 0.95:
        # Converged!
        print("Converged!")
        channel.set("final_accuracy", new_accuracy)
    else:
        # Continue iterating
        channel.set("iteration", iteration + 1)
        channel.set("accuracy", new_accuracy)
        ctx.next_iteration()

with workflow("optimization") as wf:
    wf.execute()
```

**出力例:**
```
Iteration 0: accuracy=0.65
Iteration 1: accuracy=0.78
Iteration 2: accuracy=0.88
Iteration 3: accuracy=0.96
Converged!
```

**💡 主な用途:**
- 最大試行回数付きのリトライ
- MLハイパーパラメータ調整
- 収束型アルゴリズム
- 段階的改善

### 早期終了

#### 正常終了: terminate_workflow

正常に終了したい場合:

```python
@task(inject_context=True)
def check_cache(ctx: TaskExecutionContext, key: str):
    """Check cache before processing."""
    cached = get_from_cache(key)

    if cached is not None:
        # Cache hit - no need to continue
        print(f"Cache hit: {cached}")
        ctx.terminate_workflow("Data found in cache")
        return cached

    # Cache miss - continue to next tasks
    print("Cache miss, proceeding...")
    return None

@task
def expensive_processing():
    """This won't run if cache hits."""
    print("Expensive processing...")
    return "processed"

with workflow("caching") as wf:
    check_cache(task_id="cache", key="my_key") >> expensive_processing
    wf.execute()
```

**キャッシュヒット時:**
```
Cache hit: cached_value
```

**キャッシュミス時:**
```
Cache miss, proceeding...
Expensive processing...
```

#### 異常終了: cancel_workflow

エラー時にワークフロー全体を停止したい場合:

```python
@task(inject_context=True)
def validate_data(ctx: TaskExecutionContext, data: dict):
    """Validate data before processing."""
    if not data.get("valid"):
        # Invalid data - cancel entire workflow
        ctx.cancel_workflow("Data validation failed")

    return data

@task
def process_data(data: dict):
    print("Processing data...")
    return data

with workflow("validation") as wf:
    validate = validate_data(task_id="validate", data={"valid": False})
    validate >> process_data

    try:
        wf.execute()
    except Exception as e:
        print(f"Workflow canceled: {e}")
```

**出力:**
```
Workflow canceled: Data validation failed
```

**違い:**

| メソッド | タスク完了? | 後続実行? | エラー発生? |
|--------|----------------|---------------|-------------|
| `terminate_workflow` | ✅ Yes | ❌ No | ❌ No |
| `cancel_workflow` | ❌ No | ❌ No | ✅ Yes (GraflowWorkflowCanceledError) |

**💡 重要ポイント:**
- `next_task(task)` はタスクをキューに入れて後続に進む
- `next_task(task, goto=True)` はジャンプして後続をスキップ
- `next_iteration()` は自己ループでリトライ/収束
- `terminate_workflow()` は正常終了
- `cancel_workflow()` はエラー終了

---

## ベストプラクティス

### 1. 再利用性のためにタスクインスタンスを使う

```python
# ✅ Good - Reusable task definition
@task
def fetch_data(source: str):
    return fetch(source)

api = fetch_data(task_id="api", source="api")
db = fetch_data(task_id="db", source="database")

# ❌ Avoid - Duplicated definitions
@task
def fetch_api():
    return fetch("api")

@task
def fetch_db():
    return fetch("database")
```

### 2. 型ヒントを必ず使う

```python
# ✅ Good
@task
def process(value: int, multiplier: int = 2) -> int:
    return value * multiplier

# ❌ Avoid
@task
def process(value, multiplier=2):
    return value * multiplier
```

### 3. コンテキスト注入は必要なときだけ

```python
# ✅ Simple computation - no context needed
@task
def add(x: int, y: int) -> int:
    return x + y

# ✅ Inter-task communication - needs context
@task(inject_context=True)
def share_data(ctx: TaskExecutionContext, value: int):
    ctx.get_channel().set("shared", value)
```

### 4. 分かりやすいタスクIDを使う

```python
# ✅ Good - Clear and descriptive
fetch_user_profile = fetch(task_id="fetch_user_profile")
validate_email = validate(task_id="validate_email")

# ❌ Avoid - Generic names
task1 = fetch(task_id="t1")
task2 = validate(task_id="t2")
```

### 5. ret_context で結果を取得する

```python
# ✅ Good - Access all task results
_, ctx = wf.execute(ret_context=True)
result_a = ctx.get_result("task_a")
result_b = ctx.get_result("task_b")

# ⚠️ Limited - Only final result
result = wf.execute()  # Only last task's result
```

---

## まとめ

### 学習パス

1. **ここから開始**: [レベル1](#レベル1-最初のタスク) - 最初のタスク
2. **ワークフロー構築**: [レベル2](#レベル2-最初のワークフロー) - タスク接続
3. **合成**: [レベル3](#レベル3-タスク合成) - 直列/並列
4. **データ受け渡し**: [レベル4](#レベル4-パラメータの受け渡し) - パラメータとチャンネル
5. **再利用**: [レベル5](#レベル5-タスクインスタンス) - タスクインスタンス
6. **状態共有**: [レベル6](#レベル6-チャンネルとコンテキスト) - チャンネルとコンテキスト
7. **実行制御**: [レベル7](#レベル7-実行パターン) - 実行パターン
8. **複雑パターン**: [レベル8](#レベル8-複雑なワークフロー) - ダイヤモンド/複数インスタンス
9. **高度な機能**: [レベル9](#レベル9-動的タスク生成) - 動的タスク

### パラメータの優先順位

パラメータ解決の優先順は以下の通りです (高いほど優先):

```
Injection > Bound > Channel
   (ctx)    (task_id)  (initial_channel)
```

### 次のステップ

**例を確認:**
- `examples/01_basics/` - 基本タスクパターン
- `examples/02_workflows/` - ワークフロー合成
- `examples/07_dynamic_tasks/` - 動的タスク生成

**高度な機能:**
- [Checkpoint & Resume](checkpoint/checkpoint_resume_design.md) - 障害耐性
- [HITL](hitl/hitl_design.md) - Human-in-the-loopワークフロー
- [Distributed Execution](scaling/redis_distributed_execution_redesign.md) - スケーリング

**主要ファイル:**
- `graflow/core/task.py` - タスク実装
- `graflow/core/workflow.py` - ワークフローコンテキスト
- `graflow/core/engine.py` - 実行エンジン

---

**Graflowチームより ❤️**
