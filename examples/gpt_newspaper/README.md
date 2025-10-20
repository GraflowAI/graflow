# GPT Newspaper - Graflow Implementation

An autonomous newspaper generation agent built with Graflow, demonstrating advanced workflow patterns including runtime iteration, conditional branching, and parallel processing.

## 🔍 Overview

This example showcases a complete newspaper generation workflow using Graflow's dynamic task features:

- **7 Specialized Agents**: Search, Curator, Writer, Critique, Designer, Editor, Publisher
- **Write-Critique Iteration Loop**: Uses `context.next_task()` for iterative refinement
- **Conditional Branching**: Dynamic workflow adaptation based on critique feedback
- **Parallel Processing**: Multiple article queries processed concurrently with ThreadPoolExecutor
- **LLM Abstraction**: Uses litellm for provider-agnostic LLM calls

## 🌟 Key Graflow Patterns

### 1. Sequential Task Flow with Conditional Branching

The workflow uses a clean task chain: `search >> curate >> write >> critique >> designer`

Each task is a separate, focused unit:

```python
# Build workflow graph: search >> curate >> write >> critique >> design
search_task >> curate_task >> write_task >> critique_task >> design_task
```

### 2. Loop-Back Pattern with `goto=True`

The critique agent uses `goto=True` to loop back to the existing write task. When approved, natural graph flow continues to design:

```python
@task(inject_context=True)
def critique_task(context: TaskExecutionContext, article: Dict):
    result = critique_agent.run(article)

    if result.get("critique") is not None:
        # Store article with feedback in channel
        channel.set("article", result)
        channel.set("iteration", iteration + 1)

        # Loop back to write_task using goto=True
        # Flow: write -> critique -> write -> critique -> ... -> design
        context.next_task(write_task, goto=True)
        return result

    # Approved - natural flow continues to design_task from graph
    return result
```

### 3. Benefits of Declarative Graph with `goto=True`

Using a static graph with `goto=True` loop-back:
- ✅ All tasks defined upfront with @task decorator (no dynamic TaskWrapper)
- ✅ Clear, declarative workflow structure visible in graph
- ✅ Reuses existing write_task for iterations
- ✅ Natural flow: write >> critique >> (goto write) >> critique >> design
- ✅ Design task automatically runs when critique approves
- ✅ Demonstrates goto pattern from `examples/07_dynamic_tasks/runtime_dynamic_tasks.py`

### 4. Channel-Based State Management

State persists across task iterations using channels:

```python
channel = context.get_channel()
channel.set("article", article)
channel.set("iteration", iteration + 1)
```

### 5. Parallel Workflow Execution

Multiple article workflows execute in parallel using `ThreadPoolExecutor`:

```python
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    completed_articles = list(
        executor.map(
            lambda args: execute_article_workflow(*args),
            zip(queries, article_ids, [output_dir] * len(queries))
        )
    )
```

This allows multiple articles to be processed concurrently, significantly reducing total execution time.

## 🚀 Getting Started

### Prerequisites

1. **Tavily API Key** - For web search
   - Sign up at: https://tavily.com/

2. **LLM API Key** - OpenAI, Anthropic, or any litellm-supported provider
   - OpenAI: https://platform.openai.com/
   - Anthropic: https://www.anthropic.com/
   - Or configure any provider supported by litellm

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or using uv
uv pip install -r requirements.txt
```

### Configuration

```bash
# Required: Tavily API key
export TAVILY_API_KEY=<your_tavily_key>

# Required: LLM API key (OpenAI example)
export OPENAI_API_KEY=<your_openai_key>

# Or use other providers supported by litellm
# export ANTHROPIC_API_KEY=<your_anthropic_key>
# export COHERE_API_KEY=<your_cohere_key>
# etc.
```

### Running the Example

```bash
# From the graflow root directory
make py examples/gpt_newspaper/backend/newspaper_workflow.py

# Or directly
PYTHONPATH=. uv run python examples/gpt_newspaper/backend/newspaper_workflow.py
```

## 📋 Workflow Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     For Each Query                            │
├──────────────────────────────────────────────────────────────┤
│                                                                │
│  Search Agent                                                  │
│       │                                                        │
│       ▼                                                        │
│  Curator Agent                                                 │
│       │                                                        │
│       ▼                                                        │
│  Writer Agent                                                  │
│       │                                                        │
│       ▼                                                        │
│  Critique Agent ──────────────────────────┐                   │
│       │                                    │                   │
│       │ Has Feedback?                      │                   │
│       │    Yes: next_task(Writer)          │                   │
│       │         next_task(Critique) ───────┘                   │
│       │                                                        │
│       │ No (Approved):                                         │
│       │    next_task(Designer)                                 │
│       │                                                        │
│       ▼                                                        │
│  Designer Agent                                                │
│                                                                │
└────────────────────────────────────────────────────────────────┘
                         │
                         ▼
              ┌──────────────────┐
              │  Editor Agent    │
              │  (Compile All)   │
              └──────────────────┘
                         │
                         ▼
              ┌──────────────────┐
              │ Publisher Agent  │
              │  (Save HTML)     │
              └──────────────────┘
```

## 📂 Project Structure

```
gpt_newspaper/
├── backend/                    # Backend application
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── search.py          # Searches web for news
│   │   ├── curator.py         # Selects relevant sources (uses LiteLLMClient)
│   │   ├── writer.py          # Writes/revises articles (uses LiteLLMClient)
│   │   ├── critique.py        # Provides feedback (uses LiteLLMClient)
│   │   ├── designer.py        # Creates HTML layout
│   │   ├── editor.py          # Compiles newspaper
│   │   └── publisher.py       # Saves final output
│   ├── utils/
│   │   ├── __init__.py
│   │   └── litellm.py         # LiteLLM wrapper with better error handling
│   ├── templates/
│   │   ├── article/
│   │   │   └── index.html     # Article template
│   │   └── newspaper/
│   │       └── layouts/       # Newspaper layout options
│   │           ├── layout_1.html
│   │           ├── layout_2.html
│   │           └── layout_3.html
│   ├── api.py                  # FastAPI backend server
│   ├── newspaper_workflow.py  # Main workflow
│   ├── config.py               # Configuration
│   ├── test_workflow.py        # Tests
│   ├── requirements.txt        # Python dependencies
│   └── .env.example            # Environment variables example
├── frontend/                   # Frontend application (React + TypeScript)
│   ├── src/
│   │   ├── components/        # React components
│   │   └── services/          # API client services
│   ├── public/
│   ├── package.json            # Node.js dependencies
│   ├── nginx.conf              # Nginx configuration for Docker
│   ├── vite.config.ts          # Vite configuration
│   └── tsconfig.json           # TypeScript configuration
├── outputs/                    # Generated newspapers (gitignored)
├── Dockerfile.backend          # Docker configuration for backend
├── Dockerfile.frontend         # Docker configuration for frontend
├── docker-compose.yml          # Docker Compose orchestration
├── .dockerignore               # Docker ignore file
├── DOCKER.md                   # Docker setup guide
├── WEB_APP.md                  # Web application documentation
└── README.md                   # This file
```

## 🎨 Customization

### Change Layout

Modify the `layout` parameter in `run_newspaper_workflow()`:

```python
run_newspaper_workflow(
    queries=queries,
    layout="layout_2.html",  # Try layout_2.html or layout_3.html
    model="gpt-4o-mini"
)
```

### Change LLM Model

Agents are initialized with model parameter (supports any litellm model):

```python
# In newspaper_workflow.py
writer_agent = WriterAgent(model="claude-3-5-sonnet-20241022")
curator_agent = CuratorAgent(model="gpt-4o")
critique_agent = CritiqueAgent(model="gpt-4o-mini")
```

All agents use the `LiteLLMClient` wrapper from `utils/litellm.py`, which:
- Provides better error messages if litellm is not installed
- Offers convenient `chat_text()` method for simple text responses
- Centralizes model configuration
- Makes it easy to swap LLM providers

### Add Custom Queries

Edit the queries list in `main()`:

```python
queries = [
    "Your custom topic 1",
    "Your custom topic 2",
    "Your custom topic 3",
]
```

## 🔧 Advanced Features

### Parallel Execution Control

Control the number of parallel workers:

```python
run_newspaper_workflow(
    queries=queries,
    layout="layout_1.html",
    max_workers=4  # Limit to 4 parallel articles (default: None = CPU count)
)
```

### Max Iterations Control

Control how many write-critique cycles are allowed:

```python
wf.execute(
    f"search_{article_id}",
    max_steps=30  # Adjust to allow more/fewer write-critique cycles
)
```

The critique agent also has a built-in safety limit (currently 5 iterations) to prevent infinite loops.

### Output Directory

Outputs are saved to timestamped directories:

```
outputs/
└── run_1234567890/
    ├── article_0.html
    ├── article_1.html
    ├── article_2.html
    └── newspaper.html
```

## 🌐 Web Application

A full-stack web application is available with FastAPI backend and React/TypeScript frontend.

### Quick Start

**Terminal 1 - Backend:**
```bash
cd examples/gpt_newspaper/backend
export TAVILY_API_KEY=<your_key>
export OPENAI_API_KEY=<your_key>
uvicorn api:app --reload
```

**Terminal 2 - Frontend:**
```bash
cd examples/gpt_newspaper/frontend
npm install
npm run dev
```

Then open http://localhost:5173 in your browser.

### Docker Setup

For a production-ready setup with Docker:

```bash
cd examples/gpt_newspaper

# Create .env file with your API keys
cp backend/.env.example .env
# Edit .env with your actual keys

# Build and start all services
docker-compose up --build

# Access the application
# Frontend: http://localhost:3000
# Backend API: http://localhost:8000
# API Docs: http://localhost:8000/docs
```

See **[DOCKER.md](DOCKER.md)** for detailed Docker setup, configuration, and troubleshooting.

### Features

- 🎨 **Modern UI** - React 19 + TypeScript + Material-UI
- ⚡ **FastAPI Backend** - Pydantic models for type safety
- 🔄 **Real-time Progress** - Loading states with step tracking
- 📱 **Responsive Design** - Mobile-friendly layout
- 🎯 **Type-Safe API** - Axios client with full TypeScript support
- 🚀 **Hot Reload** - Instant feedback during development
- 📚 **Storybook** - Component playground and documentation
- ♿ **Accessible** - WCAG compliant with jsx-a11y

See **[WEB_APP.md](WEB_APP.md)** for detailed setup and architecture.

See **[frontend/README.md](frontend/README.md)** for frontend-specific documentation.

## 🌐 Related Examples

- **[07_dynamic_tasks/runtime_dynamic_tasks.py](../07_dynamic_tasks/runtime_dynamic_tasks.py)** - Core patterns for `next_iteration()` and `next_task()`
- **[02_workflows/simple_pipeline.py](../02_workflows/simple_pipeline.py)** - Basic workflow patterns
- **[03_data_flow/channels.py](../03_data_flow/channels.py)** - Channel usage patterns

## 📝 Notes

- First run may take longer as LLMs generate content
- Critique iterations typically complete in 1-3 rounds
- Multiple articles are processed sequentially (parallel execution can be added)
- HTML outputs can be opened directly in a browser

## 🤝 Contributing

This example is part of the Graflow project. See the main repository for contribution guidelines.

## 📄 License

Same as Graflow main project.
