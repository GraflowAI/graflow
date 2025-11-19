# GPT Newspaper

Multi-agent workflow that researches topics, drafts and critiques articles, then renders a newspaper-style HTML output. The project couples a FastAPI backend powered by Graflow with a Vite/React frontend for requesting new newspapers and previewing recent runs.

## Highlights
- **Dynamic Graflow workflow** – parallel article generation, iterative critique loops, and template-based publishing.
- **FastAPI gateway** – `/api/generate` endpoint plus static serving of generated HTML under `/outputs`.
- **Modern frontend** – Material UI + Vite with Storybook-driven components.
- **Live log streaming** – WebSocket console mirrors backend stdout while the workflow runs so you can watch each agent step.
- **LLM observability** – Optional Langfuse integration with automatic trace propagation for workflows, LLM calls, and agent tools.
- **Container-first** – reproducible Dockerfiles and a compose script scoped to this example directory.

## Requirements
- Python ≥ 3.11 (backend) and `make install` or `pip install -e .`
- Node.js ≥ 20 (frontend) with npm
- Tavily and LLM provider credentials (see `.env.example`)
- Optional: Docker + Docker Compose v2 for containerized runs
- Optional: Langfuse account for LLM observability and tracing

## Configure Secrets
```bash
cp examples/gpt_newspaper/backend/.env.example examples/gpt_newspaper/backend/.env
# edit the file and add:
# TAVILY_API_KEY=<your key>
# OPENAI_API_KEY=<or any LiteLLM-compatible key>

# Optional: Langfuse credentials for LLM observability
# LANGFUSE_PUBLIC_KEY=pk-lf-...
# LANGFUSE_SECRET_KEY=sk-lf-...
# LANGFUSE_HOST=https://cloud.langfuse.com  # or http://localhost:3000 for local
```
The backend reads this file automatically via `python-dotenv`. Frontend dev server can point to any backend via the `VITE_API_BASE_URL` env var (defaults to `http://localhost:8000` in `vite.config.ts`).

### Langfuse Setup (Optional)

Graflow integrates with [Langfuse](https://langfuse.com/) for complete observability of your LLM workflows using **OpenTelemetry context propagation** to automatically link:
- **Graflow workflow/task traces** (via `LangFuseTracer`)
- **LLM API calls** (via LiteLLM's Langfuse callback for simple/dynamic workflows, or Google ADK for agent workflow)

**No manual trace ID passing needed** - traces are automatically nested!

#### Quick Start
1. **Get API keys** from [cloud.langfuse.com](https://cloud.langfuse.com/) (free tier available)
2. **Add to `.env`**:
   ```bash
   LANGFUSE_PUBLIC_KEY=pk-lf-...
   LANGFUSE_SECRET_KEY=sk-lf-...
   LANGFUSE_HOST=https://cloud.langfuse.com
   ```
3. **Run any workflow** - traces appear automatically in Langfuse UI

#### What You'll See
- **Workflow structure**: Task hierarchy with dependencies
- **LLM calls**: Prompts, completions, tokens, latency
- **Agent tool usage**: Tool calls with inputs/outputs (agent workflow only)
- **Performance metrics**: Task durations, costs, token usage
- **Error tracking**: Failed tasks with stack traces

#### Dependencies
Required packages (already in `requirements.txt`):
```bash
langfuse>=3.8.1              # Langfuse SDK
opentelemetry-api>=1.37.0    # OpenTelemetry context API
opentelemetry-sdk>=1.37.0    # OpenTelemetry SDK
litellm>=1.72.6              # LLM wrapper with langfuse_otel callback
```

For agent workflow (Google ADK):
```bash
google-adk>=0.9.0            # Agent Development Kit
```

#### Reference Documentation
- [LiteLLM Langfuse Integration](https://docs.litellm.ai/docs/observability/langfuse_integration)
- [Google ADK Langfuse Integration](https://langfuse.com/integrations/frameworks/google-adk)
- [Graflow LLM Integration Design](../../docs/llm_integration_design.md)
- [Full Setup Guide](backend/LANGFUSE_SETUP.md)

## Running Locally (without Docker)
1. **Backend**
   ```bash
   make install               # or pip install -e . from repo root
   cd examples/gpt_newspaper/backend
   pip install -r requirements.txt
   uvicorn api:app --reload --port 8000
   ```
2. **Frontend**
   ```bash
   cd examples/gpt_newspaper/frontend
   npm install
   VITE_API_BASE_URL=http://localhost:8000 npm run dev -- --host 0.0.0.0 --port 5173
   ```
3. Visit `http://localhost:5173`, submit topics, and watch outputs appear under `backend/outputs/`.

## Docker Compose Workflow
The repo ships with reproducible Dockerfiles plus a helper script that pins the compose project directory to this example. From anywhere in the repository:
```bash
examples/gpt_newspaper/docker-compose.sh up --build
```
- Backend runs on `http://localhost:8000`.
- Frontend dev server runs on `http://localhost:5173`.
- Generated HTML is stored on the host at `examples/gpt_newspaper/backend/outputs/`.
Stop containers with `examples/gpt_newspaper/docker-compose.sh down`.

## Live Log Streaming
- The frontend opens a WebSocket to `ws://<backend>/ws/logs/{runId}` before each request.
- Backend stdout/stderr (the same prints you see when running `uvicorn`) are mirrored in the UI log console in near real time.
- Each request displays the active run ID so you can correlate it with files in `backend/outputs/run_*`.
- Click the 📤 icon next to "Live logs" to open logs in a dedicated popup window with:
  - Live WebSocket streaming (independent of main window)
  - Auto-scroll toggle button (ON by default)
  - Connection status indicator
  - Dark terminal-style theme optimized for log viewing
- Closing or navigating away from the page automatically stops the stream (connections end server-side when the run completes).

## Tests
- Backend workflow smoke test: `uv run pytest examples/gpt_newspaper/backend/test_workflow.py`
- Frontend unit tests: `cd examples/gpt_newspaper/frontend && npm run test`
- Storybook: `npm run storybook`

## Project Layout
```
backend/   # FastAPI service, Graflow workflow, agents, templates, outputs
frontend/  # Vite app, React components, Storybook stories
docker-compose.yml
docker-compose.sh
```

## How It Works

The frontend runs on port 5173 and communicates with the backend on port 8000:
- **API requests**: HTTP calls to `http://localhost:8000/api/*`
- **WebSocket logs**: Real-time streaming via `ws://localhost:8000/ws/logs/{runId}`
- **Generated files**: Static HTML served from `http://localhost:8000/outputs/*`

When you click "Open HTML file" or an item in "Recent outputs", the frontend constructs the full backend URL automatically.

## Troubleshooting
- If Docker builds fail on the frontend due to Storybook/Vite peer warnings, the Dockerfile already uses `npm ci --legacy-peer-deps`.
- Ensure the `.env` file in `backend/.env` contains valid API keys before generating content; missing keys trigger FastAPI 500 errors with helpful messages.
- If live logs show "Waiting for backend output…", check the browser console (F12) for WebSocket connection errors. The WebSocket should connect to `ws://localhost:8000/ws/logs/{runId}`.
- If the frontend cannot connect to the API, verify that `VITE_API_BASE_URL` in docker-compose.yml is set to `http://localhost:8000` (not `http://backend:8000`).
- If clicking "Open HTML file" or "Recent outputs" opens a blank page or 404, ensure the backend is running on port 8000 and the outputs directory is mounted correctly.

### Langfuse Tracing Issues
- **Traces not appearing**: Verify `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are set correctly in `.env`. Check backend logs for "Langfuse initialized" message.
- **Missing LLM traces**: Ensure `langfuse>=3.8.1` and `litellm>=1.72.6` are installed. LiteLLM's `langfuse_otel` callback requires OpenTelemetry packages.
- **Disconnected traces**: If workflow traces and LLM calls appear separately, check that OpenTelemetry context propagation is working (automatic with `LangFuseTracer`).
- **Agent tool traces missing**: For agent workflow, ensure `google-adk>=0.9.0` is installed and ADK tracing is configured via `setup_adk_tracing()`.
- **Local Langfuse**: If using self-hosted Langfuse, set `LANGFUSE_HOST=http://localhost:3000` and ensure the server is running.
