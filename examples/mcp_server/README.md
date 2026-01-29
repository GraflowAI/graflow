# Company Intelligence MCP Server

An MCP server that generates company intelligence reports. Accessible via REST API using FastMCP.

## Quick Start

```bash
# 1. Navigate to directory
cd examples/mcp_server

# 2. Install dependencies
uv add fastmcp tavily-python

# 3. Set environment variables
cp .env.example .env
# Set TAVILY_API_KEY and OPENAI_API_KEY in .env

# 4. Start server
uv run python server.py

# 5. Run workflow only (without server)
uv run python workflow.py "Treasure Data"

# 6. Client example
uv run python client_example.py "Treasure Data"
```

## Overview

This MCP server executes the following workflow using a company name as input to generate an intelligence report for business meeting preparation:

```
Search → Curate → Write → Critique → Output
       ↑                    ↓
       └──── Feedback Loop ─┘
```

### Workflow Details

1. **Search**: Collect information from multiple sources using Tavily API
   - Company news
   - Industry trends
   - Company profile
   - Competitor information

2. **Curate**: Filter and organize search results with LLM
   - Relevance scoring
   - Deduplication
   - Category classification

3. **Write**: Generate intelligence report with LLM
   - Executive summary
   - Detailed sections
   - Key talking points for meetings

4. **Critique**: Evaluate the report with LLM
   - Accuracy, completeness, relevance, structure, source quality
   - Sends back to Write if criteria not met (max 3 iterations)

5. **Output**: Export final report in Markdown format

## Environment Setup

### 1. Create .env file

```bash
cd examples/mcp_server
cp .env.example .env
# Edit .env file to set your API keys
```

### Required Environment Variables

```bash
TAVILY_API_KEY=your-tavily-api-key
OPENAI_API_KEY=your-openai-api-key
```

### Optional (Langfuse Tracing)

```bash
LANGFUSE_PUBLIC_KEY=your-langfuse-public-key
LANGFUSE_SECRET_KEY=your-langfuse-secret-key
LANGFUSE_HOST=http://localhost:3000
ENABLE_TRACING=true
```

### Other Settings

```bash
GRAFLOW_LLM_MODEL=gpt-5-mini      # Default model
WRITER_MODEL=gpt-5-mini           # Model for Writer
CRITIQUE_MODEL=gpt-5-mini         # Model for Critique
MAX_CRITIQUE_ITERATIONS=3          # Maximum revision count
MAX_SEARCH_RESULTS=10              # Maximum search results
MCP_SERVER_HOST=0.0.0.0            # Server host
MCP_SERVER_PORT=9100               # Server port
```

## Starting the Server

### Server Startup

```bash
# Navigate to directory
cd examples/mcp_server

# Start server
uv run python server.py

# Or start with uvicorn
uv run uvicorn server:app --host 0.0.0.0 --port 9100
```

### Run Workflow Only (without server)

```bash
cd examples/mcp_server
uv run python workflow.py "Treasure Data"
```

## API Endpoints

### REST API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/config` | GET | Get current configuration |
| `/mcp` | POST | MCP SSE endpoint |

### MCP Tools

#### generate_company_intelligence

Generate a comprehensive intelligence report for a company.

**Input:**
```json
{
  "company_name": "Treasure Data",
  "enable_tracing": true
}
```

**Output:**
```json
{
  "company_name": "Treasure Data",
  "report_markdown": "# Treasure Data Intelligence Report\n...",
  "executive_summary": "...",
  "key_takeaways": ["Point 1", "Point 2"],
  "sections": [...],
  "sources_count": 15,
  "iterations": 2,
  "generated_at": "2024-01-15T10:30:00",
  "critique_score": 0.85
}
```

#### search_company_news

Search for the latest news about a company (lightweight version).

**Input:**
```json
{
  "company_name": "Salesforce",
  "max_results": 10
}
```

#### search_industry_trends

Search for industry trends.

**Input:**
```json
{
  "company_name": "Sony",
  "industry": "Electronics",
  "max_results": 10
}
```

## Client Usage

### Python Client Example

```bash
cd examples/mcp_server

# After starting the server, in another terminal
uv run python client_example.py "Treasure Data"

# News search only
uv run python client_example.py "Salesforce" --action news

# Industry trends search
uv run python client_example.py "Sony" --action trends
```

### Using curl

```bash
# Health check
curl http://localhost:9100/health

# Generate report (via SSE)
curl http://localhost:9100/mcp
```

### Using httpx

```python
import httpx

# Health check
response = httpx.get("http://localhost:9100/health")
print(response.json())

# Check configuration
response = httpx.get("http://localhost:9100/config")
print(response.json())
```

## Claude Code MCP Setup

You can use this MCP server from Claude Code.

### MCP Installation Scopes

MCP servers can be configured in three scopes:

| Scope | Storage Location | Use Case |
|-------|-----------------|----------|
| `local` | `~/.claude.json` (under project path) | Personal use, current project only (default) |
| `project` | `.mcp.json` at project root | Team sharing (commit to version control) |
| `user` | `~/.claude.json` | Available across all projects |

### Method 1: CLI Installation (Recommended)

First, start the server:
```bash
cd examples/mcp_server && uv run python server.py
```

In another terminal, register MCP:
```bash
# Local scope (default) - current project only
claude mcp add --transport http company-intel http://localhost:9100/mcp

# Project scope - shared with team (saved to .mcp.json)
claude mcp add --transport http --scope project company-intel http://localhost:9100/mcp

# User scope - available across all projects
claude mcp add --transport http --scope user company-intel http://localhost:9100/mcp
```

### Method 2: Project Configuration File (Team Sharing)

Create `.mcp.json` in your project root (commit to version control):

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

Environment variables can be referenced using `${VAR}` format (actual values taken from user's environment).

### Managing MCP Servers

```bash
# List registered servers
claude mcp list

# Get server details
claude mcp get company-intel

# Remove a server
claude mcp remove company-intel
```

### Verification

Restart Claude Code and use the `/mcp` command to check connection status.

## Langfuse Tracing

Workflow execution can be traced with Langfuse:

1. Create an account at [Langfuse Cloud](https://cloud.langfuse.com)
2. Create a project and obtain API keys
3. Set environment variables in the `.env` file
4. After starting the server, view traces in the Langfuse dashboard

Traces include:
- Total workflow execution time
- Execution time for each task (search, curate, write, critique)
- LLM call details (prompts, responses, token counts)
- Errors and stack traces

## Project Structure

```
examples/mcp_server/
├── __init__.py              # Module initialization
├── config.py                # Configuration management (dotenv support)
├── workflow.py              # Graflow workflow definition
├── server.py                # FastMCP server
├── client_example.py        # Client example
├── .env.example             # Environment variable template
├── claude_mcp_config.json   # Claude Code MCP config example
├── README.md                # This file
├── README_ja.md             # Japanese README
└── agents/
    ├── __init__.py
    ├── search.py        # Search agent (Tavily)
    ├── curator.py       # Curation agent (LLM)
    ├── writer.py        # Writer agent (LLM)
    └── critique.py      # Critique agent (LLM)
```

## Dependencies

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

Installation:
```bash
uv add fastmcp tavily-python python-dotenv
```

## License

Apache License 2.0
