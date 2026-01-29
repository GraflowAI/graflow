"""FastMCP Server for Company Intelligence Workflow.

This server exposes the company intelligence workflow as MCP tools.

Usage:
    cd examples/mcp_server
    uv run python server.py

    # Or with uvicorn
    uv run uvicorn server:app --host 0.0.0.0 --port 9100
"""

import asyncio
import json
from collections.abc import Callable
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

from config import Config
from fastapi import FastAPI
from fastmcp import Context, FastMCP
from workflow import run_workflow

# Initialize FastMCP server
mcp = FastMCP(
    name="company-intelligence",
    instructions="""Company Intelligence MCP Server.

This server provides tools for generating comprehensive intelligence reports
about companies for business meeting preparation.

Workflow: Search → Curate → Write → Critique → Output

Tools:
- generate_company_intelligence: Full report generation with AI critique
- search_company_news: Quick news search
- search_industry_trends: Industry trend search
""",
)


def create_progress_callback(
    ctx: Context | None,
    loop: asyncio.AbstractEventLoop | None,
) -> Callable[[int, int, str], None]:
    """Create a progress callback that reports to MCP client.

    Args:
        ctx: FastMCP Context for progress reporting
        loop: Event loop for running async code from sync context

    Returns:
        Callback function that reports progress
    """

    def progress_callback(current: int, total: int, message: str) -> None:
        """Report progress to MCP client.

        Args:
            current: Current step number (1-indexed)
            total: Total number of steps
            message: Progress message to display
        """
        log_message = f"[{current}/{total}] {message}"
        print(log_message)

        if ctx is not None and loop is not None:
            try:
                # Send debug message
                future = asyncio.run_coroutine_threadsafe(
                    ctx.debug(f"Step {current}/{total} starting"),
                    loop,
                )
                future.result(timeout=1.0)

                # Send progress notification to MCP client
                future = asyncio.run_coroutine_threadsafe(
                    ctx.report_progress(progress=current, total=total),
                    loop,
                )
                future.result(timeout=1.0)

                # Send info message for client visibility
                future = asyncio.run_coroutine_threadsafe(
                    ctx.info(log_message),
                    loop,
                )
                future.result(timeout=1.0)
            except Exception as e:
                print(f"Warning: Failed to report progress to MCP: {e}")

    return progress_callback


@mcp.tool()
async def generate_company_intelligence(
    company_name: str,
    enable_tracing: bool = True,
    ctx: Context | None = None,
) -> str:
    """Generate a comprehensive intelligence report for a company in Markdown format.

    This tool searches for news, industry trends, company profile, and competitor
    information, then generates a structured Markdown report.

    IMPORTANT: This tool returns the full report as a Markdown string.
    Use this tool when you need detailed company intelligence for business meetings.

    Args:
        company_name: Name of the company to research (e.g., "Anthropic", "OpenAI", "トレジャーデータ")
        enable_tracing: Whether to enable tracing (default: True)

    Returns:
        A Markdown-formatted intelligence report containing:
        - Executive summary
        - Recent news and developments
        - Industry trends
        - Competitive landscape
        - Key takeaways for business use
    """
    print(f"\n{'=' * 60}")
    print("MCP Tool: generate_company_intelligence")
    print(f"Company: {company_name}")
    print(f"Tracing: {enable_tracing}")
    print(f"{'=' * 60}\n")

    # Get the current event loop for progress reporting
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    # Create progress callback
    progress_callback = create_progress_callback(ctx, loop)

    # Log start to MCP client
    if ctx is not None:
        await ctx.debug(f"Initializing workflow for company: {company_name}")
        await ctx.info(f"Starting company intelligence workflow for: {company_name}")

    try:
        # Run the synchronous workflow in a thread pool to avoid blocking
        result = await asyncio.to_thread(
            run_workflow,
            company_name=company_name,
            enable_tracing=enable_tracing,
            progress_callback=progress_callback,
        )

        report = result.get("report", {})
        critique = result.get("critique", {})

        sources_count = result.get("sources_count", 0)
        report_markdown = result.get("report_markdown", "")

        if sources_count == 0 or not report_markdown:
            if ctx is not None:
                await ctx.warning(f"No information found for: {company_name}")
            return f"""# {company_name} - 情報なし

{company_name}に関する情報を収集できませんでした。

## 考えられる原因
- 会社名のスペルが正しくない可能性があります
- 別の表記（英語名/日本語名）をお試しください
- 非公開企業や小規模企業の場合、情報が限られる場合があります

生成日時: {datetime.now().isoformat()}
"""

        if ctx is not None:
            await ctx.debug(f"Workflow completed: {sources_count} sources, report length: {len(report_markdown)}")
            await ctx.info(f"Report generated successfully with {sources_count} sources")

        # Return the markdown report directly
        return report_markdown

    except Exception as e:
        print(f"Error in generate_company_intelligence: {e}")
        if ctx is not None:
            await ctx.error(f"Error generating report: {e}")
        return f"""# {company_name} - レポート生成エラー

レポート生成中にエラーが発生しました: {e}

## 考えられる原因
- API接続の問題
- 検索結果が見つからなかった
- レート制限

生成日時: {datetime.now().isoformat()}
"""


@mcp.tool()
def search_company_news(company_name: str, max_results: int = 10) -> dict[str, Any]:
    """Search for the latest news about a company.

    This is a lightweight tool that only performs the search step.

    Args:
        company_name: Name of the company to search
        max_results: Maximum number of results (default: 10)

    Returns:
        Dictionary containing news articles with title, url, content, date
    """
    from agents import SearchAgent

    try:
        agent = SearchAgent()
        response = agent.search_company_news(company_name, max_results=max_results)

        news_list = [
            {
                "title": r.title,
                "url": r.url,
                "content": r.content,
                "published_date": r.published_date,
                "score": r.score,
            }
            for r in response.results
        ]
        count = len(news_list)

        if count == 0:
            return {
                "success": False,
                "company_name": company_name,
                "message": f"{company_name}に関するニュースが見つかりませんでした。別の会社名や表記をお試しください。",
                "news": [],
                "count": 0,
            }

        return {
            "success": True,
            "company_name": company_name,
            "message": f"{company_name}に関するニュースを{count}件取得しました。",
            "news": news_list,
            "count": count,
        }

    except Exception as e:
        return {
            "success": False,
            "company_name": company_name,
            "message": f"ニュース検索中にエラーが発生しました: {e}",
            "news": [],
            "count": 0,
        }


@mcp.tool()
def search_industry_trends(
    company_name: str,
    industry: str | None = None,
    max_results: int = 10,
) -> dict[str, Any]:
    """Search for industry trends related to a company.

    Args:
        company_name: Name of the company
        industry: Industry name (optional, will be inferred if not provided)
        max_results: Maximum number of results (default: 10)

    Returns:
        Dictionary containing trend articles
    """
    from agents import SearchAgent

    try:
        agent = SearchAgent()
        response = agent.search_industry_trends(
            company_name,
            industry=industry,
            max_results=max_results,
        )

        trends_list = [
            {
                "title": r.title,
                "url": r.url,
                "content": r.content,
                "published_date": r.published_date,
                "score": r.score,
            }
            for r in response.results
        ]
        count = len(trends_list)

        if count == 0:
            industry_info = f"（業界: {industry}）" if industry else ""
            return {
                "success": False,
                "company_name": company_name,
                "industry": industry or "推定",
                "message": f"{company_name}{industry_info}に関する業界トレンド情報が見つかりませんでした。",
                "trends": [],
                "count": 0,
            }

        industry_info = f"（業界: {industry}）" if industry else ""
        return {
            "success": True,
            "company_name": company_name,
            "industry": industry or "推定",
            "message": f"{company_name}{industry_info}に関する業界トレンドを{count}件取得しました。",
            "trends": trends_list,
            "count": count,
        }

    except Exception as e:
        return {
            "success": False,
            "company_name": company_name,
            "industry": industry or "推定",
            "message": f"業界トレンド検索中にエラーが発生しました: {e}",
            "trends": [],
            "count": 0,
        }


@mcp.resource("config://settings")
def get_config_settings() -> str:
    """Get current server configuration settings."""
    return json.dumps(
        {
            "default_model": Config.DEFAULT_MODEL,
            "writer_model": Config.WRITER_MODEL,
            "critique_model": Config.CRITIQUE_MODEL,
            "max_critique_iterations": Config.MAX_CRITIQUE_ITERATIONS,
            "max_search_results": Config.MAX_SEARCH_RESULTS,
            "tracing_enabled": Config.ENABLE_TRACING,
            "langfuse_host": Config.LANGFUSE_HOST,
        },
        indent=2,
    )


# Get the MCP ASGI app
mcp_app = mcp.http_app()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager that includes MCP lifespan."""
    try:
        async with mcp_app.lifespan(app):
            yield
    except asyncio.CancelledError:
        # Gracefully handle shutdown cancellation
        print("\nServer shutting down...")
        raise


# Create FastAPI app with proper lifespan
app = FastAPI(
    title="Company Intelligence MCP Server",
    description="MCP server for generating company intelligence reports",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "company-intelligence",
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/config")
async def get_config():
    """Get current configuration."""
    return {
        "default_model": Config.DEFAULT_MODEL,
        "writer_model": Config.WRITER_MODEL,
        "critique_model": Config.CRITIQUE_MODEL,
        "max_critique_iterations": Config.MAX_CRITIQUE_ITERATIONS,
        "max_search_results": Config.MAX_SEARCH_RESULTS,
        "tracing_enabled": Config.ENABLE_TRACING,
    }


# Mount MCP app at root (FastMCP http_app already has /mcp route)
app.mount("/", mcp_app)


def main():
    """Run the MCP server."""
    import uvicorn

    Config.display()

    if not Config.validate():
        print("\nConfiguration validation failed!")
        print("Please set the required environment variables in .env file")
        return

    print("\nStarting Company Intelligence MCP Server")
    print(f"   Host: {Config.MCP_SERVER_HOST}")
    print(f"   Port: {Config.MCP_SERVER_PORT}")
    print("\nREST API endpoints:")
    print("   GET  /health - Health check")
    print("   GET  /config - Get configuration")
    print("   GET  /mcp - MCP SSE endpoint")
    print("\nAvailable MCP tools:")
    print("   - generate_company_intelligence")
    print("   - search_company_news")
    print("   - search_industry_trends")
    print()

    uvicorn.run(
        app,
        host=Config.MCP_SERVER_HOST,
        port=Config.MCP_SERVER_PORT,
        log_level="info",
    )


if __name__ == "__main__":
    main()
