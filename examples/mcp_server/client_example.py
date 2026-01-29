"""Example client for Company Intelligence MCP Server.

This script demonstrates how to call the MCP server using FastMCP Client.

Usage:
    # Make sure the server is running first:
    # cd examples/mcp_server && uv run python server.py

    # Then run this client:
    # cd examples/mcp_server && uv run uv run python client_example.py "トレジャーデータ"
"""

import argparse
import asyncio
import json
import sys
from typing import Any

import httpx
from fastmcp import Client


async def call_tool_async(
    server_url: str,
    tool_name: str,
    arguments: dict[str, Any],
) -> Any:
    """Call an MCP tool via FastMCP Client.

    Args:
        server_url: URL of the MCP server
        tool_name: Name of the tool to call
        arguments: Tool arguments

    Returns:
        Tool response
    """
    # FastMCP client connects to the /mcp path
    mcp_url = f"{server_url}/mcp"
    client = Client(mcp_url)

    async with client:
        result = await client.call_tool(tool_name, arguments)
        return result


def call_tool(server_url: str, tool_name: str, arguments: dict[str, Any]) -> Any:
    """Synchronous wrapper for call_tool_async."""
    return asyncio.run(call_tool_async(server_url, tool_name, arguments))


async def list_tools_async(server_url: str) -> list[dict[str, Any]]:
    """List available MCP tools.

    Args:
        server_url: URL of the MCP server

    Returns:
        List of tool definitions
    """
    mcp_url = f"{server_url}/mcp"
    client = Client(mcp_url)

    async with client:
        tools = await client.list_tools()
        return [
            {
                "name": tool.name,
                "description": tool.description,
            }
            for tool in tools
        ]


def list_tools(server_url: str) -> list[dict[str, Any]]:
    """Synchronous wrapper for list_tools_async."""
    return asyncio.run(list_tools_async(server_url))


def health_check(base_url: str) -> dict[str, Any]:
    """Check server health.

    Args:
        base_url: Base URL of the MCP server

    Returns:
        Health status
    """
    url = f"{base_url}/health"

    with httpx.Client() as client:
        response = client.get(url)
        response.raise_for_status()
        return response.json()


def generate_report(server_url: str, company_name: str, enable_tracing: bool = True) -> Any:
    """Generate company intelligence report.

    Args:
        server_url: URL of the MCP server
        company_name: Company to research
        enable_tracing: Whether to enable Langfuse tracing

    Returns:
        Generated report
    """
    return call_tool(
        server_url=server_url,
        tool_name="generate_company_intelligence",
        arguments={
            "company_name": company_name,
            "enable_tracing": enable_tracing,
        },
    )


def search_news(server_url: str, company_name: str, max_results: int = 10) -> Any:
    """Search for company news.

    Args:
        server_url: URL of the MCP server
        company_name: Company to search
        max_results: Maximum results

    Returns:
        Search results
    """
    return call_tool(
        server_url=server_url,
        tool_name="search_company_news",
        arguments={
            "company_name": company_name,
            "max_results": max_results,
        },
    )


def search_trends(
    server_url: str,
    company_name: str,
    industry: str | None = None,
    max_results: int = 10,
) -> Any:
    """Search for industry trends.

    Args:
        server_url: URL of the MCP server
        company_name: Company name
        industry: Industry name
        max_results: Maximum results

    Returns:
        Search results
    """
    args = {
        "company_name": company_name,
        "max_results": max_results,
    }
    if industry:
        args["industry"] = industry

    return call_tool(
        server_url=server_url,
        tool_name="search_industry_trends",
        arguments=args,
    )


def extract_text_content(result: Any) -> str:
    """Extract text content from MCP tool result."""
    if hasattr(result, "content"):
        # Result is a list of content blocks
        texts = []
        for block in result.content:
            if hasattr(block, "text"):
                texts.append(block.text)
        return "\n".join(texts)
    return str(result)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Company Intelligence MCP Client",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate full report
  uv run python client_example.py "トレジャーデータ"

  # Search news only
  uv run python client_example.py "Salesforce" --action news

  # Search industry trends
  uv run python client_example.py "Sony" --action trends --industry エレクトロニクス

  # List available tools
  uv run python client_example.py --list-tools
        """,
    )
    parser.add_argument(
        "company_name",
        nargs="?",
        default="トレジャーデータ",
        help="Company name to research",
    )
    parser.add_argument(
        "--host",
        default="localhost",
        help="MCP server host (default: localhost)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=9100,
        help="MCP server port (default: 9100)",
    )
    parser.add_argument(
        "--action",
        choices=["report", "news", "trends"],
        default="report",
        help="Action to perform (default: report)",
    )
    parser.add_argument(
        "--industry",
        help="Industry name for trends search",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=10,
        help="Maximum search results (default: 10)",
    )
    parser.add_argument(
        "--no-tracing",
        action="store_true",
        help="Disable Langfuse tracing",
    )
    parser.add_argument(
        "--list-tools",
        action="store_true",
        help="List available MCP tools",
    )
    parser.add_argument(
        "--health",
        action="store_true",
        help="Check server health",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON",
    )

    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"

    try:
        if args.health:
            result = health_check(base_url)
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return

        if args.list_tools:
            tools = list_tools(base_url)
            print("Available MCP Tools:")
            print("-" * 40)
            for tool in tools:
                print(f"\n{tool['name']}")
                if tool.get("description"):
                    desc = tool["description"][:100]
                    print(f"  {desc}...")
            return

        print(f"Company: {args.company_name}")
        print(f"Action: {args.action}")
        print(f"Server: {base_url}")
        print()

        if args.action == "report":
            print("Generating intelligence report...")
            print("(This may take a minute or two)")
            print()

            result = generate_report(
                server_url=base_url,
                company_name=args.company_name,
                enable_tracing=not args.no_tracing,
            )

            text = extract_text_content(result)

            if args.json:
                # Try to parse as JSON
                try:
                    data = json.loads(text)
                    print(json.dumps(data, indent=2, ensure_ascii=False))
                except json.JSONDecodeError:
                    print(text)
            else:
                # Try to extract and format the report
                try:
                    data = json.loads(text)
                    print("=" * 60)
                    print(data.get("report_markdown", text))
                    print("=" * 60)
                    if data.get("critique_score"):
                        print(f"\nCritique Score: {data['critique_score']}")
                    if data.get("iterations"):
                        print(f"Iterations: {data['iterations']}")
                    if data.get("sources_count"):
                        print(f"Sources: {data['sources_count']}")
                except json.JSONDecodeError:
                    print(text)

        elif args.action == "news":
            print("Searching for company news...")
            result = search_news(
                server_url=base_url,
                company_name=args.company_name,
                max_results=args.max_results,
            )

            text = extract_text_content(result)

            if args.json:
                print(text)
            else:
                try:
                    data = json.loads(text)
                    print(f"\nFound {data.get('count', 0)} news articles:\n")
                    for i, news in enumerate(data.get("news", []), 1):
                        print(f"{i}. {news.get('title', 'No title')}")
                        print(f"   URL: {news.get('url', '')}")
                        if news.get("published_date"):
                            print(f"   Date: {news['published_date']}")
                        print()
                except json.JSONDecodeError:
                    print(text)

        elif args.action == "trends":
            print("Searching for industry trends...")
            result = search_trends(
                server_url=base_url,
                company_name=args.company_name,
                industry=args.industry,
                max_results=args.max_results,
            )

            text = extract_text_content(result)

            if args.json:
                print(text)
            else:
                try:
                    data = json.loads(text)
                    print(f"\nFound {data.get('count', 0)} trend articles:\n")
                    for i, trend in enumerate(data.get("trends", []), 1):
                        print(f"{i}. {trend.get('title', 'No title')}")
                        print(f"   URL: {trend.get('url', '')}")
                        print()
                except json.JSONDecodeError:
                    print(text)

    except httpx.ConnectError:
        print(f"Could not connect to server at {base_url}")
        print("Make sure the server is running:")
        print("  cd examples/mcp_server && uv run python server.py")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
