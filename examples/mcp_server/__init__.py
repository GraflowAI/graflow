"""Company Intelligence MCP Server.

This module provides an MCP server that exposes company intelligence
workflow as tools that can be called from MCP clients.

Example usage:
    # Start the server
    PYTHONPATH=. python -m examples.mcp_company_intel.server

    # Or with uvicorn
    PYTHONPATH=. uvicorn examples.mcp_company_intel.server:app --host 0.0.0.0 --port 9100
"""

from .config import Config
from .workflow import create_company_intelligence_workflow, run_workflow

__all__ = [
    "Config",
    "create_company_intelligence_workflow",
    "run_workflow",
]
