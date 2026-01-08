"""Agent serialization helpers for distributed execution.

This module provides helpers to serialize Google ADK LlmAgent to YAML format
for distributed execution across worker processes. ADK agents can be complex
objects with tools, sub-agents, and configuration. We use Pydantic's model_dump
and model_validate methods for serialization.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import yaml

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from google.adk.agents import LlmAgent

# Optional imports
try:
    from google.adk.agents import LlmAgent

    ADK_AVAILABLE = True
except ImportError:
    ADK_AVAILABLE = False


def agent_to_yaml(agent: LlmAgent) -> str:  # type: ignore[valid-type]
    """Convert Google ADK LlmAgent to YAML string.

    This function uses Pydantic's model_dump() to convert a LlmAgent
    (with all its tools, sub-agents, and configuration) to a YAML string
    that can be safely pickled and sent to worker processes.

    Args:
        agent: Google ADK LlmAgent instance

    Returns:
        YAML string representation of the agent

    Raises:
        ImportError: If Google ADK is not installed
        TypeError: If agent is not a LlmAgent instance

    Example:
        ```python
        from google.adk.agents import LlmAgent
        from graflow.llm.serialization import agent_to_yaml

        agent = LlmAgent(
            name="supervisor",
            model="gemini-2.5-flash",
            tools=[search_tool]
        )

        yaml_str = agent_to_yaml(agent)
        # Can now pickle yaml_str and send to workers
        ```
    """
    if not ADK_AVAILABLE:
        raise ImportError("Google ADK is not installed. Install with: pip install google-adk")

    if not isinstance(agent, LlmAgent):
        raise TypeError(f"agent must be a LlmAgent instance, got {type(agent)}")

    try:
        # Use Pydantic's model_dump to get dict representation
        config_dict = agent.model_dump(mode="python")
        # Convert dict to YAML string
        yaml_str = yaml.dump(config_dict, default_flow_style=False, sort_keys=False)
        logger.debug(f"Serialized agent '{agent.name}' to YAML ({len(yaml_str)} chars)")
        return yaml_str
    except Exception as e:
        logger.error(f"Failed to serialize agent to YAML: {e}")
        raise


def yaml_to_agent(yaml_str: str) -> LlmAgent:  # type: ignore[valid-type]
    """Convert YAML string back to Google ADK LlmAgent.

    This function reconstructs a LlmAgent from a YAML string created by
    agent_to_yaml(). It uses Pydantic's model_validate() method which handles
    deserialization of tools, sub-agents, and all configuration.

    Args:
        yaml_str: YAML string representation of the agent

    Returns:
        Reconstructed LlmAgent instance

    Raises:
        ImportError: If Google ADK is not installed
        ValueError: If YAML is invalid or cannot be deserialized

    Example:
        ```python
        from graflow.llm.serialization import yaml_to_agent

        # In worker process, reconstruct agent from YAML
        agent = yaml_to_agent(yaml_str)
        result = agent.run("What is the weather?")
        ```
    """
    if not ADK_AVAILABLE:
        raise ImportError("Google ADK is not installed. Install with: pip install google-adk")

    try:
        # Parse YAML to dict
        config_dict = yaml.safe_load(yaml_str)
        if not isinstance(config_dict, dict):
            raise ValueError("YAML must deserialize to a dictionary")

        # Reconstruct agent from dict using Pydantic's model_validate
        agent = LlmAgent.model_validate(config_dict)
        logger.debug(f"Deserialized agent '{agent.name}' from YAML")
        return agent
    except Exception as e:
        logger.error(f"Failed to deserialize agent from YAML: {e}")
        raise ValueError(f"Invalid agent YAML: {e}") from e
