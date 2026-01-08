"""Advanced example of PydanticLLMAgent with tools and automatic keyword argument resolution.

This example demonstrates:
1. Using create_pydantic_ai_agent_with_litellm() helper with LiteLLM backend
2. Registering tools with @agent.tool decorator
3. Multi-turn conversation with message history
4. Automatic keyword argument resolution from channel (resolve_keyword_args=True by default)
5. The 'query' and 'history' parameters are automatically resolved from the channel

Requirements:
    pip install graflow[pydantic-ai]
    export OPENAI_API_KEY=sk-...

Run:
    PYTHONPATH=. python examples/11_llm_integration/pydantic_agent_with_tools.py
"""

from datetime import datetime

from pydantic import BaseModel
from pydantic_ai import RunContext

from graflow.core.context import TaskExecutionContext
from graflow.core.decorators import task
from graflow.core.workflow import workflow
from graflow.llm.agents import PydanticLLMAgent, create_pydantic_ai_agent_with_litellm


# Define structured output
class WeatherReport(BaseModel):
    """Weather report for a location."""

    location: str
    temperature: float
    condition: str
    timestamp: str


# Create agent using helper function (LiteLLM backend)
# Alternative: You can create Agent directly with Agent('openai/gpt-5-mini', ...)
pydantic_agent = create_pydantic_ai_agent_with_litellm(
    model="openai/gpt-5-mini",  # LiteLLM format: 'provider/model'
    system_prompt="You are a helpful weather assistant with access to weather data.",
)


# Register tools with the agent
@pydantic_agent.tool  # type: ignore
def get_weather(ctx: RunContext[int], city: str) -> dict:
    """Get current weather for a city.

    Args:
        city: Name of the city

    Returns:
        Weather data dictionary
    """
    # Simulate weather API call
    weather_data = {
        "Tokyo": {"temp": 15.5, "condition": "Partly cloudy"},
        "New York": {"temp": 8.2, "condition": "Rainy"},
        "London": {"temp": 12.0, "condition": "Foggy"},
        "Paris": {"temp": 10.8, "condition": "Clear"},
    }

    data = weather_data.get(city, {"temp": 20.0, "condition": "Unknown"})
    return {
        "location": city,
        "temperature": data["temp"],
        "condition": data["condition"],
        "timestamp": datetime.now().isoformat(),
    }


@pydantic_agent.tool  # type: ignore
def get_forecast(ctx: RunContext[int], city: str, days: int = 3) -> dict:
    """Get weather forecast for a city.

    Args:
        city: Name of the city
        days: Number of days to forecast (default: 3)

    Returns:
        Forecast data dictionary
    """
    # Simulate forecast API call
    return {
        "location": city,
        "days": days,
        "forecast": [{"day": i + 1, "temp": 15 + i * 0.5, "condition": "Partly cloudy"} for i in range(days)],
    }


def main() -> None:
    """Run weather assistant example."""
    # Wrap agent for Graflow
    agent = PydanticLLMAgent(pydantic_agent, name="weather_assistant")

    print(f"Agent: {agent.name}")
    print(f"Tools: {len(agent.tools)} registered")
    print(f"Metadata: {agent.metadata}")
    print("=" * 80)

    # Create and execute workflow
    with workflow("weather_assistant") as wf:
        # Define task that uses automatic keyword argument resolution
        # Both 'query' and 'history' are automatically resolved from channel
        @task(inject_context=True)
        def ask_weather(ctx: TaskExecutionContext, query: str, history: list | None = None) -> None:
            """Ask the weather assistant a question.

            Args:
                query: User query (automatically resolved from channel)
                history: Message history (automatically resolved from channel if present)
            """
            result = agent.run(query, message_history=history)

            print(f"Response: {result['output']}")
            if result["metadata"].get("usage"):
                print(f"Tokens: {result['metadata']['usage'].get('total_tokens', 'N/A')}")

            # Store result for next interaction
            channel = ctx.get_channel()  # Access channel through context
            channel.set("last_messages", result["metadata"]["messages"])

        ask_tokyo = ask_weather(task_id="ask_tokyo", query="What's the weather in Tokyo?")
        ask_multi_city = ask_weather(
            task_id="ask_multi_city", query="Compare the weather in Tokyo, New York, and London"
        )
        ask_multi_turn = ask_weather(task_id="ask_multi_turn", query="What's the weather in Paris?")
        ask_follow_up = ask_weather(
            task_id="ask_follow_up",
            query="And what about the 5-day forecast?",
            history=ask_multi_turn.outputs["metadata"]["messages"],
        )
        _ = ask_tokyo >> ask_multi_city >> ask_multi_turn >> ask_follow_up
        wf.execute()


if __name__ == "__main__":
    main()
