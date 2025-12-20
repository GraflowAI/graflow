"""Basic example of using PydanticLLMAgent with structured output and automatic keyword argument resolution.

This example demonstrates:
1. Creating a Pydantic AI agent with structured output
2. Wrapping it with PydanticLLMAgent for Graflow integration
3. Using it in a Graflow task with automatic keyword argument resolution from channel
4. The 'text' parameter is automatically resolved from the channel (resolve_keyword_args=True by default)

Requirements:
    pip install graflow[pydantic-ai]
    export OPENAI_API_KEY=sk-...

Run:
    PYTHONPATH=. python examples/11_llm_integration/pydantic_agent_basic.py
"""

from typing import List

from pydantic import BaseModel
from pydantic_ai import Agent

from graflow.core.context import TaskExecutionContext
from graflow.core.decorators import task
from graflow.core.workflow import workflow
from graflow.llm.agents import PydanticLLMAgent


# Define structured output schema
class SentimentAnalysis(BaseModel):
    """Sentiment analysis result."""

    sentiment: str  # "positive", "negative", or "neutral"
    confidence: float  # 0.0 to 1.0
    explanation: str


def main() -> None:
    """Run sentiment analysis example."""
    # Create Pydantic AI agent with structured output
    # Direct provider syntax: 'provider:model'
    pydantic_agent = Agent(
        model="openai:gpt-5-mini",
        output_type=SentimentAnalysis,
        system_prompt="You are a sentiment analysis expert. Analyze the sentiment of the given text.",
    )

    # Wrap for Graflow
    agent = PydanticLLMAgent(pydantic_agent, name="sentiment_analyzer")

    # Define task that uses automatic keyword argument resolution
    # The 'text' parameter will be automatically resolved from the channel
    def analyze_sentiment(text: str) -> None:
        """Analyze sentiment of text.

        Args:
            text: Text to analyze (automatically resolved from channel)
        """
        print(f"\nText: {text}")
        print("-" * 80)

        # Run agent on text
        result = agent.run(text)
        # result["output"] is a validated SentimentAnalysis instance
        output: SentimentAnalysis = result["output"]

        print(f"Sentiment: {output.sentiment}")
        print(f"Confidence: {output.confidence:.2f}")
        print(f"Explanation: {output.explanation}")

        if result["metadata"].get("usage"):
            usage = result["metadata"]["usage"]
            print(f"Tokens: {usage.get('total_tokens', 'N/A')}")

    # Define setup task to populate channel with test data
    @task(inject_context=True)
    def setup_test_data(context: TaskExecutionContext) -> None:
        """Setup test data in channel."""
        texts = [
            "I absolutely loved this product! It exceeded all my expectations.",
            "This is the worst experience I've ever had. Completely disappointed.",
            "It's okay, nothing special but does the job.",
        ]

        print("Sentiment Analysis Results")
        print("=" * 80)

        # Store texts in channel for tasks to consume
        channel = context.get_channel()
        channel.set("texts", texts)

    # Create and execute workflow
    with workflow("sentiment_analysis") as wf:
        # Define task to process each text
        # Note: Recommended to define tasks within the workflow context while tasks defined outside also work
        @task
        def process_texts(texts: List[str]) -> None:
            """Process all test texts."""
            for text in texts:
                # Execute analysis (text automatically resolved from channel)
                analyze_sentiment(text)

        _ = setup_test_data >> process_texts
        wf.execute()


if __name__ == "__main__":
    main()
