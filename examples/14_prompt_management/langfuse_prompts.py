"""
Langfuse Prompt Management Example

This example demonstrates:
- Loading prompts from Langfuse cloud/server
- Fetching prompts by label (production, staging)
- Fetching prompts by version number
- Rendering prompts with variable substitution
- Configuring fetch timeout and retries

Prerequisites:
    1. Install langfuse: pip install langfuse (or pip install graflow[tracing])
    2. Set environment variables:
       export LANGFUSE_PUBLIC_KEY=pk-lf-...
       export LANGFUSE_SECRET_KEY=sk-lf-...
       export LANGFUSE_HOST=http://localhost:3000  # For local Langfuse

    3. Create prompts in Langfuse dashboard:
       - Name: "greeting" (text type)
         Content: "Hello {{name}}, welcome to {{product}}!"
         Label: "production"

       - Name: "assistant" (chat type)
         Content:
           - role: system, content: "You are a helpful assistant."
           - role: user, content: "Help me with {{task}}."
         Label: "production"

Usage:
    PYTHONPATH=. uv run python examples/14_prompt_management/langfuse_prompts.py
"""

import os
import sys

from dotenv import load_dotenv

from graflow.prompts.factory import PromptManagerFactory


def main():
    # Load environment variables from .env file
    load_dotenv()

    # Check if langfuse credentials are set
    if not os.getenv("LANGFUSE_PUBLIC_KEY") or not os.getenv("LANGFUSE_SECRET_KEY"):
        print("Error: Langfuse credentials not set.")
        print()
        print("Please set the following environment variables:")
        print("  export LANGFUSE_PUBLIC_KEY=pk-lf-...")
        print("  export LANGFUSE_SECRET_KEY=sk-lf-...")
        print("  export LANGFUSE_HOST=https://cloud.langfuse.com  # optional")
        sys.exit(1)

    # Create Langfuse prompt manager
    try:
        pm = PromptManagerFactory.create(
            "langfuse",
            fetch_timeout_seconds=10,  # 10 second timeout
            max_retries=2,  # Retry up to 2 times on failure
        )
    except ValueError as e:
        print(f"Error: {e}")
        print("Install langfuse with: pip install langfuse")
        sys.exit(1)

    print("=== Langfuse Prompt Management ===\n")

    # Example 1: Get text prompt by label (default: production)
    print("1. Text Prompt (production label):")
    try:
        greeting = pm.get_text_prompt("greeting")
        rendered = greeting.render(name="Alice", product="Graflow")
        print(f"   Content: {rendered}")
        print(f"   Version: {greeting.version}")
    except Exception as e:
        print(f"   Error: {e}")
        print("   (Create a 'greeting' prompt in Langfuse dashboard)")

    # Example 2: Get prompt by specific version
    print("\n2. Text Prompt (by version):")
    try:
        greeting_v1 = pm.get_text_prompt("greeting", version=1)
        print(f"   Version 1 content: {greeting_v1.content}")
    except Exception as e:
        print(f"   Error: {e}")

    # Example 3: Get chat prompt for LLM
    print("\n3. Chat Prompt:")
    try:
        assistant = pm.get_chat_prompt("assistant")
        messages = assistant.render(task="debugging Python code")
        print(f"   Messages ({len(messages)}):")
        for msg in messages:
            print(f"     [{msg['role']}] {msg['content'][:50]}...")
    except Exception as e:
        print(f"   Error: {e}")
        print("   (Create an 'assistant' chat prompt in Langfuse dashboard)")

    # Example 4: Using cache TTL
    print("\n4. With custom cache TTL:")
    try:
        # Cache for 5 minutes
        prompt = pm.get_prompt("greeting", cache_ttl_seconds=300)
        print(f"   Cached prompt: {prompt.name} (v{prompt.version})")
    except Exception as e:
        print(f"   Error: {e}")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
