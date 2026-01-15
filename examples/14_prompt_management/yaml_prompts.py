"""
YAML Prompt Management Example

This example demonstrates:
- Loading prompts from YAML files using YAMLPromptManager
- Fetching prompts by label (production, staging)
- Fetching prompts by version number
- Rendering prompts with variable substitution
- Using chat prompts for LLM conversations

Expected Output:
    === Text Prompts ===
    Production greeting: Hello Alice, welcome to Graflow!
    Staging greeting: Hi Alice! Testing Graflow features.
    Farewell: Goodbye Alice, thanks for using Graflow!

    === Chat Prompts ===
    Assistant prompt (2 messages):
      [system] You are a helpful assistant specializing in Python.
      [user] Please help me with debugging.

    === Prompt Metadata ===
    Name: greeting
    Label: production
    Version: 2
    Created at: 2024-01-15T10:00:00
    Metadata: {'author': 'team@example.com'}
"""

from pathlib import Path

from graflow.prompts.factory import PromptManagerFactory


def main():
    # Get the prompts directory relative to this file
    prompts_dir = Path(__file__).parent / "prompts"

    # Create YAML prompt manager
    pm = PromptManagerFactory.create("yaml", prompts_dir=str(prompts_dir))

    print("=== Text Prompts ===")

    # Get production prompt (default label)
    greeting = pm.get_text_prompt("greeting")
    rendered = greeting.render(name="Alice", product="Graflow")
    print(f"Production greeting: {rendered}")

    # Get staging prompt by label
    staging_greeting = pm.get_text_prompt("greeting", label="staging")
    rendered = staging_greeting.render(name="Alice", product="Graflow")
    print(f"Staging greeting: {rendered}")

    # Get farewell prompt
    farewell = pm.get_text_prompt("farewell")
    rendered = farewell.render(name="Alice", product="Graflow")
    print(f"Farewell: {rendered}")

    print("\n=== Chat Prompts ===")

    # Get chat prompt for LLM
    assistant = pm.get_chat_prompt("assistant")
    messages = assistant.render(domain="Python", task="debugging")
    print(f"Assistant prompt ({len(messages)} messages):")
    for msg in messages:
        print(f"  [{msg['role']}] {msg['content']}")

    print("\n=== Prompt Metadata ===")

    # Access prompt metadata
    prompt = pm.get_prompt("greeting")
    print(f"Name: {prompt.name}")
    print(f"Label: {prompt.label}")
    print(f"Version: {prompt.version}")
    print(f"Created at: {prompt.created_at}")
    print(f"Metadata: {prompt.metadata}")


if __name__ == "__main__":
    main()
