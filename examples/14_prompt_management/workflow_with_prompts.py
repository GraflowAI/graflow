"""
Workflow with Prompts Example

This example demonstrates:
- Using prompts within Graflow tasks via context injection
- Passing prompt_manager to workflow context
- Accessing prompts through TaskExecutionContext
- Using channels for parameter sharing between tasks

Expected Output:
    === Customer Onboarding Workflow ===

    [setup] Initializing workflow parameters

    [greet_customer] Greeting:
    Hello Alice, welcome to Graflow!

    [generate_assistant_messages] Assistant messages (2):
      [system] You are a helpful assistant specializing in Python.
      [user] Please help me with onboarding.

    [send_farewell] Farewell:
    Goodbye Alice, thanks for using Graflow!

    === Workflow Complete ===
"""

from pathlib import Path

from graflow.core.context import TaskExecutionContext
from graflow.core.decorators import task
from graflow.core.workflow import workflow
from graflow.prompts.factory import PromptManagerFactory


def main():
    """Run workflow with prompt templates."""
    # Get the prompts directory relative to this file
    prompts_dir = Path(__file__).parent / "prompts"

    # Create YAML prompt manager
    pm = PromptManagerFactory.create("yaml", prompts_dir=str(prompts_dir))

    print("=== Customer Onboarding Workflow ===\n")

    # Create workflow with prompt manager
    # All tasks within this context can access pm via context.prompt_manager
    with workflow("customer_onboarding", prompt_manager=pm) as ctx:

        @task(inject_context=True)
        def setup(context: TaskExecutionContext):
            """Initialize workflow parameters in channel."""
            print("[setup] Initializing workflow parameters\n")
            channel = context.get_channel()
            channel.set("customer_name", "Alice")
            channel.set("product_name", "Graflow")
            channel.set("domain", "Python")
            channel.set("task_description", "onboarding")

        @task(inject_context=True)
        def greet_customer(context: TaskExecutionContext) -> str:
            """Greet customer using a prompt template."""
            pm = context.prompt_manager
            channel = context.get_channel()

            customer_name = channel.get("customer_name")
            product_name = channel.get("product_name")

            # Get production greeting prompt
            prompt = pm.get_text_prompt("greeting", label="production")

            # Render with variables
            greeting = prompt.render(name=customer_name, product=product_name)

            print("[greet_customer] Greeting:")
            print(greeting)

            channel.set("greeting", greeting)
            return greeting

        @task(inject_context=True)
        def generate_assistant_messages(context: TaskExecutionContext) -> list:
            """Generate assistant messages using a chat prompt template."""
            pm = context.prompt_manager
            channel = context.get_channel()

            domain = channel.get("domain")
            task_description = channel.get("task_description")

            # Get chat prompt for assistant
            prompt = pm.get_chat_prompt("assistant", label="production")

            # Render to message list (ready to send to LLM API)
            messages = prompt.render(domain=domain, task=task_description)

            print(f"\n[generate_assistant_messages] Assistant messages ({len(messages)}):")
            for msg in messages:
                print(f"  [{msg['role']}] {msg['content']}")

            channel.set("messages", messages)
            return messages

        @task(inject_context=True)
        def send_farewell(context: TaskExecutionContext) -> str:
            """Send farewell message using prompt template."""
            pm = context.prompt_manager
            channel = context.get_channel()

            customer_name = channel.get("customer_name")
            product_name = channel.get("product_name")

            # Get farewell prompt
            prompt = pm.get_text_prompt("farewell", label="production")

            # Render farewell
            farewell = prompt.render(name=customer_name, product=product_name)

            print("\n[send_farewell] Farewell:")
            print(farewell)

            channel.set("farewell", farewell)
            return farewell

        # Define workflow: setup -> greet -> generate messages -> farewell
        setup >> greet_customer >> generate_assistant_messages >> send_farewell  # type: ignore

        # Execute the workflow and get execution context
        _, exec_context = ctx.execute("setup", ret_context=True)

    # Retrieve results from channel
    channel = exec_context.channel
    greeting = channel.get("greeting")
    messages = channel.get("messages")
    farewell = channel.get("farewell")

    print("\n=== Workflow Complete ===")
    print(f"Greeting: {greeting}")
    print(f"Messages: {len(messages)} messages generated")
    print(f"Farewell: {farewell}")


if __name__ == "__main__":
    main()
