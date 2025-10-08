#!/usr/bin/env python3
"""
TypedChannel Data Exchange Example

This example demonstrates how to use get_typed_channel() for type-safe
inter-task communication with structured data schemas.
"""

import traceback
from typing import List

from typing_extensions import TypedDict

from graflow.core.context import TaskExecutionContext
from graflow.core.decorators import task
from graflow.core.workflow import workflow


# Simple TypedDict schemas
class UserData(TypedDict):
    """Schema for user information."""
    user_id: str
    name: str
    email: str
    age: int


class ProcessedData(TypedDict):
    """Schema for processed data."""
    user_id: str
    processed_name: str
    age_category: str
    email_domain: str


def demo_simple_typed_exchange():
    """Simple demonstration of typed data exchange."""
    print("🔒 Simple TypedChannel Data Exchange Demo")
    print("=" * 50)

    with workflow("simple_typed_exchange") as wf:

        @task(inject_context=True)
        def create_user_data(context: TaskExecutionContext) -> None:
            """Create user data and send via TypedChannel."""
            print("📊 Creating user data...")

            # Get typed channel for user data
            user_channel = context.get_typed_channel(UserData)

            # Sample users
            users: List[UserData] = [
                {"user_id": "U001", "name": "Alice", "email": "alice@example.com", "age": 25},
                {"user_id": "U002", "name": "Bob", "email": "bob@gmail.com", "age": 35}
            ]

            # Send each user via typed channel
            for user in users:
                user_channel.send(f"user_{user['user_id']}", user)
                print(f"  📤 Sent user data for {user['user_id']}: {user['name']}")

        @task(inject_context=True)
        def process_user_data(context: TaskExecutionContext) -> None:
            """Process user data using TypedChannel."""
            print("\n⚙️ Processing user data...")

            # Get typed channels
            user_channel = context.get_typed_channel(UserData)
            processed_channel = context.get_typed_channel(ProcessedData)

            # Process all available user data
            for key in user_channel.keys():
                if key.startswith("user_"):
                    user_data = user_channel.receive(key)
                    if user_data:
                        # Process the data
                        processed_data: ProcessedData = {
                            "user_id": user_data["user_id"],
                            "processed_name": user_data["name"].upper(),
                            "age_category": "adult" if user_data["age"] >= 18 else "minor",
                            "email_domain": user_data["email"].split("@")[1]
                        }

                        # Send processed data
                        processed_channel.send(f"processed_{user_data['user_id']}", processed_data)
                        print(f"  ⚙️ Processed {user_data['user_id']}: "
                              f"{processed_data['processed_name']} ({processed_data['age_category']})")

        @task(inject_context=True)
        def display_results(context: TaskExecutionContext) -> None:
            """Display processed results."""
            print("\n📋 Displaying results...")

            # Get processed data channel
            processed_channel = context.get_typed_channel(ProcessedData)

            # Display all processed data
            for key in processed_channel.keys():
                if key.startswith("processed_"):
                    processed_data = processed_channel.receive(key)
                    if processed_data:
                        print(f"  📄 {processed_data['user_id']}: "
                              f"Name={processed_data['processed_name']}, "
                              f"Category={processed_data['age_category']}, "
                              f"Domain={processed_data['email_domain']}")

        # Build workflow
        create_user_data >> process_user_data >> display_results # type: ignore

        # Execute workflow
        wf.execute("create_user_data")


def demo_type_validation():
    """Demonstrate type validation with TypedChannel."""
    print("\n🛡️ Type Validation Demo")
    print("=" * 50)

    with workflow("type_validation_demo") as wf:

        @task(inject_context=True)
        def test_valid_data(context: TaskExecutionContext) -> None:
            """Test sending valid data."""
            print("✅ Testing valid data...")

            user_channel = context.get_typed_channel(UserData)

            # Valid user data
            valid_user: UserData = {
                "user_id": "V001",
                "name": "Valid User",
                "email": "valid@example.com",
                "age": 30
            }

            try:
                user_channel.send("valid_user", valid_user)
                print("  ✅ Valid data sent successfully")
            except Exception as e:
                print(f"  ❌ Unexpected error: {e}")

        @task(inject_context=True)
        def check_received_data(context: TaskExecutionContext) -> None:
            """Check what data was received."""
            print("\n🔍 Checking received data...")

            user_channel = context.get_typed_channel(UserData)

            # Check available data
            for key in user_channel.keys():
                data = user_channel.receive(key)
                if data:
                    print(f"  📥 Received {key}: {data}")

        # Build workflow
        test_valid_data >> check_received_data # type: ignore

        # Execute workflow
        wf.execute("test_valid_data")


def main():
    """Run TypedChannel examples."""
    print("🔒 TypedChannel Inter-Task Communication Examples")
    print("This demonstrates type-safe data exchange using get_typed_channel()")

    try:
        # Simple demo
        demo_simple_typed_exchange()

        # Type validation demo
        demo_type_validation()

        print("\n🎉 All TypedChannel examples completed successfully!")

    except Exception as e:
        print(f"❌ Error running examples: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
