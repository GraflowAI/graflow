"""
Typed Channels Example
=======================

This example demonstrates how to use TypedChannels for type-safe inter-task
communication. TypedChannels provide:
- Compile-time type checking
- IDE autocomplete and IntelliSense
- Self-documenting message schemas
- Runtime validation (when desired)

Concepts Covered:
-----------------
1. Defining message schemas with TypedDict
2. Creating TypedChannel instances
3. Type-safe message passing
4. IDE support and autocomplete
5. Multiple message types in one workflow
6. Schema documentation

Expected Output:
----------------
=== Typed Channels Demo ===

Starting execution from: collect_user_data
👤 Collect User Data
   Created user profile: {'user_id': 'user_123', 'name': 'Alice', 'email': 'alice@example.com', 'age': 30}

📊 Calculate Metrics
   Retrieved user profile for: Alice
   Created metrics: {'user_id': 'user_123', 'login_count': 42, 'last_active': 1234567890.0, 'premium': True}

📝 Generate Report
   Retrieved user profile for: Alice (alice@example.com)
   Retrieved metrics: 42 logins, Premium: True
   === User Report ===
   User: Alice (user_123)
   Email: alice@example.com
   Age: 30
   Logins: 42
   Premium: ✅ Yes

Execution completed after 3 steps

Type-safe communication successful! 🎉
"""

import time
from typing import TypedDict

from graflow.core.context import TaskExecutionContext
from graflow.core.decorators import task
from graflow.core.workflow import workflow

# ============================================================================
# Message Schema Definitions
# ============================================================================

class UserProfile(TypedDict):
    """User profile information.

    This schema defines the structure of user profile messages.
    TypedDict provides type hints and IDE support.
    """
    user_id: str
    name: str
    email: str
    age: int


class UserMetrics(TypedDict):
    """User activity metrics.

    This schema defines user engagement and activity data.
    """
    user_id: str
    login_count: int
    last_active: float
    premium: bool


# ============================================================================
# Workflow Tasks
# ============================================================================

def main():
    """Demonstrate typed channel operations."""
    print("=== Typed Channels Demo ===\n")

    with workflow("typed_channel_demo") as ctx:

        @task(inject_context=True)
        def collect_user_data(context: TaskExecutionContext):
            """Collect user data and store in typed channel."""
            print("👤 Collect User Data")

            # Get a type-safe channel for UserProfile messages
            # TypedChannel provides type checking and IDE autocomplete
            profile_channel = context.get_typed_channel(UserProfile)

            # Create a user profile message
            # IDE will autocomplete the fields and check types
            user_profile: UserProfile = {
                "user_id": "user_123",
                "name": "Alice",
                "email": "alice@example.com",
                "age": 30
            }

            # Store the typed message
            profile_channel.set("current_user", user_profile)
            print(f"   Created user profile: {user_profile}\n")

        @task(inject_context=True)
        def calculate_metrics(context: TaskExecutionContext):
            """Calculate metrics for the user."""
            print("📊 Calculate Metrics")

            # Retrieve user profile from typed channel
            profile_channel = context.get_typed_channel(UserProfile)
            user_profile = profile_channel.get("current_user")

            # IDE knows the structure of user_profile
            # You get autocomplete for .name, .email, etc.
            print(f"   Retrieved user profile for: {user_profile['name']}")

            # Create metrics using the typed channel
            metrics_channel = context.get_typed_channel(UserMetrics)

            user_metrics: UserMetrics = {
                "user_id": user_profile["user_id"],
                "login_count": 42,
                "last_active": time.time(),
                "premium": True
            }

            metrics_channel.set("user_metrics", user_metrics)
            print(f"   Created metrics: {user_metrics}\n")

        @task(inject_context=True)
        def generate_report(context: TaskExecutionContext):
            """Generate a report combining user data and metrics."""
            print("📝 Generate Report")

            # Retrieve both typed messages
            profile_channel = context.get_typed_channel(UserProfile)
            metrics_channel = context.get_typed_channel(UserMetrics)

            user_profile = profile_channel.get("current_user")
            user_metrics = metrics_channel.get("user_metrics")

            # IDE knows the types and provides autocomplete
            print(f"   Retrieved user profile for: {user_profile['name']} ({user_profile['email']})")
            print(f"   Retrieved metrics: {user_metrics['login_count']} logins, Premium: {user_metrics['premium']}")

            # Generate report
            print("   === User Report ===")
            print(f"   User: {user_profile['name']} ({user_profile['user_id']})")
            print(f"   Email: {user_profile['email']}")
            print(f"   Age: {user_profile['age']}")
            print(f"   Logins: {user_metrics['login_count']}")
            print(f"   Premium: {'✅ Yes' if user_metrics['premium'] else '❌ No'}\n")

        # Define workflow
        collect_user_data >> calculate_metrics >> generate_report

        # Execute
        ctx.execute("collect_user_data")

    print("Type-safe communication successful! 🎉")


if __name__ == "__main__":
    main()


# ============================================================================
# Key Takeaways:
# ============================================================================
#
# 1. **Defining Message Schemas**
#    class MyMessage(TypedDict):
#        field1: str
#        field2: int
#
#    - Use TypedDict to define message structure
#    - Add docstrings to document the schema
#    - Fields are type-annotated
#
# 2. **Creating TypedChannels**
#    typed_channel = context.get_typed_channel(MyMessage)
#
#    - Pass the TypedDict class (not an instance)
#    - Creates a type-safe wrapper around the regular channel
#    - One TypedChannel per message type
#
# 3. **Setting Typed Messages**
#    message: MyMessage = {"field1": "value", "field2": 42}
#    typed_channel.set("key", message)
#
#    - IDE will autocomplete field names
#    - Type checker will verify field types
#    - Runtime behavior is the same as regular channels
#
# 4. **Getting Typed Messages**
#    message = typed_channel.get("key")
#    value = message["field1"]  # IDE knows this is a str
#
#    - IDE knows the return type
#    - Autocomplete works on message fields
#    - Type safety throughout your code
#
# 5. **Benefits of TypedChannels**
#    ✅ Compile-time type checking
#    ✅ IDE autocomplete and IntelliSense
#    ✅ Self-documenting code
#    ✅ Catch errors before runtime
#    ✅ Easier refactoring
#
# 6. **When to Use TypedChannels**
#    ✅ Complex message structures
#    ✅ Team development (shared schemas)
#    ✅ Long-lived codebases
#    ✅ API-like contracts between tasks
#
# 7. **When to Use Regular Channels**
#    ✅ Simple key-value pairs
#    ✅ Prototyping and quick experiments
#    ✅ Dynamic data structures
#    ✅ Single-developer projects
#
# ============================================================================
# Try Experimenting:
# ============================================================================
#
# 1. Define your own message schema:
#    class OrderMessage(TypedDict):
#        order_id: str
#        items: list[str]
#        total: float
#        status: str
#
# 2. Use optional fields with NotRequired:
#    from typing import NotRequired
#
#    class ConfigMessage(TypedDict):
#        required_field: str
#        optional_field: NotRequired[str]
#
# 3. Nest TypedDicts:
#    class Address(TypedDict):
#        street: str
#        city: str
#
#    class User(TypedDict):
#        name: str
#        address: Address
#
# 4. Create multiple typed channels:
#    profile_ch = ctx.get_typed_channel(UserProfile)
#    metrics_ch = ctx.get_typed_channel(UserMetrics)
#    orders_ch = ctx.get_typed_channel(OrderData)
#
# 5. Document your schemas:
#    class ApiResponse(TypedDict):
#        """Response from external API.
#
#        Fields:
#            status_code: HTTP status code
#            data: Response payload
#            timestamp: Request timestamp
#        """
#        status_code: int
#        data: dict
#        timestamp: float
#
# ============================================================================
# Real-World Use Cases:
# ============================================================================
#
# **API Integration**:
# Define schemas for API requests and responses, ensuring type safety
#
# **Data Pipelines**:
# Each stage has defined input/output schemas, creating clear contracts
#
# **Event Processing**:
# Define event schemas for different event types in your system
#
# **Configuration Management**:
# Typed config messages ensure all required fields are present
#
# **Multi-Team Development**:
# Shared schemas serve as API contracts between team components
#
# **Microservices Communication**:
# Message schemas define the contract between services
#
# ============================================================================
# Type Checking Tools:
# ============================================================================
#
# Use these tools to enforce type safety:
#
# **mypy**:
#   pip install mypy
#   mypy your_script.py
#
# **pyright** (used by VS Code):
#   npm install -g pyright
#   pyright your_script.py
#
# **IDE Integration**:
# - VS Code: Install Python extension
# - PyCharm: Built-in type checking
# - Both provide real-time type hints and errors
#
# ============================================================================
