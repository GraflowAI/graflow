"""
Complete ETL Data Pipeline
===========================

This example demonstrates a real-world ETL (Extract-Transform-Load) pipeline
with data validation and transformation.

Prerequisites:
--------------
None (uses in-memory data)

Concepts Covered:
-----------------
1. Sequential ETL workflow pattern
2. Data validation
3. Data transformation
4. Result aggregation using channels
5. Pipeline summary

Expected Output:
----------------
=== ETL Data Pipeline ===

🚀 Starting ETL pipeline...

📥 Extract: Reading from data sources
   ✅ Customers: 5 records
   ✅ Orders: 8 records
   ✅ Products: 3 records

✓ Validate: Checking data quality
   ✅ Customers: 5 valid (0 invalid)
   ✅ Orders: 8 valid (0 invalid)
   ✅ Products: 3 valid (0 invalid)

🔄 Transform: Enriching data
   ✅ Added email domains to customers
   ✅ Calculated order totals with tax
   ✅ Applied discounts to products

🔀 Aggregate: Combining all data
   ✅ Total records: 16

💾 Load: Saving to warehouse
   ✅ Load complete

📊 Summary: Generating report

=== Pipeline Summary ===
Status: SUCCESS
Total Records: 16
├─ Customers: 5
├─ Orders: 8
└─ Products: 3
Processing Time: 0.4s
✅ Pipeline completed successfully
"""

import time

from graflow.core.context import TaskExecutionContext
from graflow.core.decorators import task
from graflow.core.workflow import workflow


def main():
    """Run the ETL pipeline."""
    print("=== ETL Data Pipeline ===\n")

    with workflow("etl_pipeline") as ctx:

        @task(inject_context=True)
        def extract_data(context: TaskExecutionContext):
            """Extract data from multiple sources."""
            print("🚀 Starting ETL pipeline...\n")
            print("📥 Extract: Reading from data sources")
            time.sleep(0.1)

            # Extract customers
            customers = [
                {"id": 1, "name": "Alice", "email": "alice@example.com"},
                {"id": 2, "name": "Bob", "email": "bob@example.com"},
                {"id": 3, "name": "Charlie", "email": "charlie@example.com"},
                {"id": 4, "name": "Diana", "email": "diana@example.com"},
                {"id": 5, "name": "Eve", "email": "eve@example.com"},
            ]
            print(f"   ✅ Customers: {len(customers)} records")

            # Extract orders
            orders = [
                {"order_id": 101, "customer_id": 1, "amount": 150.0},
                {"order_id": 102, "customer_id": 2, "amount": 200.0},
                {"order_id": 103, "customer_id": 1, "amount": 75.0},
                {"order_id": 104, "customer_id": 3, "amount": 300.0},
                {"order_id": 105, "customer_id": 4, "amount": 125.0},
                {"order_id": 106, "customer_id": 2, "amount": 180.0},
                {"order_id": 107, "customer_id": 5, "amount": 220.0},
                {"order_id": 108, "customer_id": 3, "amount": 90.0},
            ]
            print(f"   ✅ Orders: {len(orders)} records")

            # Extract products
            products = [
                {"product_id": 1, "name": "Widget", "price": 50.0},
                {"product_id": 2, "name": "Gadget", "price": 75.0},
                {"product_id": 3, "name": "Service", "price": 100.0},
            ]
            print(f"   ✅ Products: {len(products)} records\n")

            # Store in channel
            channel = context.get_channel()
            channel.set("customers_raw", customers)
            channel.set("orders_raw", orders)
            channel.set("products_raw", products)

        @task(inject_context=True)
        def validate_data(context: TaskExecutionContext):
            """Validate extracted data."""
            print("✓ Validate: Checking data quality")
            channel = context.get_channel()

            # Validate customers
            customers = channel.get("customers_raw")
            valid_customers = [r for r in customers if "id" in r and "email" in r and "@" in r["email"]]
            print(f"   ✅ Customers: {len(valid_customers)} valid ({len(customers) - len(valid_customers)} invalid)")

            # Validate orders
            orders = channel.get("orders_raw")
            valid_orders = [r for r in orders if "order_id" in r and "amount" in r and r["amount"] > 0]
            print(f"   ✅ Orders: {len(valid_orders)} valid ({len(orders) - len(valid_orders)} invalid)")

            # Validate products
            products = channel.get("products_raw")
            valid_products = [r for r in products if "product_id" in r and "price" in r and r["price"] > 0]
            print(f"   ✅ Products: {len(valid_products)} valid ({len(products) - len(valid_products)} invalid)\n")

            # Store validated data
            channel.set("customers_valid", valid_customers)
            channel.set("orders_valid", valid_orders)
            channel.set("products_valid", valid_products)

        @task(inject_context=True)
        def transform_data(context: TaskExecutionContext):
            """Transform and enrich data."""
            print("🔄 Transform: Enriching data")
            channel = context.get_channel()

            # Transform customers - add email domain
            customers = channel.get("customers_valid")
            customers_transformed = [{**r, "email_domain": r["email"].split("@")[1]} for r in customers]
            print("   ✅ Added email domains to customers")

            # Transform orders - calculate totals with tax
            orders = channel.get("orders_valid")
            orders_transformed = [{**r, "tax": r["amount"] * 0.1, "total": r["amount"] * 1.1} for r in orders]
            print("   ✅ Calculated order totals with tax")

            # Transform products - apply discounts
            products = channel.get("products_valid")
            products_transformed = [{**r, "discounted_price": r["price"] * 0.9} for r in products]
            print("   ✅ Applied discounts to products\n")

            # Store transformed data
            channel.set("customers_transformed", customers_transformed)
            channel.set("orders_transformed", orders_transformed)
            channel.set("products_transformed", products_transformed)

        @task(inject_context=True)
        def aggregate_data(context: TaskExecutionContext):
            """Aggregate all transformed data."""
            print("🔀 Aggregate: Combining all data")
            channel = context.get_channel()

            customers = channel.get("customers_transformed")
            orders = channel.get("orders_transformed")
            products = channel.get("products_transformed")

            total_records = len(customers) + len(orders) + len(products)
            print(f"   ✅ Total records: {total_records}\n")

            # Store aggregated stats
            channel.set("total_records", total_records)
            channel.set("customer_count", len(customers))
            channel.set("order_count", len(orders))
            channel.set("product_count", len(products))

        @task(inject_context=True)
        def load_data(context: TaskExecutionContext):
            """Load data to warehouse."""
            print("💾 Load: Saving to warehouse")
            time.sleep(0.1)  # Simulate I/O
            print("   ✅ Load complete\n")

            channel = context.get_channel()
            channel.set("load_status", "SUCCESS")

        @task(inject_context=True)
        def generate_report(context: TaskExecutionContext):
            """Generate final report."""
            channel = context.get_channel()

            # Get stats from channel
            total = channel.get("total_records")
            customers = channel.get("customer_count")
            orders = channel.get("order_count")
            products = channel.get("product_count")
            status = channel.get("load_status")

            print("📊 Summary: Generating report\n")
            print("=== Pipeline Summary ===")
            print(f"Status: {status}")
            print(f"Total Records: {total}")
            print(f"├─ Customers: {customers}")
            print(f"├─ Orders: {orders}")
            print(f"└─ Products: {products}")

        # Define workflow: sequential pipeline
        extract_data >> validate_data >> transform_data >> aggregate_data >> load_data >> generate_report

        # Execute pipeline
        start_time = time.time()
        ctx.execute("extract_data")
        end_time = time.time()

        # Show timing
        elapsed = end_time - start_time
        print(f"Processing Time: {elapsed:.1f}s")
        print("\n✅ Pipeline completed successfully")


if __name__ == "__main__":
    main()


# ============================================================================
# Key Takeaways:
# ============================================================================
#
# 1. **Sequential ETL Pattern**
#    Extract → Validate → Transform → Aggregate → Load → Report
#    - Clear linear flow
#    - Easy to debug and monitor
#    - Suitable for small to medium datasets
#
# 2. **Data Sharing with Channels**
#    - Tasks use context.get_channel() to access shared storage
#    - channel.set(key, value) to store data
#    - channel.get(key) to retrieve data
#    - Data persists across tasks in the same execution
#
# 3. **Workflow Dependencies**
#    - >> operator for sequential dependencies
#    - Each task completes before the next starts
#    - Simple and predictable execution order
#
# 4. **Real-World ETL Steps**
#    ✅ Extract from multiple sources
#    ✅ Validate data quality
#    ✅ Transform and enrich
#    ✅ Aggregate results
#    ✅ Load to destination
#    ✅ Generate reports
#
# 5. **Error Handling** (Not shown but recommended)
#    - Add try/except in tasks
#    - Store error info in channel
#    - Implement retry logic
#    - Log failures
#
# ============================================================================
# Production Enhancements:
# ============================================================================
#
# **Parallel Processing**:
# - For true parallel extraction, use Redis + workers
# - See examples/05_distributed/distributed_workflow.py
# - Each source can be extracted by different workers
#
# **Error Recovery**:
# - Retry failed tasks
# - Checkpoint progress
# - Dead-letter queue for failed records
# - Alert on failures
#
# **Monitoring**:
# - Track execution time per task
# - Count records at each stage
# - Calculate throughput
# - Store metrics for analysis
#
# **Data Quality**:
# - Schema validation
# - Business rule checks
# - Duplicate detection
# - Anomaly detection
# - Data profiling
#
# **Scalability**:
# - Partition large datasets
# - Use batch processing
# - Implement streaming for real-time data
# - Add caching for lookups
#
# ============================================================================
