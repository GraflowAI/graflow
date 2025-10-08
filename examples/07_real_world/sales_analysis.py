"""
Sales Analysis Workflow
========================

This example demonstrates a complete data analysis workflow for sales data,
including data loading, anomaly detection, report generation, approval process,
and executive summary creation.

Prerequisites:
--------------
pip install pandas numpy

Concepts Covered:
-----------------
1. Data loading and preparation
2. Statistical anomaly detection
3. Multi-stage report generation
4. Human-in-the-loop approval process
5. Sequential workflow with channels for data sharing

Expected Output:
----------------
=== Sales Analysis Workflow ===

Workflow structure:
Name: sales_analysis
Tasks: 5
Dependencies: 4

📊 Loading sales data...
✅ Loaded 1000 sales records
Date range: 2024-01-15 to 2025-01-14
Total sales: $1,234,567.89

🔍 Detecting anomalies...
✅ Analysis complete:
  - Outliers detected: 48
  - Suspicious days: 12
  - Detection threshold: $2,450.75

📋 Generating detailed report...
✅ Detailed report generated
⚠️  Found 48 outlier transactions
⚠️  Found 12 suspicious days

✋ Requesting approval...
=== APPROVAL REQUEST ===
Report generated at: 2025-01-14T10:30:00
Total records analyzed: 1000
Outliers found: 48
Suspicious patterns: 12

Recommendations:
1. Review 48 outlier transactions for potential errors or fraud
2. Investigate 12 days with unusually high sales

[AUTO-APPROVED for demonstration]
✅ Report approved - proceeding to final generation

📄 Generating executive summary...
✅ Executive summary generated

SALES ANALYSIS EXECUTIVE SUMMARY
Generated: 2025-01-14T10:30:00

DATA OVERVIEW:
- Total Records: 1,000
- Date Range: 2024-01-15 to 2025-01-14
- Total Sales: $1,234,567.89
- Average Sale: $1,234.57

ANOMALY DETECTION RESULTS:
- Outlier Transactions: 48
- Suspicious Days: 12

RECOMMENDED ACTIONS:
1. Review 48 outlier transactions for potential errors or fraud
2. Investigate 12 days with unusually high sales

Status: APPROVED
Next Review: Quarterly

🎉 Sales analysis workflow completed successfully!
"""

import random
from datetime import datetime, timedelta
from typing import Any, Dict

try:
    import numpy as np
    import pandas as pd
    DEPS_AVAILABLE = True
except ImportError:
    DEPS_AVAILABLE = False

from graflow.core.context import TaskExecutionContext
from graflow.core.decorators import task
from graflow.core.workflow import workflow


def check_dependencies():
    """Check if required dependencies are installed."""
    if not DEPS_AVAILABLE:
        print("❌ Missing dependencies")
        print("\nThis example requires pandas and numpy:")
        print("  pip install pandas numpy")
        print("\nSkipping sales analysis example...")
        return False
    return True


def generate_sample_sales_data(num_records: int = 1000) -> 'pd.DataFrame':
    """Generate sample sales data for demonstration."""
    start_date = datetime.now() - timedelta(days=365)

    data = []
    for _ in range(num_records):
        date = start_date + timedelta(days=random.randint(0, 365))

        # Base sales amount with some seasonality
        base_amount = 1000 + 500 * np.sin(date.month * 2 * np.pi / 12)

        # Add some anomalies (5% chance)
        if random.random() < 0.05:
            amount = base_amount * random.choice([0.1, 5.0])  # Very low or very high
        else:
            amount = base_amount * (1 + random.gauss(0, 0.2))  # Normal variation

        data.append({
            'date': date.strftime('%Y-%m-%d'),
            'amount': max(0, amount),
            'customer_id': f'CUST_{random.randint(1, 100):03d}',
            'product_category': random.choice(['Electronics', 'Clothing', 'Books', 'Home']),
            'region': random.choice(['North', 'South', 'East', 'West'])
        })

    return pd.DataFrame(data)


def detect_anomalies(df: 'pd.DataFrame') -> Dict[str, Any]:
    """Detect anomalies in sales data using statistical methods."""
    anomalies = {
        'outliers': [],
        'suspicious_patterns': [],
        'summary': {}
    }

    # Calculate statistics
    mean_amount = df['amount'].mean()
    std_amount = df['amount'].std()
    threshold = mean_amount + 3 * std_amount

    # Find outliers (amounts more than 3 standard deviations from mean)
    outliers = df[df['amount'] > threshold]
    anomalies['outliers'] = outliers.to_dict('records')

    # Check for suspicious patterns
    daily_sales = df.groupby('date')['amount'].sum()
    daily_mean = daily_sales.mean()
    daily_std = daily_sales.std()

    suspicious_days = daily_sales[daily_sales > daily_mean + 2 * daily_std]
    anomalies['suspicious_patterns'] = [
        {
            'date': date,
            'total_amount': float(amount),
            'deviation': float(amount - daily_mean)
        }
        for date, amount in suspicious_days.items()
    ]

    # Summary statistics
    anomalies['summary'] = {
        'total_records': len(df),
        'outlier_count': len(outliers),
        'suspicious_days': len(suspicious_days),
        'mean_amount': float(mean_amount),
        'std_amount': float(std_amount),
        'threshold': float(threshold)
    }

    return anomalies


def generate_detailed_report(df: 'pd.DataFrame', anomalies: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a detailed analysis report."""
    report = {
        'timestamp': datetime.now().isoformat(),
        'data_summary': {
            'total_records': len(df),
            'date_range': {
                'start': df['date'].min(),
                'end': df['date'].max()
            },
            'total_sales': float(df['amount'].sum()),
            'average_sale': float(df['amount'].mean())
        },
        'anomaly_analysis': anomalies,
        'recommendations': []
    }

    # Generate recommendations based on findings
    if anomalies['summary']['outlier_count'] > 0:
        report['recommendations'].append(
            f"Review {anomalies['summary']['outlier_count']} outlier transactions for potential errors or fraud"
        )

    if anomalies['summary']['suspicious_days'] > 0:
        report['recommendations'].append(
            f"Investigate {anomalies['summary']['suspicious_days']} days with unusually high sales"
        )

    if anomalies['summary']['outlier_count'] == 0 and anomalies['summary']['suspicious_days'] == 0:
        report['recommendations'].append("No significant anomalies detected. Data appears normal.")

    return report


def request_approval(report: Dict[str, Any]) -> bool:
    """Simulate approval request process."""
    print("\n=== APPROVAL REQUEST ===")
    print(f"Report generated at: {report['timestamp']}")
    print(f"Total records analyzed: {report['data_summary']['total_records']}")
    print(f"Outliers found: {report['anomaly_analysis']['summary']['outlier_count']}")
    print(f"Suspicious patterns: {report['anomaly_analysis']['summary']['suspicious_days']}")

    print("\nRecommendations:")
    for i, rec in enumerate(report['recommendations'], 1):
        print(f"{i}. {rec}")

    # Auto-approve for demo (in real scenario, this would be human input)
    print("\n[AUTO-APPROVED for demonstration]")
    return True


def generate_final_report(detailed_report: Dict[str, Any], approved: bool) -> str:
    """Generate final executive summary report."""
    if not approved:
        return "Report generation cancelled - approval not granted."

    summary = f"""
SALES ANALYSIS EXECUTIVE SUMMARY
Generated: {detailed_report['timestamp']}

DATA OVERVIEW:
- Total Records: {detailed_report['data_summary']['total_records']:,}
- Date Range: {detailed_report['data_summary']['date_range']['start']} to {detailed_report['data_summary']['date_range']['end']}
- Total Sales: ${detailed_report['data_summary']['total_sales']:,.2f}
- Average Sale: ${detailed_report['data_summary']['average_sale']:.2f}

ANOMALY DETECTION RESULTS:
- Outlier Transactions: {detailed_report['anomaly_analysis']['summary']['outlier_count']}
- Suspicious Days: {detailed_report['anomaly_analysis']['summary']['suspicious_days']}

RECOMMENDED ACTIONS:
"""

    for i, rec in enumerate(detailed_report['recommendations'], 1):
        summary += f"{i}. {rec}\n"

    summary += "\nStatus: APPROVED\nNext Review: Quarterly"

    return summary


def main():
    """Main function to execute the sales analysis workflow."""
    print("=== Sales Analysis Workflow ===\n")

    # Check dependencies
    if not check_dependencies():
        return

    with workflow("sales_analysis") as ctx:

        @task(inject_context=True)
        def load_sales_data(context: TaskExecutionContext):
            """Load and prepare sales data."""
            print("📊 Loading sales data...")
            df = generate_sample_sales_data(1000)

            # Store in channel
            channel = context.get_channel()
            channel.set('sales_data', df)

            print(f"✅ Loaded {len(df)} sales records")
            print(f"Date range: {df['date'].min()} to {df['date'].max()}")
            print(f"Total sales: ${df['amount'].sum():,.2f}")

        @task(inject_context=True)
        def detect_data_anomalies(context: TaskExecutionContext):
            """Detect anomalies in the loaded data."""
            print("\n🔍 Detecting anomalies...")

            channel = context.get_channel()
            df = channel.get('sales_data')
            anomalies = detect_anomalies(df)
            channel.set('anomalies', anomalies)

            print("✅ Analysis complete:")
            print(f"  - Outliers detected: {anomalies['summary']['outlier_count']}")
            print(f"  - Suspicious days: {anomalies['summary']['suspicious_days']}")
            print(f"  - Detection threshold: ${anomalies['summary']['threshold']:.2f}")

        @task(inject_context=True)
        def create_detailed_report(context: TaskExecutionContext):
            """Generate detailed analysis report."""
            print("\n📋 Generating detailed report...")

            channel = context.get_channel()
            df = channel.get('sales_data')
            anomalies = channel.get('anomalies')

            report = generate_detailed_report(df, anomalies)
            channel.set('detailed_report', report)

            print("✅ Detailed report generated")

            # Show some key findings
            if report['anomaly_analysis']['summary']['outlier_count'] > 0:
                print(f"⚠️  Found {report['anomaly_analysis']['summary']['outlier_count']} outlier transactions")

            if report['anomaly_analysis']['summary']['suspicious_days'] > 0:
                print(f"⚠️  Found {report['anomaly_analysis']['summary']['suspicious_days']} suspicious days")

        @task(inject_context=True)
        def approval_process(context: TaskExecutionContext):
            """Request approval for report and recommendations."""
            print("\n✋ Requesting approval...")

            channel = context.get_channel()
            report = channel.get('detailed_report')
            approved = request_approval(report)
            channel.set('approved', approved)

            if approved:
                print("✅ Report approved - proceeding to final generation")
            else:
                print("❌ Report not approved - process halted")

        @task(inject_context=True)
        def generate_executive_summary(context: TaskExecutionContext):
            """Generate final executive summary."""
            print("\n📄 Generating executive summary...")

            channel = context.get_channel()
            report = channel.get('detailed_report')
            approved = channel.get('approved')

            final_report = generate_final_report(report, approved)
            channel.set('final_report', final_report)

            print("✅ Executive summary generated")
            print("\n" + "="*60)
            print(final_report)
            print("="*60)

        # Build the workflow pipeline
        load_sales_data >> detect_data_anomalies >> create_detailed_report >> approval_process >> generate_executive_summary

        # Show workflow structure
        print("Workflow structure:")
        ctx.show_info()
        print()

        # Execute the workflow
        ctx.execute("load_sales_data", max_steps=10)

    print("\n🎉 Sales analysis workflow completed successfully!")


if __name__ == "__main__":
    main()


# ============================================================================
# Key Takeaways:
# ============================================================================
#
# 1. **Multi-Stage Analysis Workflow**
#    Load → Detect → Report → Approve → Summarize
#    - Clear separation of concerns
#    - Each stage builds on previous results
#    - Data flows through channels
#
# 2. **Statistical Anomaly Detection**
#    - Mean and standard deviation analysis
#    - Outlier detection (3-sigma rule)
#    - Pattern recognition in time series
#    - Actionable insights generation
#
# 3. **Human-in-the-Loop Pattern**
#    - Automated analysis up to approval step
#    - Human review before final actions
#    - Simulation of approval for demo
#    - Production would integrate actual approval system
#
# 4. **Channel-Based Data Sharing**
#    - Store intermediate results in channels
#    - Share data across workflow tasks
#    - No need for global variables
#    - Clean data flow architecture
#
# 5. **Real-World Data Processing**
#    ✅ Statistical analysis
#    ✅ Anomaly detection
#    ✅ Report generation
#    ✅ Approval workflows
#    ✅ Executive summaries
#
# ============================================================================
# Production Enhancements:
# ============================================================================
#
# **Real Data Sources**:
# @task(inject_context=True)
# def load_sales_data(context):
#     # Load from database
#     df = pd.read_sql(
#         "SELECT * FROM sales WHERE date >= NOW() - INTERVAL '1 year'",
#         connection
#     )
#     context.get_channel().set('sales_data', df)
#
# **Advanced Anomaly Detection**:
# - Machine learning models (Isolation Forest, DBSCAN)
# - Seasonal decomposition
# - Multivariate analysis
# - Real-time detection
#
# **Interactive Approval**:
# def request_approval(report):
#     # Send to approval system
#     approval_id = approval_system.create_request(report)
#
#     # Wait for human approval
#     approved = approval_system.wait_for_decision(approval_id)
#     return approved
#
# **Notification System**:
# @task(inject_context=True)
# def send_notifications(context):
#     report = context.get_channel().get('detailed_report')
#
#     if report['anomaly_analysis']['summary']['outlier_count'] > 10:
#         send_email(
#             to="analyst@company.com",
#             subject="High number of anomalies detected",
#             body=format_alert(report)
#         )
#
# **Report Distribution**:
# @task(inject_context=True)
# def distribute_report(context):
#     final_report = context.get_channel().get('final_report')
#
#     # Save to shared location
#     save_to_s3(f"reports/{date}/executive_summary.txt", final_report)
#
#     # Send to stakeholders
#     email_report(final_report, recipients=["ceo@company.com", "cfo@company.com"])
#
#     # Update dashboard
#     dashboard.update_latest_report(final_report)
#
# ============================================================================
# Advanced Analysis Patterns:
# ============================================================================
#
# **Trend Analysis**:
# def analyze_trends(df):
#     # Monthly trend analysis
#     monthly = df.groupby(pd.Grouper(key='date', freq='M'))['amount'].agg([
#         ('total', 'sum'),
#         ('average', 'mean'),
#         ('count', 'count')
#     ])
#
#     # Calculate growth rates
#     monthly['growth'] = monthly['total'].pct_change()
#     return monthly
#
# **Customer Segmentation**:
# def segment_customers(df):
#     # RFM analysis
#     customer_stats = df.groupby('customer_id').agg({
#         'date': lambda x: (datetime.now() - pd.to_datetime(x).max()).days,
#         'amount': ['count', 'sum', 'mean']
#     })
#
#     # Segment based on behavior
#     segments = classify_customers(customer_stats)
#     return segments
#
# **Forecasting**:
# def forecast_sales(df, periods=30):
#     from statsmodels.tsa.holtwinters import ExponentialSmoothing
#
#     # Prepare time series
#     ts = df.groupby('date')['amount'].sum()
#
#     # Fit model
#     model = ExponentialSmoothing(ts, seasonal='add', seasonal_periods=7)
#     fit = model.fit()
#
#     # Generate forecast
#     forecast = fit.forecast(periods)
#     return forecast
#
# **Geographic Analysis**:
# def analyze_by_region(df):
#     regional_stats = df.groupby('region').agg({
#         'amount': ['sum', 'mean', 'count'],
#         'customer_id': 'nunique'
#     })
#
#     # Calculate regional performance
#     regional_stats['avg_customer_value'] = (
#         regional_stats[('amount', 'sum')] /
#         regional_stats[('customer_id', 'nunique')]
#     )
#     return regional_stats
#
# ============================================================================
# Integration Patterns:
# ============================================================================
#
# **Scheduled Execution**:
# # Using cron or scheduler
# from apscheduler.schedulers.blocking import BlockingScheduler
#
# scheduler = BlockingScheduler()
# scheduler.add_job(main, 'cron', day_of_week='mon', hour=8)
# scheduler.start()
#
# **Database Integration**:
# @task(inject_context=True)
# def save_anomalies_to_db(context):
#     anomalies = context.get_channel().get('anomalies')
#
#     for outlier in anomalies['outliers']:
#         db.execute(
#             "INSERT INTO anomaly_log (date, amount, customer_id, flagged_at) "
#             "VALUES (?, ?, ?, ?)",
#             outlier['date'], outlier['amount'],
#             outlier['customer_id'], datetime.now()
#         )
#
# **Dashboard Integration**:
# @task(inject_context=True)
# def update_dashboard(context):
#     report = context.get_channel().get('detailed_report')
#
#     # Push metrics to dashboard
#     dashboard_api.update_metrics({
#         'total_sales': report['data_summary']['total_sales'],
#         'anomaly_count': report['anomaly_analysis']['summary']['outlier_count'],
#         'last_updated': datetime.now().isoformat()
#     })
#
# ============================================================================
# Testing and Validation:
# ============================================================================
#
# **Unit Testing**:
# def test_anomaly_detection():
#     # Create test data with known anomalies
#     test_df = pd.DataFrame({
#         'amount': [100, 110, 105, 1000, 95],  # 1000 is outlier
#         'date': ['2024-01-01'] * 5
#     })
#
#     anomalies = detect_anomalies(test_df)
#     assert anomalies['summary']['outlier_count'] == 1
#
# **Integration Testing**:
# def test_full_workflow():
#     with workflow("test_sales") as ctx:
#         # Define workflow
#         load >> detect >> report >> approve >> summarize
#
#         # Execute
#         ctx.execute("load", max_steps=10)
#
#         # Verify results
#         final_report = ctx.get_channel().get('final_report')
#         assert "APPROVED" in final_report
#
# ============================================================================
