"""Sales analysis workflow example.

This workflow demonstrates:
1. Loading sales data
2. Detecting anomalies in the data
3. Generating detailed reports
4. Requesting approval for actions
5. Generating final reports
"""

import random
from datetime import datetime, timedelta
from typing import Any, Dict

import numpy as np
import pandas as pd

from graflow.core.decorators import task
from graflow.core.workflow import workflow


def generate_sample_sales_data(num_records: int = 1000) -> pd.DataFrame:
    """Generate sample sales data for demonstration."""
    start_date = datetime.now() - timedelta(days=365)

    data = []
    for i in range(num_records):
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


def detect_anomalies(df: pd.DataFrame) -> Dict[str, Any]:
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


def generate_detailed_report(df: pd.DataFrame, anomalies: Dict[str, Any]) -> Dict[str, Any]:
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

    with workflow("sales_analysis") as ctx:
        # Shared data storage
        workflow_data = {}

        @task
        def load_sales_data():
            """Load and prepare sales data."""
            print("📊 Loading sales data...")
            df = generate_sample_sales_data(1000)
            workflow_data['sales_data'] = df
            print(f"✅ Loaded {len(df)} sales records")
            print(f"Date range: {df['date'].min()} to {df['date'].max()}")
            print(f"Total sales: ${df['amount'].sum():,.2f}")

        @task
        def detect_data_anomalies():
            """Detect anomalies in the loaded data."""
            print("\n🔍 Detecting anomalies...")
            df = workflow_data['sales_data']
            anomalies = detect_anomalies(df)
            workflow_data['anomalies'] = anomalies

            print(f"✅ Analysis complete:")
            print(f"  - Outliers detected: {anomalies['summary']['outlier_count']}")
            print(f"  - Suspicious days: {anomalies['summary']['suspicious_days']}")
            print(f"  - Detection threshold: ${anomalies['summary']['threshold']:.2f}")

        @task
        def create_detailed_report():
            """Generate detailed analysis report."""
            print("\n📋 Generating detailed report...")
            df = workflow_data['sales_data']
            anomalies = workflow_data['anomalies']
            report = generate_detailed_report(df, anomalies)
            workflow_data['detailed_report'] = report
            print("✅ Detailed report generated")

            # Show some key findings
            if report['anomaly_analysis']['summary']['outlier_count'] > 0:
                print(f"⚠️  Found {report['anomaly_analysis']['summary']['outlier_count']} outlier transactions")

            if report['anomaly_analysis']['summary']['suspicious_days'] > 0:
                print(f"⚠️  Found {report['anomaly_analysis']['summary']['suspicious_days']} suspicious days")

        @task
        def approval_process():
            """Request approval for report and recommendations."""
            print("\n✋ Requesting approval...")
            report = workflow_data['detailed_report']
            approved = request_approval(report)
            workflow_data['approved'] = approved

            if approved:
                print("✅ Report approved - proceeding to final generation")
            else:
                print("❌ Report not approved - process halted")

        @task
        def generate_executive_summary():
            """Generate final executive summary."""
            print("\n📄 Generating executive summary...")
            report = workflow_data['detailed_report']
            approved = workflow_data['approved']
            final_report = generate_final_report(report, approved)
            workflow_data['final_report'] = final_report

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