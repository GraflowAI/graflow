"""Sales Data Analysis Workflow

Comprehensive workflow to analyze sales data, detect anomalies, create detailed reports, and send after approval
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from graflow.core.decorators import task
from graflow.core.workflow import workflow


class SalesAnalyzer:
    """Sales data analysis class"""

    def __init__(self):
        self.data: Optional[pd.DataFrame] = None
        self.anomalies: List[Dict] = []
        self.report: Dict[str, Any] = {}

    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load sales data"""
        try:
            self.data = pd.read_csv(file_path)
            print(f"Data loading complete: {len(self.data)} records")
            return self.data
        except Exception as e:
            print(f"Data loading error: {e}")
            raise

    def detect_anomalies(self, threshold: float = 2.0) -> List[Dict]:
        """Anomaly detection (Z-score method)"""
        if self.data is None:
            raise ValueError("Data not loaded")

        # Assume sales amount column
        if 'amount' not in self.data.columns:
            print("Warning: 'amount' column not found. Using sample data")
            return []

        z_scores = np.abs((self.data['amount'] - self.data['amount'].mean()) / self.data['amount'].std())
        anomaly_indices = z_scores > threshold

        self.anomalies = []
        for idx in self.data[anomaly_indices].index:
            anomaly = {
                'index': int(idx),
                'amount': float(self.data.loc[idx, 'amount']),
                'z_score': float(z_scores[idx]),
                'date': str(self.data.loc[idx, 'date']) if 'date' in self.data.columns else 'N/A'
            }
            self.anomalies.append(anomaly)

        print(f"Anomalies detected: {len(self.anomalies)} records")
        return self.anomalies

    def generate_report(self) -> Dict[str, Any]:
        """Generate detailed report"""
        if self.data is None:
            raise ValueError("Data not loaded")

        self.report = {
            'timestamp': datetime.now().isoformat(),
            'data_summary': {
                'total_records': len(self.data),
                'total_amount': float(self.data['amount'].sum()) if 'amount' in self.data.columns else 0,
                'avg_amount': float(self.data['amount'].mean()) if 'amount' in self.data.columns else 0,
                'max_amount': float(self.data['amount'].max()) if 'amount' in self.data.columns else 0,
                'min_amount': float(self.data['amount'].min()) if 'amount' in self.data.columns else 0
            },
            'anomaly_analysis': {
                'anomaly_count': len(self.anomalies),
                'anomaly_rate': len(self.anomalies) / len(self.data) if len(self.data) > 0 else 0,
                'anomalies': self.anomalies
            }
        }

        print(f"Report generation complete: {len(self.anomalies)} anomalies")
        return self.report


class ApprovalManager:
    """Approval management class"""

    @staticmethod
    def request_approval(report: Dict[str, Any]) -> bool:
        """Request approval (in actual implementation, integrate with external system)"""
        anomaly_count = report.get('anomaly_analysis', {}).get('anomaly_count', 0)

        if anomaly_count == 0:
            print("No anomalies - auto approval")
            return True
        elif anomaly_count <= 5:
            print(f"Minor anomalies ({anomaly_count} records) - auto approval")
            return True
        else:
            print(f"Critical anomalies ({anomaly_count} records) - manual approval required")
            # In actual implementation, call external approval system API
            return False


class ReportSender:
    """Report sending class"""

    @staticmethod
    def send_report(report: Dict[str, Any], approved: bool) -> bool:
        """Send report"""
        try:
            # In actual implementation, send to external systems (email, Slack, etc.)
            output_file = f"sales_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

            report_data = {
                'report': report,
                'approval_status': 'approved' if approved else 'pending',
                'sent_at': datetime.now().isoformat()
            }

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, ensure_ascii=False, indent=2)

            print(f"Report sending complete: {output_file}")
            return True
        except Exception as e:
            print(f"Sending error: {e}")
            return False


def main():
    """Main execution function"""
    print("=== Sales Data Analysis Workflow ===\n")

    # Shared data storage
    analyzer = SalesAnalyzer()
    workflow_data = {}

    with workflow("sales_analysis_pipeline") as ctx:

        @task
        def load_sales_data():
            """Load sales data"""
            print("📊 Loading sales data...")

            # Create sample data (in actual use, specify external file)
            sample_data = pd.DataFrame({
                'date': pd.date_range('2024-01-01', periods=100, freq='D'),
                'amount': np.random.normal(10000, 2000, 100)  # mean 10000, std 2000
            })

            # Intentionally add anomalies
            sample_data.loc[10, 'amount'] = 50000  # abnormally high value
            sample_data.loc[20, 'amount'] = 1000   # abnormally low value

            analyzer.data = sample_data
            workflow_data['data_loaded'] = True
            print("✅ Data loading complete")

        @task
        def analyze_anomalies():
            """Anomaly analysis"""
            print("🔍 Analyzing anomalies...")

            if not workflow_data.get('data_loaded'):
                raise ValueError("Data not loaded")

            anomalies = analyzer.detect_anomalies(threshold=2.0)
            workflow_data['anomalies_detected'] = len(anomalies) > 0
            workflow_data['anomaly_count'] = len(anomalies)
            print("✅ Anomaly analysis complete")

        @task
        def generate_detailed_report():
            """Generate detailed report"""
            print("📋 Generating detailed report...")

            if not workflow_data.get('anomalies_detected', False):
                print("ℹ️  No anomalies detected")

            report = analyzer.generate_report()
            workflow_data['report'] = report
            workflow_data['report_generated'] = True
            print("✅ Report generation complete")

        @task
        def request_approval():
            """Request approval"""
            print("✋ Requesting approval...")

            if not workflow_data.get('report_generated'):
                raise ValueError("Report not generated")

            report = workflow_data['report']
            approved = ApprovalManager.request_approval(report)
            workflow_data['approved'] = approved
            print(f"✅ Approval result: {'Approved' if approved else 'Requires approval'}")

        @task
        def send_report():
            """Send report"""
            print("📤 Sending report...")

            if not workflow_data.get('approved', False):
                print("❌ Skipping send due to lack of approval")
                return

            report = workflow_data['report']
            success = ReportSender.send_report(report, True)
            workflow_data['sent'] = success

            if success:
                print("✅ Report sending complete")
            else:
                print("❌ Report sending failed")

        @task
        def cleanup():
            """Cleanup"""
            print("🧹 Cleaning up...")

            # Delete temporary files, log records, etc.
            summary = {
                'Data Loading': workflow_data.get('data_loaded', False),
                'Anomalies Detected': workflow_data.get('anomaly_count', 0),
                'Report Generated': workflow_data.get('report_generated', False),
                'Approval Status': workflow_data.get('approved', False),
                'Send Complete': workflow_data.get('sent', False)
            }

            print("📊 Workflow Execution Summary:")
            for key, value in summary.items():
                print(f"  {key}: {value}")

            print("✅ Cleanup complete")

        # Build workflow
        load_sales_data >> analyze_anomalies >> generate_detailed_report >> request_approval >> send_report >> cleanup

        # Display workflow information
        ctx.show_info()

        # Execute
        print("\n🚀 Starting workflow execution\n")
        ctx.execute("load_sales_data")

    print("\n🎉 Sales data analysis workflow complete!")


if __name__ == "__main__":
    main()
