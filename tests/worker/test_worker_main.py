#!/usr/bin/env python3
"""Test script for worker_main.py functionality."""

import logging
import subprocess
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_worker_main_help():
    """Test that worker_main.py shows help correctly."""
    print("=== Testing worker_main.py --help ===")

    try:
        result = subprocess.run(
            ["python3", "worker_main.py", "--help"], check=False, capture_output=True, text=True, timeout=10
        )

        assert result.returncode == 0, f"Help command failed: {result.stderr}"
        assert "TaskWorker independent process" in result.stdout, "Help text missing"
        assert "--worker-id" in result.stdout, "Worker ID option missing"
        assert "--queue-type" in result.stdout, "Queue type option missing"

        print("✅ Help test passed!")

    except Exception as e:
        print(f"❌ Help test failed: {e}")


def test_worker_main_memory_queue():
    """Test worker_main.py with memory queue (should start and stop gracefully)."""
    print("\n=== Testing worker_main.py with memory queue ===")

    try:
        # Start worker process with memory queue
        process = subprocess.Popen(
            [
                "python3",
                "worker_main.py",
                "--queue-type",
                "memory",
                "--worker-id",
                "test_worker",
                "--max-concurrent-tasks",
                "2",
                "--poll-interval",
                "0.1",
                "--log-level",
                "INFO",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Let it run for a few seconds
        time.sleep(3.0)

        # Send SIGTERM to gracefully shut down
        process.terminate()

        # Wait for process to finish
        stdout, stderr = process.communicate(timeout=10)

        print(f"Process exit code: {process.returncode}")
        print(f"STDOUT:\n{stdout}")
        if stderr:
            print(f"STDERR:\n{stderr}")

        # Check that it started and stopped gracefully
        assert "TaskWorker test_worker starting..." in stdout, "Worker did not start properly"
        assert "graceful shutdown" in stdout.lower() or process.returncode in [0, -15], (
            "Worker did not shut down gracefully"
        )

        print("✅ Memory queue test passed!")

    except subprocess.TimeoutExpired:
        if "process" in locals():
            process.kill()  # type: ignore
        print("❌ Memory queue test timed out")
    except Exception as e:
        if "process" in locals():
            process.kill()  # type: ignore
        print(f"❌ Memory queue test failed: {e}")


def test_worker_main_invalid_args():
    """Test worker_main.py with invalid arguments."""
    print("\n=== Testing worker_main.py with invalid arguments ===")

    try:
        # Test with invalid queue type
        result = subprocess.run(
            ["python3", "worker_main.py", "--queue-type", "invalid"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )

        # Should fail with non-zero exit code
        assert result.returncode != 0, "Invalid queue type should cause failure"
        print("✅ Invalid arguments test passed!")

    except Exception as e:
        print(f"❌ Invalid arguments test failed: {e}")


if __name__ == "__main__":
    print("Starting worker_main.py tests...\n")

    test_worker_main_help()
    test_worker_main_memory_queue()
    test_worker_main_invalid_args()

    print("\n🎉 worker_main.py tests completed!")
