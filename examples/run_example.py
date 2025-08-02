#!/usr/bin/env python3
"""Simple runner for graflow examples."""

import os
import sys

# Add the parent directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if __name__ == "__main__":
    if len(sys.argv) != 2:  # noqa: PLR2004
        print("Usage: python run_example.py <example_name>")
        print("Available examples:")
        examples_dir = os.path.dirname(__file__)
        for file in os.listdir(examples_dir):
            if file.endswith(".py"):
                print(f"  - {file[:-3]}")
        sys.exit(1)

    example_name = sys.argv[1]
    if not example_name.endswith(".py"):
        example_name += ".py"

    example_path = os.path.join(os.path.dirname(__file__), example_name)

    if not os.path.exists(example_path):
        print(f"Error: Example '{example_name}' not found")
        sys.exit(1)

    print(f"=== Running {example_name} ===")
    with open(example_path) as f:
        code = f.read()

    exec(code)
