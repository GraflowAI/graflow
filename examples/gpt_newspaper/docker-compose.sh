#!/usr/bin/env bash

# Helper wrapper to ensure docker compose commands run with the
# gpt_newspaper directory as the base (project) directory.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

exec docker compose "$@"
