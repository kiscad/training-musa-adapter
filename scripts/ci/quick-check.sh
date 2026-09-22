#!/usr/bin/env bash
# Level 1（git commit）：快速静态检查，控制在几秒内。
set -euo pipefail
cd "$(dirname "$0")/../.."

ruff check src tests
black --check --quiet src tests
isort --check-only src tests
