#!/usr/bin/env bash
# git push（pre-push）与 CI 共用的统一入口。
set -euo pipefail
cd "$(dirname "$0")/../.."

echo "==> Full lint (ruff / black / isort / mypy)"
bash scripts/ci/lint.sh

echo "==> Unit tests"
bash scripts/ci/unit-tests.sh

echo "==> All pre-push checks passed"
