#!/usr/bin/env bash
# 新开发环境一次性初始化：安装 .git/hooks/pre-commit 与 pre-push。
set -euo pipefail

if ! command -v git >/dev/null 2>&1; then
    echo "ERROR: git is not installed in the development container."
    exit 1
fi
if ! command -v pre-commit >/dev/null 2>&1; then
    echo "ERROR: pre-commit is not installed in the development container."
    exit 1
fi

pre-commit install --install-hooks

echo
echo "Git hooks installed successfully:"
echo "  - pre-commit  -> scripts/ci/quick-check.sh"
echo "  - pre-push    -> scripts/ci/lint.sh + scripts/ci/unit-tests.sh"
