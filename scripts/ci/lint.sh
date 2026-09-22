#!/usr/bin/env bash
# Level 2（git push）与 CI：完整静态检查 + 类型检查。
set -euo pipefail
cd "$(dirname "$0")/../.."

ruff check src tests
black --check --quiet src tests
isort --check-only src tests
mypy          # 配置见 pyproject.toml [tool.mypy]（python3.10, src, ignore_missing_imports）

# 接入 C/C++ 原生扩展时在此加入：
# clang-format --dry-run --Werror $(git ls-files '*.c' '*.h' '*.cu')
