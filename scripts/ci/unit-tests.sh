#!/usr/bin/env bash
# Level 2（git push）与 CI：实际收集整个 tests，不做 marker/目录过滤。
# *_smoke.py 虽不直接被 pytest 收集，仍可能被测试作为子进程启动。
# MUSA/集成用例按各自条件执行；运行前检查设备与 TMA_RUN_INTEGRATION 等开关。
set -euo pipefail
cd "$(dirname "$0")/../.."

python3 -m pytest tests -q
