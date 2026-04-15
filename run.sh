#!/bin/bash
# =============================================================================
# Master script: setup -> agent -> judge
#
# Usage:
#   CUDA_VISIBLE_DEVICES=1 bash run.sh /data3/ishaplan/pref_model_env
#
# Steps:
#   1. Set up the environment (download data, fine-tune ViT, create sandbox)
#   2. Run the Claude agent inside the sandbox
#   3. Run the judge on the agent's output
# =============================================================================

set -euo pipefail

ENV_ROOT="${1:?Usage: run.sh <env_root>}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "==============================="
echo "  Step 1: Environment Setup"
echo "==============================="
python "${SCRIPT_DIR}/environment/setup.py" --env_root "${ENV_ROOT}"

# Copy prompt into the sandbox
cp "${SCRIPT_DIR}/agent/prompt.txt" "${ENV_ROOT}/prompt.txt"
mkdir -p "${ENV_ROOT}/results"

echo ""
echo "==============================="
echo "  Step 2: Run Agent"
echo "==============================="
bash "${SCRIPT_DIR}/agent/run_agent.sh" "${ENV_ROOT}"

echo ""
echo "==============================="
echo "  Step 3: Judge"
echo "==============================="
python "${SCRIPT_DIR}/judge/judge.py" --env_root "${ENV_ROOT}"

echo ""
echo "==============================="
echo "  Done!"
echo "==============================="
