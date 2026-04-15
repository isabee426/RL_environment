#!/bin/bash
# =============================================================================
# Run the Claude Code agent inside the sandboxed environment.
#
# The agent ONLY sees the env_root directory. It cannot touch anything else.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=1 bash agent/run_agent.sh /data3/ishaplan/pref_model_env
# =============================================================================

set -euo pipefail

ENV_ROOT="${1:?Usage: run_agent.sh <env_root>}"
PROMPT_FILE="${ENV_ROOT}/prompt.txt"

if [ ! -f "$PROMPT_FILE" ]; then
    echo "ERROR: prompt.txt not found at ${PROMPT_FILE}"
    echo "Did you run setup.py first?"
    exit 1
fi

# Ensure output directory exists
mkdir -p "${ENV_ROOT}/output"

echo "============================================="
echo "  Starting agent in: ${ENV_ROOT}"
echo "  GPU: ${CUDA_VISIBLE_DEVICES:-all}"
echo "  Prompt: ${PROMPT_FILE}"
echo "============================================="

# The key: cd into the env_root so the agent's working directory is the sandbox.
# All relative paths in the prompt (e.g., /model/, /data/) resolve within env_root.
# --allowedTools restricts to bash + file ops (no web, no agents).
cd "${ENV_ROOT}"

# Replace absolute paths in prompt with relative ones for the sandbox
PROMPT=$(cat "$PROMPT_FILE" | sed "s|/model/|./model/|g" | sed "s|/data/|./data/|g" | sed "s|/output/|./output/|g")

# Run Claude Code in non-interactive print mode
# --max-turns limits how many tool calls the agent can make (prevent runaway)
claude -p "$PROMPT" \
    --allowedTools "Bash(command:*),Read,Write,Edit,Glob,Grep" \
    --output-format json \
    > "${ENV_ROOT}/results/trajectory.json" 2>&1

echo ""
echo "============================================="
echo "  Agent finished."
echo "  Trajectory: ${ENV_ROOT}/results/trajectory.json"
echo "  Output:     ${ENV_ROOT}/output/"
echo "============================================="

# Check what the agent produced
echo ""
echo "Agent outputs:"
ls -la "${ENV_ROOT}/output/" 2>/dev/null || echo "  (no output files)"
