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
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
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
cd "${ENV_ROOT}"

# Replace absolute paths in prompt with relative ones for the sandbox
PROMPT=$(cat "$PROMPT_FILE" | sed "s|/model/|./model/|g" | sed "s|/data/|./data/|g" | sed "s|/output/|./output/|g")

# Create a CLAUDE.md in the sandbox that enforces boundaries
cat > "${ENV_ROOT}/CLAUDE.md" << 'SANDBOX_EOF'
# SANDBOX RULES — MANDATORY

You are running inside a sandboxed environment for an ML research task.

## Hard constraints
- You MUST only read/write/execute files within the current working directory.
- You MUST NOT cd to any parent directory or absolute path outside this directory.
- You MUST NOT access /home, /etc, /data3, or any path outside this sandbox.
- You MUST NOT use network access (no curl, wget, pip install, git clone, etc).
- You MUST NOT spawn subprocesses that escape this directory.
- All python scripts you write must use relative paths (./model/, ./data/, ./output/).
- If you need a package, it is already installed. Do not try to install anything.

## Available tools
- Python with: torch, torchvision, timm, numpy, scipy, scikit-learn, pandas, PIL, matplotlib
- One GPU (CUDA device 0)
- Bash for running python scripts within this directory
SANDBOX_EOF

# Create .claude/settings.local.json with the sandbox guard hook
GUARD_SCRIPT="${SCRIPT_DIR}/sandbox_guard.sh"
mkdir -p "${ENV_ROOT}/.claude"
cat > "${ENV_ROOT}/.claude/settings.local.json" << HOOKEOF
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": ".*",
        "hooks": [
          {
            "type": "command",
            "command": "SANDBOX_ROOT=${ENV_ROOT} CLAUDE_TOOL_NAME=\$CLAUDE_TOOL_NAME bash ${GUARD_SCRIPT}"
          }
        ]
      }
    ]
  }
}
HOOKEOF

export SANDBOX_ROOT="${ENV_ROOT}"

# Snapshot the workspace every 60s in the background
SNAPSHOT_DIR="${ENV_ROOT}/results/snapshots"
mkdir -p "$SNAPSHOT_DIR"
(
    i=0
    while true; do
        sleep 60
        ts=$(date +%Y%m%d_%H%M%S)
        snap="${SNAPSHOT_DIR}/snap_${i}_${ts}"
        mkdir -p "$snap"
        find "${ENV_ROOT}/output" -type f 2>/dev/null | head -50 > "$snap/output_files.txt"
        find "${ENV_ROOT}" -maxdepth 2 -name "*.py" -newer "$PROMPT_FILE" 2>/dev/null > "$snap/new_scripts.txt"
        for f in $(find "${ENV_ROOT}/output" -type f -size -1M 2>/dev/null); do
            cp "$f" "$snap/" 2>/dev/null || true
        done
        i=$((i + 1))
    done
) &
SNAPSHOT_PID=$!

# Ensure fnm/node/claude are on PATH
export FNM_PATH="/data3/ishaplan/.fnm"
if [ -d "$FNM_PATH" ]; then
    export PATH="$FNM_PATH:$PATH"
    eval "$(fnm env --shell bash)"
fi

# Run Claude Code in non-interactive print mode
# Run twice: once readable for the terminal, saving json separately
# Use text output so you can watch it live
claude -p "$PROMPT" --verbose \
    --allowedTools "Bash,Read,Write,Edit,Glob,Grep" \
    --output-format stream-json \
    > "${ENV_ROOT}/results/trajectory.jsonl" 2>&1 &
CLAUDE_PID=$!

# Print progress while claude runs
echo "Agent running (PID: $CLAUDE_PID)..."
while kill -0 $CLAUDE_PID 2>/dev/null; do
    echo "[$(date +%H:%M:%S)] Agent working... Files in output/:"
    ls -1 "${ENV_ROOT}/output/" 2>/dev/null || echo "  (none yet)"
    echo "  Scripts created:"
    ls -1 "${ENV_ROOT}"/*.py 2>/dev/null || echo "  (none yet)"
    echo ""
    sleep 30
done
wait $CLAUDE_PID

# Stop the snapshot process
kill $SNAPSHOT_PID 2>/dev/null || true

echo ""
echo "============================================="
echo "  Agent finished."
echo "  Trajectory: ${ENV_ROOT}/results/trajectory.json"
echo "  Output:     ${ENV_ROOT}/output/"
echo "============================================="

echo ""
echo "Agent outputs:"
ls -la "${ENV_ROOT}/output/" 2>/dev/null || echo "  (no output files)"
