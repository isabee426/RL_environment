#!/bin/bash
# =============================================================================
# Sandbox Guard Hook
# Runs before every tool call. Kills the agent if it tries to escape.
#
# Reads the tool input from stdin (JSON), checks all file paths and commands
# against the allowed sandbox root. Exits non-zero to block the call.
# =============================================================================

SANDBOX_ROOT="${SANDBOX_ROOT:?SANDBOX_ROOT must be set}"

# Read the hook input JSON from stdin
INPUT=$(cat)

# Extract the tool name from the hook context
TOOL_NAME="${CLAUDE_TOOL_NAME:-}"

block_and_die() {
    echo "SANDBOX VIOLATION: $1" >&2
    echo "Tool: ${TOOL_NAME}" >&2
    echo "Input: ${INPUT}" >&2
    echo "BLOCKED" >&2
    exit 2
}

check_path() {
    local path="$1"
    # Resolve to absolute path
    local resolved
    resolved=$(realpath -m "$path" 2>/dev/null || echo "$path")

    # Must start with sandbox root
    if [[ "$resolved" != "${SANDBOX_ROOT}"* ]]; then
        block_and_die "Path escapes sandbox: $path (resolved: $resolved)"
    fi
}

case "$TOOL_NAME" in
    Bash)
        # Extract command from JSON input
        cmd=$(echo "$INPUT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('command',''))" 2>/dev/null)

        # Block commands that navigate outside
        if echo "$cmd" | grep -qE '(^|\s)(cd|pushd)\s+[/~]'; then
            # Check if the cd target is within sandbox
            target=$(echo "$cmd" | grep -oE '(cd|pushd)\s+\S+' | head -1 | awk '{print $2}')
            if [[ -n "$target" ]]; then
                check_path "$target"
            fi
        fi

        # Block network commands
        if echo "$cmd" | grep -qE '(^|\s|;|&&|\|)(curl|wget|ssh|scp|rsync|nc|ncat|pip|pip3|npm|git\s+clone|git\s+push|git\s+pull)\s'; then
            block_and_die "Network/install command blocked: $cmd"
        fi

        # Block dangerous commands
        if echo "$cmd" | grep -qE '(^|\s|;|&&|\|)(rm\s+-rf\s+/|chmod|chown|kill|pkill|killall|shutdown|reboot)\s'; then
            block_and_die "Dangerous command blocked: $cmd"
        fi

        # Check any absolute paths referenced in the command
        for path in $(echo "$cmd" | grep -oE '/[a-zA-Z0-9_./-]+' | grep -v '^/dev/' | grep -v '^/proc/' | grep -v '^/tmp'); do
            check_path "$path"
        done
        ;;

    Read|Write|Edit)
        # Extract file_path from JSON
        filepath=$(echo "$INPUT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('file_path',''))" 2>/dev/null)
        if [[ -n "$filepath" ]]; then
            check_path "$filepath"
        fi
        ;;

    Glob|Grep)
        # Extract path from JSON
        searchpath=$(echo "$INPUT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('path',''))" 2>/dev/null)
        if [[ -n "$searchpath" ]]; then
            check_path "$searchpath"
        fi
        ;;
esac

# All checks passed
exit 0
