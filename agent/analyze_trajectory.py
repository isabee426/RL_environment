"""
Analyze Agent Trajectory
========================
Parses the Claude Code JSON trajectory and produces:
1. A human-readable timeline (timeline.md)
2. Metrics over time: what code was written, what was run, what results came back
3. A summary of the agent's strategy and decision points

Usage:
    python3 agent/analyze_trajectory.py --env_root /data3/ishaplan/pref_model_env
"""

import argparse
import json
import re
from pathlib import Path
from datetime import datetime


def parse_trajectory(trajectory_path: Path) -> list[dict]:
    """Parse Claude Code JSON output into a list of events."""
    with open(trajectory_path) as f:
        data = json.load(f)

    # Claude Code --output-format json returns a list of message objects
    # Each has: role, content (list of blocks), possibly tool calls/results
    events = []

    if isinstance(data, list):
        messages = data
    elif isinstance(data, dict) and "messages" in data:
        messages = data["messages"]
    else:
        # Try to handle single message or other formats
        messages = [data] if isinstance(data, dict) else data

    step = 0
    for msg in messages:
        if not isinstance(msg, dict):
            continue

        role = msg.get("role", "unknown")
        content = msg.get("content", "")

        # Handle content as string or list of blocks
        if isinstance(content, str):
            events.append({
                "step": step,
                "role": role,
                "type": "text",
                "content": content,
            })
            step += 1
            continue

        if isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue

                block_type = block.get("type", "")

                if block_type == "text":
                    events.append({
                        "step": step,
                        "role": role,
                        "type": "reasoning",
                        "content": block.get("text", ""),
                    })

                elif block_type == "tool_use":
                    tool_name = block.get("name", "unknown")
                    tool_input = block.get("input", {})
                    events.append({
                        "step": step,
                        "role": role,
                        "type": "tool_call",
                        "tool": tool_name,
                        "input": tool_input,
                    })

                elif block_type == "tool_result":
                    result_content = block.get("content", "")
                    if isinstance(result_content, list):
                        result_content = "\n".join(
                            b.get("text", "") for b in result_content
                            if isinstance(b, dict)
                        )
                    events.append({
                        "step": step,
                        "role": role,
                        "type": "tool_result",
                        "tool_use_id": block.get("tool_use_id", ""),
                        "content": str(result_content)[:2000],  # truncate long outputs
                    })

                step += 1

    return events


def extract_code_blocks(events: list[dict]) -> list[dict]:
    """Extract all code the agent wrote or executed."""
    code_blocks = []
    for e in events:
        if e["type"] == "tool_call":
            if e["tool"] == "Bash":
                code_blocks.append({
                    "step": e["step"],
                    "action": "execute",
                    "command": e["input"].get("command", ""),
                })
            elif e["tool"] == "Write":
                code_blocks.append({
                    "step": e["step"],
                    "action": "write_file",
                    "path": e["input"].get("file_path", ""),
                    "content_preview": e["input"].get("content", "")[:500],
                })
            elif e["tool"] == "Edit":
                code_blocks.append({
                    "step": e["step"],
                    "action": "edit_file",
                    "path": e["input"].get("file_path", ""),
                    "old": e["input"].get("old_string", "")[:200],
                    "new": e["input"].get("new_string", "")[:200],
                })
    return code_blocks


def extract_strategy_phases(events: list[dict]) -> list[dict]:
    """Try to identify distinct phases in the agent's approach."""
    phases = []
    current_phase = None

    keywords = {
        "exploration": ["read", "load", "config", "metadata", "inspect", "look"],
        "probing": ["probe", "linear", "classifier", "logistic", "sklearn"],
        "activation_analysis": ["activation", "attention", "head", "layer", "CLS", "token"],
        "causal_intervention": ["patch", "ablat", "causal", "swap", "intervene"],
        "fine_tuning": ["fine-tune", "finetune", "train", "optim", "gradient", "backward"],
        "evaluation": ["eval", "accuracy", "group", "val", "test", "metric"],
        "saving": ["save", "model_fixed", "output", "torch.save"],
    }

    for e in events:
        content = ""
        if e["type"] == "reasoning":
            content = e["content"].lower()
        elif e["type"] == "tool_call":
            content = json.dumps(e.get("input", {})).lower()

        for phase_name, phase_keywords in keywords.items():
            if any(kw in content for kw in phase_keywords):
                if current_phase != phase_name:
                    phases.append({
                        "phase": phase_name,
                        "started_at_step": e["step"],
                    })
                    current_phase = phase_name
                break

    return phases


def generate_timeline(events: list[dict], code_blocks: list[dict],
                      phases: list[dict]) -> str:
    """Generate a human-readable timeline markdown document."""
    lines = ["# Agent Trajectory Analysis\n"]
    lines.append(f"Generated: {datetime.now().isoformat()}\n")
    lines.append(f"Total events: {len(events)}\n")
    lines.append(f"Tool calls: {sum(1 for e in events if e['type'] == 'tool_call')}\n")
    lines.append(f"Files written: {sum(1 for c in code_blocks if c['action'] == 'write_file')}\n")
    lines.append(f"Commands executed: {sum(1 for c in code_blocks if c['action'] == 'execute')}\n")

    # Phase summary
    lines.append("\n## Strategy Phases\n")
    if phases:
        for p in phases:
            lines.append(f"- **{p['phase']}** (step {p['started_at_step']})")
    else:
        lines.append("Could not identify distinct phases.\n")

    # Detailed timeline
    lines.append("\n## Detailed Timeline\n")

    for e in events:
        if e["type"] == "reasoning":
            # Only show first 300 chars of reasoning
            text = e["content"][:300].replace("\n", " ")
            lines.append(f"\n### Step {e['step']} — Agent Reasoning\n")
            lines.append(f"> {text}{'...' if len(e['content']) > 300 else ''}\n")

        elif e["type"] == "tool_call":
            lines.append(f"\n### Step {e['step']} — {e['tool']}\n")
            if e["tool"] == "Bash":
                cmd = e["input"].get("command", "")
                lines.append(f"```bash\n{cmd}\n```\n")
            elif e["tool"] == "Write":
                path = e["input"].get("file_path", "")
                lines.append(f"**Wrote file:** `{path}`\n")
                preview = e["input"].get("content", "")[:500]
                lines.append(f"```python\n{preview}\n...\n```\n")
            elif e["tool"] == "Edit":
                path = e["input"].get("file_path", "")
                lines.append(f"**Edited:** `{path}`\n")
            elif e["tool"] == "Read":
                path = e["input"].get("file_path", "")
                lines.append(f"**Read:** `{path}`\n")
            else:
                lines.append(f"```json\n{json.dumps(e['input'], indent=2)[:500]}\n```\n")

        elif e["type"] == "tool_result":
            content = e["content"][:500]
            lines.append(f"<details><summary>Result (step {e['step']})</summary>\n")
            lines.append(f"```\n{content}\n```\n")
            lines.append(f"</details>\n")

    # Code inventory
    lines.append("\n## Files Created by Agent\n")
    written = [c for c in code_blocks if c["action"] == "write_file"]
    if written:
        for w in written:
            lines.append(f"- `{w['path']}` (step {w['step']})")
    else:
        lines.append("No files written.\n")

    lines.append("\n## Commands Executed\n")
    executed = [c for c in code_blocks if c["action"] == "execute"]
    if executed:
        for i, ex in enumerate(executed):
            cmd_preview = ex["command"][:100]
            lines.append(f"{i+1}. Step {ex['step']}: `{cmd_preview}`")
    else:
        lines.append("No commands executed.\n")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_root", type=str, required=True)
    args = parser.parse_args()

    env_root = Path(args.env_root)
    trajectory_path = env_root / "results" / "trajectory.json"
    output_dir = env_root / "results"

    if not trajectory_path.exists():
        print(f"ERROR: No trajectory found at {trajectory_path}")
        return

    print("Parsing trajectory...")
    events = parse_trajectory(trajectory_path)
    print(f"  Found {len(events)} events")

    code_blocks = extract_code_blocks(events)
    print(f"  Found {len(code_blocks)} code actions")

    phases = extract_strategy_phases(events)
    print(f"  Identified {len(phases)} strategy phases")

    # Generate timeline
    timeline = generate_timeline(events, code_blocks, phases)
    timeline_path = output_dir / "timeline.md"
    timeline_path.write_text(timeline)
    print(f"  Timeline written to {timeline_path}")

    # Save structured data
    analysis = {
        "total_events": len(events),
        "tool_calls": sum(1 for e in events if e["type"] == "tool_call"),
        "files_written": sum(1 for c in code_blocks if c["action"] == "write_file"),
        "commands_executed": sum(1 for c in code_blocks if c["action"] == "execute"),
        "phases": phases,
        "code_blocks": code_blocks,
    }
    analysis_path = output_dir / "trajectory_analysis.json"
    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"  Analysis written to {analysis_path}")


if __name__ == "__main__":
    main()
