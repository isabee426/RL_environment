"""
Behavioral Analysis of Agent Research Strategy
===============================================
Analyzes the agent's trajectory and outputs to characterize its research behavior.

Works with claude -p --output-format json which produces a summary object,
plus the actual files the agent created in the sandbox.

Usage:
    python3 agent/behavioral_analysis.py --env_root /data3/ishaplan/pref_model_env
"""

import argparse
import json
import re
from pathlib import Path
from datetime import datetime
from collections import defaultdict


def load_trajectory(env_root: Path) -> dict:
    """Load the trajectory — handles both json and stream-json (jsonl) formats."""
    # Try jsonl first (stream-json: one JSON object per line)
    jsonl_path = env_root / "results" / "trajectory.jsonl"
    if jsonl_path.exists() and jsonl_path.stat().st_size > 0:
        events = []
        result_obj = {}
        for line in jsonl_path.read_text().strip().split("\n"):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                events.append(obj)
                # The last line with type=result has the summary
                if obj.get("type") == "result":
                    result_obj = obj
            except json.JSONDecodeError:
                continue
        # Merge: put summary fields at top level, keep events list
        result_obj["events"] = events
        return result_obj

    # Fallback: single json summary
    json_path = env_root / "results" / "trajectory.json"
    if json_path.exists() and json_path.stat().st_size > 0:
        with open(json_path) as f:
            return json.load(f)

    return {}


def analyze_events(traj: dict) -> dict:
    """Analyze per-message events from stream-json trajectory."""
    events = traj.get("events", [])
    if not events:
        return {"has_events": False}

    assistant_msgs = [e for e in events if e.get("type") == "assistant"]
    user_msgs = [e for e in events if e.get("type") == "user"]

    # Count tool calls by type
    tool_calls = defaultdict(int)
    reasoning_chunks = []
    for e in assistant_msgs:
        msg = e.get("message", {})
        content = msg.get("content", [])
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "tool_use":
                        tool_calls[block.get("name", "unknown")] += 1
                    elif block.get("type") == "text":
                        reasoning_chunks.append(block.get("text", ""))

    # Count errors
    errors = 0
    for e in user_msgs:
        msg = e.get("message", {})
        content = msg.get("content", [])
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("is_error"):
                    errors += 1
        # Also check tool_use_result in the event itself
        result = e.get("tool_use_result", "")
        if isinstance(result, dict) and result.get("stderr"):
            if "error" in result["stderr"].lower() or "traceback" in result["stderr"].lower():
                errors += 1

    total_reasoning = sum(len(r) for r in reasoning_chunks)

    return {
        "has_events": True,
        "total_events": len(events),
        "assistant_messages": len(assistant_msgs),
        "tool_calls_by_type": dict(tool_calls),
        "total_tool_calls": sum(tool_calls.values()),
        "errors_encountered": errors,
        "reasoning_chunks": len(reasoning_chunks),
        "total_reasoning_chars": total_reasoning,
        "reasoning_per_tool_call": round(total_reasoning / max(sum(tool_calls.values()), 1)),
    }


def analyze_trajectory_metadata(traj: dict) -> dict:
    """Extract high-level stats from the trajectory summary."""
    return {
        "duration_ms": traj.get("duration_ms", 0),
        "duration_min": round(traj.get("duration_ms", 0) / 60000, 1),
        "num_turns": traj.get("num_turns", 0),
        "total_cost_usd": traj.get("total_cost_usd", 0),
        "stop_reason": traj.get("stop_reason", "unknown"),
        "model_used": list(traj.get("modelUsage", {}).keys()),
        "output_tokens": traj.get("usage", {}).get("output_tokens", 0),
        "fast_mode": traj.get("fast_mode_state", "unknown"),
    }


def analyze_agent_scripts(env_root: Path) -> list[dict]:
    """Analyze the Python scripts the agent wrote."""
    scripts = []
    for py_file in sorted(env_root.glob("*.py")):
        code = py_file.read_text()
        scripts.append({
            "name": py_file.name,
            "size_bytes": len(code),
            "num_lines": len(code.splitlines()),
            "imports": extract_imports(code),
            "techniques": detect_techniques(code),
            "has_training_loop": "for epoch" in code.lower() or "for step" in code.lower(),
            "has_validation": "val" in code.lower() and ("mse" in code.lower() or "loss" in code.lower()),
            "has_comparison": "baseline" in code.lower() or "sae" in code.lower(),
            "has_visualization": "plt." in code or "matplotlib" in code,
        })
    return scripts


def extract_imports(code: str) -> list[str]:
    imports = []
    for line in code.splitlines():
        line = line.strip()
        if line.startswith("import ") or line.startswith("from "):
            imports.append(line)
    return imports


def detect_techniques(code: str) -> list[str]:
    """Detect ML/interp techniques used in the code."""
    techniques = {
        "skip_transcoder": ["w_skip", "skip_connection", "skip connection", "affine skip"],
        "topk_sparsity": ["topk", "top_k", "top-k"],
        "sae_baseline": ["baseline_sae", "baseline sae", "sae"],
        "auroc_evaluation": ["auroc", "roc_auc", "auc"],
        "linear_probing": ["logistic", "linear probe", "sklearn"],
        "cosine_schedule": ["cosineanneal", "cosine_anneal", "cosine schedule"],
        "decoder_normalization": ["norm", "normalize", "unit_norm", "column_norm"],
        "mean_initialization": ["mean", "b_skip.*mean", "empirical mean"],
        "gradient_clipping": ["clip_grad", "grad_norm", "gradient clip"],
        "weight_decay": ["weight_decay", "l2_reg"],
    }

    found = []
    code_lower = code.lower()
    for tech, keywords in techniques.items():
        if any(kw in code_lower for kw in keywords):
            found.append(tech)
    return found


def analyze_agent_analysis(env_root: Path) -> dict:
    """Parse the agent's own analysis.txt for research quality signals."""
    analysis_path = env_root / "output" / "analysis.txt"
    if not analysis_path.exists():
        return {"exists": False}

    text = analysis_path.read_text()
    text_lower = text.lower()

    quality_signals = {
        "states_hypothesis": any(kw in text_lower for kw in
            ["hypothes", "expect", "likely", "suggests that", "this is because"]),
        "compares_models": any(kw in text_lower for kw in
            ["transcoder vs", "vs sae", "compared to", "baseline", "comparison"]),
        "reports_metrics": any(kw in text_lower for kw in
            ["mse", "auroc", "accuracy", "loss"]),
        "discusses_architecture": any(kw in text_lower for kw in
            ["skip connection", "w_skip", "topk", "dictionary", "encoder", "decoder"]),
        "discusses_initialization": any(kw in text_lower for kw in
            ["zero-init", "zero init", "mean of", "empirical mean", "xavier"]),
        "discusses_training": any(kw in text_lower for kw in
            ["epoch", "learning rate", "optimizer", "batch size", "cosine"]),
        "interprets_results": any(kw in text_lower for kw in
            ["this suggests", "this means", "this indicates", "interesting",
             "as expected", "likely because", "this is because"]),
        "per_concept_analysis": any(kw in text_lower for kw in
            ["per-concept", "per concept", "each concept", "concept-level"]),
        "discusses_limitations": any(kw in text_lower for kw in
            ["limitation", "caveat", "however", "challenge", "difficult"]),
        "vision_vs_language": any(kw in text_lower for kw in
            ["vision", "language model", "adapting", "domain", "visual"]),
    }

    return {
        "exists": True,
        "length_chars": len(text),
        "num_sections": text.count("\n---") + text.count("\n===") + text.count("\n("),
        "quality_signals": quality_signals,
        "quality_score": sum(quality_signals.values()) / len(quality_signals),
    }


def analyze_feature_mapping(env_root: Path) -> dict:
    """Analyze the feature mapping output."""
    mapping_path = env_root / "output" / "feature_mapping.json"
    if not mapping_path.exists():
        return {"exists": False}

    with open(mapping_path) as f:
        mapping = json.load(f)

    tc_data = mapping.get("transcoder", {})
    sae_data = mapping.get("baseline_sae", {})

    tc_aurocs = [v["auroc"] if isinstance(v, dict) else v for v in tc_data.values()]
    sae_aurocs = [v["auroc"] if isinstance(v, dict) else v for v in sae_data.values()]

    return {
        "exists": True,
        "transcoder_concepts": len(tc_data),
        "sae_concepts": len(sae_data),
        "tc_mean_auroc": round(sum(tc_aurocs) / max(len(tc_aurocs), 1), 4) if tc_aurocs else 0,
        "sae_mean_auroc": round(sum(sae_aurocs) / max(len(sae_aurocs), 1), 4) if sae_aurocs else 0,
    }


def analyze_iteration_pattern(env_root: Path) -> dict:
    """Did the agent iterate? Look at v1, v2, etc. scripts."""
    scripts = sorted(env_root.glob("*.py"))
    names = [s.name for s in scripts]

    has_versions = any("v2" in n or "v3" in n or "_2" in n for n in names)

    return {
        "num_scripts": len(scripts),
        "script_names": names,
        "iterated": has_versions,
        "iteration_count": sum(1 for n in names if any(v in n for v in ["v2", "v3", "_2", "_3"])),
    }


def generate_report(env_root: Path, traj: dict) -> str:
    """Generate the full behavioral analysis report."""
    meta = analyze_trajectory_metadata(traj)
    events_analysis = analyze_events(traj)
    scripts = analyze_agent_scripts(env_root)
    analysis = analyze_agent_analysis(env_root)
    mapping = analyze_feature_mapping(env_root)
    iteration = analyze_iteration_pattern(env_root)

    # Load judge results if available
    judge_path = env_root / "output" / "judge_results.json"
    judge = {}
    if judge_path.exists():
        with open(judge_path) as f:
            judge = json.load(f)

    lines = []
    lines.append("# Agent Behavioral Analysis Report")
    lines.append(f"\nGenerated: {datetime.now().isoformat()}")

    # 1. Execution Summary
    lines.append("\n## 1. Execution Summary")
    lines.append(f"- Duration: {meta['duration_min']} minutes")
    lines.append(f"- Turns (tool calls): {meta['num_turns']}")
    lines.append(f"- Output tokens: {meta['output_tokens']}")
    lines.append(f"- Cost: ${meta['total_cost_usd']:.2f}")
    lines.append(f"- Stop reason: {meta['stop_reason']}")
    lines.append(f"- Models used: {', '.join(meta['model_used'])}")

    if events_analysis["has_events"]:
        lines.append(f"\n### Event-Level Breakdown")
        lines.append(f"- Total events: {events_analysis['total_events']}")
        lines.append(f"- Assistant messages: {events_analysis['assistant_messages']}")
        lines.append(f"- Total tool calls: {events_analysis['total_tool_calls']}")
        lines.append(f"- Tool calls by type:")
        for tool, count in sorted(events_analysis["tool_calls_by_type"].items()):
            lines.append(f"  - {tool}: {count}")
        lines.append(f"- Errors encountered: {events_analysis['errors_encountered']}")
        lines.append(f"- Reasoning chunks: {events_analysis['reasoning_chunks']}")
        lines.append(f"- Chars of reasoning per tool call: {events_analysis['reasoning_per_tool_call']}")

    # 2. Strategy & Iteration
    lines.append("\n## 2. Strategy & Iteration")
    lines.append(f"- Scripts written: {iteration['num_scripts']}")
    lines.append(f"- Script names: {', '.join(iteration['script_names'])}")
    lines.append(f"- Iterated on approach: {'Yes' if iteration['iterated'] else 'No'}")
    if iteration['iterated']:
        lines.append(f"  - Created {iteration['iteration_count']} revised version(s)")
        lines.append("  - **Assessment: Iterative researcher** — refined approach after initial results")
    else:
        lines.append("  - **Assessment: Single-pass** — got it right first try or didn't iterate")

    # 3. Technical Implementation
    lines.append("\n## 3. Technical Implementation")
    all_techniques = set()
    for s in scripts:
        all_techniques.update(s["techniques"])
        lines.append(f"\n### {s['name']} ({s['num_lines']} lines)")
        lines.append(f"- Techniques: {', '.join(s['techniques']) or 'none detected'}")
        lines.append(f"- Has training loop: {s['has_training_loop']}")
        lines.append(f"- Has validation: {s['has_validation']}")
        lines.append(f"- Has baseline comparison: {s['has_comparison']}")

    lines.append(f"\n**All techniques used:** {', '.join(sorted(all_techniques))}")

    # 4. Research Quality
    lines.append("\n## 4. Research Quality (from agent's analysis.txt)")
    if analysis["exists"]:
        lines.append(f"- Report length: {analysis['length_chars']} characters")
        lines.append(f"- Quality score: {analysis['quality_score']:.0%}")
        lines.append(f"- Quality signals present:")
        for signal, present in analysis["quality_signals"].items():
            marker = "YES" if present else "no"
            lines.append(f"  - {signal}: {marker}")
    else:
        lines.append("- No analysis.txt produced")

    # 5. Results Quality
    lines.append("\n## 5. Results")
    if judge:
        score = judge.get("score", 0)
        lines.append(f"- **Judge score: {score:.2f}/1.0**")
        breakdown = judge.get("details", {}).get("score_breakdown", {})
        if breakdown:
            for component, val in breakdown.items():
                lines.append(f"  - {component}: {val}")

        recon = judge.get("details", {}).get("reconstruction", {})
        if recon:
            lines.append(f"- Transcoder MSE: {recon.get('transcoder_mse', '?')}")
            lines.append(f"- Baseline SAE MSE: {recon.get('baseline_sae_mse', '?')}")
            lines.append(f"- Beats baseline on reconstruction: {recon.get('beats_baseline', '?')}")

        comp = judge.get("details", {}).get("comparison", {})
        if comp:
            lines.append(f"- Transcoder wins {comp.get('transcoder_wins', '?')}/{comp.get('shared_concepts', '?')} concepts")

    if mapping["exists"]:
        lines.append(f"- Transcoder mean AUROC: {mapping['tc_mean_auroc']}")
        lines.append(f"- SAE mean AUROC: {mapping['sae_mean_auroc']}")

    # 6. Overall Assessment
    lines.append("\n## 6. Overall Assessment")

    strengths = []
    weaknesses = []

    if meta["num_turns"] > 0 and meta["stop_reason"] == "end_turn":
        strengths.append("Completed the task successfully")
    if iteration["iterated"]:
        strengths.append("Iterated on approach (wrote multiple script versions)")
    if "skip_transcoder" in all_techniques:
        strengths.append("Correctly implemented skip transcoder architecture")
    if "topk_sparsity" in all_techniques:
        strengths.append("Used TopK sparsity as specified")
    if "cosine_schedule" in all_techniques:
        strengths.append("Used learning rate scheduling (cosine annealing)")
    if "mean_initialization" in all_techniques:
        strengths.append("Properly initialized b_skip with empirical mean")
    if "decoder_normalization" in all_techniques:
        strengths.append("Applied decoder column normalization")
    if "auroc_evaluation" in all_techniques:
        strengths.append("Evaluated interpretability via AUROC")
    if analysis.get("exists") and analysis.get("quality_score", 0) >= 0.5:
        strengths.append(f"Wrote substantive analysis report (quality: {analysis['quality_score']:.0%})")
    if analysis.get("quality_signals", {}).get("interprets_results"):
        strengths.append("Interprets results with hypotheses")
    if analysis.get("quality_signals", {}).get("vision_vs_language"):
        strengths.append("Discusses vision vs. language domain adaptation")

    if judge.get("score", 0) >= 0.7:
        strengths.append(f"Strong judge score ({judge['score']:.2f})")
    elif judge.get("score", 0) >= 0.4:
        strengths.append(f"Moderate judge score ({judge['score']:.2f})")

    recon = judge.get("details", {}).get("reconstruction", {})
    if recon.get("beats_baseline"):
        strengths.append("Transcoder beats SAE on reconstruction (core claim of the paper)")

    comp = judge.get("details", {}).get("comparison", {})
    win_rate = comp.get("win_rate", 0)
    if win_rate < 0.5:
        weaknesses.append(f"Transcoder loses to SAE on interpretability ({win_rate:.0%} win rate)")
    if not analysis.get("quality_signals", {}).get("discusses_limitations"):
        weaknesses.append("Does not discuss limitations of the approach")
    if meta["duration_min"] < 3 and meta["duration_min"] > 0:
        weaknesses.append("Very fast completion — may have rushed")
    if not any(s.get("has_visualization") for s in scripts):
        weaknesses.append("No visualizations generated")

    if events_analysis["has_events"] and events_analysis["errors_encountered"] > 0:
        weaknesses.append(f"Encountered {events_analysis['errors_encountered']} errors during execution")
    if events_analysis["has_events"] and events_analysis["errors_encountered"] == 0:
        strengths.append("Zero errors during execution")

    lines.append("\n**Strengths:**")
    for s in strengths:
        lines.append(f"- {s}")
    lines.append("\n**Weaknesses:**")
    for w in weaknesses:
        lines.append(f"- {w}")

    # 7. Key Finding
    lines.append("\n## 7. Key Research Finding")
    result_text = traj.get("result", "")
    if result_text:
        # Format the result text with proper line breaks
        for line in result_text.split("\n"):
            lines.append(f"> {line}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_root", type=str, required=True)
    args = parser.parse_args()

    env_root = Path(args.env_root)

    print("Loading trajectory...")
    traj = load_trajectory(env_root)

    print("Running behavioral analysis...")
    report = generate_report(env_root, traj)

    report_path = env_root / "results" / "behavioral_analysis.md"
    report_path.write_text(report)
    print(f"Report saved to {report_path}")

    # Also save structured data
    structured = {
        "metadata": analyze_trajectory_metadata(traj),
        "events": analyze_events(traj),
        "scripts": analyze_agent_scripts(env_root),
        "analysis_quality": analyze_agent_analysis(env_root),
        "feature_mapping": analyze_feature_mapping(env_root),
        "iteration": analyze_iteration_pattern(env_root),
    }
    json_path = env_root / "results" / "behavioral_analysis.json"
    with open(json_path, "w") as f:
        json.dump(structured, f, indent=2, default=str)
    print(f"Structured data saved to {json_path}")

    print("\n" + report)


if __name__ == "__main__":
    main()
