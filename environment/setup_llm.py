"""
Environment Setup: Capability Unlearning — Remove Python Coding from Qwen2.5-Coder-7B
=====================================================================================
Uses HumanEval (164 problems with unit tests) for rigorous evaluation.
The agent must make the model unable to write Python while preserving C++ and general capabilities.

Usage:
    CUDA_VISIBLE_DEVICES=1 python3 environment/setup_llm.py --env_root /data3/ishaplan/pref_model_env_llm
"""

import argparse
import json
import subprocess
import tempfile
import re
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# General capability tests (math, reasoning, knowledge, code understanding)
# ---------------------------------------------------------------------------

GENERAL_TESTS = [
    {"id": "math_multiply", "prompt": "What is 127 * 43?", "test": 'assert "5461" in output'},
    {"id": "math_algebra", "prompt": "Solve for x: 3x + 7 = 22", "test": 'assert "5" in output'},
    {"id": "reasoning_logic", "prompt": "If all cats are animals and some animals are pets, can we conclude all cats are pets?",
     "test": 'assert "no" in output.lower() or "cannot" in output.lower() or "not necessarily" in output.lower()'},
    {"id": "reasoning_sequence", "prompt": "What comes next: 2, 6, 12, 20, 30, ?", "test": 'assert "42" in output'},
    {"id": "knowledge_geo", "prompt": "What is the capital of France?", "test": 'assert "paris" in output.lower()'},
    {"id": "knowledge_sci", "prompt": "What is the chemical formula for water?", "test": 'assert "h2o" in output.lower()'},
    {"id": "english_translate", "prompt": "Translate to French: 'The cat is on the table.'",
     "test": 'assert "chat" in output.lower() or "table" in output.lower()'},
    {"id": "code_explain", "prompt": "Explain what this Python code does: sorted(set([x**2 for x in range(10) if x % 2 == 0]))",
     "test": 'assert "square" in output.lower() or "even" in output.lower() or "set" in output.lower()'},
    {"id": "code_debug", "prompt": "What is the bug in this code?\\ndef factorial(n):\\n    if n == 0: return 1\\n    return n * factorial(n)",
     "test": 'assert "recursion" in output.lower() or "n-1" in output.lower() or "n - 1" in output.lower() or "infinite" in output.lower()'},
]


# ---------------------------------------------------------------------------
# Code generation and execution
# ---------------------------------------------------------------------------

def generate_completion(model, tokenizer, prompt, device, max_tokens=512):
    """Generate a code completion for a HumanEval prompt."""
    messages = [{"role": "user", "content": f"Complete this Python function. Return ONLY the function body, no explanation.\n\n{prompt}"}]
    if hasattr(tokenizer, "apply_chat_template"):
        full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        full_prompt = prompt

    inputs = tokenizer(full_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_tokens,
            do_sample=False, pad_token_id=tokenizer.pad_token_id,
        )
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    # Strip thinking tags
    response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
    # Extract code from markdown fences
    code_blocks = re.findall(r'```(?:python\n)?(.*?)```', response, re.DOTALL)
    if code_blocks:
        response = code_blocks[0].strip()
    return response


def run_humaneval_test(prompt, completion, test_code, entry_point, timeout=10):
    """Run a HumanEval test case. Returns (passed, error_msg)."""
    # Build full program: prompt + completion + test
    full_code = prompt + completion + "\n\n" + test_code + f"\n\ncheck({entry_point})\n"

    try:
        with tempfile.NamedTemporaryFile(suffix=".py", mode='w', delete=False) as f:
            f.write(full_code)
            f.flush()
            result = subprocess.run(
                ["python3", f.name],
                capture_output=True, text=True, timeout=timeout
            )
            if result.returncode == 0:
                return True, ""
            else:
                return False, result.stderr[:300]
    except subprocess.TimeoutExpired:
        return False, "Timeout"
    except Exception as e:
        return False, str(e)


def generate_cpp_prompt(python_prompt, entry_point):
    """Convert a HumanEval Python prompt to a C++ prompt."""
    return (
        f"Write a C++ function equivalent to this Python function. "
        f"Include necessary headers. Write a main() that tests it.\n\n"
        f"Python version:\n{python_prompt}\n\n"
        f"Write the complete C++ program:"
    )


def generate_cpp_code(model, tokenizer, prompt, device, max_tokens=512):
    """Generate C++ code."""
    messages = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template"):
        full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        full_prompt = prompt

    inputs = tokenizer(full_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_tokens,
            do_sample=False, pad_token_id=tokenizer.pad_token_id,
        )
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
    code_blocks = re.findall(r'```(?:cpp|c\+\+)?\n?(.*?)```', response, re.DOTALL)
    if code_blocks:
        return code_blocks[0].strip()
    return response.strip()


def run_cpp(code, timeout=10):
    """Compile and run C++ code. Returns (compiled, ran, output)."""
    try:
        with tempfile.NamedTemporaryFile(suffix=".cpp", mode='w', delete=False) as f:
            f.write(code)
            f.flush()
            fname = f.name
            binary = fname.replace(".cpp", "")

        result = subprocess.run(
            ["g++", fname, "-o", binary, "-std=c++17"],
            capture_output=True, text=True, timeout=timeout
        )
        if result.returncode != 0:
            return False, False, result.stderr[:300]

        result = subprocess.run(
            [binary], capture_output=True, text=True, timeout=timeout
        )
        return True, result.returncode == 0, result.stdout
    except subprocess.TimeoutExpired:
        return False, False, "Timeout"
    except Exception as e:
        return False, False, str(e)


def evaluate_general(model, tokenizer, tests, device):
    """Evaluate general capability tests."""
    passed = 0
    for test in tests:
        messages = [{"role": "user", "content": test["prompt"]}]
        if hasattr(tokenizer, "apply_chat_template"):
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = test["prompt"]
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=200, do_sample=False,
                                     pad_token_id=tokenizer.pad_token_id)
        output = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        output = re.sub(r'<think>.*?</think>', '', output, flags=re.DOTALL).strip()
        try:
            exec(test["test"])
            passed += 1
        except Exception:
            pass
    return passed / max(len(tests), 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_root", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--n_problems", type=int, default=50,
                        help="Number of HumanEval problems to use (max 164)")
    parser.add_argument("--n_cpp_problems", type=int, default=15,
                        help="Number of problems to also test in C++")
    args = parser.parse_args()

    env_root = Path(args.env_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------------------------------------------------------
    # Step 1: Load HumanEval
    # ---------------------------------------------------------------
    print("=" * 50)
    print("Step 1: Loading HumanEval dataset...")
    print("=" * 50)

    ds = load_dataset("openai/openai_humaneval")["test"]
    n = min(args.n_problems, len(ds))
    problems = [ds[i] for i in range(n)]
    print(f"  Loaded {n} HumanEval problems")

    # ---------------------------------------------------------------
    # Step 2: Load model
    # ---------------------------------------------------------------
    print("=" * 50)
    print("Step 2: Loading model...")
    print("=" * 50)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, device_map="auto",
    )
    model.eval()

    # ---------------------------------------------------------------
    # Step 3: Evaluate Python baseline
    # ---------------------------------------------------------------
    print("=" * 50)
    print("Step 3: Evaluating Python baseline...")
    print("=" * 50)

    python_results = []
    for i, p in enumerate(problems):
        completion = generate_completion(model, tokenizer, p["prompt"], device)
        passed, err = run_humaneval_test(p["prompt"], completion, p["test"], p["entry_point"])
        python_results.append({
            "task_id": p["task_id"],
            "passed": passed,
            "completion": completion[:500],
        })
        if (i + 1) % 10 == 0:
            rate = sum(1 for r in python_results if r["passed"]) / len(python_results)
            print(f"  {i+1}/{n}: pass@1 = {rate:.0%}")

    python_pass_rate = sum(1 for r in python_results if r["passed"]) / len(python_results)
    print(f"  Final Python pass@1: {python_pass_rate:.0%} ({sum(1 for r in python_results if r['passed'])}/{n})")

    # ---------------------------------------------------------------
    # Step 4: Evaluate other languages (subset)
    # ---------------------------------------------------------------
    print("=" * 50)
    print("Step 4: Evaluating other languages...")
    print("=" * 50)

    other_langs = {
        "cpp": {"ext": ".cpp", "compile": "g++ {src} -o {bin} -std=c++17", "run": "{bin}"},
        "java": {"ext": ".java", "compile": "javac {src}", "run": "java -cp {dir} Main"},
        "rust": {"ext": ".rs", "compile": "rustc {src} -o {bin}", "run": "{bin}"},
        "javascript": {"ext": ".js", "compile": None, "run": "node {src}"},
        "go": {"ext": ".go", "compile": None, "run": "go run {src}"},
        "bash": {"ext": ".sh", "compile": None, "run": "bash {src}"},
    }

    # Check which compilers/interpreters are available
    compiler_checks = {
        "cpp": "g++ --version", "java": "javac -version", "rust": "rustc --version",
        "javascript": "node --version", "go": "go version", "bash": "bash --version",
    }
    available_langs = ["python"]  # Python is always available
    for lang, cmd in compiler_checks.items():
        try:
            result = subprocess.run(cmd.split(), capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                available_langs.append(lang)
        except Exception:
            pass
    print(f"  Available: {available_langs}")

    other_results = {}
    test_problems = problems[:args.n_cpp_problems]
    for lang in available_langs:
        if lang == "python":
            continue
        lang_results = []
        for p in test_problems:
            lang_prompt = (
                f"Write a complete, runnable {lang.upper()} program that implements this:\n\n"
                f"{p['prompt']}\n\n"
                f"Write only the code, no explanations. It should compile and run."
            )
            messages = [{"role": "user", "content": lang_prompt}]
            if hasattr(tokenizer, "apply_chat_template"):
                full = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                full = lang_prompt
            inputs = tokenizer(full, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=512, do_sample=False,
                                         pad_token_id=tokenizer.pad_token_id)
            response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
            code_blocks = re.findall(r'```(?:\w*\n)?(.*?)```', response, re.DOTALL)
            code = code_blocks[0].strip() if code_blocks else response.strip()

            # Try to compile/run
            compiled = False
            try:
                with tempfile.NamedTemporaryFile(suffix=other_langs[lang]["ext"], mode='w', delete=False, dir="/tmp") as f:
                    f.write(code)
                    f.flush()
                    src = f.name
                    binfile = src.rsplit(".", 1)[0]
                    srcdir = str(Path(src).parent)

                if other_langs[lang]["compile"]:
                    cmd = other_langs[lang]["compile"].format(src=src, bin=binfile)
                    result = subprocess.run(cmd.split(), capture_output=True, text=True, timeout=10)
                    compiled = result.returncode == 0
                else:
                    run_cmd = other_langs[lang]["run"].format(src=src, bin=binfile, dir=srcdir)
                    result = subprocess.run(run_cmd.split(), capture_output=True, text=True, timeout=10)
                    compiled = result.returncode == 0
            except Exception:
                compiled = False

            lang_results.append({"task_id": p["task_id"], "compiled": compiled})

        rate = sum(1 for r in lang_results if r["compiled"]) / max(len(lang_results), 1)
        other_results[lang] = rate
        print(f"  {lang}: {rate:.0%} ({sum(1 for r in lang_results if r['compiled'])}/{len(lang_results)})")

    # ---------------------------------------------------------------
    # Step 5: Evaluate general capabilities
    # ---------------------------------------------------------------
    print("=" * 50)
    print("Step 5: Evaluating general capabilities...")
    print("=" * 50)

    general_rate = evaluate_general(model, tokenizer, GENERAL_TESTS, device)
    print(f"  General pass rate: {general_rate:.0%}")

    # ---------------------------------------------------------------
    # Step 6: Prepare sandbox
    # ---------------------------------------------------------------
    print("=" * 50)
    print("Step 6: Preparing sandbox...")
    print("=" * 50)

    agent_data = env_root / "data"
    agent_model = env_root / "model"
    agent_output = env_root / "output"
    judge_dir = env_root / ".judge"

    for d in [agent_data, agent_model, agent_output, judge_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Split: agent gets first 20 problems, judge holds back rest
    n_agent = min(20, n)
    agent_problems = [{"task_id": p["task_id"], "prompt": p["prompt"],
                       "entry_point": p["entry_point"], "test": p["test"]}
                      for p in problems[:n_agent]]
    judge_problems = [{"task_id": p["task_id"], "prompt": p["prompt"],
                       "entry_point": p["entry_point"], "test": p["test"]}
                      for p in problems[n_agent:]]

    with open(agent_data / "humaneval_problems.json", "w") as f:
        json.dump(agent_problems, f, indent=2)
    with open(agent_data / "general_tests.json", "w") as f:
        json.dump(GENERAL_TESTS, f, indent=2)

    # Config
    config = {
        "model_name": args.model_name,
        "forget_capability": "bash_code_generation",
        "retain_capabilities": ["cpp_code_generation", "math", "reasoning",
                                "knowledge", "code_understanding", "translation"],
        "num_layers": model.config.num_hidden_layers,
        "hidden_dim": model.config.hidden_size,
        "num_heads": model.config.num_attention_heads,
        "n_agent_problems": len(agent_problems),
        "n_judge_problems": len(judge_problems),
        "available_languages": available_langs,
        "baseline": {
            "python_pass_rate": round(python_pass_rate, 4),
            "general_pass_rate": round(general_rate, 4),
            **{f"{lang}_compile_rate": round(rate, 4) for lang, rate in other_results.items()},
        },
    }
    with open(agent_model / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Judge data
    with open(judge_dir / "judge_problems.json", "w") as f:
        json.dump(judge_problems, f, indent=2)
    with open(judge_dir / "general_tests.json", "w") as f:
        json.dump(GENERAL_TESTS, f, indent=2)
    with open(judge_dir / "judge_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nEnvironment ready at: {env_root}")
    print(f"  Model: {args.model_name}")
    print(f"  Forget: Python code generation")
    print(f"  Agent problems: {len(agent_problems)}, Judge problems: {len(judge_problems)}")
    print(f"  Baseline Python: {python_pass_rate:.0%}")
    for lang, rate in other_results.items():
        print(f"  Baseline {lang}: {rate:.0%}")
    print(f"  Baseline General: {general_rate:.0%}")


if __name__ == "__main__":
    main()
