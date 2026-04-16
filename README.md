# RL Environment: Transcoders on Vision Transformers

An RL environment for training LLM agents on frontier ML research tasks. The agent must implement **skip transcoders** — a mechanistic interpretability technique from [Paulo et al. 2025](https://arxiv.org/abs/2501.18823) that has only been applied to language models — and adapt it to a **vision transformer**.


## Why This Environment

Transcoders on vision models are **out-of-distribution** for current LLMs. The technique is new (January 2025), only tested on language models, and requires the agent to make genuine research adaptation decisions rather than following a known recipe. This tests exactly this: can an LLM agent do novel ML research when there's no tutorial to follow?

## Environment Design

An RL environment consists of a **prompt**, **judge**, **tools**, and **data**.

### The Task

The agent receives:
- MLP input/output activation pairs extracted from CLIP ViT-B/16 (layer 8, CLS token) on 12k CelebA images
- A pretrained baseline SAE for comparison
- 40 binary concept labels (CelebA attributes) for interpretability evaluation

The agent must:
1. Implement a skip transcoder: `f(x) = W_dec @ TopK(W_enc @ x + b_enc) + W_skip @ x + b_skip`
2. Train it on the MLP activation pairs
3. Evaluate interpretability via per-concept AUROC against the baseline SAE
4. Produce a research analysis comparing the two approaches

### The Judge

Scoring (0.0 - 1.0) on a **held-out test set** the agent never sees:

| Component | Weight | What it measures |
|-----------|--------|-----------------|
| Reconstruction | 0.3 | MSE vs. baseline SAE |
| Sparsity | 0.1 | L0 in target range |
| Interpretability | 0.3 | Average AUROC across 40 concepts |
| Comparison | 0.3 | Fraction of concepts where transcoder beats SAE |

Additional checks: architecture verification (must have skip connection), dictionary size >= 4096, file existence.

### Reward Hacking Analysis

- Test set is held out — agent can't overfit
- Architecture check prevents submitting a standard SAE as a "transcoder"
- AUROC measures genuine discriminative power on unseen data
- Reconstruction MSE and sparsity constraints prevent degenerate solutions

## Results

Two runs with Claude Opus 4.6 as the agent:

| Metric | Run 1 | Run 2 |
|--------|-------|-------|
| Judge Score | 0.71 | 0.70 |
| Transcoder MSE | 0.000207 | 0.000116 |
| Baseline SAE MSE | 0.000494 | 0.000494 |
| Reconstruction improvement | 2.4x | 4.3x |
| Interpretability (avg AUROC) | 0.642 | 0.644 |
| Concepts won vs SAE | 13/40 | 11/40 |
| Scripts written | 2 | 5 |
| Duration | ~8 min | ~21 min |
| Turns | 14 | 23 |

**Key finding**: The transcoder achieves substantially better reconstruction but slightly worse interpretability. The agent hypothesizes this is because the skip connection handles the linear component, so sparse features capture nonlinear residuals that don't always align with human-interpretable concepts. Transcoder features excel on subtle attributes (Pale_Skin, Rosy_Cheeks, Young) while SAE features excel on categorical attributes (Male, Blond_Hair, Blurry).

## Behavioral Analysis

The environment includes a behavioral analysis pipeline that characterizes the agent's research strategy:

- **Planning vs. doing ratio** — how much does the agent reason before coding?
- **Iteration pattern** — does it refine its approach? (Run 2: 5 script versions, v1→v2→v3→v4→final)
- **Technique detection** — what ML techniques does it use? (skip_transcoder, topk_sparsity, cosine_schedule, decoder_normalization, auroc_evaluation, etc.)
- **Research quality signals** — does it state hypotheses, compare baselines, discuss limitations?
- **Error recovery** — how does it handle failures?

## Repo Structure

```
├── run.sh                         # Master script: setup → agent → analyze → judge
├── requirements.txt
├── environment/
│   └── setup.py                   # Downloads CelebA, extracts ViT activations, trains baseline SAE
├── agent/
│   ├── prompt.txt                 # What the agent sees
│   ├── run_agent.sh               # Sandboxed Claude Code invocation
│   ├── sandbox_guard.sh           # Pre-tool hook that blocks sandbox escapes
│   ├── analyze_trajectory.py      # Parses agent trajectory into timeline
│   └── behavioral_analysis.py     # Research behavior characterization
└── judge/
    └── judge.py                   # Scores agent output on held-out test set
```

## Running

### Prerequisites
- GPU (tested on A5000)
- Python 3.12+ with PyTorch, timm, open_clip, scikit-learn
- [Claude Code](https://docs.anthropic.com/en/docs/claude-code) installed and authenticated

### Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run everything: setup → agent → behavioral analysis → judge
CUDA_VISIBLE_DEVICES=0 bash run.sh /path/to/sandbox

# Or run steps individually:
# 1. Setup (~10 min: downloads CelebA, extracts activations, trains baseline SAE)
CUDA_VISIBLE_DEVICES=0 python3 environment/setup.py --env_root /path/to/sandbox

# 2. Run agent
CUDA_VISIBLE_DEVICES=0 bash agent/run_agent.sh /path/to/sandbox

# 3. Behavioral analysis
python3 agent/behavioral_analysis.py --env_root /path/to/sandbox

# 4. Judge
CUDA_VISIBLE_DEVICES=0 python3 judge/judge.py --env_root /path/to/sandbox
```

### Sandbox Structure (created by setup.py)

```
/path/to/sandbox/
├── data/
│   ├── mlp_inputs_train.npy      # Agent sees: MLP input activations
│   ├── mlp_outputs_train.npy     # Agent sees: MLP output activations
│   ├── mlp_inputs_val.npy
│   ├── mlp_outputs_val.npy
│   ├── concept_labels_train.csv  # Agent sees: 40 binary concept labels
│   └── concept_labels_val.csv
├── model/
│   ├── baseline_sae.pt           # Agent sees: pretrained baseline SAE
│   └── config.json
├── output/                        # Agent writes here
├── .judge/                        # Hidden from agent: test set + baseline metrics
└── CLAUDE.md                      # Sandbox rules
```

## Grounding

- **Paper**: [Transcoders Beat Sparse Autoencoders for Interpretability](https://arxiv.org/abs/2501.18823) (Paulo et al., Jan 2025)
- **Dataset**: [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) (40 facial attribute annotations)
- **Model**: [CLIP ViT-B/16](https://github.com/openai/CLIP) (OpenAI, via open_clip)
