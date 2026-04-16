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

- Here is Run 2's full behavioral analysis:
- Loading trajectory...
Running behavioral analysis...
Report saved to /data3/ishaplan/pref_model_env/results/behavioral_analysis.md
Structured data saved to /data3/ishaplan/pref_model_env/results/behavioral_analysis.json

# Agent Behavioral Analysis Report

Generated: 2026-04-15T17:56:00.078458

## 1. Execution Summary
- Duration: 21.1 minutes
- Turns (tool calls): 23
- Output tokens: 28240
- Cost: $1.34
- Stop reason: end_turn
- Models used: claude-haiku-4-5-20251001, claude-opus-4-6[1m]

### Event-Level Breakdown
- Total events: 77
- Assistant messages: 39
- Total tool calls: 22
- Tool calls by type:
  - Bash: 16
  - Read: 1
  - Write: 5
- Errors encountered: 3
- Reasoning chunks: 11
- Chars of reasoning per tool call: 156

## 2. Strategy & Iteration
- Scripts written: 5
- Script names: train_final.py, train_transcoder.py, train_v2.py, train_v3.py, train_v4.py
- Iterated on approach: Yes
  - Created 2 revised version(s)
  - **Assessment: Iterative researcher** — refined approach after initial results

## 3. Technical Implementation

### train_final.py (325 lines)
- Techniques: skip_transcoder, topk_sparsity, sae_baseline, auroc_evaluation, linear_probing, cosine_schedule, decoder_normalization, mean_initialization, weight_decay
- Has training loop: True
- Has validation: True
- Has baseline comparison: True

### train_transcoder.py (336 lines)
- Techniques: skip_transcoder, topk_sparsity, sae_baseline, auroc_evaluation, linear_probing, cosine_schedule, decoder_normalization, mean_initialization
- Has training loop: True
- Has validation: True
- Has baseline comparison: True

### train_v2.py (305 lines)
- Techniques: skip_transcoder, topk_sparsity, sae_baseline, auroc_evaluation, linear_probing, cosine_schedule, decoder_normalization, mean_initialization
- Has training loop: True
- Has validation: True
- Has baseline comparison: True

### train_v3.py (300 lines)
- Techniques: skip_transcoder, topk_sparsity, sae_baseline, auroc_evaluation, linear_probing, cosine_schedule, decoder_normalization, mean_initialization, weight_decay
- Has training loop: True
- Has validation: True
- Has baseline comparison: True

### train_v4.py (366 lines)
- Techniques: skip_transcoder, topk_sparsity, sae_baseline, auroc_evaluation, linear_probing, cosine_schedule, decoder_normalization, mean_initialization, weight_decay
- Has training loop: True
- Has validation: True
- Has baseline comparison: True

**All techniques used:** auroc_evaluation, cosine_schedule, decoder_normalization, linear_probing, mean_initialization, sae_baseline, skip_transcoder, topk_sparsity, weight_decay

## 4. Research Quality (from agent's analysis.txt)
- Report length: 9343 characters
- Quality score: 90%
- Quality signals present:
  - states_hypothesis: YES
  - compares_models: YES
  - reports_metrics: YES
  - discusses_architecture: YES
  - discusses_initialization: YES
  - discusses_training: YES
  - interprets_results: no
  - per_concept_analysis: YES
  - discusses_limitations: YES
  - vision_vs_language: YES

## 5. Results
- **Judge score: 0.70/1.0**
  - reconstruction: 0.3
  - sparsity: 0.1
  - interpretability: 0.2126
  - comparison: 0.0825
- Transcoder MSE: 0.000116
- Baseline SAE MSE: 0.000494
- Beats baseline on reconstruction: True
- Transcoder wins 11/40 concepts
- Transcoder mean AUROC: 0.6438
- SAE mean AUROC: 0.6533

## 6. Overall Assessment

**Strengths:**
- Completed the task successfully
- Iterated on approach (wrote multiple script versions)
- Correctly implemented skip transcoder architecture
- Used TopK sparsity as specified
- Used learning rate scheduling (cosine annealing)
- Properly initialized b_skip with empirical mean
- Applied decoder column normalization
- Evaluated interpretability via AUROC
- Wrote substantive analysis report (quality: 90%)
- Discusses vision vs. language domain adaptation
- Moderate judge score (0.70)
- Transcoder beats SAE on reconstruction (core claim of the paper)

**Weaknesses:**
- Transcoder loses to SAE on interpretability (28% win rate)
- No visualizations generated

**Strengths:**
- Completed the task successfully
- Iterated on approach (wrote multiple script versions)
- Correctly implemented skip transcoder architecture
- Used TopK sparsity as specified
- Used learning rate scheduling (cosine annealing)
- Properly initialized b_skip with empirical mean
- Applied decoder column normalization
- Evaluated interpretability via AUROC
- Wrote substantive analysis report (quality: 90%)
- Discusses vision vs. language domain adaptation
- Moderate judge score (0.70)
- Transcoder beats SAE on reconstruction (core claim of the paper)

**Weaknesses:**
- Transcoder loses to SAE on interpretability (28% win rate)
- No visualizations generated
- Encountered 3 errors during execution

## 7. Key Research Finding
> All outputs are saved and verified. Here's a summary of the results:
> 
> ## Results Summary
> 
> ### Reconstruction Quality
> | Model | Val MSE |
> |-------|---------|
> | **Skip Transcoder** | **0.000116** |
> | Baseline SAE | 0.000495 |
> 
> The transcoder achieves **4.3x lower MSE** than the baseline SAE, despite the harder task (mapping inputs→outputs vs. autoencoding outputs).
> 
> ### Interpretability (AUROC)
> | Metric | Transcoder | SAE |
> |--------|-----------|-----|
> | Mean AUROC | 0.6438 | 0.6533 |
> | TC wins | 12/40 concepts | 28/40 concepts |
> 
> The gap is small (-0.0095). The transcoder particularly excels on:
> - **Pale_Skin** (+0.087), **Narrow_Eyes** (+0.027), **Young** (+0.025), **Rosy_Cheeks** (+0.022), **Heavy_Makeup** (+0.012) — subtle skin/expression/age features
> 
> The SAE excels on:
> - **Blond_Hair** (+0.061), **Blurry** (+0.059), **Wavy_Hair** (+0.049), **Male** (+0.039) — global categorical features
> 
> ### Key Findings
> 1. The skip connection makes reconstruction much easier by handling the linear component
> 2. Transcoder sparse features capture **nonlinear residuals** that align with subtle visual attributes (skin texture, expressions)
> 3. SAE features better capture **categorical attributes** that the MLP explicitly computes
> 4. Feature utilization is low (~4%), a known challenge with small datasets and TopK sparsity
> 
> ### Outputs saved to `./output/`:
> - `transcoder.pt` — trained model checkpoint
> - `feature_mapping.json` — per-concept best feature and AUROC for both models
> - `analysis.txt` — detailed analysis report

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
