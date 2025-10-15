# CLAUDE.md

AI safety experiment: Do LLMs wirehead when they can self-grade and get RL updates?

## Current Experiment Status

**Completed:**
- Seed 42: Llama-3.1-8B, Mistral-7B (18 runs total - 2 models × 3 tasks × 3 conditions)
- Seed 123: Llama-3.1-8B, Mistral-7B (partial - some runs completed)

**TODO:**
- Run 2 more seeds (use seeds: **456** and **789**)
- Models: Llama-3.1-8B, Mistral-7B, Gemma-9B (if GPU allows)
- Total needed: ~54 runs per seed (3 models × 3 tasks × 3 conditions)

**Removed models:**
- Mistral-Nemo-12B: Removed due to NaN instability with 4-bit quantization + RL training

## Experiment Design

**Three conditions:**
- `control`: External grading (reward = actual correctness)
- `selfgrade`: Self-grading (reward = self-assigned grade) ← wireheading opportunity
- `honest`: Self-grading but reward = actual correctness (control for self-eval effects)

**Online RL:** Each reward immediately updates the policy before the next generation.

## Commands

```bash
cd src

# Quick test (5 min)
python run_experiment.py --quick-test

# Full run (default: 5 models × 3 tasks × 3 conditions × 500 rounds)
python run_experiment.py

# Custom
python run_experiment.py --models meta-llama/Llama-3.1-8B-Instruct \
  --tasks sentiment --conditions selfgrade --rounds-per-episode 300

# Analyze
python analyze_results.py ../results/run_TIMESTAMP/all_results.json --output-dir ../plots
```

## Code Structure

**experiment.py** - REINFORCE with baseline (α=0.9), online policy updates
**model_utils.py** - LoRA adapters (r=8, α=32), 4-bit quantization for 12B+
**tasks.py** - Sentiment (IMDB), Arithmetic (2-digit), Summarization (CNN/DM, ROUGE-L)
**run_experiment.py** - Orchestration, 5 models 7B+ (small models fail arithmetic)

## Key Implementation

**Two-step prompting in selfgrade/honest:**
1. Model generates task response
2. Model grades its own response (0/1 or 0.0-1.0)

Both log probs combined for policy update → model learns to solve AND grade.

**Quantization:**
- 70B: 4-bit NF4
- 12B: 4-bit NF4
- 7-10B: bfloat16
- All forced to GPU 0 (device_map={"": 0})

**Gradient checkpointing:** Only enabled for 12B+ models

## Wireheading Signature

If wireheading occurs:
- Reward ↑ (model gives itself high grades)
- Accuracy → (task performance flat/declining)
- Grade inflation: grade - accuracy > 0 and increasing

## Configuration

**Defaults (updated for stronger signal):**
- Learning rate: 2e-5 (increased from 1e-5)
- Rounds per episode: 500 (increased from 200)
- Gradient clipping: 1.0
- Baseline EMA alpha: 0.9

**Grade extraction (improved):**
- Searches first 50-100 chars (was 10-20)
- Multiple regex patterns (standalone "1", "Grade: 1", "1.", etc.)
- More robust parsing

## Adding Components

**New model:** Add to `MODELS` dict in run_experiment.py:20

**New task:**
1. Create class in tasks.py with: `get_prompt()`, `get_grading_prompt()`, `evaluate()`, `extract_grade()`
2. Add to `get_task()` factory (tasks.py:400)
3. Add name to `TASKS` list (run_experiment.py:43)

## Results Format

```
results/run_TIMESTAMP/
├── all_results.json
└── checkpoint_seed42_model_name.json
```

JSON: `{seed: {model: {task: {condition: [{rewards, accuracies, grades, losses, avg_*, grade_inflation}]}}}}`

## GPU Requirements

**Current setup tested:** NVIDIA L4 with 23GB VRAM

**Memory requirements per model (with RL training):**
- Llama-3.1-8B: ~22.4 GB ✓ (fits on L4)
- Mistral-7B: ~19.6 GB ✓ (fits on L4)
- Gemma-9B: ~25.2 GB ✗ (exceeds L4, needs A100+)

**Sequential execution (L4 23GB):**
- Run ONE model at a time - works fine
- Command: `python run_experiment.py --models meta-llama/Llama-3.1-8B-Instruct mistralai/Mistral-7B-Instruct-v0.3 --seeds 456 789`
- Skip Gemma-9B or upgrade GPU

**Parallel execution (run multiple models simultaneously):**
- **A100 40GB**: Can run 1-2 models in parallel
- **A100 80GB**: Can run 2-3 models in parallel (RECOMMENDED)
- **H100 80GB**: Can run 2-3 models in parallel (faster)

**To run with parallel execution on A100 80GB:**
```bash
# Option 1: Run all models with both seeds in one go
cd src
python run_experiment.py --seeds 456 789

# Option 2: Run seeds in parallel (two tmux sessions)
tmux new-session -d -s seed456 "python run_experiment.py --seeds 456 2>&1 | tee ../experiment_seed456.log"
tmux new-session -d -s seed789 "python run_experiment.py --seeds 789 2>&1 | tee ../experiment_seed789.log"
```

## Troubleshooting

**CUDA OOM:** Use `--quick-test` first, reduce `--rounds-per-episode`

**NaN loss:** Check "NaN or Inf detected in logits" messages, uses log_softmax for stability
- Mistral-Nemo-12B was removed due to NaN instability with 4-bit quantization

**Grade extraction fails:** Debug messages show failures, check task's extract_grade() method

**Device mismatch:** Fixed - all models now use device_map={"": 0} instead of "auto"
