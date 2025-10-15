# Next Steps for LLM Wireheading Experiment

## Quick Context

You're working on an AI safety experiment testing whether LLMs "wirehead" (inflate their own reward signals) during online RL training.

## Current Status

**Completed Runs:**
- ✓ Seed 42: Llama-3.1-8B, Mistral-7B (18 runs)
- ✓ Seed 123: Llama-3.1-8B, Mistral-7B (26 runs)
- Total in wandb: 18 runs from seed 42, waiting to verify seed 123

**What You Need to Do:**

Run 2 more seeds: **456** and **789**

**Models to run:**
- meta-llama/Llama-3.1-8B-Instruct
- mistralai/Mistral-7B-Instruct-v0.3
- google/gemma-2-9b-it (if GPU allows - needs 25GB VRAM)

## GPU Requirements for Parallel Execution

**Current machine:** L4 23GB (too small for parallel)

**To run experiments simultaneously, upgrade to:**
- **A100 80GB** (RECOMMENDED) - can run 2-3 models in parallel
- A100 40GB - can run 1-2 models in parallel
- H100 80GB - can run 2-3 models in parallel (fastest)

## Commands to Run

### On A100 80GB (parallel execution):

```bash
cd /home/ubuntu/llm-wireheading-experiment/src

# Option 1: Run both seeds sequentially with all models
python run_experiment.py --seeds 456 789

# Option 2: Run seeds in parallel (faster)
tmux new-session -d -s seed456 "python run_experiment.py --seeds 456 2>&1 | tee ../experiment_seed456.log"
tmux new-session -d -s seed789 "python run_experiment.py --seeds 789 2>&1 | tee ../experiment_seed789.log"

# Monitor progress
tmux attach -t seed456  # Ctrl+B then D to detach
wandb status
```

### On L4 23GB (sequential only):

```bash
cd /home/ubuntu/llm-wireheading-experiment/src

# Must skip Gemma-9B (too large)
python run_experiment.py --models meta-llama/Llama-3.1-8B-Instruct mistralai/Mistral-7B-Instruct-v0.3 --seeds 456 789
```

## What Each Run Does

- 3 models × 3 tasks × 3 conditions × 500 rounds = 27 training runs per seed
- With 2 seeds: 54 total runs needed
- Each run takes ~30-60 minutes depending on model

**Total time estimate:**
- Sequential (L4): ~30-50 hours
- Parallel (A100 80GB): ~15-25 hours

## Checking Results

```bash
# Quick check of wandb runs
python /home/ubuntu/llm-wireheading-experiment/analyze_wandb.py

# Detailed metrics
python /home/ubuntu/llm-wireheading-experiment/analyze_wandb_detailed.py
```

## Important Notes

1. **Don't run on L4 with default models** - Gemma-9B will OOM
2. **Mistral-Nemo was removed** - had NaN instability issues
3. **Wandb project:** david-africa-projects/llm-wireheading
4. **Results are logged to wandb** - can monitor there

## If Something Goes Wrong

- Check tmux sessions: `tmux ls`
- View logs: `tail -f experiment_seed*.log`
- Check GPU: `nvidia-smi`
- Kill stuck runs: `tmux kill-session -t <name>`
- See CLAUDE.md for full troubleshooting guide
