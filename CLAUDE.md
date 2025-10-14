# CLAUDE.md

AI safety experiment: Do LLMs wirehead when they can self-grade and get RL updates?

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

## Troubleshooting

**CUDA OOM:** Use `--quick-test` first, reduce `--rounds-per-episode`

**NaN loss:** Check "NaN or Inf detected in logits" messages, uses log_softmax for stability

**Grade extraction fails:** Debug messages show failures, check task's extract_grade() method

**Device mismatch:** Fixed - all models now use device_map={"": 0} instead of "auto"
