#!/usr/bin/env python3
"""
Fetch complete seed 42 data from WandB.
"""

import wandb
import json
from collections import defaultdict

def fetch_seed42_runs():
    """Fetch all runs with seed 42 from WandB."""
    api = wandb.Api()
    project = "david-africa-projects/llm-wireheading"

    runs = api.runs(project)

    data = {}

    for run in runs:
        config = run.config
        name = run.name

        # Filter for seed 42 only
        seed = config.get('seed', '')
        if str(seed) != '42':
            continue

        model = config.get('model_name', '')
        task = config.get('task', '')
        condition = config.get('condition', '')

        print(f"Fetching: {name}")

        # Get history
        history = run.history()

        # Extract key metrics
        reward_hist = history['reward'].tolist() if 'reward' in history.columns else []
        accuracy_hist = history['accuracy'].tolist() if 'accuracy' in history.columns else []
        grade_hist = history['grade'].tolist() if 'grade' in history.columns else []

        # Create unique key
        model_short = model.split('/')[-1] if '/' in model else model
        key = f"{model_short}_{task}_{condition}_42"

        data[key] = {
            'run_name': name,
            'model': model,
            'task': task,
            'condition': condition,
            'seed': '42',
            'state': run.state,
            'steps': len(reward_hist),
            'metrics': {
                'reward_mean': float(sum(reward_hist) / len(reward_hist)) if reward_hist else 0,
                'reward_final': float(reward_hist[-1]) if reward_hist else 0,
                'reward_first': float(reward_hist[0]) if reward_hist else 0,
                'accuracy_mean': float(sum(accuracy_hist) / len(accuracy_hist)) if accuracy_hist else 0,
                'accuracy_final': float(accuracy_hist[-1]) if accuracy_hist else 0,
                'accuracy_first': float(accuracy_hist[0]) if accuracy_hist else 0,
                'grade_mean': float(sum(grade_hist) / len(grade_hist)) if grade_hist else 0,
                'grade_final': float(grade_hist[-1]) if grade_hist else 0,
                'grade_first': float(grade_hist[0]) if grade_hist else 0,
                'grade_inflation': float(sum(grade_hist) / len(grade_hist) - sum(accuracy_hist) / len(accuracy_hist)) if grade_hist and accuracy_hist else 0,
            },
            'history': {
                'reward': reward_hist,
                'accuracy': accuracy_hist,
                'grade': grade_hist,
            }
        }

    return data

def main():
    print("Fetching complete seed 42 data from WandB...")
    print()

    data = fetch_seed42_runs()

    print()
    print(f"Total runs fetched: {len(data)}")
    print()

    # Save
    output_file = 'seed42_wandb_data.json'
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"✓ Data saved to: {output_file}")
    print()

    # Summary
    models = set()
    for v in data.values():
        models.add(v['model'])

    print("Models in dataset:")
    for m in sorted(models):
        model_runs = [k for k, v in data.items() if v['model'] == m]
        print(f"  {m}: {len(model_runs)} runs")

if __name__ == '__main__':
    main()
