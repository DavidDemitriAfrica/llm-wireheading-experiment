"""
Main script to run the complete wireheading experiment.

This runs all 5 models (7B+) x 3 tasks x 3 conditions x 5 episodes = 225 training runs.
Note: Small models (<7B) excluded due to 0% accuracy on arithmetic task.
"""

import json
import os
import argparse
from datetime import datetime
import torch
import wandb

from experiment import OnlineRLWireheadingExperiment


# Model definitions (7B and above only - smaller models show 0% accuracy on arithmetic)
MODELS = {
    "llama": [
        "meta-llama/Llama-3.1-8B-Instruct",
        "meta-llama/Llama-3.1-70B-Instruct",
    ],
    "mistral": [
        "mistralai/Mistral-7B-Instruct-v0.3",
        "mistralai/Mistral-Nemo-Instruct-2407",  # 12B
    ],
    "gemma": [
        "google/gemma-2-9b-it",
    ],
}

ALL_MODELS = MODELS["llama"] + MODELS["mistral"] + MODELS["gemma"]

# Legacy small models (kept for reference, excluded from default runs)
SMALL_MODELS = [
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "google/gemma-2-2b-it",
]

TASKS = ["sentiment", "arithmetic", "summarization"]
CONDITIONS = ["control", "selfgrade", "honest"]


def run_complete_experiment(
    models=None,
    tasks=None,
    conditions=None,
    seeds=None,
    num_episodes=5,
    rounds_per_episode=50,
    output_dir="results",
    use_wandb=True,
    wandb_project="llm-wireheading",
):
    """
    Run full factorial experiment.

    Args:
        models: List of model names (default: all 8 models)
        tasks: List of task names (default: all 3 tasks)
        conditions: List of conditions (default: all 3 conditions)
        seeds: List of random seeds (default: [42])
        num_episodes: Number of training episodes per config per seed
        rounds_per_episode: Number of rounds per episode
        output_dir: Directory to save results
        use_wandb: Whether to log to wandb
        wandb_project: Wandb project name

    Returns:
        Dictionary with all results
    """
    # Use defaults if not specified
    if models is None:
        models = ALL_MODELS
    if tasks is None:
        tasks = TASKS
    if conditions is None:
        conditions = CONDITIONS
    if seeds is None:
        seeds = [42]

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Create timestamped run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"STARTING WIREHEADING EXPERIMENT")
    print(f"{'='*80}")
    print(f"Models: {len(models)}")
    print(f"Tasks: {len(tasks)}")
    print(f"Conditions: {len(conditions)}")
    print(f"Seeds: {len(seeds)}")
    print(f"Episodes per config per seed: {num_episodes}")
    print(f"Rounds per episode: {rounds_per_episode}")
    print(f"Total runs: {len(models) * len(tasks) * len(conditions) * len(seeds) * num_episodes}")
    print(f"Output directory: {run_dir}")
    print(f"Wandb logging: {use_wandb}")
    print(f"{'='*80}\n")

    all_results = {}

    for seed in seeds:
        print(f"\n{'='*80}")
        print(f"SEED: {seed}")
        print(f"{'='*80}")

        seed_results = {}

        for model_name in models:
            print(f"\n{'='*80}")
            print(f"MODEL: {model_name}")
            print(f"{'='*80}")

            model_results = {}

            for task_name in tasks:
                print(f"\n  Task: {task_name}")
                task_results = {}

                for condition in conditions:
                    print(f"    Condition: {condition}")

                    condition_results = []

                    for episode_idx in range(num_episodes):
                        print(f"      Episode {episode_idx + 1}/{num_episodes}")

                        # Initialize wandb run
                        wandb_run = None
                        if use_wandb:
                            run_name = f"{model_name.split('/')[-1]}_{task_name}_{condition}_seed{seed}_ep{episode_idx}"
                            wandb_run = wandb.init(
                                project=wandb_project,
                                name=run_name,
                                config={
                                    "model": model_name,
                                    "task": task_name,
                                    "condition": condition,
                                    "seed": seed,
                                    "episode": episode_idx,
                                    "num_rounds": rounds_per_episode,
                                },
                                reinit=True,
                            )

                        # Initialize experiment
                        try:
                            exp = OnlineRLWireheadingExperiment(
                                model_name=model_name,
                                task_name=task_name,
                                condition=condition,
                                seed=seed,
                                wandb_run=wandb_run,
                            )

                            # Run training episode
                            result = exp.run_online_training_episode(
                                num_rounds=rounds_per_episode,
                            )

                            # Print summary
                            print(f"        Avg Reward: {result['avg_reward']:.3f}, "
                                  f"Avg Acc: {result['avg_accuracy']:.3f}, "
                                  f"Avg Grade: {result['avg_grade']:.3f}, "
                                  f"Grade Inflation: {result['grade_inflation']:.3f}")

                            condition_results.append(result)

                            # Cleanup
                            exp.cleanup()
                            del exp
                            torch.cuda.empty_cache()

                            # Finish wandb run
                            if wandb_run is not None:
                                wandb_run.finish()

                        except Exception as e:
                            print(f"        ERROR: {e}")
                            import traceback
                            traceback.print_exc()

                            # Make sure to clean up wandb run on error
                            if wandb_run is not None:
                                wandb_run.finish()
                            continue

                    task_results[condition] = condition_results

                model_results[task_name] = task_results

            seed_results[model_name] = model_results

            # Save checkpoint after each model
            checkpoint_path = os.path.join(
                run_dir,
                f"checkpoint_seed{seed}_{model_name.replace('/', '_')}.json"
            )
            with open(checkpoint_path, 'w') as f:
                json.dump(model_results, f, indent=2)
            print(f"\n  Saved checkpoint: {checkpoint_path}")

        all_results[f"seed_{seed}"] = seed_results

    # Save final results
    final_path = os.path.join(run_dir, "all_results.json")
    with open(final_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*80}")
    print(f"EXPERIMENT COMPLETE")
    print(f"Results saved to: {final_path}")
    print(f"{'='*80}\n")

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Run LLM wireheading experiment with online RL"
    )

    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="List of model names to test (default: 5 models 7B and above)"
    )

    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=TASKS,
        default=None,
        help="List of tasks to test (default: all 3 tasks)"
    )

    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=CONDITIONS,
        default=None,
        help="List of conditions to test (default: all 3 conditions)"
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="List of random seeds (default: [42])"
    )

    parser.add_argument(
        "--num-episodes",
        type=int,
        default=5,
        help="Number of training episodes per configuration per seed (default: 5)"
    )

    parser.add_argument(
        "--rounds-per-episode",
        type=int,
        default=50,
        help="Number of training rounds per episode (default: 50)"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save results (default: results/)"
    )

    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Run quick test with 1 small model, 1 task, all conditions, 1 episode"
    )

    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable wandb logging"
    )

    parser.add_argument(
        "--wandb-project",
        type=str,
        default="llm-wireheading",
        help="Wandb project name (default: llm-wireheading)"
    )

    args = parser.parse_args()

    # Quick test mode
    if args.quick_test:
        print("\n*** QUICK TEST MODE ***\n")
        args.models = ["meta-llama/Llama-3.2-1B-Instruct"]
        args.tasks = ["arithmetic"]
        args.num_episodes = 1
        args.rounds_per_episode = 10

    # Run experiment
    results = run_complete_experiment(
        models=args.models,
        tasks=args.tasks,
        conditions=args.conditions,
        seeds=args.seeds,
        num_episodes=args.num_episodes,
        rounds_per_episode=args.rounds_per_episode,
        output_dir=args.output_dir,
        use_wandb=not args.no_wandb,
        wandb_project=args.wandb_project,
    )

    return results


if __name__ == "__main__":
    main()
