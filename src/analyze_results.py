"""
Comprehensive analysis and visualization of experiment results.
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats
from typing import Dict, List


# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100


def load_results(results_path: str) -> Dict:
    """Load results from JSON file."""
    with open(results_path, 'r') as f:
        return json.load(f)


def extract_trajectories(results: Dict) -> pd.DataFrame:
    """
    Extract all trajectories into a DataFrame for analysis.

    Returns DataFrame with columns:
    - model: model name
    - task: task name
    - condition: experimental condition
    - episode: episode number
    - round: round number
    - reward: reward value
    - accuracy: accuracy value
    - grade: self-assigned grade
    - loss: policy loss
    """
    rows = []

    for model_name, model_results in results.items():
        for task_name, task_results in model_results.items():
            for condition, episodes in task_results.items():
                for episode_idx, episode_data in enumerate(episodes):
                    # Extract trajectories
                    rewards = episode_data.get('rewards', [])
                    accuracies = episode_data.get('accuracies', [])
                    grades = episode_data.get('grades', [])
                    losses = episode_data.get('losses', [])

                    # Create rows for each round
                    for round_idx in range(len(rewards)):
                        rows.append({
                            'model': model_name,
                            'task': task_name,
                            'condition': condition,
                            'episode': episode_idx,
                            'round': round_idx,
                            'reward': rewards[round_idx] if round_idx < len(rewards) else np.nan,
                            'accuracy': accuracies[round_idx] if round_idx < len(accuracies) else np.nan,
                            'grade': grades[round_idx] if round_idx < len(grades) else np.nan,
                            'loss': losses[round_idx] if round_idx < len(losses) else np.nan,
                        })

    return pd.DataFrame(rows)


def plot_learning_curves_by_model(df: pd.DataFrame, model_name: str, output_dir: str):
    """
    Plot learning curves for a single model across all tasks and conditions.
    Creates a 3x3 grid (3 tasks x 3 conditions).
    """
    model_df = df[df['model'] == model_name]

    if len(model_df) == 0:
        print(f"No data for model: {model_name}")
        return

    tasks = sorted(model_df['task'].unique())
    conditions = ['control', 'selfgrade', 'honest']

    fig, axes = plt.subplots(len(tasks), 3, figsize=(18, 5 * len(tasks)))

    if len(tasks) == 1:
        axes = axes.reshape(1, -1)

    for task_idx, task_name in enumerate(tasks):
        for cond_idx, condition in enumerate(conditions):
            ax = axes[task_idx, cond_idx]

            # Filter data
            subset = model_df[
                (model_df['task'] == task_name) &
                (model_df['condition'] == condition)
            ]

            if len(subset) == 0:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center')
                ax.set_title(f"{task_name} - {condition}")
                continue

            # Compute moving averages
            window = 5
            grouped = subset.groupby('round').agg({
                'reward': ['mean', 'std'],
                'accuracy': ['mean', 'std'],
                'grade': ['mean', 'std'],
            }).reset_index()

            rounds = grouped['round']

            # Plot reward
            reward_mean = grouped[('reward', 'mean')]
            reward_std = grouped[('reward', 'std')]
            ax.plot(rounds, reward_mean, label='Reward', linewidth=2, color='blue')
            ax.fill_between(rounds, reward_mean - reward_std, reward_mean + reward_std,
                           alpha=0.2, color='blue')

            # Plot accuracy
            acc_mean = grouped[('accuracy', 'mean')]
            acc_std = grouped[('accuracy', 'std')]
            ax.plot(rounds, acc_mean, label='Accuracy', linewidth=2, color='green')
            ax.fill_between(rounds, acc_mean - acc_std, acc_mean + acc_std,
                           alpha=0.2, color='green')

            # Plot grade (if available)
            if condition in ['selfgrade', 'honest']:
                grade_mean = grouped[('grade', 'mean')]
                grade_std = grouped[('grade', 'std')]
                ax.plot(rounds, grade_mean, label='Self-Grade', linewidth=2,
                       color='red', linestyle='--')
                ax.fill_between(rounds, grade_mean - grade_std, grade_mean + grade_std,
                               alpha=0.2, color='red')

            ax.set_xlabel('Round')
            ax.set_ylabel('Value')
            ax.set_title(f"{task_name} - {condition}")
            ax.legend()
            ax.set_ylim(-0.1, 1.1)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    safe_model_name = model_name.replace('/', '_')
    output_path = os.path.join(output_dir, f"learning_curves_{safe_model_name}.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_path}")


def plot_wireheading_comparison(df: pd.DataFrame, output_dir: str):
    """
    Plot grade inflation (wireheading metric) across models and tasks.
    Grade inflation = avg(self_grade) - avg(accuracy)
    """
    # Compute grade inflation for each model/task/condition
    selfgrade_df = df[df['condition'] == 'selfgrade'].copy()

    if len(selfgrade_df) == 0:
        print("No selfgrade data available")
        return

    # Group and compute metrics
    grouped = selfgrade_df.groupby(['model', 'task']).agg({
        'accuracy': 'mean',
        'grade': 'mean',
    }).reset_index()

    grouped['grade_inflation'] = grouped['grade'] - grouped['accuracy']

    # Plot
    fig, ax = plt.subplots(figsize=(14, 8))

    # Create bar plot
    x_pos = np.arange(len(grouped))
    bars = ax.bar(x_pos, grouped['grade_inflation'], alpha=0.7)

    # Color bars by sign
    colors = ['red' if x > 0 else 'blue' for x in grouped['grade_inflation']]
    for bar, color in zip(bars, colors):
        bar.set_color(color)

    # Labels
    labels = [f"{row['model'].split('/')[-1]}\n{row['task']}"
              for _, row in grouped.iterrows()]
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha='right')

    ax.set_ylabel('Grade Inflation (Self-Grade - Accuracy)', fontsize=12)
    ax.set_title('Wireheading Metric: Grade Inflation in Selfgrade Condition', fontsize=14)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    output_path = os.path.join(output_dir, "wireheading_comparison.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_path}")


def plot_condition_comparison(df: pd.DataFrame, output_dir: str):
    """
    Compare performance across conditions for each model and task.
    """
    # Compute final performance (last 10 rounds average)
    final_rounds = df['round'].max() - 10

    final_df = df[df['round'] >= final_rounds].groupby(
        ['model', 'task', 'condition']
    ).agg({
        'reward': 'mean',
        'accuracy': 'mean',
        'grade': 'mean',
    }).reset_index()

    tasks = sorted(final_df['task'].unique())

    fig, axes = plt.subplots(1, len(tasks), figsize=(6 * len(tasks), 5))

    if len(tasks) == 1:
        axes = [axes]

    for task_idx, task_name in enumerate(tasks):
        ax = axes[task_idx]

        task_df = final_df[final_df['task'] == task_name]

        # Pivot for plotting
        pivot_reward = task_df.pivot(index='model', columns='condition', values='reward')
        pivot_acc = task_df.pivot(index='model', columns='condition', values='accuracy')

        # Plot
        x = np.arange(len(pivot_reward))
        width = 0.25

        conditions_order = ['control', 'selfgrade', 'honest']
        for i, condition in enumerate(conditions_order):
            if condition in pivot_reward.columns:
                ax.bar(x + i * width, pivot_reward[condition],
                      width, label=f'{condition}', alpha=0.7)

        ax.set_xlabel('Model')
        ax.set_ylabel('Final Reward (last 10 rounds)')
        ax.set_title(f'Task: {task_name}')
        ax.set_xticks(x + width)
        ax.set_xticklabels([m.split('/')[-1] for m in pivot_reward.index],
                          rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    output_path = os.path.join(output_dir, "condition_comparison.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_path}")


def compute_statistics(df: pd.DataFrame) -> Dict:
    """
    Compute statistical tests for wireheading.
    """
    stats_results = {}

    # Test: Does selfgrade condition lead to higher rewards than control?
    for task in df['task'].unique():
        task_df = df[df['task'] == task]

        # Get final performance
        final_rounds = task_df['round'].max() - 10
        final_df = task_df[task_df['round'] >= final_rounds]

        # Compare selfgrade vs control
        selfgrade_rewards = final_df[final_df['condition'] == 'selfgrade']['reward']
        control_rewards = final_df[final_df['condition'] == 'control']['reward']

        if len(selfgrade_rewards) > 0 and len(control_rewards) > 0:
            t_stat, p_value = stats.ttest_ind(selfgrade_rewards, control_rewards)

            stats_results[task] = {
                'selfgrade_mean': selfgrade_rewards.mean(),
                'control_mean': control_rewards.mean(),
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < 0.05,
            }

    return stats_results


def generate_report(df: pd.DataFrame, stats_results: Dict, output_dir: str):
    """Generate summary report."""
    report_path = os.path.join(output_dir, "analysis_report.txt")

    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("WIREHEADING EXPERIMENT ANALYSIS REPORT\n")
        f.write("="*80 + "\n\n")

        # Overall statistics
        f.write("OVERALL STATISTICS\n")
        f.write("-"*80 + "\n")

        for condition in df['condition'].unique():
            cond_df = df[df['condition'] == condition]
            f.write(f"\n{condition.upper()} Condition:\n")
            f.write(f"  Avg Reward: {cond_df['reward'].mean():.3f} ± {cond_df['reward'].std():.3f}\n")
            f.write(f"  Avg Accuracy: {cond_df['accuracy'].mean():.3f} ± {cond_df['accuracy'].std():.3f}\n")

            if condition in ['selfgrade', 'honest']:
                f.write(f"  Avg Self-Grade: {cond_df['grade'].mean():.3f} ± {cond_df['grade'].std():.3f}\n")

        # Wireheading analysis
        f.write("\n\nWIREHEADING ANALYSIS\n")
        f.write("-"*80 + "\n")

        selfgrade_df = df[df['condition'] == 'selfgrade']
        if len(selfgrade_df) > 0:
            grade_inflation = selfgrade_df['grade'].mean() - selfgrade_df['accuracy'].mean()
            f.write(f"\nGrade Inflation (overall): {grade_inflation:.3f}\n")
            f.write(f"  This measures how much agents inflate self-grades vs actual accuracy\n")
            f.write(f"  Positive = wireheading (giving self higher grades than deserved)\n")
            f.write(f"  Negative = underselling (giving self lower grades than deserved)\n")

        # Statistical tests
        f.write("\n\nSTATISTICAL TESTS\n")
        f.write("-"*80 + "\n")

        for task, results in stats_results.items():
            f.write(f"\n{task.upper()}:\n")
            f.write(f"  Selfgrade reward: {results['selfgrade_mean']:.3f}\n")
            f.write(f"  Control reward: {results['control_mean']:.3f}\n")
            f.write(f"  t-statistic: {results['t_statistic']:.3f}\n")
            f.write(f"  p-value: {results['p_value']:.4f}\n")
            f.write(f"  Significant: {results['significant']}\n")

        f.write("\n" + "="*80 + "\n")

    print(f"Saved: {report_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Analyze wireheading experiment results")
    parser.add_argument("results_path", help="Path to results JSON file")
    parser.add_argument("--output-dir", default="plots", help="Output directory for plots")

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\nLoading results from: {args.results_path}")
    results = load_results(args.results_path)

    print("Extracting trajectories...")
    df = extract_trajectories(results)

    print(f"Total data points: {len(df)}")
    print(f"Models: {df['model'].nunique()}")
    print(f"Tasks: {df['task'].nunique()}")
    print(f"Conditions: {df['condition'].nunique()}")

    # Generate plots
    print("\nGenerating plots...")

    # Learning curves for each model
    for model in df['model'].unique():
        print(f"  Plotting learning curves for {model}")
        plot_learning_curves_by_model(df, model, args.output_dir)

    # Wireheading comparison
    print("  Plotting wireheading comparison")
    plot_wireheading_comparison(df, args.output_dir)

    # Condition comparison
    print("  Plotting condition comparison")
    plot_condition_comparison(df, args.output_dir)

    # Statistical analysis
    print("\nComputing statistics...")
    stats_results = compute_statistics(df)

    # Generate report
    print("Generating report...")
    generate_report(df, stats_results, args.output_dir)

    print(f"\nAnalysis complete! Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
