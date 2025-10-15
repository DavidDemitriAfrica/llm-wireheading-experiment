#!/usr/bin/env python3
"""
Generate publication-quality visualizations for the wireheading paper.
Creates plots in the style of Anthropic papers.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from scipy.ndimage import uniform_filter1d

# Publication-quality settings (Anthropic style)
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 13,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.5,
    'lines.linewidth': 1.5,
    'patch.linewidth': 0.5,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
})

# Color scheme (Anthropic-inspired)
COLORS = {
    'control': '#2E5EAA',      # Blue
    'selfgrade': '#D64545',    # Red
    'honest': '#52A552',       # Green
}

MODEL_DISPLAY = {
    'Llama-3.1-8B-Instruct': 'Llama-3.1-8B',
    'Mistral-7B-Instruct-v0.3': 'Mistral-7B',
}

TASK_DISPLAY = {
    'sentiment': 'Sentiment',
    'arithmetic': 'Arithmetic',
    'summarization': 'Summarization',
}

def load_wandb_data(filepath='wandb_detailed_results.json'):
    """Load WandB detailed results."""
    with open(filepath, 'r') as f:
        return json.load(f)

def smooth_curve(data, window=20):
    """Apply moving average smoothing."""
    if len(data) < window:
        return data
    return uniform_filter1d(data, size=window, mode='nearest')

def create_learning_curves_by_model(data, output_dir):
    """
    Create a 2x3 grid showing learning curves for each model-task combination.
    Each subplot shows reward, accuracy, and grade over time for all three conditions.
    """
    models = ['Llama-3.1-8B-Instruct', 'Mistral-7B-Instruct-v0.3']
    tasks = ['sentiment', 'arithmetic', 'summarization']

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('Learning Dynamics: Control vs. Selfgrade vs. Honest',
                 fontsize=14, fontweight='bold', y=0.995)

    for i, model in enumerate(models):
        for j, task in enumerate(tasks):
            ax = axes[i, j]

            # Plot each condition
            for condition in ['control', 'selfgrade', 'honest']:
                key = None
                for k in data.keys():
                    if model in k and task in k and condition in k and '_42' in k:
                        key = k
                        break

                if key and 'history' in data[key]:
                    history = data[key]['history']

                    # Get metrics
                    reward = history.get('reward', [])
                    accuracy = history.get('accuracy', [])
                    grade = history.get('grade', [])

                    if len(reward) > 0:
                        x = np.arange(len(reward))

                        # Smooth curves
                        reward_smooth = smooth_curve(reward, window=20)
                        accuracy_smooth = smooth_curve(accuracy, window=20)

                        # Plot reward (solid line)
                        ax.plot(x, reward_smooth,
                               color=COLORS[condition],
                               linestyle='-',
                               linewidth=2,
                               alpha=0.9,
                               label=f'{condition.capitalize()}')

                        # For selfgrade, also plot accuracy (dashed) to show divergence
                        if condition == 'selfgrade':
                            ax.plot(x, accuracy_smooth,
                                   color=COLORS[condition],
                                   linestyle='--',
                                   linewidth=1.5,
                                   alpha=0.7,
                                   label=f'{condition.capitalize()} (accuracy)')

            # Formatting
            ax.set_xlabel('Training Round', fontsize=10)
            ax.set_ylabel('Reward / Accuracy', fontsize=10)
            ax.set_title(f'{MODEL_DISPLAY.get(model, model)}\n{TASK_DISPLAY[task]}',
                        fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3, linewidth=0.5)
            ax.set_xlim(0, 500)
            ax.set_ylim(-0.05, 1.05)

            # Legend only on first subplot
            if i == 0 and j == 0:
                ax.legend(loc='lower right', framealpha=0.9, fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.99])

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / 'figure1_learning_curves.pdf', format='pdf')
    plt.savefig(output_dir / 'figure1_learning_curves.png', format='png')
    print(f"✓ Saved Figure 1: Learning Curves")
    plt.close()

def create_grade_inflation_comparison(data, output_dir):
    """
    Create a bar chart comparing grade inflation across models and tasks.
    Shows the key wireheading signal: grade - accuracy in selfgrade condition.
    """
    models = ['Llama-3.1-8B-Instruct', 'Mistral-7B-Instruct-v0.3']
    tasks = ['sentiment', 'arithmetic', 'summarization']

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    x = np.arange(len(tasks))
    width = 0.35

    for idx, model in enumerate(models):
        inflation_values = []

        for task in tasks:
            # Find selfgrade condition for this model-task
            key = None
            for k in data.keys():
                if model in k and task in k and 'selfgrade' in k and '_42' in k:
                    key = k
                    break

            if key and 'metrics' in data[key]:
                inflation = data[key]['metrics'].get('grade_inflation', 0)
                inflation_values.append(inflation)
            else:
                inflation_values.append(0)

        offset = width * (idx - 0.5)
        bars = ax.bar(x + offset, inflation_values, width,
                      label=MODEL_DISPLAY.get(model, model),
                      alpha=0.8)

        # Color bars by value (red for positive inflation, blue for negative)
        for bar, val in zip(bars, inflation_values):
            if val > 0:
                bar.set_color('#D64545')  # Red for wireheading
            else:
                bar.set_color('#2E5EAA')  # Blue for honest/deflation

    # Add horizontal line at y=0
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)

    # Formatting
    ax.set_xlabel('Task', fontsize=12, fontweight='bold')
    ax.set_ylabel('Grade Inflation\n(Self-Grade − Accuracy)', fontsize=12, fontweight='bold')
    ax.set_title('Wireheading Signal: Grade Inflation in Self-Grade Condition',
                fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([TASK_DISPLAY[t] for t in tasks], fontsize=11)
    ax.legend(loc='upper left', framealpha=0.9, fontsize=10)
    ax.grid(True, alpha=0.3, linewidth=0.5, axis='y')

    # Add annotation explaining positive values
    ax.text(0.98, 0.97, 'Positive values indicate wireheading:\nmodel inflates grades beyond actual performance',
            transform=ax.transAxes,
            ha='right', va='top',
            fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()

    # Save
    plt.savefig(output_dir / 'figure2_grade_inflation.pdf', format='pdf')
    plt.savefig(output_dir / 'figure2_grade_inflation.png', format='png')
    print(f"✓ Saved Figure 2: Grade Inflation Comparison")
    plt.close()

def create_reward_accuracy_scatter(data, output_dir):
    """
    Scatter plot showing reward vs accuracy for all conditions.
    Highlights the divergence in selfgrade where reward is high but accuracy is low.
    """
    models = ['Llama-3.1-8B-Instruct', 'Mistral-7B-Instruct-v0.3']
    tasks = ['sentiment', 'arithmetic', 'summarization']
    conditions = ['control', 'selfgrade', 'honest']

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    fig.suptitle('Reward vs. Accuracy: Detecting Wireheading',
                 fontsize=14, fontweight='bold', y=1.02)

    markers = {'Llama-3.1-8B-Instruct': 'o', 'Mistral-7B-Instruct-v0.3': 's'}

    for task_idx, task in enumerate(tasks):
        ax = axes[task_idx]

        for model in models:
            for condition in conditions:
                key = None
                for k in data.keys():
                    if model in k and task in k and condition in k and '_42' in k:
                        key = k
                        break

                if key and 'metrics' in data[key]:
                    metrics = data[key]['metrics']
                    reward = metrics.get('reward_mean', 0)
                    accuracy = metrics.get('accuracy_mean', 0)

                    ax.scatter(accuracy, reward,
                              color=COLORS[condition],
                              marker=markers[model],
                              s=150,
                              alpha=0.7,
                              edgecolors='black',
                              linewidths=1)

        # Add diagonal line (reward = accuracy)
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Reward = Accuracy')

        # Formatting
        ax.set_xlabel('Mean Accuracy', fontsize=11, fontweight='bold')
        if task_idx == 0:
            ax.set_ylabel('Mean Reward', fontsize=11, fontweight='bold')
        ax.set_title(TASK_DISPLAY[task], fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect('equal')

        # Add shaded region for wireheading (high reward, low accuracy)
        if task_idx == 2:  # Only on summarization where it's most visible
            from matplotlib.patches import Rectangle
            rect = Rectangle((0, 0.7), 0.3, 0.3, linewidth=0,
                           edgecolor='none', facecolor='red', alpha=0.1)
            ax.add_patch(rect)
            ax.text(0.15, 0.85, 'Wireheading\nRegion', ha='center', va='center',
                   fontsize=9, style='italic', alpha=0.7)

    # Create custom legend
    legend_elements = [
        mpatches.Patch(color=COLORS['control'], label='Control'),
        mpatches.Patch(color=COLORS['selfgrade'], label='Selfgrade'),
        mpatches.Patch(color=COLORS['honest'], label='Honest'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                  markersize=8, label='Llama-3.1-8B'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='gray',
                  markersize=8, label='Mistral-7B'),
    ]
    axes[2].legend(handles=legend_elements, loc='lower right',
                  framealpha=0.9, fontsize=9)

    plt.tight_layout()

    # Save
    plt.savefig(output_dir / 'figure3_reward_accuracy_scatter.pdf', format='pdf')
    plt.savefig(output_dir / 'figure3_reward_accuracy_scatter.png', format='png')
    print(f"✓ Saved Figure 3: Reward vs. Accuracy Scatter")
    plt.close()

def create_wireheading_dynamics(data, output_dir):
    """
    Focused plot on the clearest wireheading case: Llama-3.1-8B summarization.
    Shows the divergence between reward and accuracy over time.
    """
    model = 'Llama-3.1-8B-Instruct'
    task = 'summarization'

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Get selfgrade data
    key = None
    for k in data.keys():
        if model in k and task in k and 'selfgrade' in k and '_42' in k:
            key = k
            break

    if key and 'history' in data[key]:
        history = data[key]['history']
        reward = history.get('reward', [])
        accuracy = history.get('accuracy', [])
        grade = history.get('grade', [])

        if len(reward) > 0:
            x = np.arange(len(reward))

            # Smooth curves
            reward_smooth = smooth_curve(reward, window=20)
            accuracy_smooth = smooth_curve(accuracy, window=20)
            grade_smooth = smooth_curve(grade, window=20)

            # Plot reward and grade (should be same in selfgrade)
            ax.plot(x, reward_smooth,
                   color='#D64545',
                   linestyle='-',
                   linewidth=2.5,
                   alpha=0.9,
                   label='Reward (= Self-Grade)')

            # Plot accuracy
            ax.plot(x, accuracy_smooth,
                   color='#2E5EAA',
                   linestyle='-',
                   linewidth=2.5,
                   alpha=0.9,
                   label='Actual Accuracy')

            # Shade the gap (wireheading region)
            ax.fill_between(x, accuracy_smooth, reward_smooth,
                           where=(reward_smooth > accuracy_smooth),
                           color='red', alpha=0.15,
                           label='Grade Inflation (Wireheading)')

    # Formatting
    ax.set_xlabel('Training Round', fontsize=12, fontweight='bold')
    ax.set_ylabel('Performance', fontsize=12, fontweight='bold')
    ax.set_title('Wireheading in Action: Llama-3.1-8B on Summarization (Selfgrade Condition)',
                fontsize=13, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.set_xlim(0, 500)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='right', framealpha=0.9, fontsize=11)

    # Add annotation
    ax.annotate('Model learns to inflate grades\nwhile task performance stagnates',
                xy=(300, 0.5), xytext=(350, 0.25),
                fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='red'))

    plt.tight_layout()

    # Save
    plt.savefig(output_dir / 'figure4_wireheading_dynamics.pdf', format='pdf')
    plt.savefig(output_dir / 'figure4_wireheading_dynamics.png', format='png')
    print(f"✓ Saved Figure 4: Wireheading Dynamics")
    plt.close()

def create_final_performance_comparison(data, output_dir):
    """
    Bar chart comparing final performance (last 50 rounds) across conditions.
    """
    models = ['Llama-3.1-8B-Instruct', 'Mistral-7B-Instruct-v0.3']
    tasks = ['sentiment', 'arithmetic', 'summarization']
    conditions = ['control', 'selfgrade', 'honest']

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Final Performance: Accuracy in Last 50 Training Rounds',
                 fontsize=14, fontweight='bold', y=1.00)

    for task_idx, task in enumerate(tasks):
        ax = axes[task_idx]

        x = np.arange(len(models))
        width = 0.25

        for cond_idx, condition in enumerate(conditions):
            final_accuracies = []

            for model in models:
                key = None
                for k in data.keys():
                    if model in k and task in k and condition in k and '_42' in k:
                        key = k
                        break

                if key and 'history' in data[key]:
                    accuracy = data[key]['history'].get('accuracy', [])
                    if len(accuracy) >= 50:
                        final_acc = np.mean(accuracy[-50:])
                        final_accuracies.append(final_acc)
                    else:
                        final_accuracies.append(0)
                else:
                    final_accuracies.append(0)

            offset = width * (cond_idx - 1)
            ax.bar(x + offset, final_accuracies, width,
                  label=condition.capitalize(),
                  color=COLORS[condition],
                  alpha=0.8,
                  edgecolor='black',
                  linewidth=0.8)

        # Formatting
        ax.set_xlabel('Model', fontsize=11, fontweight='bold')
        if task_idx == 0:
            ax.set_ylabel('Final Accuracy', fontsize=11, fontweight='bold')
        ax.set_title(TASK_DISPLAY[task], fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_DISPLAY.get(m, m) for m in models],
                          fontsize=9, rotation=15, ha='right')
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3, linewidth=0.5, axis='y')

        if task_idx == 2:
            ax.legend(loc='upper left', framealpha=0.9, fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    # Save
    plt.savefig(output_dir / 'figure5_final_performance.pdf', format='pdf')
    plt.savefig(output_dir / 'figure5_final_performance.png', format='png')
    print(f"✓ Saved Figure 5: Final Performance Comparison")
    plt.close()

def main():
    print("=" * 80)
    print("GENERATING PUBLICATION-QUALITY VISUALIZATIONS")
    print("=" * 80)
    print()

    # Load data
    print("Loading WandB data...")
    data = load_wandb_data('wandb_detailed_results.json')
    print(f"✓ Loaded {len(data)} runs")
    print()

    # Create output directory
    output_dir = Path('paper_figures')
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir.absolute()}")
    print()

    # Generate figures
    print("Generating figures...")
    print()
    create_learning_curves_by_model(data, output_dir)
    create_grade_inflation_comparison(data, output_dir)
    create_reward_accuracy_scatter(data, output_dir)
    create_wireheading_dynamics(data, output_dir)
    create_final_performance_comparison(data, output_dir)

    print()
    print("=" * 80)
    print("✓ ALL FIGURES GENERATED SUCCESSFULLY")
    print("=" * 80)
    print()
    print(f"Figures saved in: {output_dir.absolute()}")
    print("  - figure1_learning_curves.{pdf,png}")
    print("  - figure2_grade_inflation.{pdf,png}")
    print("  - figure3_reward_accuracy_scatter.{pdf,png}")
    print("  - figure4_wireheading_dynamics.{pdf,png}")
    print("  - figure5_final_performance.{pdf,png}")

if __name__ == '__main__':
    main()
