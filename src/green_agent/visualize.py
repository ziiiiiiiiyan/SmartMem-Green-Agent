"""
Visualization module for SmartMem Green Agent

Generates radar charts and performance visualizations for agent evaluation results.
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Optional

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def generate_radar_chart(
    dimension_scores: Dict[str, float],
    title: str = "Agent Capability Radar",
    output_path: Optional[str] = None,
    show: bool = False
) -> Optional[str]:
    """
    Generate a capability radar chart.

    Args:
        dimension_scores: Scores for each dimension (0-100)
        title: Chart title
        output_path: Output file path (optional)
        show: Whether to display the chart

    Returns:
        Output file path if saved, None otherwise
    """
    if not HAS_MATPLOTLIB:
        return None

    if not dimension_scores:
        return None

    # Data preparation
    categories = list(dimension_scores.keys())
    values = list(dimension_scores.values())

    # Close the polygon
    num_vars = len(categories)
    angles = [n / float(num_vars) * 2 * math.pi for n in range(num_vars)]
    angles += angles[:1]
    values += values[:1]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    # Set angles
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12)

    # Set scale range
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20', '40', '60', '80', '100'], fontsize=10, color='gray')

    # Draw radar chart
    ax.plot(angles, values, 'o-', linewidth=2, color='#2E86AB', label='Score')
    ax.fill(angles, values, alpha=0.25, color='#2E86AB')

    # Add value labels
    for angle, value, cat in zip(angles[:-1], values[:-1], categories):
        ax.annotate(
            f'{value:.1f}',
            xy=(angle, value),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=10,
            color='#2E86AB',
            fontweight='bold'
        )

    # Add reference line (60% passing threshold)
    reference_values = [60] * (num_vars + 1)
    ax.plot(angles, reference_values, '--', linewidth=1, color='orange', alpha=0.7, label='Passing (60)')

    # Title
    plt.title(title, fontsize=16, fontweight='bold', pad=20)

    # Legend
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

    # Save
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')

    if show:
        plt.show()

    plt.close()

    return output_path


def generate_bar_chart(
    category_scores: Dict[str, float],
    title: str = "Category Performance",
    output_path: Optional[str] = None,
    show: bool = False
) -> Optional[str]:
    """
    Generate a bar chart for category scores.

    Args:
        category_scores: Scores for each category (0-100)
        title: Chart title
        output_path: Output file path
        show: Whether to display

    Returns:
        Output file path if saved
    """
    if not HAS_MATPLOTLIB:
        return None

    if not category_scores:
        return None

    categories = list(category_scores.keys())
    scores = list(category_scores.values())

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(categories))
    width = 0.6

    # Color based on score
    colors = ['#27AE60' if s >= 80 else '#F39C12' if s >= 60 else '#E74C3C' for s in scores]

    bars = ax.bar(x, scores, width, color=colors, edgecolor='white', linewidth=2)

    # Add value labels
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax.annotate(
            f'{score:.1f}%',
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 5),
            textcoords='offset points',
            ha='center',
            fontsize=11,
            fontweight='bold'
        )

    # Settings
    ax.set_xlabel('Category', fontsize=12)
    ax.set_ylabel('Score (%)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha='right', fontsize=10)
    ax.set_ylim(0, 110)

    # Add reference line
    ax.axhline(y=60, color='orange', linestyle='--', linewidth=1, label='Passing (60%)')

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='#27AE60', label='Excellent (>=80%)'),
        mpatches.Patch(facecolor='#F39C12', label='Average (60-80%)'),
        mpatches.Patch(facecolor='#E74C3C', label='Poor (<60%)')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')

    if show:
        plt.show()

    plt.close()

    return output_path


def generate_report_charts(
    report_data: Dict,
    output_dir: str = "artifacts",
    agent_name: str = "Purple Agent"
) -> List[str]:
    """
    Generate all charts from evaluation report data.

    Args:
        report_data: Evaluation report dictionary
        output_dir: Output directory for charts
        agent_name: Name of the agent being evaluated

    Returns:
        List of generated chart file paths
    """
    if not HAS_MATPLOTLIB:
        return []

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    generated_files = []

    # 1. Per-tag radar chart
    per_tag_scores = report_data.get("per_tag_scores", {})
    if per_tag_scores:
        radar_path = output_path / "capability_radar.png"
        result = generate_radar_chart(
            per_tag_scores,
            title=f"{agent_name} Capability Radar",
            output_path=str(radar_path)
        )
        if result:
            generated_files.append(result)

    # 2. Category bar chart
    if per_tag_scores:
        bar_path = output_path / "category_scores.png"
        result = generate_bar_chart(
            per_tag_scores,
            title=f"{agent_name} Category Performance",
            output_path=str(bar_path)
        )
        if result:
            generated_files.append(result)

    return generated_files
