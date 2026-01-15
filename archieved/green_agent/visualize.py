"""
弱点雷达图可视化模块

使用 matplotlib 生成多维能力雷达图
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Optional
import sys

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.font_manager import FontProperties
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("⚠️ matplotlib 未安装，雷达图功能不可用")
    print("  运行: pip install matplotlib numpy")


def generate_radar_chart(
    dimension_scores: Dict[str, float],
    title: str = "Agent 能力雷达图",
    output_path: Optional[str] = None,
    show: bool = False
) -> Optional[str]:
    """
    生成能力雷达图
    
    Args:
        dimension_scores: 各维度能力分数 (0-100)
        title: 图表标题
        output_path: 输出文件路径 (可选)
        show: 是否显示图表
    
    Returns:
        输出文件路径（如果保存）
    """
    if not HAS_MATPLOTLIB:
        return None
    
    # 数据准备
    categories = list(dimension_scores.keys())
    values = list(dimension_scores.values())
    
    # 闭合图形
    num_vars = len(categories)
    angles = [n / float(num_vars) * 2 * math.pi for n in range(num_vars)]
    angles += angles[:1]
    values += values[:1]
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    # 设置角度
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12)
    
    # 设置刻度范围
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20', '40', '60', '80', '100'], fontsize=10, color='gray')
    
    # 绘制雷达图
    ax.plot(angles, values, 'o-', linewidth=2, color='#2E86AB', label='能力值')
    ax.fill(angles, values, alpha=0.25, color='#2E86AB')
    
    # 添加数值标签
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
    
    # 添加参考线（60分及格线）
    reference_values = [60] * (num_vars + 1)
    ax.plot(angles, reference_values, '--', linewidth=1, color='orange', alpha=0.7, label='及格线(60)')
    
    # 标题
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    
    # 图例
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    # 保存
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"📊 雷达图已保存: {output_path}")
    
    if show:
        plt.show()
    
    plt.close()
    
    return output_path


def generate_comparison_radar(
    scores_list: List[Dict[str, float]],
    labels: List[str],
    title: str = "Agent 能力对比",
    output_path: Optional[str] = None,
    show: bool = False
) -> Optional[str]:
    """
    生成多个 Agent 的能力对比雷达图
    
    Args:
        scores_list: 各 Agent 的维度分数列表
        labels: Agent 标签列表
        title: 图表标题
        output_path: 输出文件路径
        show: 是否显示
    """
    if not HAS_MATPLOTLIB:
        return None
    
    if not scores_list:
        return None
    
    # 数据准备
    categories = list(scores_list[0].keys())
    num_vars = len(categories)
    angles = [n / float(num_vars) * 2 * math.pi for n in range(num_vars)]
    angles += angles[:1]
    
    # 颜色列表
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B']
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(polar=True))
    
    # 设置角度
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    
    # 绘制每个 Agent
    for i, (scores, label) in enumerate(zip(scores_list, labels)):
        values = [scores.get(cat, 0) for cat in categories]
        values += values[:1]
        color = colors[i % len(colors)]
        
        ax.plot(angles, values, 'o-', linewidth=2, color=color, label=label)
        ax.fill(angles, values, alpha=0.1, color=color)
    
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"📊 对比雷达图已保存: {output_path}")
    
    if show:
        plt.show()
    
    plt.close()
    
    return output_path


def generate_difficulty_bar_chart(
    difficulty_stats: Dict[str, Dict],
    title: str = "难度通过率分析",
    output_path: Optional[str] = None,
    show: bool = False
) -> Optional[str]:
    """
    生成难度通过率柱状图
    
    Args:
        difficulty_stats: 难度统计数据
        title: 图表标题
        output_path: 输出路径
        show: 是否显示
    """
    if not HAS_MATPLOTLIB:
        return None
    
    difficulties = ['easy', 'medium', 'difficult']
    pass_rates = []
    totals = []
    
    for diff in difficulties:
        stats = difficulty_stats.get(diff, {})
        pass_rates.append(stats.get('pass_rate', 0) * 100)
        totals.append(stats.get('total', 0))
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(difficulties))
    width = 0.6
    
    # 颜色根据通过率
    colors = ['#27AE60' if r >= 80 else '#F39C12' if r >= 60 else '#E74C3C' for r in pass_rates]
    
    bars = ax.bar(x, pass_rates, width, color=colors, edgecolor='white', linewidth=2)
    
    # 添加数值标签
    for bar, rate, total in zip(bars, pass_rates, totals):
        height = bar.get_height()
        ax.annotate(
            f'{rate:.1f}%\n(n={total})',
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 5),
            textcoords='offset points',
            ha='center',
            fontsize=12,
            fontweight='bold'
        )
    
    # 设置
    ax.set_xlabel('难度等级', fontsize=12)
    ax.set_ylabel('通过率 (%)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['Easy', 'Medium', 'Difficult'], fontsize=12)
    ax.set_ylim(0, 110)
    
    # 添加参考线
    ax.axhline(y=60, color='orange', linestyle='--', linewidth=1, label='及格线(60%)')
    ax.legend()
    
    # 添加图例说明
    legend_elements = [
        mpatches.Patch(facecolor='#27AE60', label='优秀 (≥80%)'),
        mpatches.Patch(facecolor='#F39C12', label='中等 (60-80%)'),
        mpatches.Patch(facecolor='#E74C3C', label='较差 (<60%)')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"📊 难度分析图已保存: {output_path}")
    
    if show:
        plt.show()
    
    plt.close()
    
    return output_path


def generate_weakness_heatmap(
    dimension_device_matrix: Dict[str, Dict[str, float]],
    title: str = "维度-设备弱点热力图",
    output_path: Optional[str] = None,
    show: bool = False
) -> Optional[str]:
    """
    生成维度-设备弱点热力图
    
    Args:
        dimension_device_matrix: 维度×设备的弱点分数矩阵
        title: 图表标题
        output_path: 输出路径
        show: 是否显示
    """
    if not HAS_MATPLOTLIB:
        return None
    
    dimensions = list(dimension_device_matrix.keys())
    if not dimensions:
        return None
    
    devices = list(dimension_device_matrix[dimensions[0]].keys())
    
    # 构建矩阵
    matrix = []
    for dim in dimensions:
        row = [dimension_device_matrix[dim].get(dev, 0) for dev in devices]
        matrix.append(row)
    
    matrix = np.array(matrix)
    
    # 创建热力图
    fig, ax = plt.subplots(figsize=(14, 8))
    
    im = ax.imshow(matrix, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=1)
    
    # 设置刻度
    ax.set_xticks(np.arange(len(devices)))
    ax.set_yticks(np.arange(len(dimensions)))
    ax.set_xticklabels(devices, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(dimensions, fontsize=10)
    
    # 添加数值
    for i in range(len(dimensions)):
        for j in range(len(devices)):
            value = matrix[i, j]
            color = 'white' if value > 0.5 else 'black'
            ax.text(j, i, f'{value:.2f}', ha='center', va='center', color=color, fontsize=9)
    
    # 颜色条
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('弱点分数 (越高越弱)', rotation=-90, va='bottom', fontsize=10)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('设备', fontsize=12)
    ax.set_ylabel('维度', fontsize=12)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"📊 热力图已保存: {output_path}")
    
    if show:
        plt.show()
    
    plt.close()
    
    return output_path


def generate_full_report_charts(
    weakness_data_path: str,
    output_dir: str = None
) -> List[str]:
    """
    根据弱点数据生成完整的图表报告
    
    Args:
        weakness_data_path: weakness_data JSON 文件路径
        output_dir: 输出目录（默认与数据文件同目录）
    
    Returns:
        生成的图表文件列表
    """
    if not HAS_MATPLOTLIB:
        print("⚠️ matplotlib 未安装，无法生成图表")
        return []
    
    # 读取数据
    with open(weakness_data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 确定输出目录
    if output_dir is None:
        output_dir = Path(weakness_data_path).parent
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = data.get('timestamp', 'unknown')
    generated_files = []
    
    # 1. 能力雷达图
    if 'radar_data' in data and 'dimensions' in data['radar_data']:
        radar_path = output_dir / f"radar_chart_{timestamp}.png"
        generate_radar_chart(
            data['radar_data']['dimensions'],
            title=f"{data.get('agent_name', 'Agent')} 能力雷达图",
            output_path=str(radar_path)
        )
        generated_files.append(str(radar_path))
    
    # 2. 难度分析图
    if 'difficulty_stats' in data:
        diff_path = output_dir / f"difficulty_chart_{timestamp}.png"
        generate_difficulty_bar_chart(
            data['difficulty_stats'],
            title=f"{data.get('agent_name', 'Agent')} 难度通过率",
            output_path=str(diff_path)
        )
        generated_files.append(str(diff_path))
    
    print(f"\n✅ 生成了 {len(generated_files)} 个图表")
    return generated_files


# ============== 命令行接口 ==============

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="生成弱点分析图表")
    parser.add_argument("data_file", help="weakness_data JSON 文件路径")
    parser.add_argument("--output-dir", "-o", help="输出目录")
    parser.add_argument("--show", "-s", action="store_true", help="显示图表")
    
    args = parser.parse_args()
    
    generate_full_report_charts(args.data_file, args.output_dir)


if __name__ == "__main__":
    main()
