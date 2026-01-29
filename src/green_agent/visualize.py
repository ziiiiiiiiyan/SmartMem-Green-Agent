"""
弱点雷达图可视化模块

使用 matplotlib 生成多维能力雷达图和评估报告
"""

import json
import math
import io
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .base import WeaknessProfile, TestResult, DimensionStats

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for server environments
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None  # type: ignore
    np = None  # type: ignore
    mpatches = None  # type: ignore


# ============== 雷达图生成 ==============

def generate_radar_chart(
    dimension_scores: Dict[str, float],
    title: str = "Agent 能力雷达图",
    output_path: Optional[str] = None,
    show: bool = False
) -> Optional[bytes]:
    """
    生成能力雷达图
    
    Args:
        dimension_scores: 各维度能力分数 (0-100)
        title: 图表标题
        output_path: 输出文件路径 (可选)
        show: 是否显示图表
    
    Returns:
        PNG 图片的字节数据（如果没有 output_path）
    """
    if not HAS_MATPLOTLIB or plt is None or np is None:
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
    ax.plot(angles, values, 'o-', linewidth=2, color='#2E86AB', label='Capability Score')
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
    ax.plot(angles, reference_values, '--', linewidth=1, color='orange', alpha=0.7, label='Pass Line (60)')
    
    # 标题
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    
    # 图例
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    # 保存或返回字节
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        return None
    else:
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        buf.seek(0)
        return buf.getvalue()


def generate_difficulty_bar_chart(
    difficulty_stats: Dict[str, Any],
    title: str = "Difficulty Pass Rate Analysis",
    output_path: Optional[str] = None
) -> Optional[bytes]:
    """
    生成难度通过率柱状图
    
    Args:
        difficulty_stats: 难度统计数据 {'easy': DimensionStats, ...}
        title: 图表标题
        output_path: 输出路径
    
    Returns:
        PNG bytes if no output_path specified
    """
    if not HAS_MATPLOTLIB or plt is None or np is None or mpatches is None:
        return None
    
    difficulties = ['easy', 'medium', 'difficult']
    pass_rates = []
    totals = []
    
    for diff in difficulties:
        stats = difficulty_stats.get(diff)
        if stats:
            pass_rates.append(stats.pass_rate * 100)
            totals.append(stats.total)
        else:
            pass_rates.append(0)
            totals.append(0)
    
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
    ax.set_xlabel('Difficulty Level', fontsize=12)
    ax.set_ylabel('Pass Rate (%)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['Easy', 'Medium', 'Difficult'], fontsize=12)
    ax.set_ylim(0, 110)
    
    # 添加参考线
    ax.axhline(y=60, color='orange', linestyle='--', linewidth=1, label='Pass Line (60%)')
    
    # 添加图例说明
    legend_elements = [
        mpatches.Patch(facecolor='#27AE60', label='Excellent (≥80%)'),
        mpatches.Patch(facecolor='#F39C12', label='Medium (60-80%)'),
        mpatches.Patch(facecolor='#E74C3C', label='Poor (<60%)')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        return None
    else:
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        buf.seek(0)
        return buf.getvalue()


# ============== 报告生成 ==============

class ReportGenerator:
    """弱点报告生成器"""
    
    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = Path(output_dir) if output_dir else Path("./results")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_report(
        self, 
        profile: "WeaknessProfile",
        round_history: List[dict],
        all_results: List["TestResult"],
        agent_name: str = "Purple Agent"
    ) -> Dict[str, Any]:
        """
        生成完整的评估报告
        
        Args:
            profile: 弱点画像
            round_history: 测试轮次历史
            all_results: 所有测试结果
            agent_name: 被测 Agent 名称
        
        Returns:
            包含 'text', 'data', 'charts' 的字典
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. 计算统计数据
        total_cases = len(all_results)
        total_passed = sum(1 for r in all_results if r.passed)
        total_score = sum(r.score for r in all_results)
        max_score = sum(r.max_score for r in all_results)
        
        # 2. 计算雷达图数据 (能力分 = 1 - 弱点分, 转换为百分制)
        radar_data = {}
        for dim, stats in profile.by_dimension.items():
            if stats.total > 0:
                radar_data[dim] = (1 - stats.weakness_score) * 100
            else:
                radar_data[dim] = 50  # 未测试的维度给中等分
        
        # 3. 生成文本报告
        text_report = self._generate_text_report(
            agent_name=agent_name,
            timestamp=timestamp,
            total_cases=total_cases,
            total_passed=total_passed,
            total_score=total_score,
            max_score=max_score,
            profile=profile,
            round_history=round_history
        )
        
        # 4. 生成结构化数据
        structured_data = {
            "agent_name": agent_name,
            "timestamp": timestamp,
            "summary": {
                "total_cases": total_cases,
                "passed": total_passed,
                "failed": total_cases - total_passed,
                "pass_rate": total_passed / max(1, total_cases),
                "total_score": total_score,
                "max_score": max_score,
                "score_rate": total_score / max(1, max_score)
            },
            "radar_data": {
                "dimensions": radar_data
            },
            "dimension_stats": {
                dim: {
                    "total": stats.total,
                    "passed": stats.passed,
                    "failed": stats.failed,
                    "pass_rate": stats.pass_rate,
                    "weakness_score": stats.weakness_score
                }
                for dim, stats in profile.by_dimension.items()
            },
            "difficulty_stats": {
                diff: {
                    "total": stats.total,
                    "passed": stats.passed,
                    "pass_rate": stats.pass_rate
                }
                for diff, stats in profile.by_difficulty.items()
            },
            "boundaries": profile.boundary_found,
            "top_weaknesses": profile.failed_cases[:10] if profile.failed_cases else [],
            "round_history": round_history
        }
        
        # 5. 生成图表
        charts = {}
        
        if HAS_MATPLOTLIB and radar_data:
            # 雷达图
            radar_bytes = generate_radar_chart(
                radar_data,
                title=f"{agent_name} Capability Radar"
            )
            if radar_bytes:
                charts['radar'] = radar_bytes
            
            # 难度柱状图
            difficulty_bytes = generate_difficulty_bar_chart(
                profile.by_difficulty,
                title=f"{agent_name} Difficulty Analysis"
            )
            if difficulty_bytes:
                charts['difficulty'] = difficulty_bytes
        
        return {
            "text": text_report,
            "data": structured_data,
            "charts": charts
        }
    
    def _generate_text_report(
        self,
        agent_name: str,
        timestamp: str,
        total_cases: int,
        total_passed: int,
        total_score: float,
        max_score: float,
        profile: "WeaknessProfile",
        round_history: List[dict]
    ) -> str:
        """生成 Markdown 文本报告"""
        
        lines = []
        lines.append(f"# 🎯 Agent Capability Assessment Report")
        lines.append(f"\n**Agent**: {agent_name}")
        lines.append(f"**Assessment Time**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"**Test Rounds**: {len(round_history)}")
        
        # 总体统计
        lines.append(f"\n## 📊 Overall Statistics\n")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Total Test Cases | {total_cases} |")
        lines.append(f"| Passed | {total_passed} |")
        lines.append(f"| Failed | {total_cases - total_passed} |")
        lines.append(f"| Pass Rate | {total_passed/max(1,total_cases)*100:.1f}% |")
        lines.append(f"| Score | {total_score:.1f} / {max_score:.1f} ({total_score/max(1,max_score)*100:.1f}%) |")
        
        # 维度分析
        lines.append(f"\n## 🔍 Dimension Analysis\n")
        lines.append(f"| Dimension | Tests | Passed | Pass Rate | Weakness |")
        lines.append(f"|-----------|-------|--------|-----------|----------|")
        
        for dim, stats in sorted(profile.by_dimension.items(), key=lambda x: x[1].weakness_score, reverse=True):
            weakness_indicator = "🔴" if stats.weakness_score > 0.5 else "🟡" if stats.weakness_score > 0.3 else "🟢"
            lines.append(f"| {dim} | {stats.total} | {stats.passed} | {stats.pass_rate*100:.1f}% | {weakness_indicator} {stats.weakness_score:.2f} |")
        
        # 难度分析
        lines.append(f"\n## 📈 Difficulty Analysis\n")
        lines.append(f"| Difficulty | Tests | Pass Rate |")
        lines.append(f"|------------|-------|-----------|")
        
        for diff in ['easy', 'medium', 'difficult']:
            stats = profile.by_difficulty.get(diff)
            if stats and stats.total > 0:
                lines.append(f"| {diff.capitalize()} | {stats.total} | {stats.pass_rate*100:.1f}% |")
        
        # 能力边界
        if profile.boundary_found:
            lines.append(f"\n## ⚠️ Capability Boundaries Detected\n")
            for dim, diff in profile.boundary_found.items():
                lines.append(f"- **{dim}**: Performance drops significantly at `{diff}` level")
        
        # 建议
        lines.append(f"\n## 💡 Recommendations\n")
        
        # 获取 top 3 弱点
        weaknesses = []
        for dim, stats in profile.by_dimension.items():
            if stats.total > 0:
                weaknesses.append((dim, stats.weakness_score))
        weaknesses.sort(key=lambda x: x[1], reverse=True)
        
        if weaknesses:
            lines.append("Based on the assessment, focus improvement on:")
            for i, (dim, score) in enumerate(weaknesses[:3], 1):
                lines.append(f"{i}. **{dim}** (weakness score: {score:.2f})")
        
        return "\n".join(lines)
    
    def save_report(
        self,
        report: Dict[str, Any],
        prefix: str = "assessment"
    ) -> Dict[str, str]:
        """
        保存报告到文件
        
        Returns:
            文件路径字典 {'text': path, 'data': path, 'radar': path, ...}
        """
        timestamp = report['data'].get('timestamp', datetime.now().strftime("%Y%m%d_%H%M%S"))
        saved_files = {}
        
        # 保存文本报告
        text_path = self.output_dir / f"{prefix}_report_{timestamp}.md"
        text_path.write_text(report['text'], encoding='utf-8')
        saved_files['text'] = str(text_path)
        
        # 保存 JSON 数据
        data_path = self.output_dir / f"{prefix}_data_{timestamp}.json"
        # 清理不能序列化的数据
        clean_data = self._clean_for_json(report['data'])
        data_path.write_text(json.dumps(clean_data, indent=2, ensure_ascii=False), encoding='utf-8')
        saved_files['data'] = str(data_path)
        
        # 保存图表
        for chart_name, chart_bytes in report.get('charts', {}).items():
            chart_path = self.output_dir / f"{prefix}_{chart_name}_{timestamp}.png"
            chart_path.write_bytes(chart_bytes)
            saved_files[chart_name] = str(chart_path)
        
        return saved_files
    
    def _clean_for_json(self, data: Any) -> Any:
        """清理数据以便 JSON 序列化"""
        if isinstance(data, dict):
            return {k: self._clean_for_json(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._clean_for_json(item) for item in data]
        elif hasattr(data, '__dict__'):
            # Convert dataclass/object to dict
            return self._clean_for_json(vars(data))
        elif isinstance(data, (int, float, str, bool, type(None))):
            return data
        else:
            return str(data)


def create_artifact_parts(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    将报告转换为 AgentBeats artifact parts 格式
    
    Returns:
        List of Part dicts for use with TaskUpdater.add_artifact()
    """
    parts = []
    
    # 1. 文本报告
    parts.append({
        "type": "text",
        "text": report['text']
    })
    
    # 2. 结构化数据
    # 清理不可序列化的对象
    clean_data = {}
    for key, value in report['data'].items():
        if key == 'top_weaknesses':
            # TestResult 对象需要转换
            clean_data[key] = []  # 简化处理
        else:
            clean_data[key] = value
    
    parts.append({
        "type": "data",
        "data": clean_data
    })
    
    # 3. 图表（如果有）
    for chart_name, chart_bytes in report.get('charts', {}).items():
        # 图表以 base64 编码存储
        import base64
        parts.append({
            "type": "file",
            "name": f"{chart_name}_chart.png",
            "mime_type": "image/png",
            "data": base64.b64encode(chart_bytes).decode('utf-8')
        })
    
    return parts
