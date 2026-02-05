"""
RankHead 网络结构可视化
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np
from pathlib import Path

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = Path(__file__).resolve().parent / "results"
OUTPUT_DIR.mkdir(exist_ok=True)


def draw_network_architecture():
    """绘制完整的 TabPFN-RankHead 架构"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 10))

    # ============ 左图: DeepRankingHead ============
    ax1 = axes[0]
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 12)
    ax1.axis('off')
    ax1.set_title('DeepRankingHead 结构', fontsize=14, fontweight='bold', pad=20)

    # 输入
    ax1.add_patch(FancyBboxPatch((3.5, 10.5), 3, 0.8, boxstyle="round,pad=0.05",
                                  facecolor='#E3F2FD', edgecolor='#1976D2', linewidth=2))
    ax1.text(5, 10.9, 'TabPFN 预测值', fontsize=11, ha='center', va='center', fontweight='bold')
    ax1.text(5, 10.6, 'ŷ_tabpfn (1维)', fontsize=9, ha='center', va='center', color='#666')

    # 箭头
    ax1.annotate('', xy=(5, 10), xytext=(5, 10.5),
                arrowprops=dict(arrowstyle='->', color='#455A64', lw=2))

    # Layer 1: Linear(1, 64) + ReLU + Dropout
    ax1.add_patch(FancyBboxPatch((2, 8.5), 6, 1.2, boxstyle="round,pad=0.05",
                                  facecolor='#C8E6C9', edgecolor='#388E3C', linewidth=2))
    ax1.text(5, 9.3, 'Linear(1 → 64) + ReLU + Dropout(0.1)', fontsize=10, ha='center', va='center')
    ax1.text(5, 8.8, '64 neurons', fontsize=9, ha='center', va='center', color='#666')

    ax1.annotate('', xy=(5, 8), xytext=(5, 8.5),
                arrowprops=dict(arrowstyle='->', color='#455A64', lw=2))

    # Layer 2: Linear(64, 64) + ReLU + Dropout
    ax1.add_patch(FancyBboxPatch((2, 6.5), 6, 1.2, boxstyle="round,pad=0.05",
                                  facecolor='#C8E6C9', edgecolor='#388E3C', linewidth=2))
    ax1.text(5, 7.3, 'Linear(64 → 64) + ReLU + Dropout(0.1)', fontsize=10, ha='center', va='center')
    ax1.text(5, 6.8, '64 neurons', fontsize=9, ha='center', va='center', color='#666')

    ax1.annotate('', xy=(5, 6), xytext=(5, 6.5),
                arrowprops=dict(arrowstyle='->', color='#455A64', lw=2))

    # Layer 3: Linear(64, 32) + ReLU
    ax1.add_patch(FancyBboxPatch((2, 4.5), 6, 1.2, boxstyle="round,pad=0.05",
                                  facecolor='#C8E6C9', edgecolor='#388E3C', linewidth=2))
    ax1.text(5, 5.3, 'Linear(64 → 32) + ReLU', fontsize=10, ha='center', va='center')
    ax1.text(5, 4.8, '32 neurons', fontsize=9, ha='center', va='center', color='#666')

    ax1.annotate('', xy=(5, 4), xytext=(5, 4.5),
                arrowprops=dict(arrowstyle='->', color='#455A64', lw=2))

    # Layer 4: Linear(32, 1)
    ax1.add_patch(FancyBboxPatch((2, 2.5), 6, 1.2, boxstyle="round,pad=0.05",
                                  facecolor='#BBDEFB', edgecolor='#1976D2', linewidth=2))
    ax1.text(5, 3.3, 'Linear(32 → 1)', fontsize=10, ha='center', va='center')
    ax1.text(5, 2.8, '输出层', fontsize=9, ha='center', va='center', color='#666')

    ax1.annotate('', xy=(5, 2), xytext=(5, 2.5),
                arrowprops=dict(arrowstyle='->', color='#455A64', lw=2))

    # 输出
    ax1.add_patch(FancyBboxPatch((3.5, 0.8), 3, 0.8, boxstyle="round,pad=0.05",
                                  facecolor='#F8BBD9', edgecolor='#C2185B', linewidth=2))
    ax1.text(5, 1.2, 'ŷ_final (1维)', fontsize=11, ha='center', va='center', fontweight='bold')

    # 参数统计
    ax1.text(5, 0.2, '总参数: 1×64 + 64×64 + 64×32 + 32×1 = 6,241', fontsize=9,
             ha='center', va='center', color='#666', style='italic')

    # ============ 右图: MonotonicRankingHead ============
    ax2 = axes[1]
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 12)
    ax2.axis('off')
    ax2.set_title('MonotonicRankingHead 结构 (单调约束)', fontsize=14, fontweight='bold', pad=20)

    # 输入
    ax2.add_patch(FancyBboxPatch((3.5, 10.5), 3, 0.8, boxstyle="round,pad=0.05",
                                  facecolor='#E3F2FD', edgecolor='#1976D2', linewidth=2))
    ax2.text(5, 10.9, 'TabPFN 预测值', fontsize=11, ha='center', va='center', fontweight='bold')
    ax2.text(5, 10.6, 'ŷ_tabpfn (1维)', fontsize=9, ha='center', va='center', color='#666')

    ax2.annotate('', xy=(5, 10), xytext=(5, 10.5),
                arrowprops=dict(arrowstyle='->', color='#455A64', lw=2))

    # Layer 1: Monotonic Linear (正权重)
    ax2.add_patch(FancyBboxPatch((1.5, 7.5), 7, 2, boxstyle="round,pad=0.05",
                                  facecolor='#FFF9C4', edgecolor='#F9A825', linewidth=2))
    ax2.text(5, 9.0, 'Monotonic Linear(1 → 32) + ReLU', fontsize=10, ha='center', va='center', fontweight='bold')
    ax2.text(5, 8.5, 'W₁ = softplus(W₁_raw)', fontsize=10, ha='center', va='center', color='#E65100')
    ax2.text(5, 8.0, '保证权重 > 0 → 单调递增', fontsize=9, ha='center', va='center', color='#666')

    ax2.annotate('', xy=(5, 7), xytext=(5, 7.5),
                arrowprops=dict(arrowstyle='->', color='#455A64', lw=2))

    # Layer 2: Monotonic Linear (正权重)
    ax2.add_patch(FancyBboxPatch((1.5, 4.5), 7, 2, boxstyle="round,pad=0.05",
                                  facecolor='#FFF9C4', edgecolor='#F9A825', linewidth=2))
    ax2.text(5, 6.0, 'Monotonic Linear(32 → 1)', fontsize=10, ha='center', va='center', fontweight='bold')
    ax2.text(5, 5.5, 'W₂ = softplus(W₂_raw)', fontsize=10, ha='center', va='center', color='#E65100')
    ax2.text(5, 5.0, '保证权重 > 0 → 单调递增', fontsize=9, ha='center', va='center', color='#666')

    ax2.annotate('', xy=(5, 4), xytext=(5, 4.5),
                arrowprops=dict(arrowstyle='->', color='#455A64', lw=2))

    # 输出
    ax2.add_patch(FancyBboxPatch((3.5, 2.5), 3, 0.8, boxstyle="round,pad=0.05",
                                  facecolor='#F8BBD9', edgecolor='#C2185B', linewidth=2))
    ax2.text(5, 2.9, 'ŷ_final (1维)', fontsize=11, ha='center', va='center', fontweight='bold')

    # 单调性说明
    ax2.add_patch(FancyBboxPatch((1.5, 0.8), 7, 1.2, boxstyle="round,pad=0.05",
                                  facecolor='#E8F5E9', edgecolor='#4CAF50', linewidth=1.5))
    ax2.text(5, 1.7, '单调性保证:', fontsize=10, ha='center', va='center', fontweight='bold')
    ax2.text(5, 1.2, '若 ŷ_tabpfn↑ 则 ŷ_final↑ (排序不变)', fontsize=9, ha='center', va='center')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'rankhead_architecture.png', dpi=300, bbox_inches='tight', facecolor='white')
    print(f"结构图已保存: {OUTPUT_DIR / 'rankhead_architecture.png'}")
    plt.close()


def draw_full_pipeline():
    """绘制完整的 TabPFN-RankHead Pipeline"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')

    ax.set_title('TabPFN-RankHead 完整流程', fontsize=16, fontweight='bold', pad=20)

    # ============ 输入层 ============
    ax.add_patch(FancyBboxPatch((0.5, 7), 2.5, 1.5, boxstyle="round,pad=0.05",
                                 facecolor='#E3F2FD', edgecolor='#1976D2', linewidth=2))
    ax.text(1.75, 8.0, '光谱特征', fontsize=11, ha='center', va='center', fontweight='bold')
    ax.text(1.75, 7.5, 'X (n × 37)', fontsize=10, ha='center', va='center', color='#666')

    # 箭头
    ax.annotate('', xy=(3.5, 7.75), xytext=(3, 7.75),
               arrowprops=dict(arrowstyle='->', color='#455A64', lw=2))

    # ============ TabPFN (冻结) ============
    ax.add_patch(FancyBboxPatch((3.5, 6.5), 3, 2.5, boxstyle="round,pad=0.05",
                                 facecolor='#BBDEFB', edgecolor='#1565C0', linewidth=2))
    ax.text(5, 8.3, 'TabPFN', fontsize=12, ha='center', va='center', fontweight='bold')
    ax.text(5, 7.8, '(预训练, 冻结)', fontsize=10, ha='center', va='center', color='#666')
    ax.text(5, 7.3, '256 estimators', fontsize=9, ha='center', va='center', color='#888')
    ax.text(5, 6.8, '🔒 不更新权重', fontsize=9, ha='center', va='center', color='#C62828')

    ax.annotate('', xy=(7, 7.75), xytext=(6.5, 7.75),
               arrowprops=dict(arrowstyle='->', color='#455A64', lw=2))

    # ============ 中间预测 ============
    ax.add_patch(FancyBboxPatch((7, 7), 2, 1.5, boxstyle="round,pad=0.05",
                                 facecolor='#E8F5E9', edgecolor='#388E3C', linewidth=2))
    ax.text(8, 8.0, 'ŷ_tabpfn', fontsize=11, ha='center', va='center', fontweight='bold')
    ax.text(8, 7.5, '(n × 1)', fontsize=10, ha='center', va='center', color='#666')

    ax.annotate('', xy=(9.5, 7.75), xytext=(9, 7.75),
               arrowprops=dict(arrowstyle='->', color='#455A64', lw=2))

    # ============ RankHead (可训练) ============
    ax.add_patch(FancyBboxPatch((9.5, 6.5), 3.5, 2.5, boxstyle="round,pad=0.05",
                                 facecolor='#FFF9C4', edgecolor='#F9A825', linewidth=2))
    ax.text(11.25, 8.3, 'RankHead', fontsize=12, ha='center', va='center', fontweight='bold')
    ax.text(11.25, 7.8, '(排序优化层)', fontsize=10, ha='center', va='center', color='#666')
    ax.text(11.25, 7.3, 'MLP: 1→64→64→32→1', fontsize=9, ha='center', va='center', color='#888')
    ax.text(11.25, 6.8, '🔓 可训练', fontsize=9, ha='center', va='center', color='#388E3C')

    # ============ 损失函数 ============
    ax.add_patch(FancyBboxPatch((4, 3), 6, 2.5, boxstyle="round,pad=0.05",
                                 facecolor='#FFCCBC', edgecolor='#E64A19', linewidth=2))
    ax.text(7, 5.0, '组合损失函数', fontsize=11, ha='center', va='center', fontweight='bold')
    ax.text(7, 4.4, 'L = (1-α-β)·MSE + α·Pairwise + β·Spearman', fontsize=10, ha='center', va='center')
    ax.text(7, 3.8, 'α=0.3, β=0.5', fontsize=9, ha='center', va='center', color='#666')
    ax.text(7, 3.3, '优化排序 > 优化数值', fontsize=9, ha='center', va='center', color='#C62828')

    # 损失函数箭头
    ax.annotate('', xy=(10, 6.5), xytext=(8.5, 5.5),
               arrowprops=dict(arrowstyle='->', color='#E64A19', lw=1.5, connectionstyle='arc3,rad=-0.2'))
    ax.text(9.5, 6.2, '反向传播', fontsize=8, ha='center', va='center', color='#E64A19')

    # 标签箭头
    ax.add_patch(FancyBboxPatch((0.5, 3.5), 2.5, 1.5, boxstyle="round,pad=0.05",
                                 facecolor='#F3E5F5', edgecolor='#7B1FA2', linewidth=2))
    ax.text(1.75, 4.5, '真实标签', fontsize=11, ha='center', va='center', fontweight='bold')
    ax.text(1.75, 4.0, 'y (D_conv)', fontsize=10, ha='center', va='center', color='#666')

    ax.annotate('', xy=(4, 4.25), xytext=(3, 4.25),
               arrowprops=dict(arrowstyle='->', color='#7B1FA2', lw=1.5))

    # ============ 输出 ============
    ax.add_patch(FancyBboxPatch((10, 1), 3, 1.5, boxstyle="round,pad=0.05",
                                 facecolor='#F8BBD9', edgecolor='#C2185B', linewidth=2))
    ax.text(11.5, 2.0, 'ŷ_final', fontsize=12, ha='center', va='center', fontweight='bold')
    ax.text(11.5, 1.5, '排序优化后的预测', fontsize=9, ha='center', va='center', color='#666')

    ax.annotate('', xy=(11.5, 2.5), xytext=(11.5, 6.5),
               arrowprops=dict(arrowstyle='->', color='#455A64', lw=2))

    # ============ 评估 ============
    ax.add_patch(FancyBboxPatch((0.5, 0.5), 8, 1.8, boxstyle="round,pad=0.05",
                                 facecolor='#E8F5E9', edgecolor='#4CAF50', linewidth=2))
    ax.text(4.5, 1.8, '评估指标', fontsize=11, ha='center', va='center', fontweight='bold')
    ax.text(4.5, 1.2, 'Spearman ρ | Kendall τ | R² | 匹配排名', fontsize=10, ha='center', va='center')
    ax.text(4.5, 0.7, '目标: Spearman = 1.0', fontsize=9, ha='center', va='center', color='#388E3C')

    ax.annotate('', xy=(8.5, 1.4), xytext=(10, 1.4),
               arrowprops=dict(arrowstyle='->', color='#4CAF50', lw=1.5))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'tabpfn_rankhead_pipeline.png', dpi=300, bbox_inches='tight', facecolor='white')
    print(f"流程图已保存: {OUTPUT_DIR / 'tabpfn_rankhead_pipeline.png'}")
    plt.close()


def draw_loss_components():
    """绘制损失函数组件"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # ============ MSE Loss ============
    ax1 = axes[0]
    x = np.linspace(-1, 1, 100)
    y = x ** 2
    ax1.plot(x, y, 'b-', linewidth=2)
    ax1.fill_between(x, y, alpha=0.3)
    ax1.set_title('MSE Loss', fontsize=12, fontweight='bold')
    ax1.set_xlabel('预测误差 (ŷ - y)', fontsize=10)
    ax1.set_ylabel('损失', fontsize=10)
    ax1.text(0, 0.8, 'L = (ŷ - y)²', fontsize=11, ha='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax1.grid(True, alpha=0.3)

    # ============ Pairwise Ranking Loss ============
    ax2 = axes[1]
    x = np.linspace(-0.5, 0.5, 100)
    margin = 0.02
    y = np.maximum(0, x + margin)
    ax2.plot(x, y, 'g-', linewidth=2)
    ax2.fill_between(x, y, alpha=0.3, color='green')
    ax2.axvline(x=-margin, color='red', linestyle='--', alpha=0.7)
    ax2.set_title('Pairwise Ranking Loss', fontsize=12, fontweight='bold')
    ax2.set_xlabel('排序差 (ŷᵢ - ŷⱼ), 当 yᵢ < yⱼ', fontsize=10)
    ax2.set_ylabel('损失', fontsize=10)
    ax2.text(0.15, 0.3, 'L = ReLU(ŷᵢ - ŷⱼ + m)', fontsize=10, ha='center',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    ax2.text(-margin, 0.4, 'margin', fontsize=9, ha='center', color='red')
    ax2.grid(True, alpha=0.3)

    # ============ Spearman Loss ============
    ax3 = axes[2]
    rho = np.linspace(-1, 1, 100)
    loss = 1 - rho
    ax3.plot(rho, loss, 'purple', linewidth=2)
    ax3.fill_between(rho, loss, alpha=0.3, color='purple')
    ax3.set_title('Spearman Loss', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Spearman ρ', fontsize=10)
    ax3.set_ylabel('损失', fontsize=10)
    ax3.text(0, 1.5, 'L = 1 - ρ', fontsize=11, ha='center',
             bbox=dict(boxstyle='round', facecolor='plum', alpha=0.5))
    ax3.axhline(y=0, color='green', linestyle='--', alpha=0.7)
    ax3.text(0.9, 0.1, 'ρ=1时\n损失=0', fontsize=9, ha='center', color='green')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'loss_components.png', dpi=300, bbox_inches='tight', facecolor='white')
    print(f"损失函数图已保存: {OUTPUT_DIR / 'loss_components.png'}")
    plt.close()


def main():
    print("=" * 70)
    print("生成 RankHead 网络结构可视化")
    print("=" * 70)

    draw_network_architecture()
    draw_full_pipeline()
    draw_loss_components()

    print("\n输出文件:")
    print(f"  - {OUTPUT_DIR / 'rankhead_architecture.png'}")
    print(f"  - {OUTPUT_DIR / 'tabpfn_rankhead_pipeline.png'}")
    print(f"  - {OUTPUT_DIR / 'loss_components.png'}")


if __name__ == "__main__":
    main()
