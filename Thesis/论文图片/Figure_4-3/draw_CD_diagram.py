import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import scienceplots
import os

# 1. 顶刊级环境配置：确保全论文视觉统一
plt.style.use(['science', 'nature', 'no-latex'])
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "stix",
    "figure.figsize": (7, 2.8),  # CD图建议采用扁平比例
    "figure.dpi": 300,
    "savefig.bbox": "tight",
})

# ── 数据定义 ─────────────────────────────────────────────────────────────
# 基于你对 TabPFN 与其他算法性能对比的研究需求
models = ['TabPFN', 'CatBoost', 'Ridge', 'RF', 'PLSR', 'SVR']
avg_rank = [2.00, 2.69, 3.08, 3.77, 4.00, 5.46]
CD = 2.09

# 统一色系配置
C_TABPFN = '#4477AA'  # 核心模型色（科技蓝）
C_CLIQUE = '#AA4499'  # 显著性分组色（波尔多红）
C_OTHER = '#666666'  # 其他模型色（中性灰）

# ── 绘图核心 ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots()

# 轴范围与反转（Rank越小越靠右，性能越优）
ax.set_xlim(6.2, 0.8)
ax.set_ylim(-0.8, 1.6)

# 隐藏冗余边框与刻度（符合极简学术标准）
for sp in ['top', 'right', 'left']:
    ax.spines[sp].set_visible(False)
ax.set_yticks([])
ax.tick_params(axis='x', which='both', top=False, direction='in', length=4)  # 移除顶部刻度

# ── 绘制数据点与引导线 ──────────────────────────────────────────────────────
y_dot = 0.5
for i, (m, r) in enumerate(zip(models, avg_rank)):
    is_tabpfn = (m == 'TabPFN')
    col = C_TABPFN if is_tabpfn else C_OTHER
    fw = 'bold' if is_tabpfn else 'normal'

    # 绘制 rank 点：白色描边增加矢量感
    ax.plot(r, y_dot, 'o', color=col, markersize=8, zorder=5,
            markeredgewidth=0.6, markeredgecolor='white')

    # 标注模型名称 (上方)
    ax.text(r, y_dot + 0.28, m, va='bottom', ha='center',
            fontsize=9.5, fontweight=fw, color=col)

    # 标注 Rank 具体数值 (下方)
    ax.text(r, y_dot - 0.18, f'{r:.2f}', va='top', ha='center',
            fontsize=8, color=col, alpha=0.8)

# ── 绘制无显著差异的分组条 (Clique Bar) ───────────────────────────────────
# 显示与 TabPFN 无显著差异的模型梯队
clique_indices = [0, 1, 2, 3]
clique_ranks = [avg_rank[i] for i in clique_indices]
bar_y = 1.3

# 绘制圆角的显著性分组条
ax.hlines(bar_y, min(clique_ranks), max(clique_ranks),
          colors=C_CLIQUE, linewidth=3.0, zorder=4, capstyle='round')

# 绘制引导虚线：(0, (3, 3)) 为精细的点状线
for r in clique_ranks:
    ax.vlines(r, y_dot + 0.28, bar_y,
              colors=C_CLIQUE, linewidth=0.6, linestyle=(0, (3, 3)), alpha=0.4)

# ── 关键标注：CD 值指示箭头 (移动至更清爽的位置) ───────────────────────────
arrow_y = -0.45
arrow_x_end = 5.8
arrow_x_start = arrow_x_end - CD

ax.annotate('', xy=(arrow_x_start, arrow_y), xytext=(arrow_x_end, arrow_y),
            arrowprops=dict(arrowstyle='<->', color='#555555',
                            lw=0.8, mutation_scale=6))

# 使用 LaTeX 渲染 CD 值，增加学术严谨性
ax.text((arrow_x_start + arrow_x_end) / 2, arrow_y - 0.12,
        r'$CD = 2.09$', ha='center', va='top', fontsize=9, color='#555555')

# ── X 轴美化 ─────────────────────────────────────────────────────────────
ax.set_xlabel('Average Rank', fontsize=11, labelpad=8, fontweight='bold')
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))

# ── 保存与导出 ───────────────────────────────────────────────────────────
plt.subplots_adjust(bottom=0.32, top=0.92)

output_dir = 'F:/all_exp/Thesis/论文图片/Figure_4-3'
os.makedirs(output_dir, exist_ok=True)

# 同时保存 PNG (预览) 与 PDF (投稿)
plt.savefig(f'{output_dir}/CD_Diagram_Final.png', dpi=600)
plt.savefig(f'{output_dir}/CD_Diagram_Final.pdf')

plt.show()
print("Success: Final CD Diagram saved with Bordeaux Red theme.")