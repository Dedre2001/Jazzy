import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scienceplots
import os

# 1. 环境配置：延续 Agri-AI 2.0 顶刊视觉 DNA
plt.style.use(['science', 'nature', 'no-latex'])
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "figure.figsize": (7.2, 5.2),  # 回归稳重的 4:3 比例，增加视觉分量
    "figure.dpi": 600,
    "axes.linewidth": 1.25,        # 力量感轴宽
    "font.size": 11,
    "axes.labelsize": 13,
})

# 2. Agri-AI 2.0 扩展色库：采用模型重要性编组
# TabPFN 使用 Highlight (靛青紫)，其他模型使用互补的中性与灰调色
COLORS = {
    'TabPFN':   '#332288',  # 靛青紫 (Agri-AI Highlight)
    'CatBoost': '#44AA99',  # 鼠尾草绿 (Agri-AI Tolerant)
    'Ridge':    '#88CCEE',  # 青黛色
    'PLSR':     '#DDCC77',  # 琥珀金
    'RF':       '#CC6677',  # 灰瑰红
    'SVR':      '#AA4499',  # 波尔多红
}

# 3. ── 数据处理 ──
models  = ['TabPFN', 'CatBoost', 'Ridge', 'PLSR', 'RF', 'SVR']
metrics = ['$R^2$', 'Spearman $\\rho$', 'Pairwise Acc', 'Hit@3']
data = {
    '$R^2$':            [0.948, 0.935, 0.907, 0.856, 0.851, 0.420],
    'Spearman $\\rho$': [1.000, 0.978, 0.940, 0.978, 0.967, 0.940],
    'Pairwise Acc':     [1.000, 0.949, 0.910, 0.962, 0.949, 0.897],
    'Hit@3':            [1.000, 1.000, 0.670, 1.000, 1.000, 1.000],
}

# ── 4. 布局计算 ──
n_models, n_metrics = len(models), len(metrics)
x = np.arange(n_metrics)
width = 0.13  # 稍微加宽柱子，减少视觉上的“纤细感”
offsets = np.linspace(-(n_models - 1) / 2, (n_models - 1) / 2, n_models) * width

# ── 5. 绘图核心 ──
fig, ax = plt.subplots()

for i, model in enumerate(models):
    vals = [data[m][i] for m in metrics]
    color = COLORS[model]
    bars = ax.bar(x + offsets[i], vals, width,
                  label=model, color=color,
                  edgecolor='white', linewidth=0.5, zorder=3)

    # 仅针对核心模型 TabPFN 进行数据标注，提升其存在感
    if model == 'TabPFN':
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.015,
                    f'{val:.3f}',
                    ha='center', va='bottom',
                    fontsize=8.5, fontweight='bold', color=color)

# ── 6. 细节修正：Nature 风格封闭刻度 ──
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=11)
ax.set_ylabel('Performance Score', fontweight='bold', labelpad=12)

# 封闭式内向刻度
ax.tick_params(direction='in', top=True, right=True, which='both', length=6, width=1.2)
ax.minorticks_off() # 柱状图无需次级刻度

# 设置 y 轴范围与参考线
ax.set_ylim(0.3, 1.15)
ax.axhline(1.0, color='#888888', linestyle='--', lw=0.8, alpha=0.5, zorder=2)

# ── 7. 图例优化：横向排列以节省垂直空间 ──
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
          ncol=3, frameon=False, fontsize=9.5, columnspacing=1.0)

# ── 8. 导出：PDF + PNG 双重保障 ──
plt.tight_layout()
out_dir = 'F:/all_exp/Thesis/论文图片/Figure_4-1'
if not os.path.exists(out_dir): os.makedirs(out_dir)

plt.savefig(f'{out_dir}/Figure_4-1_Model_Comparison.pdf', bbox_inches='tight')
plt.savefig(f'{out_dir}/Figure_4-1_Model_Comparison.png', dpi=600, bbox_inches='tight', facecolor='white')

plt.show()
print(f"Success: Figure 4-1 saved to {out_dir}")