"""Figure 4-2 示意版：学习曲线（手绘趋势，非实测数据）"""
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Times New Roman'

# 示意性数据：基于各模型已知特性手工设定趋势
# 横轴：训练样本数 9, 18, 27, 36
xs = np.array([9, 18, 27, 36])

# TabPFN: 小样本鲁棒，从高位平稳上升
tabpfn = np.array([0.82, 0.91, 0.97, 1.00])
tabpfn_std = np.array([0.10, 0.07, 0.04, 0.02])

# CatBoost: 小样本略差，中等样本后追上
catboost = np.array([0.68, 0.84, 0.93, 0.98])
catboost_std = np.array([0.14, 0.10, 0.06, 0.03])

# PLSR: 线性模型，中等样本表现稳定
plsr = np.array([0.62, 0.80, 0.90, 0.97])
plsr_std = np.array([0.16, 0.11, 0.07, 0.04])

# RF: 集成模型，需要更多样本
rf = np.array([0.58, 0.78, 0.88, 0.97])
rf_std = np.array([0.18, 0.12, 0.08, 0.04])

# Ridge: 正则化线性，小样本尚可但上限低
ridge = np.array([0.55, 0.72, 0.82, 0.94])
ridge_std = np.array([0.20, 0.14, 0.10, 0.06])

# SVR: 小样本极差，超参敏感
svr = np.array([0.35, 0.58, 0.75, 0.94])
svr_std = np.array([0.25, 0.18, 0.12, 0.06])

models_data = {
    'TabPFN':   (tabpfn,   tabpfn_std,   '#2166AC', 'o'),
    'CatBoost': (catboost, catboost_std, '#4DAC26', 's'),
    'PLSR':     (plsr,     plsr_std,     '#E08214', 'D'),
    'RF':       (rf,       rf_std,       '#1A9850', 'v'),
    'Ridge':    (ridge,    ridge_std,    '#762A83', '^'),
    'SVR':      (svr,      svr_std,      '#D73027', 'x'),
}

fig, ax = plt.subplots(figsize=(7, 4.5))

for name, (mean, std, color, marker) in models_data.items():
    ax.plot(xs, mean, marker=marker, color=color, linewidth=1.8,
            markersize=7, label=name, zorder=3)
    ax.fill_between(xs, mean - std, mean + std,
                    color=color, alpha=0.10, zorder=2)

ax.set_xlabel('Number of training samples', fontsize=11)
ax.set_ylabel('Spearman $\\rho$ (variety-level)', fontsize=11)
ax.set_xticks([9, 18, 27, 36])
ax.set_xticklabels(['9\n(3 var)', '18\n(6 var)', '27\n(9 var)', '36\n(12 var)'])
ax.set_ylim(0.15, 1.08)
ax.yaxis.grid(True, linestyle='--', alpha=0.4)
ax.set_axisbelow(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(loc='lower right', fontsize=9, ncol=2, framealpha=0.9)

# 标注示意性
ax.text(0.02, 0.97, 'Schematic (illustrative)', transform=ax.transAxes,
        fontsize=8, color='gray', va='top', style='italic')

ax.set_title('Figure 4-2  Learning curves: Spearman ρ vs. training sample size',
             fontsize=11, pad=10)

plt.tight_layout()
plt.savefig('F:/all_exp/Thesis/论文图片/Figure_4-2/Figure_4-2_learning_curve_schematic.png',
            dpi=300, bbox_inches='tight')
plt.close()
print("Figure 4-2 schematic saved.")
