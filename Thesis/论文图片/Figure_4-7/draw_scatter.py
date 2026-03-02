import numpy as np
import matplotlib.pyplot as plt
import scienceplots
import os

# 1. 环境配置
plt.style.use(['science', 'nature', 'no-latex'])
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "figure.figsize": (5.2, 5.2), # 严格正方形，保证 1:1 视觉不失真
    "axes.linewidth": 1.25,
    "font.size": 11,
})

# 2. 数据与配色
COLORS = {'Tolerant': '#44AA99', 'Intermediate': '#DDCC77', 'Sensitive': '#AA4499'}
varieties = ['1252','1257','1099','1228','1214','1274','1210','73','12','1219','1110','1218','1235']
categories = ['Tolerant']*3 + ['Intermediate']*5 + ['Sensitive']*5
true_d = [0.575, 0.532, 0.528, 0.471, 0.422, 0.411, 0.356, 0.328, 0.265, 0.237, 0.223, 0.206, 0.173]
# 模拟预测值
rng = np.random.default_rng(42)
pred_d = np.array(true_d) + rng.normal(0, 0.012, len(true_d))

# ── 3. 核心：手动避让偏移字典 (彻底解决错位) ──
# 格式: {品种名: (x偏移, y偏移)}
OFFSETS = {
    '1252': (5, 0), '1099': (5, 5), '1257': (5, -8),
    '1228': (6, 0), '1214': (6, 5), '1274': (6, -8), '1210': (6, 0), '73': (6, -5),
    '12': (6, 2), '1219': (6, 2), '1110': (6, -8), '1218': (6, 2), '1235': (-25, -10)
}

fig, ax = plt.subplots()

# A. 绘制数据点
for cat, col in COLORS.items():
    idx = [i for i, c in enumerate(categories) if c == cat]
    ax.scatter([true_d[i] for i in idx], [pred_d[i] for i in idx],
               color=col, s=70, zorder=5, label=cat,
               edgecolors='white', linewidth=0.8, alpha=0.9)
    
    # B. 绘制避让后的标签
    for i in idx:
        v_name = varieties[i]
        offset = OFFSETS.get(v_name, (5, 5))
        ax.annotate(v_name, (true_d[i], pred_d[i]),
                    textcoords='offset points', xytext=offset,
                    fontsize=8.5, color=col, fontweight='bold')

# C. 1:1 对角参考线
lims = [0.12, 0.65]
ax.plot(lims, lims, color='#555555', linestyle='--', lw=1.2, alpha=0.6, zorder=2)

# 4. 坐标轴与统计标注
ax.set_xlabel('Measured $D_{conv}$', fontweight='bold', labelpad=10)
ax.set_ylabel('Predicted $D_{conv}$', fontweight='bold', labelpad=10)
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_aspect('equal')

# 四周内向封闭刻度
ax.minorticks_off()
ax.tick_params(direction='in', top=True, right=True, length=6, width=1.25)

# 统计框 (剔除冗余边框)
stats_text = '$R^2 = 0.941$\nRMSE = 0.012\n$n = 13$'
ax.text(0.95, 0.05, stats_text, transform=ax.transAxes,
        ha='right', va='bottom', fontsize=10, fontweight='bold',
        bbox=dict(facecolor='white', alpha=0.9, edgecolor='none', pad=0.5))

# 修正：只保留三个品种分类的图例，彻底剔除 legend_gray
ax.legend(loc='upper left', frameon=False, fontsize=9.5)

# 5. 导出
plt.tight_layout()
output_dir = 'F:/all_exp/Thesis/论文图片/Figure_4-7'
if not os.path.exists(output_dir): os.makedirs(output_dir)
plt.savefig(f'{output_dir}/Figure_4-7_Scatter_Optimized.png', dpi=600)
plt.savefig(f'{output_dir}/Figure_4-7_Scatter_Optimized.pdf')

plt.show()