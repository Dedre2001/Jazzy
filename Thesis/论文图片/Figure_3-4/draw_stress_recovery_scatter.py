import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import scienceplots
import os

# 1. 环境配置
plt.style.use(['science', 'nature', 'no-latex'])
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "stix",
    "figure.figsize": (7.8, 7.2),
    "figure.dpi": 300,
    "font.size": 11,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "savefig.bbox": "tight",
})

# ── 2. 数据准备 ──
data = {
    '1252': (0.6471, 0.5023, 'Tolerant'),
    '1257': (0.6884, 0.3750, 'Tolerant'),
    '1099': (0.3480, 0.7087, 'Tolerant'),
    '1228': (0.6668, 0.2761, 'Intermediate'),
    '1214': (0.5935, 0.2514, 'Intermediate'),
    '1274': (0.5522, 0.2692, 'Intermediate'),
    '1210': (0.3130, 0.3993, 'Intermediate'),
    '73': (0.3391, 0.3163, 'Intermediate'),
    '12': (0.2427, 0.2872, 'Sensitive'),
    '1219': (0.3988, 0.0752, 'Sensitive'),
    '1110': (0.3435, 0.1029, 'Sensitive'),
    '1218': (0.3094, 0.1029, 'Sensitive'),
    '1235': (0.2353, 0.1109, 'Sensitive'),
}

C_TOLERANT, C_INTERMEDIATE, C_SENSITIVE = '#44AA99', '#DDCC77', '#AA4499'
C_LINE = '#888888'
color_map = {'Tolerant': C_TOLERANT, 'Intermediate': C_INTERMEDIATE, 'Sensitive': C_SENSITIVE}

# ── 3. 绘图核心 ──
fig, ax = plt.subplots()

d_stress_vals = np.array([v[0] for v in data.values()])
d_recovery_vals = np.array([v[1] for v in data.values()])
mean_s, mean_r = np.mean(d_stress_vals), np.mean(d_recovery_vals)

# A. 绘制象限虚线 (中轴线)
ax.axvline(mean_s, color=C_LINE, linestyle='--', linewidth=0.9, zorder=1)
ax.axhline(mean_r, color=C_LINE, linestyle='--', linewidth=0.9, zorder=1)

# B. 绘制散点 + 深度避让标注
# 核心修改：增加对 73 和 1274 的垂直偏移，确保不被中轴线挡住
special_offsets = {
    '1235': (-0.015, -0.045),  # 向下避开 1218
    '1218': (0.015, 0.040),  # 向上避开 1235
    '12': (0.025, 0.025),  # 向右上方偏移，彻底避开图例
    '1210': (0.015, 0.040),  # 向上偏移
    '73': (0.015, -0.055),  # 显著向下偏移，避开水平中轴线
    '1274': (-0.015, 0.045),  # 显著向上偏移，避开水平中轴线
    '1214': (0.015, -0.050),  # 向下偏移，与 1274 形成错位
    '1219': (0.015, 0.020),  # 向上微调
}

for name, (ds, dr, cat) in data.items():
    c = color_map[cat]
    ax.scatter(ds, dr, c=c, s=100, zorder=5, edgecolors='white', linewidths=0.7, alpha=0.9)

    # 应用手动避让逻辑
    dx, dy = special_offsets.get(name, (0.015, 0.015))
    ax.text(ds + dx, dr + dy, name, fontsize=10.5, fontweight='bold', color=c,
            ha='left' if dx > 0 else 'right', va='center', zorder=10)

# ── 4. 象限策略标注 (Strategic Annotations) ──
label_style = dict(fontsize=10, color='#444444', fontweight='bold', fontstyle='italic')
ax.text(0.97, 0.97, 'Comprehensive Tolerance\n(High Resilience / High Recovery)',
        ha='right', va='top', transform=ax.transAxes, **label_style)
ax.text(0.03, 0.97, 'Resilience-Recovery\n(Low Resilience / High Recovery)',
        ha='left', va='top', transform=ax.transAxes, **label_style)
ax.text(0.97, 0.03, 'Drought Avoidance\n(High Resilience / Low Recovery)',
        ha='right', va='bottom', transform=ax.transAxes, **label_style)
ax.text(0.03, 0.03, 'Sensitive / Vulnerable\n(Low Resilience / Low Recovery)',
        ha='left', va='bottom', transform=ax.transAxes, **label_style)

# ── 5. 坐标轴与图例精修 ──
ax.set_xlabel(r'$D_{stress}$ (Stress Maintenance Dimension)', labelpad=12)
ax.set_ylabel(r'$D_{recovery}$ (Recovery Efficiency Dimension)', labelpad=12)
ax.tick_params(direction='in', which='both', length=6, width=1.1)

# 优化轴范围：为左侧图例和各类标注预留安全边际
ax.set_xlim(0.12, 0.85)
ax.set_ylim(-0.05, 0.85)

# 再次调整图例：移动到 Q2 象限的最左上角，确保不干扰数据点 12
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Tolerant', markerfacecolor=C_TOLERANT, markersize=8),
    Line2D([0], [0], marker='o', color='w', label='Intermediate', markerfacecolor=C_INTERMEDIATE, markersize=8),
    Line2D([0], [0], marker='o', color='w', label='Sensitive', markerfacecolor=C_SENSITIVE, markersize=8),
]
ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.01, 0.72),
          frameon=False, fontsize=10.5, handletextpad=0.2)

# ── 6. 保存 ──
plt.tight_layout()
output_dir = r'F:\all_exp\Thesis\论文图片\Figure_3-4'
os.makedirs(output_dir, exist_ok=True)
plt.savefig(f'{output_dir}/Figure_3-4_Strategic_Final_v2.png', dpi=600)
plt.savefig(f'{output_dir}/Figure_3-4_Strategic_Final_v2.pdf')

plt.show()