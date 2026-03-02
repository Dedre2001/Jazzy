"""
Figure 3-6: OJIP曲线对比（品种1252 vs 1235）
基于OJIP_cleaned_wide.csv的完整原始荧光诱导曲线
数据源: OJIP_cleaned_wide.csv（118时间点，实测数据）
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from pathlib import Path

rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False

# 读取原始OJIP曲线数据
raw = pd.read_csv(r'F:\all_exp\data\OJIP_cleaned_wide.csv')

# 时间列（秒）→ 毫秒
time_cols = [c for c in raw.columns if c not in ['Variety', 'Treatment']]
time_ms = np.array([float(c) * 1000 for c in time_cols])

# 清理Treatment列中的中文标注
raw['Treatment_clean'] = raw['Treatment'].str.extract(r'(CK1|D1|RD2)')

varieties = [(1252, '#2ca02c', '1252 (抗旱型)'),
             (1235, '#d62728', '1235 (敏感型)')]

fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

for v_id, color, v_label in varieties:
    for trt, trt_label, ls, lw in [('CK1', 'CK1', '-',  2.2),
                                     ('D1',  'D1',  '--', 2.0),
                                     ('RD2', 'RD2', ':',  1.8)]:
        mask = (raw['Variety'] == v_id) & (raw['Treatment_clean'] == trt)
        subset = raw[mask]
        if len(subset) == 0:
            print(f"警告: 品种{v_id} {trt} 无数据")
            continue

        # 取该品种×处理的均值曲线
        curve = subset[time_cols].mean().values
        label = f'{v_label} - {trt_label}'
        ax.plot(time_ms, curve, ls, color=color, linewidth=lw,
                label=label, zorder=3)

ax.set_xscale('log')
ax.set_xlabel('时间 (ms, 对数坐标)', fontsize=12)
ax.set_ylabel('荧光强度 (a.u.)', fontsize=12)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 标注OJIP关键时间点
key_times = {'O': 0.05, 'J': 2, 'I': 30, 'P': None}
# P点用实际Fm时间（约200-240ms）
for name, t in [('O\n(0.05ms)', 0.05), ('J\n(2ms)', 2), ('I\n(30ms)', 30)]:
    ax.axvline(x=t, color='#cccccc', linestyle=':', linewidth=0.8, zorder=1)
    ax.annotate(name, xy=(t, ax.get_ylim()[0]),
                xytext=(t, -0.02), textcoords=('data', 'axes fraction'),
                fontsize=8, ha='center', color='#888888')

# 标注相位区域
phase_regions = [
    (0.05, 2, 'O-J相'),
    (2, 30, 'J-I相'),
    (30, 300, 'I-P相'),
]
ylim = ax.get_ylim()
for x0, x1, label in phase_regions:
    mid_x = np.sqrt(x0 * x1)
    ax.annotate(label, xy=(mid_x, ylim[1] * 0.95),
                fontsize=9, ha='center', color='#888888',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                          edgecolor='#cccccc', alpha=0.8))

ax.legend(fontsize=9, loc='lower right')

plt.tight_layout()
out = Path(__file__).parent / 'Figure_3-6_ojip_curve.png'
plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"已保存: {out}")
