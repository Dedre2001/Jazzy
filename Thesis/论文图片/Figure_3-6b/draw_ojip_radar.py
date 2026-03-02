"""
Figure 3-6b: 三类代表品种OJIP参数雷达图
6轴: Fv/Fm, PIabs, Vi, Vj, TRo/RC, ETo/RC
以CK1为100%标准化，展示D1和RD2下的相对变化
数据源: data/raw/ojip.csv（含三处理）
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from pathlib import Path

rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False

df = pd.read_csv(r'F:\all_exp\data\raw\ojip.csv')

# TRo/RC = ETo/RC / (1 - Vj)
df['OJIP_TRo_RC'] = df['OJIP_ETo_RC'] / (1 - df['OJIP_Vj'])

param_cols = ['OJIP_FvFm', 'OJIP_PIabs', 'OJIP_Vi', 'OJIP_Vj', 'OJIP_TRo_RC', 'OJIP_ETo_RC']
param_labels = ['Fv/Fm', 'PIabs', 'Vi', 'Vj', 'TRo/RC', 'ETo/RC']

varieties = [(1252, '#2ca02c', '1252 (抗旱型)'),
             (1228, '#ff7f0e', '1228 (中间型)'),
             (1235, '#d62728', '1235 (敏感型)')]

# 计算D1/CK1和RD2/CK1百分比
ratios = {}
for v_id, _, _ in varieties:
    ck1_vals = df[(df['Variety'] == v_id) & (df['Treatment'] == 'CK1')][param_cols].mean()
    for trt in ['D1', 'RD2']:
        trt_vals = df[(df['Variety'] == v_id) & (df['Treatment'] == trt)][param_cols].mean()
        ratios[(v_id, trt)] = (trt_vals / ck1_vals * 100).values

n_axes = len(param_cols)
angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(8, 8), dpi=300,
                        subplot_kw=dict(polar=True))

ref_vals = [100] * n_axes + [100]
ax.plot(angles, ref_vals, '-', color='#aaaaaa', linewidth=1.5,
        label='CK1 (100%)', zorder=2)

for v_id, color, v_label in varieties:
    for trt, ls, lw in [('D1', '--', 2.0), ('RD2', ':', 1.6)]:
        values = ratios[(v_id, trt)].tolist()
        values += values[:1]
        label = f'{v_label} - {trt}'
        ax.plot(angles, values, ls, color=color, linewidth=lw,
                label=label, zorder=3)
        ax.fill(angles, values, color=color, alpha=0.05)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(param_labels, fontsize=10)
ax.set_title('D1/RD2相对CK1变化率 (%)', fontsize=10, pad=20)
ax.legend(fontsize=9, loc='upper right', bbox_to_anchor=(1.35, 1.1))

plt.tight_layout()
out = r'F:\all_exp\Thesis\论文图片\Figure_3-6b\Figure_3-6b_ojip_radar.png'
plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"已保存: {out}")
