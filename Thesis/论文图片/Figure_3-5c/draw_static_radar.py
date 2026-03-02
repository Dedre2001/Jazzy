"""
Figure 3-5c: 稳态荧光LC化雷达图（三品种 × 三处理）
参数：4波段 + 6比值 = 10轴
LC化：CK1基线 = 1，D1/RD2 为相对倍数
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from pathlib import Path

rcParams['font.family'] = 'DejaVu Sans'
rcParams['axes.unicode_minus'] = False

# ── 数据加载（使用修正后数据：RF↓ GF↑ 符合文献直觉）──────────
df = pd.read_csv(r'F:\all_exp\data\raw\static_corrected.csv')

def get_raw_vals(sub):
    """返回4波段 + 6比值，共10个值"""
    bf  = sub['BF(F440)'].mean()
    gf  = sub['GF(F520)'].mean()
    rf  = sub['RF(F690)'].mean()
    frf = sub['FrF(f740)'].mean()
    return [bf, gf, rf, frf,
            bf/rf, bf/gf, bf/frf,
            gf/rf, gf/frf, rf/frf]

PARAM_LABELS = [
    'BF\n(F440)', 'GF\n(F520)', 'RF\n(F690)', 'FrF\n(F740)',
    'F440/\nF690',  'F440/\nF520',  'F440/\nF740',
    'F520/\nF690',  'F520/\nF740',  'F690/\nF740',
]
N_AXES = len(PARAM_LABELS)

VARIETIES = [
    (1252, '#2ca02c', '1252  Drought-tolerant'),
    (1228, '#ff7f0e', '1228  Intermediate'),
    (1235, '#d62728', '1235  Drought-sensitive'),
]

# ── LC化计算（相对CK1归一化）────────────────────────────────
lc_data = {}
for v_id, _, _ in VARIETIES:
    ck_vals = get_raw_vals(df[(df['Variety'] == v_id) & (df['Treatment'] == 'CK1')])
    lc_data[(v_id, 'CK1')] = [1.0] * N_AXES
    for trt in ['D1', 'RD2']:
        raw = get_raw_vals(df[(df['Variety'] == v_id) & (df['Treatment'] == trt)])
        lc_data[(v_id, trt)] = [r / c for r, c in zip(raw, ck_vals)]

# ── 雷达角度 ──────────────────────────────────────────────
angles = np.linspace(0, 2 * np.pi, N_AXES, endpoint=False).tolist()
angles_closed = angles + angles[:1]

# ── 绘图 ──────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 7), dpi=300,
                         subplot_kw=dict(polar=True))
fig.patch.set_facecolor('white')

YLIM = 2.3
YTICKS = [0.5, 1.0, 1.5, 2.0]

for ax, (v_id, color, v_label) in zip(axes, VARIETIES):
    # CK1 基线（灰色实线，全部=1）
    baseline = [1.0] * (N_AXES + 1)
    ax.plot(angles_closed, baseline, '-', color='#888888',
            linewidth=1.5, alpha=0.9, label='CK1 (baseline=1)')
    ax.fill(angles_closed, baseline, color='#cccccc', alpha=0.15)

    # D1
    d1_vals = lc_data[(v_id, 'D1')] + [lc_data[(v_id, 'D1')][0]]
    ax.plot(angles_closed, d1_vals, '--', color=color,
            linewidth=2.2, label='D1')
    ax.fill(angles_closed, d1_vals, color=color, alpha=0.10)

    # RD2
    rd2_vals = lc_data[(v_id, 'RD2')] + [lc_data[(v_id, 'RD2')][0]]
    ax.plot(angles_closed, rd2_vals, ':', color=color,
            linewidth=2.0, label='RD2')
    ax.fill(angles_closed, rd2_vals, color=color, alpha=0.06)

    ax.set_xticks(angles)
    ax.set_xticklabels(PARAM_LABELS, fontsize=8.5)
    ax.set_ylim(0, YLIM)
    ax.set_yticks(YTICKS)
    ax.set_yticklabels([str(y) for y in YTICKS], fontsize=7, color='#555555')
    ax.grid(True, alpha=0.35)
    ax.set_title(v_label, fontsize=11, pad=22, fontweight='bold', color=color)
    ax.legend(fontsize=8.5, loc='upper right',
              bbox_to_anchor=(1.38, 1.18), framealpha=0.85)

plt.suptitle(
    'Static Fluorescence — LC-Normalized Radar  (relative to CK1 = 1.0)',
    fontsize=13, y=1.03, fontweight='bold'
)
plt.tight_layout()

out = Path(r'F:\all_exp\Thesis\论文图片\Figure_3-5c') / 'Figure_3-5c_static_radar_LC.png'
plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"已保存: {out}")
