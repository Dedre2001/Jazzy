"""
Figure 3-7: 37特征Pearson相关性热力图
按模态分组: Multi(19) + Static(10) + OJIP(8)
数据源: features_40_nir_corrected.csv（已修正波段标签）
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False

df = pd.read_csv(r'F:\all_exp\Thesis\论文图片\Figure_3-5\features_40_nir_corrected.csv')

# ── 37个特征列（排除元数据和Trt哑变量）──
multi_cols = ['R460', 'R520', 'R590', 'R660', 'R680', 'R710', 'R730', 'R780',
              'R820', 'R850', 'R910',
              'VI_NDVI', 'VI_NDRE', 'VI_EVI', 'VI_SIPI', 'VI_PRI',
              'VI_MTCI', 'VI_GNDVI', 'VI_NDWI']
static_cols = ['BF(F440)', 'GF(F520)', 'RF(F690)', 'FrF(f740)',
               'SR_F690_F740', 'SR_F440_F690', 'SR_F440_F520',
               'SR_F520_F690', 'SR_F440_F740', 'SR_F520_F740']
ojip_cols = ['OJIP_FvFm', 'OJIP_PIabs', 'OJIP_TRo_RC', 'OJIP_ETo_RC',
             'OJIP_Vi', 'OJIP_Vj', 'OJIP_ABS_RC_log', 'OJIP_DIo_RC_log']

all_features = multi_cols + static_cols + ojip_cols
assert len(all_features) == 37, f"Expected 37 features, got {len(all_features)}"

corr = df[all_features].corr(method='pearson')

# ── 绘图 ──
fig, ax = plt.subplots(figsize=(14, 12), dpi=300)

im = ax.imshow(corr.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')

# 模态分隔线
n_multi = len(multi_cols)
n_static = len(static_cols)
for pos in [n_multi - 0.5, n_multi + n_static - 0.5]:
    ax.axhline(pos, color='black', linewidth=1.5, zorder=4)
    ax.axvline(pos, color='black', linewidth=1.5, zorder=4)

# 模态标签（侧边）
ax.text(-2.5, n_multi / 2, 'Multi\n(19)', fontsize=10, ha='center',
        va='center', fontweight='bold', color='#1f77b4')
ax.text(-2.5, n_multi + n_static / 2, 'Static\n(10)', fontsize=10,
        ha='center', va='center', fontweight='bold', color='#ff7f0e')
ax.text(-2.5, n_multi + n_static + len(ojip_cols) / 2, 'OJIP\n(8)',
        fontsize=10, ha='center', va='center', fontweight='bold', color='#2ca02c')

# 简化标签
short_labels = []
for c in all_features:
    c = c.replace('VI_', '').replace('OJIP_', '').replace('SR_', '')
    short_labels.append(c)

ax.set_xticks(range(37))
ax.set_xticklabels(short_labels, fontsize=6, rotation=90)
ax.set_yticks(range(37))
ax.set_yticklabels(short_labels, fontsize=6)

cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
cbar.set_label('Pearson 相关系数', fontsize=11)

plt.tight_layout()
out = r'F:\all_exp\Thesis\论文图片\Figure_3-7\Figure_3-7_correlation_heatmap.png'
plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"已保存: {out}")
