import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib.collections import LineCollection
import scienceplots
import os

# 1. 环境配置：回归纯粹的有衬线风格 (Serif Style)
plt.style.use(['science', 'nature', 'no-latex'])
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"], # 移除无衬线干扰
    "mathtext.fontset": "stix",
    "figure.figsize": (8, 5.5),
    "figure.dpi": 300,
    "font.size": 11,
    "savefig.bbox": "tight",
})

# ── 2. 数据准备 ──
varieties = ['1252', '1257', '1099', '1228', '1214', '1274',
             '1210', '73', '12', '1219', '1110', '1218', '1235']
d_conv = np.array([0.5747, 0.5317, 0.5283, 0.4714, 0.4225, 0.4107,
                    0.3562, 0.3277, 0.2650, 0.2370, 0.2232, 0.2062, 0.1731])
X = d_conv.reshape(-1, 1)
Z = linkage(X, method='ward', metric='euclidean')

# ── 3. Agri-AI (智农深研) 统一色系 ──
C_TOLERANT = '#44AA99'
C_INTERMEDIATE = '#DDCC77'
C_SENSITIVE = '#AA4499'
C_LINE = '#555555'

tolerant = {'1252', '1257', '1099'}
intermediate = {'1228', '1214', '1274', '1210', '73'}
sensitive = {'12', '1219', '1110', '1218', '1235'}

label_to_color = {v: (C_TOLERANT if v in tolerant else C_INTERMEDIATE if v in intermediate else C_SENSITIVE) for v in varieties}

# ── 4. 绘图 ──
fig, ax = plt.subplots()
merge_distances = Z[:, 2]
cut_distance = (merge_distances[-2] + merge_distances[-3]) / 2

# 绘制树状图：全英文环境
dn = dendrogram(Z, labels=varieties, leaf_rotation=0, leaf_font_size=11,
                color_threshold=cut_distance, above_threshold_color=C_LINE, ax=ax)

# ── 5. 枝干着色 ──
for d in ax.get_children():
    if isinstance(d, LineCollection):
        for seg in d.get_segments():
            x_mid = np.mean(seg[:, 0])
            idx = int((x_mid - 5) / 10)
            if idx < len(dn['ivl']):
                v_name = dn['ivl'][idx]
                d.set_colors([label_to_color[v_name]])

# ── 6. 视觉分层与英文标注 ──
# 截断线标注：改为英文，更显学术感
ax.axhline(y=cut_distance, color=C_LINE, linestyle='--', linewidth=0.8, alpha=0.7)
ax.text(ax.get_xlim()[1]*0.98, cut_distance, r'$K=3$ Cutoff',
        va='bottom', ha='right', fontsize=10, color=C_LINE, fontweight='bold')

# 品种标签着色加粗
for lbl in ax.get_xticklabels():
    v_name = lbl.get_text()
    lbl.set_color(label_to_color[v_name])
    lbl.set_fontweight('bold')

# 坐标轴美化：全英文标注
ax.set_ylabel('Cluster Distance (Ward\'s Method)', fontsize=12, fontweight='bold')
ax.set_xlabel('Rice Variety ID', fontsize=12, fontweight='bold', labelpad=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 背景分类色带
ax.axvspan(0, 30, color=C_TOLERANT, alpha=0.06, zorder=0)
ax.axvspan(30, 80, color=C_INTERMEDIATE, alpha=0.06, zorder=0)
ax.axvspan(80, 130, color=C_SENSITIVE, alpha=0.06, zorder=0)

# ── 7. 图例：采用全英文展示 ──
# 在毕业论文中，通过这种方式展示分类非常标准
legend_elements = [
    mpatches.Patch(color=C_TOLERANT, label='Tolerant'),
    mpatches.Patch(color=C_INTERMEDIATE, label='Intermediate'),
    mpatches.Patch(color=C_SENSITIVE, label='Sensitive'),
]
ax.legend(handles=legend_elements, loc='upper right', frameon=False, fontsize=10)

# ── 8. 保存 ──
plt.tight_layout()
output_dir = r'F:\all_exp\Thesis\论文图片\Figure_3-3'
os.makedirs(output_dir, exist_ok=True)
plt.savefig(f'{output_dir}/Figure_3-3_Dendrogram_English.png', dpi=600)
plt.savefig(f'{output_dir}/Figure_3-3_Dendrogram_English.pdf')

plt.show()