import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import scienceplots

# 1. 环境配置
plt.style.use(['science', 'nature', 'no-latex'])
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "stix",
    "figure.figsize": (7, 4.5),  # 略微加高以适应底部标注
    "figure.dpi": 300,
    "savefig.bbox": "tight",
})

# ── 数据定义 ─────────────────────────────────────────────────────────────
row_labels = ['Full Fusion', 'Static+OJIP', 'Multi+OJIP',
              'Multi+Static', 'OJIP-only', 'Multi-only', 'Static-only']
col_labels = ['$R^2$', 'Spearman $\\rho$', 'Pairwise Acc', 'Hit@3', 'RMSE']

r2 = [0.948, 0.924, 0.908, 0.883, 0.911, 0.798, 0.813]
rho = [1.000, 0.995, 0.978, 0.978, 0.978, 0.995, 0.923]
pacc = [1.000, 0.987, 0.949, 0.962, 0.949, 0.987, 0.897]
hit3 = [1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 0.670]
rmse = [0.030, 0.037, 0.040, 0.045, 0.039, 0.059, 0.057]

# RMSE 归一化处理（逆向）
rmse_arr = np.array(rmse)
rmse_norm = 1.0 - (rmse_arr - rmse_arr.min()) / (rmse_arr.max() - rmse_arr.min())

mat = np.column_stack([r2, rho, pacc, hit3, rmse_norm])
raw_annot = np.column_stack([r2, rho, pacc, hit3, rmse])

# ── 色系定义 ─────────────────────────────────────────────────────────────
colors = ["#AA4499", "#F0F0F0", "#4477AA"]
custom_cmap = mcolors.LinearSegmentedColormap.from_list("AgriAI_Heatmap", colors, N=256)

# ── 绘图核心 ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots()

im = ax.imshow(mat, cmap=custom_cmap, vmin=0.65, vmax=1.00,
               aspect='auto', interpolation='nearest')

# 坐标轴设置
ax.set_xticks(range(5))
ax.set_xticklabels(col_labels, fontsize=10)
ax.set_yticks(range(7))
ax.set_yticklabels(row_labels, fontsize=10)
ax.tick_params(which='both', length=0)

# ── 单元格标注 ──────────────────────────────────────────────────────────
for i in range(7):
    for j in range(5):
        norm_val = mat[i, j]
        raw_val = raw_annot[i, j]
        txt = f'{raw_val:.3f}'
        text_color = 'white' if (norm_val < 0.78 or norm_val > 0.92) else '#222222'
        ax.text(j, i, txt, ha='center', va='center',
                fontsize=9, color=text_color,
                fontweight='bold' if i == 0 else 'normal')

# ── 分隔线 ────────────────────────────────────────────────────────────
for yv in [0.5, 3.5]:
    ax.axhline(yv, color='white', linewidth=1.5, alpha=0.9, zorder=5)

# ── 2. 核心修改：左侧大括号分组（Brackets） ────────────────────────────
# 解决原代码中的标签重叠问题
group_info = [
    (0, 0, 'Full'),
    (1, 3, 'Dual'),
    (4, 6, 'Single'),
]

# 增加左侧间距以容纳大括号
plt.subplots_adjust(left=0.25)

for y_start, y_end, label in group_info:
    # 大括号的几何参数
    x_bracket = -0.28  # 括号在 axes fraction 的位置
    bracket_width = 0.03

    # 绘制括号主线（垂直）
    ax.plot([x_bracket, x_bracket], [y_start - 0.4, y_end + 0.4],
            transform=ax.get_yaxis_transform(), color='#333333', lw=1.2, clip_on=False)
    # 绘制括号端点（水平）
    for y_pos in [y_start - 0.4, y_end + 0.4]:
        ax.plot([x_bracket, x_bracket + bracket_width], [y_pos, y_pos],
                transform=ax.get_yaxis_transform(), color='#333333', lw=1.2, clip_on=False)

    # 绘制加深颜色的分组文字
    ax.text(x_bracket - 0.03, (y_start + y_end) / 2, label,
            transform=ax.get_yaxis_transform(), ha='right', va='center',
            fontsize=10, fontweight='bold', color='#333333')

# ── 3. 核心修改：RMSE 解释性标注 ─────────────────────────────────────
# 在图表下方增加逆向归一化说明
ax.text(1.0, -0.15, "*Note: RMSE is inversely normalized (smaller is darker blue).",
        transform=ax.transAxes, ha='right', va='top',
        fontsize=8, color='#555555', fontstyle='italic')

# ── Colorbar 精修 ──────────────────────────────────────────────────────────
cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.04)
cbar.set_label('Normalized Performance Index', fontsize=9, labelpad=8)
cbar.ax.tick_params(labelsize=8, direction='in', length=3)
cbar.outline.set_linewidth(0.8)

for spine in ax.spines.values():
    spine.set_linewidth(0.8)

# ── 保存 ────────────────────────────────────────────────────────────────
plt.tight_layout()
output_path = 'F:/all_exp/Thesis/论文图片/Figure_4-5'
os.makedirs(output_path, exist_ok=True)
plt.savefig(f'{output_path}/Figure_4-5_Heatmap_Pro.pdf')
plt.savefig(f'{output_path}/Figure_4-5_Heatmap_Pro.png', dpi=600)

plt.show()