import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import scienceplots
import os

# 1. 环境配置：强制使用科学绘图样式
# 即使没有安装 LaTeX，'no-latex' 也能确保字体回退到类 Times New Roman 的 serif 字体
plt.style.use(['science', 'nature', 'no-latex'])

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "stix",  # 让数学符号（如 R², ρ）看起来更专业
    "figure.dpi": 300,
    "savefig.bbox": "tight",
})

# ── 数据定义 ─────────────────────────────────────────────────────────────
configs = ['Full\nFusion', 'Static\n+OJIP', 'Multi\n+OJIP',
           'Multi\n+Static', 'OJIP\nonly', 'Multi\nonly', 'Static\nonly']
r2 = [0.941, 0.924, 0.908, 0.883, 0.911, 0.798, 0.813]
rho = [1.000, 0.995, 0.978, 0.978, 0.978, 0.995, 0.923]

# 顶刊级配色方案
C_R2 = '#4477AA'    # 经典科技蓝
C_RHO = '#AA4499'   # 波尔多红 (Bordeaux Red)

# ── 创建画布 ──────────────────────────────────────────────────────────
# 增加高度以容纳下方的分组标签
fig, ax1 = plt.subplots(figsize=(7, 4.5))
ax2 = ax1.twinx()

x = np.arange(len(configs))
w = 0.35  # 柱子宽度

# 2. 绘制柱状图
# 增加 zorder 确保柱子在网格线上方，edgecolor 提升矢量感
ax1.bar(x - w/2, r2, w, color=C_R2, edgecolor='black', linewidth=0.5, label='$R^2$', zorder=3)
ax2.bar(x + w/2, rho, w, color=C_RHO, edgecolor='black', linewidth=0.5, label='Spearman $\\rho$', zorder=3)

# ── 标注重点数值 (Full Fusion) ───────────────────────────────────────
# 针对你研究中的 Full Fusion 核心结果进行突出
ax1.text(x[0]-w/2, r2[0]+0.008, f'{r2[0]:.3f}', ha='center', fontweight='bold', color=C_R2, size=8.5)
ax2.text(x[0]+w/2, rho[0]+0.004, f'{rho[0]:.3f}', ha='center', fontweight='bold', color=C_RHO, size=8.5)

# ── 坐标轴精修 ────────────────────────────────────────────────────────
ax1.set_ylabel('$R^2$', color=C_R2, fontweight='bold', size=11, labelpad=8)
ax2.set_ylabel('Spearman $\\rho$', color=C_RHO, fontweight='bold', size=11, rotation=270, labelpad=18)

# 设定合理的 y 轴范围，为顶部的图例预留空间
ax1.set_ylim(0.65, 1.10)
ax2.set_ylim(0.88, 1.03)

# 坐标轴颜色与数据联动
ax1.spines['left'].set_color(C_R2)
ax1.spines['left'].set_linewidth(1.0)
ax2.spines['right'].set_color(C_RHO)
ax2.spines['right'].set_linewidth(1.0)
ax1.spines['top'].set_visible(False)
ax2.spines['top'].set_visible(False)

# 刻度设置
ax1.set_xticks(x)
ax1.set_xticklabels(configs, size=9)
ax1.tick_params(axis='y', colors=C_R2, which='both', direction='in')
ax2.tick_params(axis='y', colors=C_RHO, which='both', direction='in')

# ── 3. 解决“标签压图”：分组虚线与文字 ────────────────────────────────────
# 绘制垂直分隔线
ax1.vlines([0.5, 4.5], 0.65, 1.10, colors='gray', ls='--', lw=0.7, alpha=0.4, zorder=1)

# 使用 axes fraction 绝对定位，彻底解决文字遮挡问题
# group_y 控制文字在图表下方的深度
group_y = -0.22
ax1.annotate('Full', xy=(0.07, group_y), xycoords='axes fraction',
             ha='center', fontweight='bold', color='#555555', size=9)
ax1.annotate('Dual-modality', xy=(0.4, group_y), xycoords='axes fraction',
             ha='center', fontweight='bold', color='#555555', size=9)
ax1.annotate('Single-modality', xy=(0.82, group_y), xycoords='axes fraction',
             ha='center', fontweight='bold', color='#555555', size=9)

# ── 4. 图例优化：移至上方，避免遮挡数据 ─────────────────────────────
legend_elements = [
    Patch(facecolor=C_R2, edgecolor='black', linewidth=0.5, label='$R^2$'),
    Patch(facecolor=C_RHO, edgecolor='black', linewidth=0.5, label='Spearman $\\rho$')
]
# 放置在图表正上方 (upper center)，分两列显示
ax1.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.02),
           ncol=2, frameon=False, fontsize=9, handletextpad=0.5, columnspacing=1.5)

# ── 5. 保存与导出 ─────────────────────────────────────────────────────
# 预留足够的底部边距给分组文字
plt.subplots_adjust(bottom=0.25, top=0.92)

# 保存为 PNG 用于快速预览，保存为 PDF 用于论文投稿（矢量格式）
output_path = 'F:/all_exp/Thesis/论文图片/Figure_4-4'
os.makedirs(output_path, exist_ok=True)

plt.savefig(f'{output_path}/Ablation_Bar_Bordeaux.png', dpi=600)
plt.savefig(f'{output_path}/Ablation_Bar_Bordeaux.pdf')

plt.show()
print("Success: Figure saved with Bordeaux Red and optimized layout.")