import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
import os

# 1. 环境配置：Agri-AI 2.0 顶刊视觉 DNA
plt.style.use(['science', 'nature', 'no-latex'])
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "figure.figsize": (7.5, 5.8),  # 采用更紧致的比例
    "axes.linewidth": 1.25,  # 力量感轴宽
    "font.size": 10.5,
})

# 2. Agri-AI 2.0 核心配色 (品种抗性映射)
COLORS = {1252: '#44AA99', 1228: '#DDCC77', 1235: '#AA4499', 'bracket': '#555555'}
TRT_STYLES = [('CK1', '-', 'Control'), ('D1', '--', 'Stress'), ('RD2', ':', 'Recovery')]

# ── 3. 数据处理 (核心：百分比换算) ──
bands = [f'R{b}' for b in [460, 520, 590, 660, 680, 710, 730, 780, 820, 850, 910]]
wavelengths = [460, 520, 590, 660, 680, 710, 730, 780, 820, 850, 910]
x_pos = np.arange(len(wavelengths))

df = pd.read_csv(r'F:\all_exp\Thesis\论文图片\Figure_3-5\features_40_nir_corrected.csv')
df[bands] = df[bands] * 100

# ── 4. 绘图核心逻辑 ──
fig, ax = plt.subplots()

# 记录全局最高点，用于自适应高度计算
global_peak = 0

for v_id, cat in [(1252, 'Tolerant'), (1228, 'Intermediate'), (1235, 'Sensitive')]:
    for trt, ls, t_lbl in TRT_STYLES:
        sub = df[(df['Variety'] == v_id) & (df['Treatment'] == trt)][bands]
        if sub.empty: continue
        mean, se = sub.mean().values, sub.std().values / np.sqrt(len(sub))

        # 更新全局峰值 (包含 SE 阴影的上界)
        current_peak = (mean + se).max()
        if current_peak > global_peak: global_peak = current_peak

        # 绘线
        ax.plot(x_pos, mean, color=COLORS[v_id], ls=ls, lw=1.3,
                label=f'V{v_id} ({cat}) - {t_lbl}', zorder=5)
        ax.fill_between(x_pos, mean - se, mean + se, color=COLORS[v_id], alpha=0.07, zorder=4)

# ── 5. 分区标注：自适应“悬浮”逻辑 ──
# 核心修正：将支架设在数据绝对峰值上方 6% 处，确保绝不压线
y_bracket = global_peak + 5.0

regions = [(0, 3.5, 'VIS'), (3.5, 5.5, 'Red Edge'), (5.5, 10, 'NIR')]
for x0, x1, lbl in regions:
    ax.plot([x0, x1], [y_bracket, y_bracket], color=COLORS['bracket'], lw=1.1, zorder=10)
    ax.vlines([x0, x1], y_bracket - 1.5, y_bracket, color=COLORS['bracket'], lw=1.1, zorder=10)
    ax.text((x0 + x1) / 2, y_bracket + 0.8, lbl, ha='center', va='bottom',
            fontsize=9.2, fontweight='bold', color='#444444')

# ── 6. 坐标轴与“零留白”控制 ──
ax.set_xlabel('Wavelength (nm)', fontweight='bold', labelpad=12)
ax.set_ylabel('Reflectance (%)', fontweight='bold', labelpad=12)
ax.set_xticks(x_pos)
ax.set_xticklabels(wavelengths)

# 四周内向刻度
ax.minorticks_off()
ax.tick_params(direction='in', top=True, right=True, length=7, width=1.25)

# 核心修正：消除空洞。将上限锁定在支架上方 12 个单位，刚好塞下图例
ax.set_ylim(0, y_bracket + 12.5)

# 图例回归：紧贴支架上方，解决悬浮感
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.13), ncol=3,
          frameon=False, fontsize=8.0, columnspacing=0.6, handletextpad=0.3)

plt.tight_layout()
# 导出 PDF + PNG
output_path = r'F:\all_exp\Thesis\论文图片\Figure_3-5b\Figure_3-5b_UltraCompact.png'
plt.savefig(output_path, dpi=600)
plt.show()