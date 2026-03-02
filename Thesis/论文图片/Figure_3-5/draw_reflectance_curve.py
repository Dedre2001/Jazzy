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
    "figure.figsize": (7.5, 5.8),  # 优化长宽比
    "figure.dpi": 600,
    "axes.linewidth": 1.25,  # 增强轴线分量
    "font.size": 11,
    "axes.labelsize": 13,
    "savefig.bbox": "tight",
})

# 2. Agri-AI 2.0 核心配色方案
COLORS = {
    'CK1': '#44AA99',  # 鼠尾草绿
    'D1': '#AA4499',  # 波尔多红
    'RD2': '#DDCC77',  # 琥珀金
    'bracket': '#555555'
}

# 3. 数据加载与百分比换算逻辑
bands = ['R460', 'R520', 'R590', 'R660', 'R680', 'R710', 'R730', 'R780', 'R820', 'R850', 'R910']
wavelengths = [460, 520, 590, 660, 680, 710, 730, 780, 820, 850, 910]
x_pos = np.arange(len(wavelengths))

file_path = r'F:\all_exp\Thesis\论文图片\Figure_3-5\features_40_nir_corrected.csv'
if os.path.exists(file_path):
    df = pd.read_csv(file_path)
    # 核心修改：将所有反射率波段数值乘以 100 转换为百分比
    df[bands] = df[bands] * 100
else:
    # 模拟数据块 (百分比维度：0-100)
    np.random.seed(42)
    data = {'Treatment': ['CK1'] * 20 + ['D1'] * 20 + ['RD2'] * 20}
    for i, b in enumerate(bands):
        base = 52.0 if int(b[1:]) > 720 else (7.0 if b == 'R680' else 14.0)
        data[b] = np.random.normal(base, 4.0, 60)
    df = pd.DataFrame(data)

# 4. 绘图核心
fig, ax = plt.subplots()

# A. 绘制光谱曲线：增强 SE 阴影可见度
for trt, marker in [('CK1', 'o'), ('D1', 's'), ('RD2', '^')]:
    color = COLORS[trt]
    sub = df[df['Treatment'] == trt][bands]

    # 计算均值与标准误 (SE)
    mean = sub.mean().values
    se = sub.std().values / np.sqrt(len(sub))

    # 绘制实心点线条
    ax.plot(x_pos, mean, color=color, linewidth=1.8, marker=marker,
            markersize=5.5, markerfacecolor=color, markeredgecolor='white',
            markeredgewidth=0.8, label=f'{trt} Treatment', zorder=5)

    # 核心修改：绘制 SE 阴影区，调整 alpha 确保重叠区清晰
    ax.fill_between(x_pos, mean - se, mean + se, color=color, alpha=0.20, zorder=4)

# B. 顶部光谱分区标注 (全英文支架系统)
regions = [
    (0, 3.5, 'Visible (VIS)', '#F7F7F7'),
    (3.5, 5.5, 'Red Edge', '#FFF9F9'),
    (5.5, 10, 'Near-Infrared (NIR)', '#F9FFF9')
]

# 动态计算支架高度
y_min, y_max = ax.get_ylim()
y_bracket = y_max * 1.05

for x0, x1, lbl, fc in regions:
    ax.axvspan(x0, x1, color=fc, alpha=1.0, zorder=0)
    ax.plot([x0, x1], [y_bracket, y_bracket], color=COLORS['bracket'], linewidth=1.1, zorder=6)
    ax.vlines([x0, x1], y_bracket - 0.5, y_bracket, color=COLORS['bracket'], linewidth=1.1, zorder=6)
    ax.text((x0 + x1) / 2, y_bracket + 0.8, lbl, fontsize=9.5,
            ha='center', va='bottom', color='#333333', fontweight='bold')

# 5. 坐标轴修正：更新纵坐标标签
ax.set_xlabel('Wavelength (nm)', fontweight='bold', labelpad=14)
ax.set_ylabel('Reflectance (%)', fontweight='bold', labelpad=14)  # 更新为百分比标注
ax.set_xticks(x_pos)
ax.set_xticklabels(wavelengths, fontsize=10)

# 封闭刻度 (Boxed Ticks)
ax.minorticks_off()
ax.tick_params(direction='in', top=True, right=True, which='both', length=7, width=1.25)

# 设置轴范围，预留标注空间
ax.set_ylim(0, y_bracket * 1.30)

# 图例放置
ax.legend(loc='upper left', frameon=False, fontsize=10.5, borderaxespad=1.8)

# 6. 保存导出
plt.tight_layout()
output_dir = r'F:\all_exp\Thesis\论文图片\Figure_3-5'
if not os.path.exists(output_dir): os.makedirs(output_dir)
plt.savefig(f'{output_dir}/Figure_3-5_Spectral_AgriAI_Final.png', dpi=600)
plt.savefig(f'{output_dir}/Figure_3-5_Spectral_AgriAI_Final.pdf')

plt.show()