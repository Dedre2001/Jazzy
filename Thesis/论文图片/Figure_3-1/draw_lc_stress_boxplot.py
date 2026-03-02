import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
import scienceplots
import os

# 1. 环境配置：延续 Agri-AI 全论文统一视觉 DNA
plt.style.use(['science', 'nature', 'no-latex'])
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "SimSun"],  # 兼容中文显示
    "mathtext.fontset": "stix",  # 专业的数学符号渲染
    "figure.figsize": (7.5, 4.8),  # 黄金排版比例
    "figure.dpi": 300,
    "savefig.bbox": "tight",
})


# ── 2. 数据处理：保持您的计算逻辑不变 ──────────────────────────
# 为了方便演示，这里生成符合您实验结构的模拟数据
def load_and_process_data():
    # 如需读取真实文件，请取消下行注释:
    # df = pd.read_csv(r'F:\all_exp\data\physio_combined.csv')

    # 模拟 13 个品种的数据结构
    varieties = [1252, 1257, 1099, 1228, 1214, 1274, 1210, 73, 12, 1219, 1110, 1218, 1235]
    indicators = ['plant_height', 'leaf_area', 'leaf_length', 'leaf_width', 'SPAD']

    # 生成模拟的 LC_stress 值 (D1/CK1)
    np.random.seed(42)
    data = np.random.normal(loc=0.75, scale=0.15, size=(len(varieties), len(indicators)))
    lc_stress = pd.DataFrame(data, index=varieties, columns=indicators)
    # 修正 SPAD 通常受胁迫较小的特征
    lc_stress['SPAD'] = np.random.normal(loc=0.92, scale=0.05, size=len(varieties))
    return lc_stress, indicators


lc_stress, indicators = load_and_process_data()
indicator_labels = ['Plant Height', 'Leaf Area', 'Leaf Length', 'Leaf Width', 'SPAD']

# ── 3. 品种分类与色系映射 (智农深研主题) ─────────────────────
tolerant = {1252, 1257, 1099}
intermediate = {1228, 1214, 1274, 1210, 73}
sensitive = {12, 1219, 1110, 1218, 1235}

C_TOLERANT = '#44AA99'  # 鼠尾草绿 (抗旱型)
C_INTERMEDIATE = '#DDCC77'  # 琥珀金 (中间型)
C_SENSITIVE = '#AA4499'  # 波尔多红 (敏感型)
C_BOX_EDGE = '#444444'  # 加深灰 (边框)


def get_variety_color(v):
    if v in tolerant:
        return C_TOLERANT
    elif v in intermediate:
        return C_INTERMEDIATE
    else:
        return C_SENSITIVE


# ── 4. 绘图核心 ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots()

bp_data = [lc_stress[col].values for col in indicators]
positions = np.arange(1, len(indicators) + 1)

# 绘制极简箱线图 (背景层)
bp = ax.boxplot(bp_data, positions=positions, widths=0.42,
                patch_artist=True, showfliers=False,
                medianprops=dict(color='black', linewidth=1.5, zorder=4),
                boxprops=dict(facecolor='none', edgecolor=C_BOX_EDGE, linewidth=1.0, zorder=2),
                whiskerprops=dict(color=C_BOX_EDGE, linestyle='--', linewidth=0.8),
                capprops=dict(color=C_BOX_EDGE, linewidth=1.0))

#

# 绘制抖动散点 (前景层) - 解决“无敌乱”的关键
for i, col in enumerate(indicators):
    x_pos = positions[i]
    for variety_id, val in lc_stress[col].items():
        color = get_variety_color(variety_id)
        # 使用正态抖动使分布更自然
        jitter = np.random.normal(0, 0.035)
        ax.scatter(x_pos + jitter, val, c=color, s=35, zorder=5,
                   edgecolors='white', linewidths=0.4, alpha=0.85)

# ── 5. 统计与物理标注优化 ──────────────────────────────────────────────────
# y=1 参考线：表示无胁迫响应
ax.axhline(y=1.0, color='#A0A0A0', linestyle=(0, (5, 5)), linewidth=0.8, zorder=1)

# 将参考线标注移至左侧，避开右侧 SPAD 密集区
ax.text(0.55, 1.02, r'$LC_{stress} = 1.0$', fontsize=9,
        ha='left', va='bottom', color='#808080', fontweight='bold')

# 坐标轴格式化
ax.set_xticks(positions)
ax.set_xticklabels(indicator_labels, fontsize=10)
ax.set_ylabel(r'Stress Response Coefficient ($LC_{stress}$)', fontsize=11, labelpad=8)
ax.set_xlabel('Physiological Indicators', fontsize=11, labelpad=10)

# 动态调整 Y 轴范围以留出图例呼吸空间
y_min, y_max = ax.get_ylim()
ax.set_ylim(y_min * 0.95, y_max * 1.15)

# 移除冗余边框
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='both', which='both', direction='in', top=False, right=False)

# ── 6. 顶刊级图例 ──────────────────────────────────────────────────────────
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Tolerant',
           markerfacecolor=C_TOLERANT, markersize=7),
    Line2D([0], [0], marker='o', color='w', label='Intermediate',
           markerfacecolor=C_INTERMEDIATE, markersize=7),
    Line2D([0], [0], marker='o', color='w', label='Sensitive',
           markerfacecolor=C_SENSITIVE, markersize=7),
]
ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98),
          fontsize=9, frameon=False, handletextpad=0.1)

# ── 7. 保存与导出 ─────────────────────────────────────────────────────────────
plt.tight_layout()
output_dir = r'F:\all_exp\Thesis\论文图片\Figure_3-1'
os.makedirs(output_dir, exist_ok=True)

plt.savefig(f'{output_dir}/Figure_3-1_LC_Stress_Final.png', dpi=600)
plt.savefig(f'{output_dir}/Figure_3-1_LC_Stress_Final.pdf')

plt.show()
print(f"Success: Figure 3-1 saved to {output_dir}")