"""
绘制OJIP曲线对比图
展示三个代表品种(1252, 1228, 1235)在三个处理(CK1, D1, RD2)下的OJIP曲线
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 设置路径
DATA_DIR = Path("F:/all_exp/data")
INPUT_FILE = DATA_DIR / "ojip_curves.csv"
OUTPUT_DIR = Path("F:/all_exp/results/figures/chapter3")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_ojip_curves():
    """加载OJIP曲线数据"""
    df = pd.read_csv(INPUT_FILE)
    print(f"加载OJIP曲线数据: {len(df)} 条记录")
    return df

def plot_ojip_curves_comparison(df, varieties=[1252, 1228, 1235]):
    """
    绘制OJIP曲线对比图
    3×3面板: 3个品种 × 3个处理
    """
    # 获取时间点列(排除Variety和Treatment列)
    time_cols = [col for col in df.columns if col not in ['Variety', 'Treatment']]
    time_points = np.array([float(col) for col in time_cols])

    # 创建图形
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    fig.suptitle('三类品种的OJIP荧光诱导曲线对比', fontsize=16, fontweight='bold', y=0.995)

    variety_names = {1252: '抗旱型(1252)', 1228: '中间型(1228)', 1235: '敏感型(1235)'}
    treatment_names = {'CK1': '对照(CK1)', 'D1': '干旱(D1)', 'RD2': '复水(RD2)'}
    colors = {'CK1': '#2ecc71', 'D1': '#e74c3c', 'RD2': '#3498db'}
    linestyles = {'CK1': '-', 'D1': '--', 'RD2': ':'}

    for i, variety in enumerate(varieties):
        for j, treatment in enumerate(['CK1', 'D1', 'RD2']):
            ax = axes[i, j]

            # 获取该品种×处理的所有重复数据
            df_subset = df[(df['Variety'] == variety) & (df['Treatment'] == treatment)]

            if len(df_subset) == 0:
                ax.text(0.5, 0.5, '无数据', ha='center', va='center', transform=ax.transAxes)
                continue

            # 绘制每个重复的曲线
            for idx, row in df_subset.iterrows():
                fluorescence = row[time_cols].values.astype(float)

                # 归一化到Fm=1
                fm = np.max(fluorescence)
                fluorescence_norm = fluorescence / fm

                ax.semilogx(time_points * 1000, fluorescence_norm,
                           color=colors[treatment], alpha=0.6, linewidth=1.5)

            # 计算平均曲线
            fluorescence_mean = df_subset[time_cols].mean().values.astype(float)
            fm_mean = np.max(fluorescence_mean)
            fluorescence_mean_norm = fluorescence_mean / fm_mean

            ax.semilogx(time_points * 1000, fluorescence_mean_norm,
                       color=colors[treatment], linewidth=2.5,
                       label=f'{treatment_names[treatment]} (均值)')

            # 标注O-J-I-P关键点
            # O点: 约0.05 ms
            # J点: 约2 ms
            # I点: 约30 ms
            # P点: 最大值
            if treatment == 'CK1':
                # 找到关键点的索引
                idx_j = np.argmin(np.abs(time_points - 0.002))  # 2 ms
                idx_i = np.argmin(np.abs(time_points - 0.03))   # 30 ms

                ax.axvline(0.05, color='gray', linestyle=':', alpha=0.3, linewidth=0.8)
                ax.axvline(2, color='gray', linestyle=':', alpha=0.3, linewidth=0.8)
                ax.axvline(30, color='gray', linestyle=':', alpha=0.3, linewidth=0.8)

                ax.text(0.05, 0.95, 'O', transform=ax.transAxes,
                       fontsize=10, ha='left', va='top', color='gray')
                ax.text(0.25, 0.95, 'J', transform=ax.transAxes,
                       fontsize=10, ha='left', va='top', color='gray')
                ax.text(0.55, 0.95, 'I', transform=ax.transAxes,
                       fontsize=10, ha='left', va='top', color='gray')
                ax.text(0.85, 0.95, 'P', transform=ax.transAxes,
                       fontsize=10, ha='left', va='top', color='gray')

            # 设置标签和标题
            if i == 0:
                ax.set_title(treatment_names[treatment], fontsize=12, fontweight='bold')
            if j == 0:
                ax.set_ylabel(f'{variety_names[variety]}\n相对荧光强度', fontsize=11)
            if i == 2:
                ax.set_xlabel('时间 (ms)', fontsize=11)

            ax.set_xlim(0.01, 3000)
            ax.set_ylim(0, 1.1)
            ax.grid(True, alpha=0.3, linestyle='--')

            if i == 0 and j == 2:
                ax.legend(loc='lower right', fontsize=8)

    plt.tight_layout()
    output_file = OUTPUT_DIR / "fig3_8_ojip_curves.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"OJIP曲线图已保存到: {output_file}")
    plt.close()

def main():
    """主函数"""
    print("="*60)
    print("绘制OJIP曲线对比图")
    print("="*60)

    # 加载数据
    df = load_ojip_curves()

    # 绘制OJIP曲线对比图
    print("\n绘制OJIP曲线对比图...")
    plot_ojip_curves_comparison(df)

    print("\n" + "="*60)
    print("OJIP曲线绘制完成!")
    print("="*60)

if __name__ == "__main__":
    main()
