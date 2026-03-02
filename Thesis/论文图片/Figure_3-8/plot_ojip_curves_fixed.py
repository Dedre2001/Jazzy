"""
绘制OJIP曲线对比图 - 修复版
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
OUTPUT_DIR = Path("F:/all_exp/Thesis/论文图片/Figure_3-8")
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

    for i, variety in enumerate(varieties):
        for j, treatment in enumerate(['CK1', 'D1', 'RD2']):
            ax = axes[i, j]

            # 获取该品种×处理的所有重复数据
            df_subset = df[(df['Variety'] == variety) & (df['Treatment'] == treatment)]

            if len(df_subset) == 0:
                ax.text(0.5, 0.5, '无数据', ha='center', va='center', transform=ax.transAxes)
                print(f"警告: 品种{variety}, 处理{treatment}无数据")
                continue

            print(f"绘制: 品种{variety}, 处理{treatment}, {len(df_subset)}条记录")

            # 绘制每个重复的曲线
            for idx, row in df_subset.iterrows():
                fluorescence = pd.to_numeric(row[time_cols], errors='coerce').values

                # 过滤掉NaN值
                valid_mask = ~np.isnan(fluorescence)
                if valid_mask.sum() == 0:
                    print(f"  警告: 所有数据都是NaN")
                    continue

                time_valid = time_points[valid_mask]
                fluor_valid = fluorescence[valid_mask]

                # 归一化到Fm=1
                fm = np.max(fluor_valid)
                fluorescence_norm = fluor_valid / fm

                ax.semilogx(time_valid * 1000, fluorescence_norm,
                           color=colors[treatment], alpha=0.4, linewidth=1.0)

            # 计算平均曲线
            fluorescence_mean = df_subset[time_cols].apply(pd.to_numeric, errors='coerce').mean().values

            # 过滤掉NaN值
            valid_mask = ~np.isnan(fluorescence_mean)
            time_valid = time_points[valid_mask]
            fluor_mean_valid = fluorescence_mean[valid_mask]

            if len(fluor_mean_valid) > 0:
                fm_mean = np.max(fluor_mean_valid)
                fluorescence_mean_norm = fluor_mean_valid / fm_mean

                ax.semilogx(time_valid * 1000, fluorescence_mean_norm,
                           color=colors[treatment], linewidth=3.0,
                           label=f'{treatment_names[treatment]}')

            # 标注O-J-I-P关键点
            if treatment == 'CK1':
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
                ax.legend(loc='lower right', fontsize=9)

    plt.tight_layout()
    output_file = OUTPUT_DIR / "fig3_8_ojip_curves.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nOJIP曲线图已保存到: {output_file}")
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
