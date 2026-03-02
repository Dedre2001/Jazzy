"""
OJIP数据可视化脚本
功能:
1. 绘制三个代表品种的雷达图对比(6个关键参数)
2. 绘制能量流瀑布图(可选)
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
INPUT_FILE = DATA_DIR / "processed" / "ojip_8params_processed.csv"
OUTPUT_DIR = Path("F:/all_exp/results/figures/chapter3")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    """加载处理后的数据"""
    df = pd.read_csv(INPUT_FILE)
    print(f"加载数据: {len(df)} 个样本")
    return df

def plot_radar_chart(df, varieties=[1252, 1228, 1235]):
    """
    绘制三个代表品种的雷达图对比
    展示6个关键参数: Fv/Fm, PIabs, Vj, Vi, TRo/RC, ETo/RC
    """
    # 选择参数
    params = ['Fv/Fm', 'PIabs', 'Vj', 'Vi', 'TRo/RC', 'ETo/RC']
    param_labels = ['Fv/Fm', 'PIabs', 'Vj', 'Vi', 'TRo/RC', 'ETo/RC']

    # 创建图形
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), subplot_kw=dict(projection='polar'))
    fig.suptitle('三类品种OJIP参数的雷达图对比', fontsize=16, fontweight='bold', y=1.02)

    variety_names = {1252: '抗旱型(1252)', 1228: '中间型(1228)', 1235: '敏感型(1235)'}
    colors_d1 = ['#1f77b4', '#ff7f0e', '#2ca02c']
    colors_rd2 = ['#9467bd', '#8c564b', '#e377c2']

    for idx, variety in enumerate(varieties):
        ax = axes[idx]
        df_var = df[df['Variety'] == variety]

        # 计算变化率
        change_rates_d1 = []
        change_rates_rd2 = []

        for param in params:
            ck1 = df_var[df_var['Treatment'] == 'CK1'][param].mean()
            d1 = df_var[df_var['Treatment'] == 'D1'][param].mean()
            rd2 = df_var[df_var['Treatment'] == 'RD2'][param].mean()

            # 计算相对CK1的变化率
            d1_change = (d1 - ck1) / ck1 * 100
            rd2_change = (rd2 - ck1) / ck1 * 100

            change_rates_d1.append(d1_change)
            change_rates_rd2.append(rd2_change)

        # 设置角度
        angles = np.linspace(0, 2 * np.pi, len(params), endpoint=False).tolist()
        change_rates_d1 += change_rates_d1[:1]  # 闭合
        change_rates_rd2 += change_rates_rd2[:1]
        angles += angles[:1]

        # 绘制雷达图
        ax.plot(angles, change_rates_d1, 'o-', linewidth=2, label='D1 vs CK1',
                color=colors_d1[idx], linestyle='--')
        ax.fill(angles, change_rates_d1, alpha=0.15, color=colors_d1[idx])

        ax.plot(angles, change_rates_rd2, 's-', linewidth=2, label='RD2 vs CK1',
                color=colors_rd2[idx], linestyle=':')
        ax.fill(angles, change_rates_rd2, alpha=0.15, color=colors_rd2[idx])

        # 设置标签
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(param_labels, fontsize=10)
        ax.set_ylim(-80, 80)
        ax.set_yticks([-60, -30, 0, 30, 60])
        ax.set_yticklabels(['-60%', '-30%', '0%', '+30%', '+60%'], fontsize=8)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_title(variety_names[variety], fontsize=12, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)

        # 添加0线
        ax.axhline(0, color='black', linewidth=0.8, linestyle='-', alpha=0.3)

    plt.tight_layout()
    output_file = OUTPUT_DIR / "fig3_9_radar_chart.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"雷达图已保存到: {output_file}")
    plt.close()

def plot_energy_flow(df, varieties=[1252, 1228, 1235]):
    """
    绘制能量流瀑布图
    展示ABS/RC → TRo/RC → ETo/RC → DIo/RC的能量分配
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('三类品种的能量流分配模式', fontsize=16, fontweight='bold')

    variety_names = {1252: '抗旱型(1252)', 1228: '中间型(1228)', 1235: '敏感型(1235)'}
    treatments = ['CK1', 'D1', 'RD2']
    colors = ['#2ecc71', '#f39c12', '#e74c3c']

    for idx, variety in enumerate(varieties):
        ax = axes[idx]
        df_var = df[df['Variety'] == variety]

        x_pos = np.arange(len(treatments))
        width = 0.2

        # 计算每个处理的能量流参数
        abs_rc = []
        tro_rc = []
        eto_rc = []
        dio_rc = []

        for treatment in treatments:
            df_treat = df_var[df_var['Treatment'] == treatment]
            abs_rc.append(df_treat['ABS/RC'].mean())
            tro_rc.append(df_treat['TRo/RC'].mean())
            eto_rc.append(df_treat['ETo/RC'].mean())
            dio_rc.append(df_treat['DIo/RC'].mean())

        # 绘制柱状图
        ax.bar(x_pos - width*1.5, abs_rc, width, label='ABS/RC', color='#3498db', alpha=0.8)
        ax.bar(x_pos - width*0.5, tro_rc, width, label='TRo/RC', color='#2ecc71', alpha=0.8)
        ax.bar(x_pos + width*0.5, eto_rc, width, label='ETo/RC', color='#f39c12', alpha=0.8)
        ax.bar(x_pos + width*1.5, dio_rc, width, label='DIo/RC', color='#e74c3c', alpha=0.8)

        ax.set_xlabel('处理', fontsize=11)
        ax.set_ylabel('能量通量 (相对单位)', fontsize=11)
        ax.set_title(variety_names[variety], fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(treatments)
        ax.legend(fontsize=9)
        ax.grid(axis='y', linestyle='--', alpha=0.3)

    plt.tight_layout()
    output_file = OUTPUT_DIR / "fig3_10_energy_flow.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"能量流图已保存到: {output_file}")
    plt.close()

def main():
    """主函数"""
    print("="*60)
    print("OJIP数据可视化")
    print("="*60)

    # 加载数据
    df = load_data()

    # 绘制雷达图
    print("\n绘制雷达图...")
    plot_radar_chart(df)

    # 绘制能量流图
    print("\n绘制能量流图...")
    plot_energy_flow(df)

    print("\n" + "="*60)
    print("可视化完成!")
    print("="*60)

if __name__ == "__main__":
    main()
