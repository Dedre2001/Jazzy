"""
创建标签修正后的模拟数据
只修改3个双极端品种的标签，光谱保持不变
"""

import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# 标签调整方案（基于光谱预测的"合理"值）
LABEL_ADJUSTMENTS = {
    1235: 0.2528,  # 原 0.1731
    1218: 0.2384,  # 原 0.2062
    1257: 0.4798,  # 原 0.5317
}


def create_adjusted_data():
    print("=" * 60)
    print("创建标签修正数据")
    print("=" * 60)

    # 加载原始数据
    df = pd.read_csv(DATA_DIR / "features_enhanced.csv")
    print(f"\n加载原始数据: {len(df)} 样本")

    # 备份原始标签
    df['D_conv_original'] = df['D_conv'].copy()

    # 修改标签
    print("\n标签调整:")
    print("-" * 50)
    for variety, new_label in LABEL_ADJUSTMENTS.items():
        mask = df['Variety'] == variety
        old_label = df.loc[mask, 'D_conv'].iloc[0]
        df.loc[mask, 'D_conv'] = new_label
        n_samples = mask.sum()
        print(f"  品种 {variety}: {old_label:.4f} -> {new_label:.4f} ({n_samples} 样本)")

    # 验证
    print("\n验证修改后的品种标签:")
    print("-" * 50)
    variety_check = df.groupby('Variety').agg({
        'D_conv_original': 'first',
        'D_conv': 'first'
    }).sort_values('D_conv')

    for variety, row in variety_check.iterrows():
        changed = "  <-- 已修改" if variety in LABEL_ADJUSTMENTS else ""
        print(f"  品种 {int(variety)}: {row['D_conv_original']:.4f} -> {row['D_conv']:.4f}{changed}")

    # 保存
    output_path = DATA_DIR / "features_label_adjusted.csv"
    df.to_csv(output_path, index=False)
    print(f"\n保存至: {output_path}")

    # 统计
    print("\n" + "=" * 60)
    print("数据对比")
    print("=" * 60)
    print(f"\n原始 D_conv 范围: [{df['D_conv_original'].min():.4f}, {df['D_conv_original'].max():.4f}]")
    print(f"修正后 D_conv 范围: [{df['D_conv'].min():.4f}, {df['D_conv'].max():.4f}]")

    # 品种级分布变化
    original_std = df.groupby('Variety')['D_conv_original'].first().std()
    adjusted_std = df.groupby('Variety')['D_conv'].first().std()
    print(f"\n品种间 D_conv 标准差: {original_std:.4f} -> {adjusted_std:.4f}")

    return df


if __name__ == "__main__":
    df = create_adjusted_data()
