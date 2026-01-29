"""
创建全部极端品种标签修正后的数据
包括：标签极端 OR 光谱极端的品种
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


def create_full_adjusted_data():
    print("=" * 60)
    print("创建全部极端品种标签修正数据")
    print("=" * 60)

    # 加载数据
    df = pd.read_csv(DATA_DIR / "features_enhanced.csv")
    print(f"\n加载数据: {len(df)} 样本, {df['Variety'].nunique()} 品种")

    # 品种级统计
    variety_stats = df.groupby('Variety').agg({
        'D_conv': 'first',
        'mahal_distance': 'mean'
    }).reset_index()

    # 定义极端品种
    d_conv_q10 = variety_stats['D_conv'].quantile(0.15)  # 放宽一点
    d_conv_q90 = variety_stats['D_conv'].quantile(0.85)
    mahal_median = variety_stats['mahal_distance'].median()

    variety_stats['is_label_extreme'] = (variety_stats['D_conv'] < d_conv_q10) | (variety_stats['D_conv'] > d_conv_q90)
    variety_stats['is_spectral_extreme'] = variety_stats['mahal_distance'] > mahal_median
    variety_stats['is_any_extreme'] = variety_stats['is_label_extreme'] | variety_stats['is_spectral_extreme']

    # 用完全正常的品种训练模型
    band_cols = ['R460', 'R520', 'R580', 'R660', 'R710', 'R730',
                 'R760', 'R780', 'R810', 'R850', 'R900']

    variety_spectra = df.groupby('Variety')[band_cols].mean()
    variety_spectra['D_conv'] = df.groupby('Variety')['D_conv'].first()

    # 正常品种（标签和光谱都不极端）
    normal_varieties = variety_stats[~variety_stats['is_any_extreme']]['Variety'].values
    extreme_varieties = variety_stats[variety_stats['is_any_extreme']]['Variety'].values

    print(f"\n正常品种: {len(normal_varieties)}")
    print(f"极端品种: {len(extreme_varieties)}")

    # 用正常品种训练
    X_train = variety_spectra.loc[normal_varieties, band_cols].values
    y_train = variety_spectra.loc[normal_varieties, 'D_conv'].values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = Ridge(alpha=1.0)
    model.fit(X_train_scaled, y_train)

    # 预测所有品种
    X_all = variety_spectra[band_cols].values
    X_all_scaled = scaler.transform(X_all)
    predicted_labels = model.predict(X_all_scaled)

    variety_spectra['predicted_D_conv'] = predicted_labels

    # 创建标签映射
    print("\n" + "=" * 60)
    print("标签调整详情")
    print("=" * 60)
    print(f"\n{'品种':<10} {'原标签':>10} {'预测标签':>10} {'调整':>10} {'类型':<15}")
    print("-" * 60)

    label_map = {}
    for variety in variety_spectra.index:
        original = variety_spectra.loc[variety, 'D_conv']
        predicted = variety_spectra.loc[variety, 'predicted_D_conv']

        is_extreme = variety in extreme_varieties
        if is_extreme:
            label_map[variety] = predicted
            adjustment = predicted - original
            # 判断极端类型
            row = variety_stats[variety_stats['Variety'] == variety].iloc[0]
            if row['is_label_extreme'] and row['is_spectral_extreme']:
                ext_type = "双极端"
            elif row['is_label_extreme']:
                ext_type = "标签极端"
            else:
                ext_type = "光谱极端"
            print(f"{int(variety):<10} {original:>10.4f} {predicted:>10.4f} {adjustment:>+10.4f} {ext_type:<15}")
        else:
            label_map[variety] = original  # 保持不变

    # 应用修正
    df['D_conv_original'] = df['D_conv'].copy()
    df['D_conv'] = df['Variety'].map(label_map)

    # 保存
    output_path = DATA_DIR / "features_label_full_adjusted.csv"
    df.to_csv(output_path, index=False)
    print(f"\n保存至: {output_path}")

    # 统计变化
    print("\n" + "=" * 60)
    print("数据变化统计")
    print("=" * 60)
    print(f"\n原始 D_conv 范围: [{df['D_conv_original'].min():.4f}, {df['D_conv_original'].max():.4f}]")
    print(f"修正后 D_conv 范围: [{df['D_conv'].min():.4f}, {df['D_conv'].max():.4f}]")

    original_std = df.groupby('Variety')['D_conv_original'].first().std()
    adjusted_std = df.groupby('Variety')['D_conv'].first().std()
    print(f"\n品种间标准差: {original_std:.4f} -> {adjusted_std:.4f}")

    return df, extreme_varieties


if __name__ == "__main__":
    df, extreme_varieties = create_full_adjusted_data()
