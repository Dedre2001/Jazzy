"""
极端品种分析
分析极端标签与极端马氏距离的同步关系
估算极端品种的"合理"标签值
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


def load_data():
    df = pd.read_csv(DATA_DIR / "features_enhanced.csv")
    return df


def analyze_extreme_varieties(df):
    """分析极端品种"""
    print("=" * 60)
    print("极端品种分析")
    print("=" * 60)

    # 品种级统计
    variety_stats = df.groupby('Variety').agg({
        'D_conv': 'first',
        'mahal_distance': 'mean',
        'outlier_flag': 'sum'
    }).reset_index()

    variety_stats = variety_stats.sort_values('D_conv')

    # 计算分位数
    d_conv_q10 = variety_stats['D_conv'].quantile(0.1)
    d_conv_q90 = variety_stats['D_conv'].quantile(0.9)
    mahal_median = variety_stats['mahal_distance'].median()

    print(f"\n品种数: {len(variety_stats)}")
    print(f"D_conv 范围: [{variety_stats['D_conv'].min():.4f}, {variety_stats['D_conv'].max():.4f}]")
    print(f"D_conv Q10: {d_conv_q10:.4f}, Q90: {d_conv_q90:.4f}")
    print(f"马氏距离中位数: {mahal_median:.2f}")

    # 定义极端品种: D_conv在边缘 OR 马氏距离高
    variety_stats['is_label_extreme'] = (variety_stats['D_conv'] < d_conv_q10) | (variety_stats['D_conv'] > d_conv_q90)
    variety_stats['is_spectral_extreme'] = variety_stats['mahal_distance'] > mahal_median

    # 同时极端的品种
    variety_stats['is_both_extreme'] = variety_stats['is_label_extreme'] & variety_stats['is_spectral_extreme']

    print("\n" + "=" * 60)
    print("全部品种概览")
    print("=" * 60)
    print(f"\n{'品种':<10} {'D_conv':>10} {'马氏距离':>10} {'标签极端':>10} {'光谱极端':>10} {'双极端':>8}")
    print("-" * 60)

    for _, row in variety_stats.iterrows():
        label_ext = "是" if row['is_label_extreme'] else ""
        spec_ext = "是" if row['is_spectral_extreme'] else ""
        both_ext = "★" if row['is_both_extreme'] else ""
        print(f"{int(row['Variety']):<10} {row['D_conv']:>10.4f} {row['mahal_distance']:>10.2f} "
              f"{label_ext:>10} {spec_ext:>10} {both_ext:>8}")

    # 统计
    n_label_extreme = variety_stats['is_label_extreme'].sum()
    n_spectral_extreme = variety_stats['is_spectral_extreme'].sum()
    n_both_extreme = variety_stats['is_both_extreme'].sum()

    print("-" * 60)
    print(f"\n统计:")
    print(f"  标签极端品种: {n_label_extreme}")
    print(f"  光谱极端品种: {n_spectral_extreme}")
    print(f"  双极端品种: {n_both_extreme}")

    # 相关性分析
    r, p = pearsonr(variety_stats['D_conv'], variety_stats['mahal_distance'])
    print(f"\nD_conv vs 马氏距离相关性: r={r:.4f}, p={p:.4f}")

    return variety_stats


def estimate_expected_labels(df, variety_stats):
    """
    基于光谱特征估算每个品种的"预期"标签
    思路: 用非极端品种训练模型，预测极端品种的标签
    """
    print("\n" + "=" * 60)
    print("估算极端品种的'合理'标签")
    print("=" * 60)

    band_cols = ['R460', 'R520', 'R580', 'R660', 'R710', 'R730',
                 'R760', 'R780', 'R810', 'R850', 'R900']

    # 品种级光谱均值
    variety_spectra = df.groupby('Variety')[band_cols].mean()
    variety_spectra['D_conv'] = df.groupby('Variety')['D_conv'].first()

    # 分离极端和非极端品种
    extreme_varieties = variety_stats[variety_stats['is_both_extreme']]['Variety'].values
    normal_varieties = variety_stats[~variety_stats['is_both_extreme']]['Variety'].values

    print(f"\n用于训练的正常品种: {len(normal_varieties)}")
    print(f"需要估算的极端品种: {len(extreme_varieties)}")

    # 用正常品种训练简单模型
    X_train = variety_spectra.loc[normal_varieties, band_cols].values
    y_train = variety_spectra.loc[normal_varieties, 'D_conv'].values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = Ridge(alpha=1.0)
    model.fit(X_train_scaled, y_train)

    # 预测所有品种（包括极端品种）
    X_all = variety_spectra[band_cols].values
    X_all_scaled = scaler.transform(X_all)
    predicted_labels = model.predict(X_all_scaled)

    variety_spectra['predicted_D_conv'] = predicted_labels
    variety_spectra['label_gap'] = variety_spectra['D_conv'] - variety_spectra['predicted_D_conv']

    # 合并马氏距离
    mahal_map = variety_stats.set_index('Variety')['mahal_distance']
    variety_spectra['mahal_distance'] = variety_spectra.index.map(mahal_map)

    print("\n" + "=" * 60)
    print("极端品种分析")
    print("=" * 60)
    print(f"\n{'品种':<10} {'实际D_conv':>12} {'预期D_conv':>12} {'差距':>10} {'马氏距离':>10}")
    print("-" * 60)

    for variety in extreme_varieties:
        row = variety_spectra.loc[variety]
        print(f"{int(variety):<10} {row['D_conv']:>12.4f} {row['predicted_D_conv']:>12.4f} "
              f"{row['label_gap']:>10.4f} {row['mahal_distance']:>10.2f}")

    print("\n" + "=" * 60)
    print("建议的'修正'标签值")
    print("=" * 60)

    print("\n如果要让极端品种的光谱-标签关系与其他品种一致，标签应调整为:")
    print(f"\n{'品种':<10} {'当前标签':>12} {'建议标签':>12} {'调整幅度':>12}")
    print("-" * 50)

    adjustments = []
    for variety in extreme_varieties:
        row = variety_spectra.loc[variety]
        current = row['D_conv']
        suggested = row['predicted_D_conv']
        adjustment = suggested - current
        adjustments.append({
            'Variety': int(variety),
            'current': current,
            'suggested': suggested,
            'adjustment': adjustment
        })
        print(f"{int(variety):<10} {current:>12.4f} {suggested:>12.4f} {adjustment:>+12.4f}")

    return variety_spectra, adjustments


def analyze_what_if(df, adjustments):
    """分析如果修正标签会怎样"""
    print("\n" + "=" * 60)
    print("模拟实验预估")
    print("=" * 60)

    print("\n当前问题:")
    print("  - 极端品种的光谱也是极端的（高马氏距离）")
    print("  - 模型在GroupKFold下无法学到：'极端光谱 -> 极端标签'")
    print("  - 只能从正常品种学到：'正常光谱 -> 中等标签'")
    print("  - 预测极端品种时，模型用正常模式，导致向均值回归")

    print("\n如果修正标签后:")
    for adj in adjustments:
        direction = "降低" if adj['adjustment'] < 0 else "提高"
        print(f"  - 品种 {adj['Variety']}: {direction} {abs(adj['adjustment']):.4f}")
        print(f"    {adj['current']:.4f} -> {adj['suggested']:.4f}")

    print("\n预期效果:")
    print("  - 极端品种的光谱-标签关系将与其他品种一致")
    print("  - 模型从正常品种学到的模式可以适用于极端品种")
    print("  - GroupKFold R² 预计会提升")

    print("\n注意事项:")
    print("  - 这是模拟实验，用于验证'光谱-标签不一致'是否是核心问题")
    print("  - 真实场景中不能随意修改标签")
    print("  - 如果模拟有效，说明需要更多类似极端品种的样本")


def main():
    df = load_data()

    # 1. 分析极端品种
    variety_stats = analyze_extreme_varieties(df)

    # 2. 估算预期标签
    variety_spectra, adjustments = estimate_expected_labels(df, variety_stats)

    # 3. 分析修正影响
    analyze_what_if(df, adjustments)

    # 4. 输出用于模拟实验的数据
    print("\n" + "=" * 60)
    print("模拟实验数据")
    print("=" * 60)

    print("\n极端品种及建议标签值 (可用于模拟):")
    print("-" * 40)
    for adj in adjustments:
        print(f"品种 {adj['Variety']}: {adj['current']:.4f} -> {adj['suggested']:.4f}")

    return variety_stats, adjustments


if __name__ == "__main__":
    variety_stats, adjustments = main()
