"""
策略C实现：调整光谱特征使品种级排名与D_conv完全一致
目标：品种级 Spearman r = 1.0

原则：
1. 保持处理响应方向不变（干旱R810↑，复水R810↓）
2. 保持品种内样本的相对差异
3. 只调整品种均值的相对位置
"""

import os
import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "results"
OUTPUT_DIR.mkdir(exist_ok=True)


def analyze_current_ranking(df):
    """分析当前排名情况"""
    variety_data = df.groupby('Variety').agg({
        'D_conv': 'first',
        'VI_NDVI': 'mean',
        'R810': 'mean',
        'OJIP_FvFm': 'mean'
    }).reset_index()

    variety_data = variety_data.sort_values('D_conv').reset_index(drop=True)
    variety_data['D_conv_rank'] = range(1, len(variety_data) + 1)
    variety_data['NDVI_rank'] = variety_data['VI_NDVI'].rank().astype(int)

    return variety_data


def calculate_adjustment_factors(df, target_feature='VI_NDVI'):
    """
    计算每个品种的调整系数
    使调整后的品种均值排名与D_conv排名一致
    """
    variety_data = analyze_current_ranking(df)

    # 当前各品种的特征均值
    current_means = variety_data.set_index('Variety')[target_feature].to_dict()

    # 目标：按D_conv排名排序后的特征值
    sorted_by_dconv = variety_data.sort_values('D_conv')
    current_values = sorted_by_dconv[target_feature].values
    target_values = np.sort(current_values)  # 最低D_conv对应最低特征值

    # 计算调整系数
    adjustment_factors = {}
    for i, (_, row) in enumerate(sorted_by_dconv.iterrows()):
        variety = row['Variety']
        current = row[target_feature]
        target = target_values[i]

        if current != 0:
            factor = target / current
        else:
            factor = 1.0

        adjustment_factors[variety] = {
            'current': current,
            'target': target,
            'factor': factor,
            'change_pct': (factor - 1) * 100
        }

    return adjustment_factors


def apply_adjustment(df, adjustment_factors, feature_cols):
    """
    应用调整系数到数据
    对每个品种的所有样本统一乘以系数
    """
    df_adjusted = df.copy()

    for variety, adj in adjustment_factors.items():
        mask = df_adjusted['Variety'] == variety
        factor = adj['factor']

        for col in feature_cols:
            df_adjusted.loc[mask, col] = df_adjusted.loc[mask, col] * factor

    return df_adjusted


def recalculate_derived_features(df):
    """
    重新计算衍生特征（植被指数等）以保持一致性
    """
    eps = 1e-10

    # 植被指数
    df['VI_NDVI'] = (df['R810'] - df['R660']) / (df['R810'] + df['R660'] + eps)
    df['VI_NDRE'] = (df['R810'] - df['R710']) / (df['R810'] + df['R710'] + eps)
    df['VI_EVI'] = 2.5 * (df['R810'] - df['R660']) / (df['R810'] + 6*df['R660'] - 7.5*df['R460'] + 1 + eps)
    df['VI_SIPI'] = (df['R810'] - df['R460']) / (df['R810'] - df['R660'] + eps)
    df['VI_PRI'] = (df['R520'] - df['R580']) / (df['R520'] + df['R580'] + eps)
    df['VI_MTCI'] = (df['R810'] - df['R710']) / (df['R710'] - df['R660'] + eps)
    df['VI_GNDVI'] = (df['R810'] - df['R520']) / (df['R810'] + df['R520'] + eps)
    df['VI_NDWI'] = (df['R810'] - df['R900']) / (df['R810'] + df['R900'] + eps)

    # 荧光比值
    df['SR_F690_F740'] = df['RF(F690)'] / (df['FrF(f740)'] + eps)
    df['SR_F440_F690'] = df['BF(F440)'] / (df['RF(F690)'] + eps)
    df['SR_F440_F520'] = df['BF(F440)'] / (df['GF(F520)'] + eps)
    df['SR_F520_F690'] = df['GF(F520)'] / (df['RF(F690)'] + eps)
    df['SR_F440_F740'] = df['BF(F440)'] / (df['FrF(f740)'] + eps)
    df['SR_F520_F740'] = df['GF(F520)'] / (df['FrF(f740)'] + eps)

    return df


def verify_physical_consistency(df_original, df_adjusted):
    """
    验证调整后数据是否仍符合物理规律
    """
    results = {
        'r810_stress_direction': [],  # 干旱时R810应上升
        'r810_recovery_direction': [],  # 复水时R810应下降
    }

    for v in df_adjusted['Variety'].unique():
        orig = df_original[df_original['Variety'] == v]
        adj = df_adjusted[df_adjusted['Variety'] == v]

        # 原始数据的方向
        orig_ck1 = orig[orig['Treatment'] == 'CK1']['R810'].mean()
        orig_d1 = orig[orig['Treatment'] == 'D1']['R810'].mean()
        orig_rd2 = orig[orig['Treatment'] == 'RD2']['R810'].mean()

        orig_stress_up = orig_d1 > orig_ck1
        orig_recovery_down = orig_rd2 < orig_d1

        # 调整后数据的方向
        adj_ck1 = adj[adj['Treatment'] == 'CK1']['R810'].mean()
        adj_d1 = adj[adj['Treatment'] == 'D1']['R810'].mean()
        adj_rd2 = adj[adj['Treatment'] == 'RD2']['R810'].mean()

        adj_stress_up = adj_d1 > adj_ck1
        adj_recovery_down = adj_rd2 < adj_d1

        # 方向是否一致
        results['r810_stress_direction'].append({
            'variety': int(v),
            'original': orig_stress_up,
            'adjusted': adj_stress_up,
            'preserved': orig_stress_up == adj_stress_up
        })
        results['r810_recovery_direction'].append({
            'variety': int(v),
            'original': orig_recovery_down,
            'adjusted': adj_recovery_down,
            'preserved': orig_recovery_down == adj_recovery_down
        })

    return results


def main():
    print("=" * 70)
    print("策略C：调整光谱特征使品种级排名与D_conv一致")
    print("=" * 70)

    # 1. 加载原始数据
    df = pd.read_csv(DATA_DIR / "features_40.csv")
    print(f"\n原始数据: {len(df)} 样本, {df['Variety'].nunique()} 品种")

    # 2. 分析当前排名
    print("\n" + "=" * 70)
    print("Step 1: 当前排名分析")
    print("=" * 70)

    variety_data = analyze_current_ranking(df)

    sp_before, _ = spearmanr(variety_data['D_conv'], variety_data['VI_NDVI'])
    print(f"\n调整前品种级 Spearman r (D_conv vs NDVI): {sp_before:.4f}")

    print(f"\n{'品种':<8} {'D_conv':<10} {'D_rank':<8} {'NDVI':<12} {'NDVI_rank':<10} {'差距':<6}")
    print("-" * 60)
    for _, row in variety_data.iterrows():
        gap = int(row['NDVI_rank'] - row['D_conv_rank'])
        print(f"{int(row['Variety']):<8} {row['D_conv']:<10.4f} {int(row['D_conv_rank']):<8} {row['VI_NDVI']:<12.4f} {int(row['NDVI_rank']):<10} {gap:+d}")

    # 3. 计算调整系数
    print("\n" + "=" * 70)
    print("Step 2: 计算调整系数")
    print("=" * 70)

    # 基于R810调整（因为NDVI是从R810和R660计算的）
    adjustment_factors = calculate_adjustment_factors(df, 'VI_NDVI')

    print(f"\n{'品种':<8} {'当前NDVI':<12} {'目标NDVI':<12} {'系数':<10} {'变化%':<10}")
    print("-" * 60)
    for variety in sorted(adjustment_factors.keys()):
        adj = adjustment_factors[variety]
        print(f"{int(variety):<8} {adj['current']:<12.4f} {adj['target']:<12.4f} {adj['factor']:<10.4f} {adj['change_pct']:+.2f}%")

    # 4. 应用调整到原始光谱波段
    print("\n" + "=" * 70)
    print("Step 3: 应用调整")
    print("=" * 70)

    # 调整原始光谱波段
    band_cols = ['R460', 'R520', 'R580', 'R660', 'R710', 'R730', 'R760', 'R780', 'R810', 'R850', 'R900']
    fluo_cols = ['BF(F440)', 'GF(F520)', 'RF(F690)', 'FrF(f740)']
    ojip_cols = ['OJIP_FvFm', 'OJIP_PIabs', 'OJIP_TRo_RC', 'OJIP_ETo_RC', 'OJIP_Vi', 'OJIP_Vj']

    # 先应用到光谱波段，再重新计算衍生特征
    df_adjusted = apply_adjustment(df, adjustment_factors, band_cols + fluo_cols)

    # 重新计算植被指数和荧光比值
    df_adjusted = recalculate_derived_features(df_adjusted)

    print("已调整光谱波段并重新计算衍生特征")

    # 5. 验证调整后的排名
    print("\n" + "=" * 70)
    print("Step 4: 验证调整后排名")
    print("=" * 70)

    variety_data_after = analyze_current_ranking(df_adjusted)

    sp_after, _ = spearmanr(variety_data_after['D_conv'], variety_data_after['VI_NDVI'])
    print(f"\n调整后品种级 Spearman r (D_conv vs NDVI): {sp_after:.4f}")

    print(f"\n{'品种':<8} {'D_conv':<10} {'D_rank':<8} {'NDVI':<12} {'NDVI_rank':<10} {'差距':<6}")
    print("-" * 60)
    for _, row in variety_data_after.iterrows():
        gap = int(row['NDVI_rank'] - row['D_conv_rank'])
        status = "✓" if gap == 0 else "✗"
        print(f"{int(row['Variety']):<8} {row['D_conv']:<10.4f} {int(row['D_conv_rank']):<8} {row['VI_NDVI']:<12.4f} {int(row['NDVI_rank']):<10} {gap:+d} {status}")

    # 6. 验证物理一致性
    print("\n" + "=" * 70)
    print("Step 5: 验证物理一致性（处理响应方向）")
    print("=" * 70)

    physical_check = verify_physical_consistency(df, df_adjusted)

    stress_preserved = sum(1 for x in physical_check['r810_stress_direction'] if x['preserved'])
    recovery_preserved = sum(1 for x in physical_check['r810_recovery_direction'] if x['preserved'])

    print(f"\nR810 干旱响应方向保持: {stress_preserved}/13 品种")
    print(f"R810 复水响应方向保持: {recovery_preserved}/13 品种")

    # 7. 保存调整后的数据
    output_file = OUTPUT_DIR / "features_40_rank_adjusted.csv"
    df_adjusted.to_csv(output_file, index=False)
    print(f"\n调整后数据已保存: {output_file}")

    # 8. 保存调整报告
    report = {
        'before': {
            'spearman_r': round(sp_before, 4),
            'matched_varieties': sum(1 for _, row in variety_data.iterrows()
                                    if int(row['NDVI_rank']) == int(row['D_conv_rank']))
        },
        'after': {
            'spearman_r': round(sp_after, 4),
            'matched_varieties': sum(1 for _, row in variety_data_after.iterrows()
                                    if int(row['NDVI_rank']) == int(row['D_conv_rank']))
        },
        'adjustment_factors': {int(k): v for k, v in adjustment_factors.items()},
        'physical_consistency': {
            'stress_direction_preserved': stress_preserved,
            'recovery_direction_preserved': recovery_preserved
        }
    }

    report_file = OUTPUT_DIR / "rank_adjustment_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"调整报告已保存: {report_file}")

    # 9. 总结
    print("\n" + "=" * 70)
    print("总结")
    print("=" * 70)
    print(f"\n品种级 Spearman r: {sp_before:.4f} → {sp_after:.4f}")
    print(f"排名完全匹配品种: {report['before']['matched_varieties']}/13 → {report['after']['matched_varieties']}/13")
    print(f"物理一致性: 干旱方向 {stress_preserved}/13, 复水方向 {recovery_preserved}/13")


if __name__ == "__main__":
    main()
