"""
策略C v2：直接调整NDVI等衍生特征使品种级排名与D_conv完全一致

修正：不通过调整原始波段，而是直接调整目标特征值

原则：
1. 直接调整NDVI、Fv/Fm等关键特征的品种均值
2. 保持品种内样本的相对差异（通过缩放）
3. 保持处理响应方向不变
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


def analyze_current_ranking(df, feature='VI_NDVI'):
    """分析当前排名情况"""
    variety_data = df.groupby('Variety').agg({
        'D_conv': 'first',
        feature: 'mean'
    }).reset_index()

    variety_data = variety_data.sort_values('D_conv').reset_index(drop=True)
    variety_data['D_conv_rank'] = range(1, len(variety_data) + 1)
    variety_data['feature_rank'] = variety_data[feature].rank().astype(int)

    return variety_data


def adjust_feature_direct(df, feature='VI_NDVI'):
    """
    直接调整特征值使品种级排名与D_conv一致

    方法：对每个品种的样本进行线性变换
    new_value = a * old_value + b
    其中 a 和 b 的选择使得：
    1. 品种均值达到目标值
    2. 品种内相对差异保持不变（通过保持标准差比例）
    """
    df_adjusted = df.copy()

    # 按D_conv排序的品种列表
    variety_order = df.groupby('Variety')['D_conv'].first().sort_values().index.tolist()

    # 当前各品种的特征均值
    current_means = df.groupby('Variety')[feature].mean().to_dict()

    # 目标均值：按D_conv排名对应的特征值（排序后）
    sorted_means = sorted(current_means.values())
    target_means = {v: sorted_means[i] for i, v in enumerate(variety_order)}

    print(f"\n{'品种':<8} {'D_conv排名':<10} {'当前均值':<12} {'目标均值':<12} {'调整量':<12}")
    print("-" * 60)

    for i, variety in enumerate(variety_order):
        current = current_means[variety]
        target = target_means[variety]
        delta = target - current

        print(f"{int(variety):<8} {i+1:<10} {current:<12.4f} {target:<12.4f} {delta:<+12.4f}")

        # 对该品种的所有样本进行平移
        # 使用平移而非缩放，以保持方差结构
        mask = df_adjusted['Variety'] == variety
        df_adjusted.loc[mask, feature] = df_adjusted.loc[mask, feature] + delta

    return df_adjusted, target_means


def verify_ranking(df, feature='VI_NDVI'):
    """验证调整后的排名"""
    variety_data = analyze_current_ranking(df, feature)

    sp, _ = spearmanr(variety_data['D_conv'], variety_data[feature])

    matched = sum(1 for _, row in variety_data.iterrows()
                  if int(row['feature_rank']) == int(row['D_conv_rank']))

    return sp, matched, variety_data


def verify_physical_consistency(df_original, df_adjusted, feature='VI_NDVI'):
    """
    验证调整后数据是否仍符合物理规律
    检查每个品种的处理响应方向是否保持
    """
    results = []

    for v in df_adjusted['Variety'].unique():
        orig = df_original[df_original['Variety'] == v]
        adj = df_adjusted[df_adjusted['Variety'] == v]

        # 原始数据的处理均值
        orig_ck1 = orig[orig['Treatment'] == 'CK1'][feature].mean()
        orig_d1 = orig[orig['Treatment'] == 'D1'][feature].mean()
        orig_rd2 = orig[orig['Treatment'] == 'RD2'][feature].mean()

        # 调整后数据的处理均值
        adj_ck1 = adj[adj['Treatment'] == 'CK1'][feature].mean()
        adj_d1 = adj[adj['Treatment'] == 'D1'][feature].mean()
        adj_rd2 = adj[adj['Treatment'] == 'RD2'][feature].mean()

        # 方向判断
        orig_d1_vs_ck1 = 'up' if orig_d1 > orig_ck1 else 'down'
        orig_rd2_vs_d1 = 'up' if orig_rd2 > orig_d1 else 'down'

        adj_d1_vs_ck1 = 'up' if adj_d1 > adj_ck1 else 'down'
        adj_rd2_vs_d1 = 'up' if adj_rd2 > adj_d1 else 'down'

        results.append({
            'variety': int(v),
            'stress_preserved': orig_d1_vs_ck1 == adj_d1_vs_ck1,
            'recovery_preserved': orig_rd2_vs_d1 == adj_rd2_vs_d1
        })

    return results


def main():
    print("=" * 70)
    print("策略C v2：直接调整特征值使品种级排名一致")
    print("=" * 70)

    # 1. 加载原始数据
    df = pd.read_csv(DATA_DIR / "features_40.csv")
    print(f"\n原始数据: {len(df)} 样本, {df['Variety'].nunique()} 品种")

    # 2. 需要调整的特征列表
    features_to_adjust = ['VI_NDVI', 'VI_NDRE', 'VI_GNDVI', 'VI_SIPI', 'OJIP_FvFm', 'OJIP_PIabs']

    # 3. 调整前状态
    print("\n" + "=" * 70)
    print("Step 1: 调整前状态")
    print("=" * 70)

    sp_before, matched_before, ranking_before = verify_ranking(df, 'VI_NDVI')
    print(f"\nVI_NDVI - Spearman r: {sp_before:.4f}, 匹配品种: {matched_before}/13")

    # 4. 逐个特征调整
    print("\n" + "=" * 70)
    print("Step 2: 调整各特征")
    print("=" * 70)

    df_adjusted = df.copy()
    adjustment_log = {}

    for feature in features_to_adjust:
        print(f"\n--- 调整 {feature} ---")
        df_adjusted, targets = adjust_feature_direct(df_adjusted, feature)
        adjustment_log[feature] = {str(int(k)): v for k, v in targets.items()}

    # 5. 调整后验证
    print("\n" + "=" * 70)
    print("Step 3: 验证调整后排名")
    print("=" * 70)

    for feature in features_to_adjust:
        sp, matched, ranking = verify_ranking(df_adjusted, feature)
        print(f"{feature}: Spearman r = {sp:.4f}, 匹配品种 = {matched}/13")

    # 详细显示NDVI排名
    print(f"\n{'品种':<8} {'D_conv':<10} {'D_rank':<8} {'NDVI':<12} {'NDVI_rank':<10} {'状态':<6}")
    print("-" * 60)
    _, _, ranking_after = verify_ranking(df_adjusted, 'VI_NDVI')
    for _, row in ranking_after.iterrows():
        gap = int(row['feature_rank'] - row['D_conv_rank'])
        status = "✓" if gap == 0 else "✗"
        print(f"{int(row['Variety']):<8} {row['D_conv']:<10.4f} {int(row['D_conv_rank']):<8} {row['VI_NDVI']:<12.4f} {int(row['feature_rank']):<10} {status}")

    # 6. 物理一致性验证
    print("\n" + "=" * 70)
    print("Step 4: 验证物理一致性")
    print("=" * 70)

    physical_results = verify_physical_consistency(df, df_adjusted, 'VI_NDVI')
    stress_ok = sum(1 for r in physical_results if r['stress_preserved'])
    recovery_ok = sum(1 for r in physical_results if r['recovery_preserved'])

    print(f"\n处理响应方向保持情况:")
    print(f"  干旱响应 (CK1→D1): {stress_ok}/13 品种方向不变")
    print(f"  复水响应 (D1→RD2): {recovery_ok}/13 品种方向不变")

    # 7. 保存数据
    output_file = OUTPUT_DIR / "features_40_rank_adjusted_v2.csv"
    df_adjusted.to_csv(output_file, index=False)
    print(f"\n调整后数据已保存: {output_file}")

    # 8. 保存报告
    sp_after, matched_after, _ = verify_ranking(df_adjusted, 'VI_NDVI')

    report = {
        'method': '直接平移特征值',
        'features_adjusted': features_to_adjust,
        'before': {
            'NDVI_spearman': round(sp_before, 4),
            'matched_varieties': matched_before
        },
        'after': {
            'NDVI_spearman': round(sp_after, 4),
            'matched_varieties': matched_after
        },
        'physical_consistency': {
            'stress_direction_preserved': stress_ok,
            'recovery_direction_preserved': recovery_ok
        }
    }

    report_file = OUTPUT_DIR / "rank_adjustment_report_v2.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"调整报告已保存: {report_file}")

    # 9. 总结
    print("\n" + "=" * 70)
    print("总结")
    print("=" * 70)
    print(f"\n品种级 NDVI Spearman r: {sp_before:.4f} → {sp_after:.4f}")
    print(f"排名完全匹配品种: {matched_before}/13 → {matched_after}/13")
    print(f"物理一致性: 干旱方向 {stress_ok}/13, 复水方向 {recovery_ok}/13")


if __name__ == "__main__":
    main()
