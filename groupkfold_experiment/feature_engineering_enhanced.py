"""
增强特征工程脚本
添加光谱导数特征和异常检测标记

输出:
- data/features_enhanced.csv (增强特征)
- data/feature_sets_enhanced.json (包含FS4_enhanced)
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from scipy.spatial.distance import mahalanobis

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


def compute_spectral_derivatives(df, band_cols):
    """
    计算光谱一阶导数特征

    原理：消除基线漂移，增强光谱形状特征
    使用相邻波段差值作为近似导数
    """
    print("\n计算光谱导数...")

    derivatives = pd.DataFrame(index=df.index)

    # 按波长排序的波段
    sorted_bands = sorted(band_cols, key=lambda x: int(x.replace('R', '')))

    # 计算一阶导数 (差分)
    for i in range(len(sorted_bands) - 1):
        b1, b2 = sorted_bands[i], sorted_bands[i + 1]
        w1, w2 = int(b1.replace('R', '')), int(b2.replace('R', ''))
        delta_w = w2 - w1

        col_name = f"dR{w1}_{w2}"
        derivatives[col_name] = (df[b2] - df[b1]) / delta_w

    print(f"  生成 {len(derivatives.columns)} 个一阶导数特征")
    return derivatives


def compute_mahalanobis_distance(df, feature_cols):
    """
    计算每个样本的马氏距离（用于检测光谱离群点）

    马氏距离 > 3 视为潜在离群点
    """
    print("\n计算马氏距离...")

    X = df[feature_cols].values

    # 计算均值和协方差
    mean = np.mean(X, axis=0)

    # 使用正则化协方差避免奇异矩阵
    cov = np.cov(X.T)
    cov_reg = cov + np.eye(cov.shape[0]) * 1e-6

    try:
        cov_inv = np.linalg.inv(cov_reg)
    except np.linalg.LinAlgError:
        print("  [WARN] 协方差矩阵不可逆，使用伪逆")
        cov_inv = np.linalg.pinv(cov_reg)

    # 计算每个样本的马氏距离
    distances = []
    for i in range(len(X)):
        d = mahalanobis(X[i], mean, cov_inv)
        distances.append(d)

    distances = np.array(distances)

    # 统计
    print(f"  马氏距离: mean={distances.mean():.2f}, max={distances.max():.2f}")
    n_outliers = np.sum(distances > 3)
    print(f"  离群样本 (d > 3): {n_outliers} 个 ({100*n_outliers/len(distances):.1f}%)")

    return distances


def compute_sample_weights(y, q_low=0.1, q_high=0.9, boost=2.0):
    """
    计算样本权重，对边缘标签样本增加权重

    参数:
    - y: 目标变量
    - q_low: 低分位数阈值
    - q_high: 高分位数阈值
    - boost: 边缘样本权重倍数

    返回:
    - weights: 样本权重数组
    """
    low_threshold = np.quantile(y, q_low)
    high_threshold = np.quantile(y, q_high)

    weights = np.ones(len(y))
    weights[(y <= low_threshold) | (y >= high_threshold)] = boost

    return weights


def main():
    print("=" * 60)
    print("增强特征工程")
    print("=" * 60)

    # 1. 加载原始数据
    df = pd.read_csv(DATA_DIR / "features_40.csv")
    print(f"加载数据: {len(df)} 样本")

    # 2. 定义波段列
    band_cols = ['R460', 'R520', 'R580', 'R660', 'R710', 'R730',
                 'R760', 'R780', 'R810', 'R850', 'R900']
    static_cols = ['BF(F440)', 'GF(F520)', 'RF(F690)', 'FrF(f740)']

    # 3. 计算光谱导数
    derivatives = compute_spectral_derivatives(df, band_cols)

    # 4. 计算马氏距离（基于光谱波段）
    mahal_dist = compute_mahalanobis_distance(df, band_cols)
    df['mahal_distance'] = mahal_dist
    df['outlier_flag'] = (mahal_dist > 3).astype(int)

    # 5. 计算样本权重
    weights = compute_sample_weights(df['D_conv'].values)
    df['sample_weight'] = weights

    # 6. 合并特征
    df_enhanced = pd.concat([df, derivatives], axis=1)

    print(f"\n增强后特征矩阵: {df_enhanced.shape}")

    # 7. 创建增强特征集定义
    with open(DATA_DIR / "feature_sets.json", 'r', encoding='utf-8') as f:
        feature_sets = json.load(f)

    # FS4_enhanced: 原FS4 + 导数特征
    fs4_enhanced_features = feature_sets['FS4']['features'].copy()
    fs4_enhanced_features.extend(list(derivatives.columns))

    feature_sets['FS4_enhanced'] = {
        "description": "三源融合 + 光谱导数",
        "features": fs4_enhanced_features,
        "n_features": len(fs4_enhanced_features)
    }

    # FS4_derivatives_only: 只用导数替代原始光谱
    fs4_deriv_only = [c for c in feature_sets['FS4']['features'] if not c.startswith('R')]
    fs4_deriv_only.extend(list(derivatives.columns))

    feature_sets['FS4_derivatives'] = {
        "description": "三源融合（光谱用导数替代）",
        "features": fs4_deriv_only,
        "n_features": len(fs4_deriv_only)
    }

    print(f"\n特征集定义:")
    for name, info in feature_sets.items():
        print(f"  {name}: {info['n_features']} 个特征")

    # 8. 保存
    output_path = DATA_DIR / "features_enhanced.csv"
    df_enhanced.to_csv(output_path, index=False)
    print(f"\n保存增强特征: {output_path}")

    fs_path = DATA_DIR / "feature_sets_enhanced.json"
    with open(fs_path, 'w', encoding='utf-8') as f:
        json.dump(feature_sets, f, indent=2, ensure_ascii=False)
    print(f"保存特征集定义: {fs_path}")

    # 9. 品种级统计
    print("\n" + "=" * 60)
    print("品种级马氏距离统计")
    print("=" * 60)

    variety_stats = df_enhanced.groupby('Variety').agg({
        'D_conv': 'first',
        'mahal_distance': 'mean',
        'outlier_flag': 'sum'
    }).sort_values('D_conv', ascending=False)

    print("\n极端品种分析:")
    print(variety_stats.head(3))
    print("...")
    print(variety_stats.tail(3))

    return df_enhanced, feature_sets


if __name__ == "__main__":
    df_enhanced, feature_sets = main()
