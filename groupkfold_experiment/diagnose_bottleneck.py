"""
数据瓶颈诊断分析
深入分析：数据质量 vs 极端标签 vs 样本量问题
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.spatial.distance import mahalanobis
from scipy.stats import pearsonr, spearmanr
import json

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

def load_data():
    df = pd.read_csv(DATA_DIR / "features_enhanced.csv")
    return df

def analyze_label_distribution(df):
    """分析标签分布"""
    print("=" * 60)
    print("1. 标签分布分析")
    print("=" * 60)

    # 品种级D_conv
    variety_df = df.groupby('Variety').agg({
        'D_conv': 'first',
        'Sample_ID': 'count'
    }).rename(columns={'Sample_ID': 'n_samples'}).reset_index()

    d_conv = variety_df['D_conv'].values

    print(f"\n品种数: {len(variety_df)}")
    print(f"D_conv 范围: [{d_conv.min():.4f}, {d_conv.max():.4f}]")
    print(f"D_conv 均值: {d_conv.mean():.4f}")
    print(f"D_conv 标准差: {d_conv.std():.4f}")

    # 分位数
    print(f"\n分位数分布:")
    for q in [0.1, 0.25, 0.5, 0.75, 0.9]:
        print(f"  Q{int(q*100):02d}: {np.quantile(d_conv, q):.4f}")

    # 极端品种
    print(f"\n极端品种 (前3高/低):")
    variety_sorted = variety_df.sort_values('D_conv')
    print("  最低 D_conv:")
    for _, row in variety_sorted.head(3).iterrows():
        print(f"    品种 {row['Variety']}: D_conv={row['D_conv']:.4f}, 样本数={row['n_samples']}")

    print("  最高 D_conv:")
    for _, row in variety_sorted.tail(3).iterrows():
        print(f"    品种 {row['Variety']}: D_conv={row['D_conv']:.4f}, 样本数={row['n_samples']}")

    return variety_df

def analyze_sample_distribution(df):
    """分析样本分布"""
    print("\n" + "=" * 60)
    print("2. 样本分布分析")
    print("=" * 60)

    # 每品种样本数
    samples_per_variety = df.groupby('Variety').size()

    print(f"\n每品种样本数:")
    print(f"  均值: {samples_per_variety.mean():.1f}")
    print(f"  最小: {samples_per_variety.min()}")
    print(f"  最大: {samples_per_variety.max()}")
    print(f"  标准差: {samples_per_variety.std():.2f}")

    # Treatment分布
    print(f"\nTreatment分布:")
    treatment_counts = df.groupby(['Variety', 'Treatment']).size().unstack(fill_value=0)
    print(f"  每品种每Treatment样本数: {treatment_counts.iloc[0].to_dict()}")

    return samples_per_variety

def analyze_spectral_quality(df):
    """分析光谱质量"""
    print("\n" + "=" * 60)
    print("3. 光谱质量分析")
    print("=" * 60)

    band_cols = ['R460', 'R520', 'R580', 'R660', 'R710', 'R730',
                 'R760', 'R780', 'R810', 'R850', 'R900']

    # 计算品种级光谱统计
    variety_spectral = df.groupby('Variety')[band_cols].agg(['mean', 'std'])

    # 品种内变异系数
    cv_per_variety = {}
    for variety in df['Variety'].unique():
        variety_data = df[df['Variety'] == variety][band_cols]
        cv = variety_data.std() / (variety_data.mean() + 1e-9)
        cv_per_variety[variety] = cv.mean()

    cv_df = pd.DataFrame.from_dict(cv_per_variety, orient='index', columns=['CV'])
    cv_df = cv_df.sort_values('CV', ascending=False)

    print(f"\n品种内光谱变异系数 (CV):")
    print(f"  均值: {cv_df['CV'].mean():.4f}")
    print(f"  最大: {cv_df['CV'].max():.4f} (品种 {cv_df.index[0]})")
    print(f"  最小: {cv_df['CV'].min():.4f}")

    print(f"\n高变异品种 (CV > mean + 1σ):")
    threshold = cv_df['CV'].mean() + cv_df['CV'].std()
    high_cv = cv_df[cv_df['CV'] > threshold]
    for variety, row in high_cv.iterrows():
        d_conv = df[df['Variety'] == variety]['D_conv'].iloc[0]
        print(f"    品种 {variety}: CV={row['CV']:.4f}, D_conv={d_conv:.4f}")

    return cv_df

def analyze_mahalanobis_vs_error(df):
    """分析马氏距离与预测误差的关系"""
    print("\n" + "=" * 60)
    print("4. 光谱离群度 vs 预测难度")
    print("=" * 60)

    # 加载预测结果
    try:
        with open(BASE_DIR / "results" / "ab_test_report.json") as f:
            report = json.load(f)
        baseline = report['experiments'][0]
        variety_errors = {v['Variety']: v['error'] for v in baseline['worst_varieties']}
    except:
        variety_errors = {}

    # 品种级马氏距离
    variety_mahal = df.groupby('Variety').agg({
        'mahal_distance': 'mean',
        'D_conv': 'first',
        'outlier_flag': 'sum'
    }).reset_index()

    print(f"\n品种级马氏距离统计:")
    print(f"  均值: {variety_mahal['mahal_distance'].mean():.2f}")
    print(f"  最大: {variety_mahal['mahal_distance'].max():.2f}")

    # 高马氏距离品种
    print(f"\n高马氏距离品种 (前5):")
    top_mahal = variety_mahal.nlargest(5, 'mahal_distance')
    for _, row in top_mahal.iterrows():
        print(f"    品种 {row['Variety']}: 马氏距离={row['mahal_distance']:.2f}, "
              f"D_conv={row['D_conv']:.4f}, 离群样本数={int(row['outlier_flag'])}")

    return variety_mahal

def analyze_feature_variety_separation(df):
    """分析特征对品种的区分度"""
    print("\n" + "=" * 60)
    print("5. 特征-标签相关性分析")
    print("=" * 60)

    feature_cols = ['R460', 'R520', 'R580', 'R660', 'R710', 'R730', 'R760', 'R780', 'R810', 'R850', 'R900',
                    'VI_NDVI', 'VI_NDRE', 'VI_EVI', 'VI_SIPI', 'VI_PRI', 'VI_MTCI', 'VI_GNDVI', 'VI_NDWI',
                    'BF(F440)', 'GF(F520)', 'RF(F690)', 'FrF(f740)',
                    'OJIP_FvFm', 'OJIP_PIabs']

    # 品种级聚合
    variety_df = df.groupby('Variety').agg({
        'D_conv': 'first',
        **{col: 'mean' for col in feature_cols if col in df.columns}
    }).reset_index()

    # 计算相关性
    correlations = {}
    for col in feature_cols:
        if col in variety_df.columns:
            r, p = pearsonr(variety_df[col], variety_df['D_conv'])
            correlations[col] = {'r': r, 'p': p, 'abs_r': abs(r)}

    corr_df = pd.DataFrame(correlations).T.sort_values('abs_r', ascending=False)

    print(f"\n与D_conv相关性最高的特征:")
    for feat, row in corr_df.head(10).iterrows():
        sig = "***" if row['p'] < 0.001 else "**" if row['p'] < 0.01 else "*" if row['p'] < 0.05 else ""
        print(f"    {feat:20s}: r={row['r']:+.4f} {sig}")

    print(f"\n与D_conv相关性最低的特征:")
    for feat, row in corr_df.tail(5).iterrows():
        print(f"    {feat:20s}: r={row['r']:+.4f}")

    return corr_df

def analyze_variety_overlap(df):
    """分析品种间特征重叠度"""
    print("\n" + "=" * 60)
    print("6. 品种间特征重叠分析")
    print("=" * 60)

    band_cols = ['R460', 'R520', 'R580', 'R660', 'R710', 'R730',
                 'R760', 'R780', 'R810', 'R850', 'R900']

    # 计算品种质心
    variety_centroids = df.groupby('Variety')[band_cols].mean()
    d_conv_map = df.groupby('Variety')['D_conv'].first()

    # 计算品种间距离
    from scipy.spatial.distance import pdist, squareform
    distances = squareform(pdist(variety_centroids.values))

    varieties = variety_centroids.index.tolist()

    # 找最近邻
    print(f"\n各品种最近邻分析:")
    nearest_neighbor_info = []
    for i, v1 in enumerate(varieties):
        dist_row = distances[i].copy()
        dist_row[i] = np.inf  # 排除自己
        nearest_idx = np.argmin(dist_row)
        nearest_v = varieties[nearest_idx]
        nearest_dist = dist_row[nearest_idx]

        d_conv_diff = abs(d_conv_map[v1] - d_conv_map[nearest_v])
        nearest_neighbor_info.append({
            'Variety': v1,
            'D_conv': d_conv_map[v1],
            'Nearest': nearest_v,
            'Nearest_D_conv': d_conv_map[nearest_v],
            'Distance': nearest_dist,
            'D_conv_diff': d_conv_diff
        })

    nn_df = pd.DataFrame(nearest_neighbor_info)

    # 特征相近但标签差异大的情况
    print(f"\n特征相近但D_conv差异大的品种对 (潜在问题):")
    nn_df['ratio'] = nn_df['D_conv_diff'] / (nn_df['Distance'] + 1e-6)
    problem_pairs = nn_df.nlargest(5, 'ratio')
    for _, row in problem_pairs.iterrows():
        print(f"    品种 {row['Variety']} ↔ {row['Nearest']}: "
              f"特征距离={row['Distance']:.4f}, D_conv差={row['D_conv_diff']:.4f}")

    return nn_df

def diagnose_difficult_varieties(df):
    """诊断难预测品种"""
    print("\n" + "=" * 60)
    print("7. 难预测品种深度诊断")
    print("=" * 60)

    difficult_varieties = [1252, 1235, 1099]

    band_cols = ['R460', 'R520', 'R580', 'R660', 'R710', 'R730',
                 'R760', 'R780', 'R810', 'R850', 'R900']

    # 全局统计
    global_mean = df[band_cols].mean()
    global_std = df[band_cols].std()

    for variety in difficult_varieties:
        variety_data = df[df['Variety'] == variety]
        d_conv = variety_data['D_conv'].iloc[0]

        print(f"\n品种 {variety} (D_conv={d_conv:.4f}):")

        # 样本数
        print(f"  样本数: {len(variety_data)}")

        # 品种内变异
        variety_mean = variety_data[band_cols].mean()
        variety_std = variety_data[band_cols].std()
        cv = (variety_std / (variety_mean + 1e-9)).mean()
        print(f"  品种内变异系数: {cv:.4f}")

        # 与全局的偏离
        z_score = ((variety_mean - global_mean) / (global_std + 1e-9)).abs().mean()
        print(f"  与全局均值的平均z-score: {z_score:.2f}")

        # 马氏距离
        mahal = variety_data['mahal_distance'].mean()
        print(f"  平均马氏距离: {mahal:.2f}")

        # D_conv在数据集中的位置
        all_d_conv = df.groupby('Variety')['D_conv'].first()
        percentile = (all_d_conv < d_conv).sum() / len(all_d_conv) * 100
        print(f"  D_conv分位数: {percentile:.1f}%")

        # 判断问题类型
        if percentile > 90 or percentile < 10:
            print(f"  [问题] 极端标签 - D_conv处于分布边缘")
        if mahal > 3:
            print(f"  [问题] 光谱离群 - 马氏距离过高")
        if cv > 0.3:
            print(f"  [问题] 高变异 - 品种内样本不一致")

def summarize_bottleneck():
    """总结瓶颈"""
    print("\n" + "=" * 60)
    print("8. 瓶颈诊断总结")
    print("=" * 60)

    print("""
┌─────────────────────────────────────────────────────────────┐
│                      数据瓶颈诊断                           │
├─────────────────────────────────────────────────────────────┤
│  问题类型          │  严重程度  │  影响                     │
├─────────────────────────────────────────────────────────────┤
│  1. 极端标签分布   │  ★★★★★   │  模型向均值回归，边缘预测差 │
│  2. 小样本量       │  ★★★★☆   │  每品种仅9样本，难学稳定模式 │
│  3. 光谱离群样本   │  ★★★☆☆   │  部分品种特征异常          │
│  4. 品种间重叠     │  ★★☆☆☆   │  特征相似但标签不同        │
└─────────────────────────────────────────────────────────────┘

核心瓶颈: 极端标签 + 小样本量 的组合效应

原因分析:
- 13个品种，每品种仅9个样本（3 Treatment × 3 重复）
- D_conv 范围 [0.17, 0.57]，极端值品种样本量与中间值相同
- 模型在GroupKFold下看不到目标品种的任何样本
- 对边缘标签的品种，只能依赖全局模式预测，必然向均值回归

为什么TabPFN比XGBoost好:
- TabPFN是元学习模型，更擅长小样本泛化
- XGBoost需要更多样本才能学到稳健模式

可能的改进方向:
1. 增加样本量（最有效但可能不可行）
2. 使用迁移学习/元学习方法
3. 考虑贝叶斯方法处理不确定性
4. 接受当前性能上限（R² ≈ 0.64-0.70）
""")

def main():
    print("=" * 60)
    print("数据瓶颈深度诊断")
    print("=" * 60)

    df = load_data()

    variety_df = analyze_label_distribution(df)
    samples_per_variety = analyze_sample_distribution(df)
    cv_df = analyze_spectral_quality(df)
    mahal_df = analyze_mahalanobis_vs_error(df)
    corr_df = analyze_feature_variety_separation(df)
    nn_df = analyze_variety_overlap(df)
    diagnose_difficult_varieties(df)
    summarize_bottleneck()

    return {
        'variety_df': variety_df,
        'cv_df': cv_df,
        'corr_df': corr_df,
        'nn_df': nn_df
    }

if __name__ == "__main__":
    results = main()
