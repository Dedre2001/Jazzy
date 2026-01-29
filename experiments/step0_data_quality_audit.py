"""
Step 0: 数据质量审计
目的: 在正式实验前全面了解数据质量，为预处理决策提供依据

输出:
- results/data_audit/quality_report.txt
- results/data_audit/outlier_summary.csv
- results/data_audit/distribution_plots/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
import os
from pathlib import Path

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results" / "data_audit"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR / "distribution_plots", exist_ok=True)

def load_data():
    """加载所有数据文件"""
    fusion = pd.read_csv(DATA_DIR / "fusion_all.csv")
    multi = pd.read_csv(DATA_DIR / "multi.csv")
    static = pd.read_csv(DATA_DIR / "static.csv")
    ojip = pd.read_csv(DATA_DIR / "ojip.csv")
    return fusion, multi, static, ojip

def check_basic_info(df, name="fusion_all"):
    """基本信息检查"""
    report = []
    report.append(f"\n{'='*60}")
    report.append(f"数据集: {name}")
    report.append(f"{'='*60}")
    report.append(f"样本数: {len(df)}")
    report.append(f"特征数: {len(df.columns)}")
    report.append(f"列名: {list(df.columns)}")

    # 样本完整性
    if 'Variety' in df.columns and 'Treatment' in df.columns:
        varieties = df['Variety'].unique()
        treatments = df['Treatment'].unique()
        report.append(f"\n品种数: {len(varieties)} -> {sorted(varieties)}")
        report.append(f"处理数: {len(treatments)} -> {list(treatments)}")

        # 检查每个品种×处理的样本数
        cross_tab = df.groupby(['Variety', 'Treatment']).size().unstack(fill_value=0)
        report.append(f"\n品种×处理 样本数矩阵:")
        report.append(str(cross_tab))

        expected = len(varieties) * len(treatments) * 3  # 3 replicates
        report.append(f"\n预期样本数: {expected}, 实际样本数: {len(df)}")
        if len(df) == expected:
            report.append("[OK] 样本完整性检查通过")
        else:
            report.append("[WARNING] 样本数量不符，请检查")

    return "\n".join(report)

def check_missing_values(df):
    """缺失值检查"""
    report = []
    report.append(f"\n{'='*60}")
    report.append("缺失值检查")
    report.append(f"{'='*60}")

    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)

    if missing.sum() == 0:
        report.append("[OK] 无缺失值")
    else:
        missing_df = pd.DataFrame({
            '缺失数': missing[missing > 0],
            '缺失比例(%)': missing_pct[missing > 0]
        })
        report.append(str(missing_df))

    return "\n".join(report)

def check_data_types(df):
    """数据类型检查"""
    report = []
    report.append(f"\n{'='*60}")
    report.append("数据类型检查")
    report.append(f"{'='*60}")

    # 识别数值列和非数值列
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    report.append(f"数值列 ({len(numeric_cols)}): {numeric_cols[:10]}..." if len(numeric_cols) > 10 else f"数值列: {numeric_cols}")
    report.append(f"非数值列 ({len(non_numeric_cols)}): {non_numeric_cols}")

    return "\n".join(report), numeric_cols

def detect_outliers(df, numeric_cols):
    """异常值检测 (Z-score + IQR 双重方法)"""
    report = []
    report.append(f"\n{'='*60}")
    report.append("异常值检测")
    report.append(f"{'='*60}")

    outlier_summary = []

    # 排除ID列和标签列
    exclude_cols = ['Sample_ID', 'Treatment', 'Variety', 'Sample', 'Category', 'Rank']
    feature_cols = [c for c in numeric_cols if c not in exclude_cols]

    for col in feature_cols:
        data = df[col].dropna()

        # Z-score方法
        z_scores = np.abs(stats.zscore(data))
        z_outliers = (z_scores > 3).sum()

        # IQR方法
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        iqr_outliers = ((data < Q1 - 1.5*IQR) | (data > Q3 + 1.5*IQR)).sum()

        if z_outliers > 0 or iqr_outliers > 0:
            outlier_summary.append({
                'Feature': col,
                'Z_outliers': z_outliers,
                'IQR_outliers': iqr_outliers,
                'Min': data.min(),
                'Max': data.max(),
                'Mean': data.mean(),
                'Std': data.std(),
                'Skewness': stats.skew(data),
                'Kurtosis': stats.kurtosis(data)
            })

    if outlier_summary:
        outlier_df = pd.DataFrame(outlier_summary)
        outlier_df = outlier_df.sort_values('Z_outliers', ascending=False)
        report.append(f"检测到 {len(outlier_summary)} 个特征存在异常值:")
        report.append(str(outlier_df.head(20)))

        # 保存详细报告
        outlier_df.to_csv(f"{RESULTS_DIR}/outlier_summary.csv", index=False)
        report.append(f"\n详细报告已保存至: {RESULTS_DIR}/outlier_summary.csv")
    else:
        report.append("[OK] 未检测到明显异常值")

    return "\n".join(report), outlier_summary

def analyze_distributions(df, numeric_cols):
    """分布分析"""
    report = []
    report.append(f"\n{'='*60}")
    report.append("分布分析 (偏度/峰度)")
    report.append(f"{'='*60}")

    exclude_cols = ['Sample_ID', 'Treatment', 'Variety', 'Sample', 'Category', 'Rank']
    feature_cols = [c for c in numeric_cols if c not in exclude_cols]

    dist_stats = []
    for col in feature_cols:
        data = df[col].dropna()
        dist_stats.append({
            'Feature': col,
            'Mean': round(data.mean(), 6),
            'Std': round(data.std(), 6),
            'Min': round(data.min(), 6),
            'Max': round(data.max(), 6),
            'Skewness': round(stats.skew(data), 3),
            'Kurtosis': round(stats.kurtosis(data), 3)
        })

    dist_df = pd.DataFrame(dist_stats)

    # 识别高偏度特征（可能需要log变换）
    high_skew = dist_df[abs(dist_df['Skewness']) > 1]
    if len(high_skew) > 0:
        report.append(f"\n[WARNING] 高偏度特征 (|Skewness| > 1): {len(high_skew)}个")
        report.append(str(high_skew[['Feature', 'Skewness', 'Min', 'Max']]))

    # 保存完整统计
    dist_df.to_csv(f"{RESULTS_DIR}/distribution_stats.csv", index=False)
    report.append(f"\n完整分布统计已保存至: {RESULTS_DIR}/distribution_stats.csv")

    return "\n".join(report), dist_df

def analyze_ground_truth(df):
    """Ground Truth分析"""
    report = []
    report.append(f"\n{'='*60}")
    report.append("Ground Truth 分析")
    report.append(f"{'='*60}")

    # D_conv分布
    if 'D_conv' in df.columns:
        variety_dconv = df.groupby('Variety')['D_conv'].first().sort_values(ascending=False)
        report.append("\n品种 D_conv 排名:")
        for i, (v, d) in enumerate(variety_dconv.items(), 1):
            cat = df[df['Variety']==v]['Category'].iloc[0]
            report.append(f"  {i:2d}. 品种{v}: D_conv={d:.4f} ({cat})")

    # Category分布
    if 'Category' in df.columns:
        cat_counts = df.groupby('Variety')['Category'].first().value_counts()
        report.append(f"\n抗旱等级分布 (品种数):")
        for cat, cnt in cat_counts.items():
            report.append(f"  {cat}: {cnt}个品种")

    # Rank分布验证
    if 'Rank' in df.columns:
        variety_ranks = df.groupby('Variety')['Rank'].first().sort_values()
        report.append(f"\nRank分布: {list(variety_ranks.values)}")

        # 验证3/5/5分档
        top3 = variety_ranks[variety_ranks <= 3]
        mid5 = variety_ranks[(variety_ranks > 3) & (variety_ranks <= 8)]
        bot5 = variety_ranks[variety_ranks > 8]
        report.append(f"Rank分档验证: Top-3={len(top3)}, Middle-5={len(mid5)}, Bottom-5={len(bot5)}")

    return "\n".join(report)

def plot_correlation_heatmap(df, numeric_cols):
    """特征相关性热图"""
    exclude_cols = ['Sample_ID', 'Treatment', 'Variety', 'Sample', 'Category', 'Rank',
                    'D_conv', 'D_stress', 'D_recovery']
    feature_cols = [c for c in numeric_cols if c not in exclude_cols]

    # 分模态绘制
    multi_cols = [c for c in feature_cols if c.startswith('R') and c[1:].replace('.','').isdigit()]
    static_cols = [c for c in feature_cols if 'F(' in c or 'F4' in c or 'F5' in c or 'F6' in c or 'F7' in c]
    ojip_cols = [c for c in feature_cols if c.startswith('OJIP_')]

    # 如果static_cols为空，尝试其他模式
    if not static_cols:
        static_cols = [c for c in feature_cols if c in ['BF(F440)', 'GF(F520)', 'RF(F690)', 'FrF(f740)']]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Multi相关性
    if multi_cols:
        corr_multi = df[multi_cols].corr()
        sns.heatmap(corr_multi, ax=axes[0], cmap='RdBu_r', center=0,
                    xticklabels=True, yticklabels=True, annot=False)
        axes[0].set_title(f'Multi Features Correlation ({len(multi_cols)} features)')

    # Static相关性
    if static_cols:
        corr_static = df[static_cols].corr()
        sns.heatmap(corr_static, ax=axes[1], cmap='RdBu_r', center=0,
                    xticklabels=True, yticklabels=True, annot=True, fmt='.2f')
        axes[1].set_title(f'Static Features Correlation ({len(static_cols)} features)')

    # OJIP相关性
    if ojip_cols:
        corr_ojip = df[ojip_cols].corr()
        sns.heatmap(corr_ojip, ax=axes[2], cmap='RdBu_r', center=0,
                    xticklabels=True, yticklabels=True, annot=False)
        axes[2].set_title(f'OJIP Features Correlation ({len(ojip_cols)} features)')

    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/correlation_heatmap.png", dpi=150, bbox_inches='tight')
    plt.close()

    return f"相关性热图已保存至: {RESULTS_DIR}/correlation_heatmap.png"

def plot_treatment_effect(df):
    """处理效应可视化"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 选择代表性特征
    features_to_plot = ['R850', 'R710', 'OJIP_FvFm', 'OJIP_PIabs', 'BF(F440)', 'RF(F690)']
    features_available = [f for f in features_to_plot if f in df.columns]

    for i, feat in enumerate(features_available):
        ax = axes[i//3, i%3]
        df.boxplot(column=feat, by='Treatment', ax=ax)
        ax.set_title(feat)
        ax.set_xlabel('Treatment')

    plt.suptitle('Treatment Effect on Key Features', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/treatment_effect.png", dpi=150, bbox_inches='tight')
    plt.close()

    return f"处理效应图已保存至: {RESULTS_DIR}/treatment_effect.png"

def main():
    print("="*60)
    print("Step 0: 数据质量审计")
    print("="*60)

    # 加载数据
    print("\n加载数据...")
    fusion, multi, static, ojip = load_data()

    # 收集报告
    full_report = []
    full_report.append("="*60)
    full_report.append("数据质量审计报告")
    full_report.append("="*60)
    full_report.append(f"生成时间: {pd.Timestamp.now()}")

    # 1. 基本信息
    full_report.append(check_basic_info(fusion, "fusion_all"))

    # 2. 缺失值检查
    full_report.append(check_missing_values(fusion))

    # 3. 数据类型检查
    type_report, numeric_cols = check_data_types(fusion)
    full_report.append(type_report)

    # 4. 异常值检测
    outlier_report, outlier_summary = detect_outliers(fusion, numeric_cols)
    full_report.append(outlier_report)

    # 5. 分布分析
    dist_report, dist_df = analyze_distributions(fusion, numeric_cols)
    full_report.append(dist_report)

    # 6. Ground Truth分析
    full_report.append(analyze_ground_truth(fusion))

    # 7. 可视化
    print("\n生成可视化...")
    full_report.append(f"\n{'='*60}")
    full_report.append("可视化输出")
    full_report.append(f"{'='*60}")
    full_report.append(plot_correlation_heatmap(fusion, numeric_cols))
    full_report.append(plot_treatment_effect(fusion))

    # 保存报告
    report_text = "\n".join(full_report)
    with open(f"{RESULTS_DIR}/quality_report.txt", 'w', encoding='utf-8') as f:
        f.write(report_text)

    print(report_text)
    print(f"\n报告已保存至: {RESULTS_DIR}/quality_report.txt")

    # 返回关键统计供后续决策
    return {
        'n_samples': len(fusion),
        'n_features': len(numeric_cols),
        'n_outlier_features': len(outlier_summary),
        'high_skew_features': len(dist_df[abs(dist_df['Skewness']) > 1]) if len(dist_df) > 0 else 0
    }

if __name__ == "__main__":
    summary = main()
    print(f"\n审计完成: {summary}")
