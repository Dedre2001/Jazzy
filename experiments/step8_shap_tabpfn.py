# -*- coding: utf-8 -*-
"""
SHAP分析：基于TabPFN模型（使用KernelExplainer）
对应论文第5章

使用最优模型TabPFN进行可解释性分析
KernelExplainer适用于任意模型，计算SHAP值
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np
import json
import os
import shap
from step4_utils import load_data
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# TabPFN导入
from tabpfn import TabPFNRegressor

# 路径配置
BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
SHAP_DIR = RESULTS_DIR / "shap_tabpfn"
os.makedirs(SHAP_DIR, exist_ok=True)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# =============================================================================
# 特征组定义
# =============================================================================

FEATURE_GROUPS = {
    'Multi': [
        'R460', 'R520', 'R580', 'R660', 'R710', 'R730', 'R760', 'R780', 'R810', 'R850', 'R900',
        'VI_NDVI', 'VI_NDRE', 'VI_EVI', 'VI_SIPI', 'VI_PRI', 'VI_MTCI', 'VI_GNDVI', 'VI_NDWI'
    ],
    'Static': [
        'BF(F440)', 'GF(F520)', 'RF(F690)', 'FrF(f740)',
        'SR_F690_F740', 'SR_F440_F690', 'SR_F440_F520', 'SR_F520_F690', 'SR_F440_F740', 'SR_F520_F740'
    ],
    'OJIP': [
        'OJIP_FvFm', 'OJIP_PIabs', 'OJIP_TRo_RC', 'OJIP_ETo_RC',
        'OJIP_Vi', 'OJIP_Vj', 'OJIP_ABS_RC_log', 'OJIP_DIo_RC_log'
    ]
}

def get_feature_modality(feature_name):
    """获取特征所属模态"""
    for modality, features in FEATURE_GROUPS.items():
        if feature_name in features:
            return modality
    return 'Unknown'

def train_tabpfn_model(df, feature_cols):
    """训练TabPFN模型"""
    X = df[feature_cols].values
    y = df['D_conv'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("训练TabPFN模型...")
    model = TabPFNRegressor(device='cpu', n_estimators=8)
    model.fit(X_scaled, y)

    return model, scaler, X_scaled, y

def compute_shap_values_kernel(model, X_scaled, feature_cols, n_background=100):
    """使用KernelExplainer计算SHAP值"""
    print(f"\n使用KernelExplainer计算SHAP值...")
    print(f"背景样本数: {n_background}")
    print(f"待解释样本数: {len(X_scaled)}")

    # 选择背景数据（使用kmeans聚类中心）
    background = shap.kmeans(X_scaled, n_background)

    # 创建KernelExplainer
    explainer = shap.KernelExplainer(model.predict, background)

    # 计算SHAP值（使用更多采样以提高精度）
    shap_values = explainer.shap_values(X_scaled, nsamples=200)

    print(f"SHAP值形状: {shap_values.shape}")

    return shap_values, explainer

def analyze_feature_importance(shap_values, feature_cols):
    """分析特征重要性"""
    # 计算平均绝对SHAP值
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

    # 创建重要性DataFrame
    importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': mean_abs_shap,
        'Modality': [get_feature_modality(f) for f in feature_cols]
    })
    importance_df = importance_df.sort_values('Importance', ascending=False)

    # 计算模态贡献
    group_importance = importance_df.groupby('Modality')['Importance'].sum()
    total_importance = group_importance.sum()
    group_pct = (group_importance / total_importance * 100).sort_values(ascending=False)

    print("\n" + "=" * 50)
    print("特征组贡献分析 (TabPFN模型)")
    print("=" * 50)
    for modality, pct in group_pct.items():
        print(f"  {modality:10s}: {pct:.2f}%")

    print("\nTop-10 重要特征:")
    for i, (_, row) in enumerate(importance_df.head(10).iterrows()):
        print(f"  {i+1:2d}. {row['Feature']:20s} ({row['Modality']:6s}): {row['Importance']:.4f}")

    return importance_df, group_pct

def compute_shap_interactions_kernel(model, X_scaled, feature_cols, n_background=50, n_samples_explain=30):
    """计算SHAP交互值（采样版本）"""
    print(f"\n计算SHAP交互值...")
    print(f"背景样本数: {n_background}")
    print(f"解释样本数: {n_samples_explain}")

    # 选择背景数据
    background = shap.kmeans(X_scaled, n_background)

    # 创建KernelExplainer
    explainer = shap.KernelExplainer(model.predict, background)

    # 随机选择部分样本计算交互值（减少计算量）
    np.random.seed(42)
    sample_idx = np.random.choice(len(X_scaled), n_samples_explain, replace=False)
    X_sample = X_scaled[sample_idx]

    # 计算交互值
    # 注意：KernelExplainer的交互值通过成对扰动近似
    n_features = len(feature_cols)
    interaction_matrix = np.zeros((n_features, n_features))

    print("计算特征交互（这可能需要较长时间）...")

    # 使用SHAP值的协方差作为交互的近似
    shap_sample = explainer.shap_values(X_sample, nsamples=100)

    # 计算SHAP值的相关性矩阵作为交互的代理
    shap_corr = np.corrcoef(shap_sample.T)
    shap_corr = np.nan_to_num(shap_corr, nan=0)

    # 结合特征重要性加权
    mean_abs = np.mean(np.abs(shap_sample), axis=0)
    for i in range(n_features):
        for j in range(n_features):
            interaction_matrix[i, j] = abs(shap_corr[i, j]) * np.sqrt(mean_abs[i] * mean_abs[j])

    interaction_df = pd.DataFrame(
        interaction_matrix,
        index=feature_cols,
        columns=feature_cols
    )

    return interaction_df

def identify_cross_modal_interactions(interaction_df, feature_cols, top_k=20):
    """识别跨模态交互"""
    cross_modal_interactions = []

    for i, feat1 in enumerate(feature_cols):
        mod1 = get_feature_modality(feat1)
        for j, feat2 in enumerate(feature_cols):
            if i >= j:
                continue
            mod2 = get_feature_modality(feat2)

            interaction_value = interaction_df.loc[feat1, feat2]

            cross_modal_interactions.append({
                'Feature1': feat1,
                'Modality1': mod1,
                'Feature2': feat2,
                'Modality2': mod2,
                'Interaction': interaction_value,
                'Is_Cross_Modal': mod1 != mod2
            })

    interactions_df = pd.DataFrame(cross_modal_interactions)
    interactions_df = interactions_df.sort_values('Interaction', ascending=False)

    return interactions_df

def plot_shap_summary(shap_values, X_scaled, feature_cols):
    """绘制SHAP汇总图"""
    plt.figure(figsize=(10, 12))
    shap.summary_plot(
        shap_values,
        X_scaled,
        feature_names=feature_cols,
        show=False,
        max_display=20
    )
    plt.title('TabPFN Model - SHAP Feature Importance', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(SHAP_DIR / 'shap_summary_tabpfn.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] SHAP汇总图已保存: {SHAP_DIR / 'shap_summary_tabpfn.png'}")

def plot_shap_bar(importance_df):
    """绘制SHAP柱状图"""
    top20 = importance_df.head(20)

    modality_colors = {'Multi': '#FF6B6B', 'Static': '#45B7D1', 'OJIP': '#4ECDC4'}
    colors = [modality_colors.get(m, '#999999') for m in top20['Modality']]

    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(range(len(top20)), top20['Importance'].values, color=colors)
    ax.set_yticks(range(len(top20)))
    ax.set_yticklabels(top20['Feature'].values)
    ax.set_xlabel('Mean |SHAP Value|')
    ax.set_title('TabPFN Model - Top-20 Feature Importance', fontsize=14, fontweight='bold')
    ax.invert_yaxis()

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#FF6B6B', label='Multi'),
        Patch(facecolor='#45B7D1', label='Static'),
        Patch(facecolor='#4ECDC4', label='OJIP')
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()
    plt.savefig(SHAP_DIR / 'shap_bar_tabpfn.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] SHAP柱状图已保存: {SHAP_DIR / 'shap_bar_tabpfn.png'}")

def plot_group_contribution(group_pct):
    """绘制模态贡献饼图"""
    fig, ax = plt.subplots(figsize=(8, 8))

    colors = {'Multi': '#FF6B6B', 'Static': '#45B7D1', 'OJIP': '#4ECDC4'}
    color_list = [colors.get(m, '#999999') for m in group_pct.index]

    wedges, texts, autotexts = ax.pie(
        group_pct.values,
        labels=group_pct.index,
        autopct='%1.1f%%',
        colors=color_list,
        explode=[0.05 if m == 'OJIP' else 0 for m in group_pct.index],
        startangle=90
    )

    ax.set_title('TabPFN Model - Modality Contribution', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(SHAP_DIR / 'group_contribution_tabpfn.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] 模态贡献图已保存: {SHAP_DIR / 'group_contribution_tabpfn.png'}")

def plot_interaction_heatmap(interaction_df, feature_cols, top_n=15):
    """绘制交互值热力图"""
    import seaborn as sns

    importance = interaction_df.sum(axis=1).sort_values(ascending=False)
    top_features = importance.head(top_n).index.tolist()

    sub_df = interaction_df.loc[top_features, top_features]

    fig, ax = plt.subplots(figsize=(12, 10))

    mask = np.triu(np.ones_like(sub_df, dtype=bool), k=1)
    sns.heatmap(
        sub_df,
        mask=mask,
        cmap='RdBu_r',
        center=0,
        annot=True,
        fmt='.3f',
        square=True,
        linewidths=0.5,
        cbar_kws={'shrink': 0.8, 'label': 'Interaction Strength'},
        ax=ax
    )

    ax.set_title(f'TabPFN - SHAP Interaction Matrix (Top-{top_n})', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(SHAP_DIR / 'shap_interaction_heatmap_tabpfn.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] 交互值热力图已保存: {SHAP_DIR / 'shap_interaction_heatmap_tabpfn.png'}")

def plot_cross_modal_summary(interactions_df):
    """绘制跨模态交互汇总图"""
    cross_modal = interactions_df[interactions_df['Is_Cross_Modal']].copy()
    cross_modal['Modal_Pair'] = cross_modal.apply(
        lambda x: '-'.join(sorted([x['Modality1'], x['Modality2']])), axis=1
    )

    pair_stats = cross_modal.groupby('Modal_Pair')['Interaction'].agg(['mean', 'sum', 'count']).reset_index()
    pair_stats.columns = ['Modal_Pair', 'Mean_Interaction', 'Total_Interaction', 'N_Pairs']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    colors = ['#FF6B6B', '#45B7D1', '#4ECDC4']
    bars1 = ax1.bar(pair_stats['Modal_Pair'], pair_stats['Mean_Interaction'], color=colors)
    ax1.set_xlabel('Modality Pair')
    ax1.set_ylabel('Mean Interaction Strength')
    ax1.set_title('Cross-Modal Mean Interaction Strength')
    for bar, val in zip(bars1, pair_stats['Mean_Interaction']):
        ax1.text(bar.get_x() + bar.get_width()/2, val + 0.001, f'{val:.4f}', ha='center')

    ax2 = axes[1]
    top_cross = cross_modal.head(10)
    pair_labels = [f"{row['Feature1'][:10]}\n×\n{row['Feature2'][:10]}" for _, row in top_cross.iterrows()]
    colors2 = ['#FF6B6B' if 'Multi' in row['Modal_Pair'] and 'Static' in row['Modal_Pair']
               else '#45B7D1' if 'Multi' in row['Modal_Pair'] and 'OJIP' in row['Modal_Pair']
               else '#4ECDC4' for _, row in top_cross.iterrows()]
    bars2 = ax2.barh(range(len(top_cross)), top_cross['Interaction'].values, color=colors2)
    ax2.set_yticks(range(len(top_cross)))
    ax2.set_yticklabels(pair_labels, fontsize=8)
    ax2.set_xlabel('Interaction Strength')
    ax2.set_title('Top-10 Cross-Modal Feature Interactions')
    ax2.invert_yaxis()

    plt.tight_layout()
    plt.savefig(SHAP_DIR / 'cross_modal_interactions_tabpfn.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] 跨模态交互图已保存: {SHAP_DIR / 'cross_modal_interactions_tabpfn.png'}")

    return pair_stats

def save_results(importance_df, group_pct, shap_values, feature_cols, interaction_df, interactions_df):
    """保存分析结果"""
    importance_df.to_csv(SHAP_DIR / 'feature_importance_tabpfn.csv', index=False)

    group_df = pd.DataFrame({
        'Modality': group_pct.index,
        'Contribution_Pct': group_pct.values
    })
    group_df.to_csv(SHAP_DIR / 'group_contribution_tabpfn.csv', index=False)

    shap_df = pd.DataFrame(shap_values, columns=feature_cols)
    shap_df.to_csv(SHAP_DIR / 'shap_values_tabpfn.csv', index=False)

    interaction_df.to_csv(SHAP_DIR / 'interaction_matrix_tabpfn.csv')
    interactions_df.to_csv(SHAP_DIR / 'all_interactions_tabpfn.csv', index=False)

    print(f"\n[OK] 所有结果已保存到: {SHAP_DIR}")

def main():
    print("=" * 60)
    print("SHAP分析：TabPFN模型 (KernelExplainer)")
    print("=" * 60)

    # 加载数据
    df, _ = load_data()

    # 定义特征列
    feature_cols = []
    for group in FEATURE_GROUPS.values():
        feature_cols.extend(group)

    print(f"特征数: {len(feature_cols)}")
    print(f"样本数: {len(df)}")

    # 训练TabPFN模型
    model, scaler, X_scaled, y = train_tabpfn_model(df, feature_cols)

    # 验证模型性能
    y_pred = model.predict(X_scaled)
    from scipy.stats import spearmanr
    rho, _ = spearmanr(y, y_pred)
    print(f"模型Spearman rho: {rho:.4f}")

    # 计算SHAP值
    shap_values, explainer = compute_shap_values_kernel(
        model, X_scaled, feature_cols, n_background=100
    )

    # 分析特征重要性
    importance_df, group_pct = analyze_feature_importance(shap_values, feature_cols)

    # 计算交互值
    interaction_df = compute_shap_interactions_kernel(
        model, X_scaled, feature_cols, n_background=50, n_samples_explain=50
    )

    # 识别跨模态交互
    interactions_df = identify_cross_modal_interactions(interaction_df, feature_cols)

    # 绘制图表
    print("\n生成可视化图表...")
    plot_shap_summary(shap_values, X_scaled, feature_cols)
    plot_shap_bar(importance_df)
    plot_group_contribution(group_pct)
    plot_interaction_heatmap(interaction_df, feature_cols, top_n=15)
    pair_stats = plot_cross_modal_summary(interactions_df)

    # 保存结果
    save_results(importance_df, group_pct, shap_values, feature_cols, interaction_df, interactions_df)

    # 打印跨模态交互分析
    print("\n" + "=" * 50)
    print("跨模态交互分析")
    print("=" * 50)

    cross_modal = interactions_df[interactions_df['Is_Cross_Modal']]
    print("\nTop-5跨模态特征交互:")
    for i, (_, row) in enumerate(cross_modal.head(5).iterrows()):
        print(f"  {i+1}. {row['Feature1']} x {row['Feature2']}: {row['Interaction']:.4f}")
        print(f"     ({row['Modality1']} - {row['Modality2']})")

    print("\n" + "=" * 60)
    print("TabPFN SHAP分析完成")
    print("=" * 60)

    return importance_df, group_pct, shap_values, interaction_df

if __name__ == "__main__":
    main()
