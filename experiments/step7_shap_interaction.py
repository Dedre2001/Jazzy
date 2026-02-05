# -*- coding: utf-8 -*-
"""
SHAP交互值分析：跨模态协同效应挖掘
对应论文第5章 5.4节

分析内容：
1. 计算SHAP交互值矩阵
2. 识别跨模态特征协同效应
3. 解释多源融合优于单源的机制
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
from catboost import CatBoostRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# 路径配置
BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
SHAP_DIR = RESULTS_DIR / "shap_interaction"
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

def train_full_model(df, feature_cols):
    """训练完整模型用于SHAP分析"""
    X = df[feature_cols].values
    y = df['D_conv'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = CatBoostRegressor(
        iterations=500,
        learning_rate=0.05,
        depth=4,
        l2_leaf_reg=5,
        min_data_in_leaf=3,
        loss_function='RMSE',
        random_seed=42,
        verbose=False
    )
    model.fit(X_scaled, y)

    return model, scaler, X_scaled

def compute_shap_interactions(model, X_scaled, feature_cols):
    """计算SHAP交互值"""
    print("计算SHAP交互值（这可能需要几分钟）...")

    explainer = shap.TreeExplainer(model)

    # 计算交互值
    shap_interaction_values = explainer.shap_interaction_values(X_scaled)

    print(f"交互值矩阵形状: {shap_interaction_values.shape}")
    # shape: (n_samples, n_features, n_features)

    return shap_interaction_values

def analyze_interaction_matrix(shap_interaction_values, feature_cols):
    """分析交互值矩阵"""
    n_samples, n_features, _ = shap_interaction_values.shape

    # 计算平均绝对交互值
    mean_abs_interactions = np.mean(np.abs(shap_interaction_values), axis=0)

    # 创建DataFrame
    interaction_df = pd.DataFrame(
        mean_abs_interactions,
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
            if i >= j:  # 只取上三角
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

def plot_interaction_heatmap(interaction_df, feature_cols, top_n=15):
    """绘制交互值热力图（Top-N特征）"""
    # 选择Top-N重要特征
    importance = interaction_df.sum(axis=1).sort_values(ascending=False)
    top_features = importance.head(top_n).index.tolist()

    # 提取子矩阵
    sub_df = interaction_df.loc[top_features, top_features]

    # 添加模态标签
    modality_colors = {'Multi': '#FF6B6B', 'Static': '#45B7D1', 'OJIP': '#4ECDC4'}
    row_colors = [modality_colors.get(get_feature_modality(f), '#999999') for f in top_features]

    fig, ax = plt.subplots(figsize=(12, 10))

    # 绘制热力图
    mask = np.triu(np.ones_like(sub_df, dtype=bool), k=1)  # 上三角遮罩
    sns.heatmap(
        sub_df,
        mask=mask,
        cmap='RdBu_r',
        center=0,
        annot=True,
        fmt='.3f',
        square=True,
        linewidths=0.5,
        cbar_kws={'shrink': 0.8, 'label': 'Mean |SHAP Interaction|'},
        ax=ax
    )

    ax.set_title(f'SHAP交互值矩阵 (Top-{top_n}特征)', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(SHAP_DIR / 'shap_interaction_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] 交互值热力图已保存: {SHAP_DIR / 'shap_interaction_heatmap.png'}")

def plot_cross_modal_summary(interactions_df):
    """绘制跨模态交互汇总图"""
    # 按模态对分组统计
    cross_modal = interactions_df[interactions_df['Is_Cross_Modal']].copy()
    cross_modal['Modal_Pair'] = cross_modal.apply(
        lambda x: '-'.join(sorted([x['Modality1'], x['Modality2']])), axis=1
    )

    # 分组统计
    pair_stats = cross_modal.groupby('Modal_Pair')['Interaction'].agg(['mean', 'sum', 'count']).reset_index()
    pair_stats.columns = ['Modal_Pair', 'Mean_Interaction', 'Total_Interaction', 'N_Pairs']

    # 绘图
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 图1: 平均交互强度
    ax1 = axes[0]
    colors = ['#FF6B6B', '#45B7D1', '#4ECDC4']
    bars1 = ax1.bar(pair_stats['Modal_Pair'], pair_stats['Mean_Interaction'], color=colors)
    ax1.set_xlabel('模态对')
    ax1.set_ylabel('平均交互强度')
    ax1.set_title('跨模态平均交互强度')
    for bar, val in zip(bars1, pair_stats['Mean_Interaction']):
        ax1.text(bar.get_x() + bar.get_width()/2, val + 0.0005, f'{val:.4f}', ha='center')

    # 图2: Top-10跨模态交互对
    ax2 = axes[1]
    top_cross = cross_modal.head(10)
    pair_labels = [f"{row['Feature1'][:8]}\n×\n{row['Feature2'][:8]}" for _, row in top_cross.iterrows()]
    colors2 = ['#FF6B6B' if 'Multi' in row['Modal_Pair'] and 'Static' in row['Modal_Pair']
               else '#45B7D1' if 'Multi' in row['Modal_Pair'] and 'OJIP' in row['Modal_Pair']
               else '#4ECDC4' for _, row in top_cross.iterrows()]
    bars2 = ax2.barh(range(len(top_cross)), top_cross['Interaction'].values, color=colors2)
    ax2.set_yticks(range(len(top_cross)))
    ax2.set_yticklabels(pair_labels, fontsize=8)
    ax2.set_xlabel('交互强度')
    ax2.set_title('Top-10跨模态特征交互')
    ax2.invert_yaxis()

    plt.tight_layout()
    plt.savefig(SHAP_DIR / 'cross_modal_interactions.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] 跨模态交互图已保存: {SHAP_DIR / 'cross_modal_interactions.png'}")

    return pair_stats

def explain_synergy_mechanism(interactions_df, pair_stats):
    """解释协同机制"""
    print("\n" + "=" * 60)
    print("多源融合协同机制分析")
    print("=" * 60)

    # 跨模态 vs 模态内交互
    cross_modal = interactions_df[interactions_df['Is_Cross_Modal']]
    intra_modal = interactions_df[~interactions_df['Is_Cross_Modal']]

    cross_mean = cross_modal['Interaction'].mean()
    intra_mean = intra_modal['Interaction'].mean()

    print(f"\n1. 交互强度对比:")
    print(f"   跨模态平均交互: {cross_mean:.4f}")
    print(f"   模态内平均交互: {intra_mean:.4f}")
    print(f"   跨模态/模态内比值: {cross_mean/intra_mean:.2f}")

    print(f"\n2. 模态对交互强度排名:")
    for _, row in pair_stats.sort_values('Mean_Interaction', ascending=False).iterrows():
        print(f"   {row['Modal_Pair']}: {row['Mean_Interaction']:.4f} (共{row['N_Pairs']:.0f}对)")

    print(f"\n3. Top-5跨模态特征交互:")
    for i, (_, row) in enumerate(cross_modal.head(5).iterrows()):
        print(f"   {i+1}. {row['Feature1']} × {row['Feature2']}: {row['Interaction']:.4f}")
        print(f"      ({row['Modality1']} - {row['Modality2']})")

    # 保存分析结果
    analysis = {
        'cross_modal_mean': float(cross_mean),
        'intra_modal_mean': float(intra_mean),
        'cross_intra_ratio': float(cross_mean / intra_mean),
        'modal_pair_stats': pair_stats.to_dict('records'),
        'top_cross_modal': cross_modal.head(10).to_dict('records')
    }

    with open(SHAP_DIR / 'synergy_analysis.json', 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)

    return analysis

def main():
    print("=" * 60)
    print("SHAP交互值分析：跨模态协同效应挖掘")
    print("=" * 60)

    # 加载数据
    df, _ = load_data()

    # 定义特征列（不含Treatment）
    feature_cols = []
    for group in FEATURE_GROUPS.values():
        feature_cols.extend(group)

    print(f"特征数: {len(feature_cols)}")
    print(f"样本数: {len(df)}")

    # 训练模型
    print("\n训练CatBoost模型...")
    model, scaler, X_scaled = train_full_model(df, feature_cols)

    # 计算SHAP交互值
    shap_interaction_values = compute_shap_interactions(model, X_scaled, feature_cols)

    # 分析交互矩阵
    print("\n分析交互值矩阵...")
    interaction_df = analyze_interaction_matrix(shap_interaction_values, feature_cols)
    interaction_df.to_csv(SHAP_DIR / 'interaction_matrix.csv')

    # 识别跨模态交互
    interactions_df = identify_cross_modal_interactions(interaction_df, feature_cols)
    interactions_df.to_csv(SHAP_DIR / 'all_interactions.csv', index=False)

    # 绘制图表
    print("\n生成可视化图表...")
    plot_interaction_heatmap(interaction_df, feature_cols, top_n=15)
    pair_stats = plot_cross_modal_summary(interactions_df)

    # 解释协同机制
    analysis = explain_synergy_mechanism(interactions_df, pair_stats)

    print("\n" + "=" * 60)
    print("SHAP交互值分析完成")
    print("=" * 60)

    return interaction_df, interactions_df, analysis

if __name__ == "__main__":
    main()
