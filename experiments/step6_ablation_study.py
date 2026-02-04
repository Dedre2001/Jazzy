# -*- coding: utf-8 -*-
"""
消融实验：单源 vs 多源融合性能对比
对应论文第4章 4.4节

实验设计：
- 单模态：Multi-only, Static-only, OJIP-only
- 双模态：Multi+Static, Multi+OJIP, Static+OJIP
- 三模态：Multi+Static+OJIP（完整融合）

使用TabPFN作为基模型（最优模型，验证融合策略在最优模型上的有效性）
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np
import json
import os
from step4_utils import load_data, get_variety_metrics
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from tabpfn import TabPFNRegressor
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 路径配置
BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
ABLATION_DIR = RESULTS_DIR / "ablation"
os.makedirs(ABLATION_DIR, exist_ok=True)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# =============================================================================
# 特征组定义（不含Treatment）
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

# 消融实验配置
ABLATION_CONFIGS = {
    # 单模态
    'Multi-only': ['Multi'],
    'Static-only': ['Static'],
    'OJIP-only': ['OJIP'],
    # 双模态
    'Multi+Static': ['Multi', 'Static'],
    'Multi+OJIP': ['Multi', 'OJIP'],
    'Static+OJIP': ['Static', 'OJIP'],
    # 三模态（完整融合）
    'Full Fusion': ['Multi', 'Static', 'OJIP']
}

def get_features_for_config(config_groups):
    """根据配置获取特征列表"""
    features = []
    for group in config_groups:
        features.extend(FEATURE_GROUPS[group])
    return features

def run_tabpfn_kfold_cv(df, feature_cols, target_col='D_conv', n_splits=5):
    """
    使用TabPFN运行5-Fold GroupKFold交叉验证
    """
    X = df[feature_cols].values
    y = df[target_col].values
    groups = df['Variety'].values

    oof_predictions = np.zeros(len(y))
    gkfold = GroupKFold(n_splits=n_splits)

    for fold, (train_idx, test_idx) in enumerate(gkfold.split(X, y, groups=groups)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # 折内标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 训练TabPFN模型
        model = TabPFNRegressor(device='cpu', n_estimators=8)
        model.fit(X_train_scaled, y_train)

        # 预测
        y_pred = model.predict(X_test_scaled)

        if hasattr(y_pred, 'ndim') and y_pred.ndim > 1:
            y_pred = y_pred.ravel()

        oof_predictions[test_idx] = y_pred

    # 聚合到品种层
    df_result = df[['Variety', target_col]].copy()
    df_result['pred'] = oof_predictions

    variety_agg = df_result.groupby('Variety').agg({
        target_col: 'first',
        'pred': 'mean'
    }).reset_index()

    # 计算品种层指标
    metrics = get_variety_metrics(
        variety_agg[target_col].values,
        variety_agg['pred'].values
    )

    return metrics, variety_agg


def run_ablation_experiment():
    """运行消融实验"""
    print("=" * 60)
    print("消融实验：单源 vs 多源融合性能对比 (TabPFN)")
    print("=" * 60)

    # 加载数据
    df, _ = load_data()
    print(f"样本数: {len(df)}")
    print(f"品种数: {df['Variety'].nunique()}")
    print(f"基模型: TabPFN (最优模型)")

    # 存储结果
    results = []

    for config_name, groups in ABLATION_CONFIGS.items():
        print(f"\n{'─' * 40}")
        print(f"配置: {config_name}")
        print(f"模态: {', '.join(groups)}")

        # 获取特征
        feature_cols = get_features_for_config(groups)
        print(f"特征数: {len(feature_cols)}")

        # 检查特征是否存在
        missing = [f for f in feature_cols if f not in df.columns]
        if missing:
            print(f"[警告] 缺失特征: {missing}")
            continue

        # 运行TabPFN交叉验证
        metrics, variety_agg = run_tabpfn_kfold_cv(df, feature_cols)

        # 记录结果
        result = {
            'Config': config_name,
            'Modalities': '+'.join(groups),
            'N_Features': len(feature_cols),
            'R2': metrics['R2'],
            'RMSE': metrics['RMSE'],
            'Spearman': metrics['Spearman'],
            'Pairwise_Acc': metrics['Pairwise_Acc'],
            'Hit@3': metrics['Hit@3'],
            'Hit@5': metrics['Hit@5']
        }
        results.append(result)

        print(f"  R2: {metrics['R2']:.4f}")
        print(f"  RMSE: {metrics['RMSE']:.4f}")
        print(f"  Spearman rho: {metrics['Spearman']:.4f}")
        print(f"  Hit@3: {metrics['Hit@3']:.2%}")

    # 转换为DataFrame
    results_df = pd.DataFrame(results)

    return results_df

def calculate_fusion_gain(results_df):
    """计算融合增益"""
    print("\n" + "=" * 60)
    print("融合增益分析")
    print("=" * 60)

    # 获取各配置的Spearman值
    full_fusion = results_df[results_df['Config'] == 'Full Fusion']['Spearman'].values[0]

    gains = []
    for _, row in results_df.iterrows():
        if row['Config'] != 'Full Fusion':
            gain = full_fusion - row['Spearman']
            gain_pct = (gain / row['Spearman']) * 100 if row['Spearman'] > 0 else 0
            gains.append({
                'Config': row['Config'],
                'Spearman': row['Spearman'],
                'Full_Fusion': full_fusion,
                'Gain': gain,
                'Gain_Pct': gain_pct
            })

    gains_df = pd.DataFrame(gains)

    print("\n相对于完整融合的Spearman增益:")
    for _, row in gains_df.iterrows():
        print(f"  {row['Config']:15s}: {row['Spearman']:.4f} → {row['Full_Fusion']:.4f} (+{row['Gain']:.4f}, +{row['Gain_Pct']:.1f}%)")

    return gains_df

def plot_ablation_results(results_df):
    """绘制消融实验结果图"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 按Spearman排序
    results_sorted = results_df.sort_values('Spearman', ascending=True)

    # 颜色映射
    colors = []
    for config in results_sorted['Config']:
        if 'only' in config:
            colors.append('#FF6B6B')  # 单模态 - 红色
        elif config == 'Full Fusion':
            colors.append('#4ECDC4')  # 完整融合 - 青色
        else:
            colors.append('#45B7D1')  # 双模态 - 蓝色

    # 图1: Spearman相关系数
    ax1 = axes[0]
    bars1 = ax1.barh(results_sorted['Config'], results_sorted['Spearman'], color=colors)
    ax1.set_xlabel('Spearman ρ')
    ax1.set_title('排序相关性')
    ax1.set_xlim(0, 1.05)
    for bar, val in zip(bars1, results_sorted['Spearman']):
        ax1.text(val + 0.02, bar.get_y() + bar.get_height()/2, f'{val:.3f}', va='center')

    # 图2: R2
    ax2 = axes[1]
    bars2 = ax2.barh(results_sorted['Config'], results_sorted['R2'], color=colors)
    ax2.set_xlabel('R²')
    ax2.set_title('回归精度')
    ax2.set_xlim(0, 1.05)
    for bar, val in zip(bars2, results_sorted['R2']):
        ax2.text(val + 0.02, bar.get_y() + bar.get_height()/2, f'{val:.3f}', va='center')

    # 图3: Hit@3
    ax3 = axes[2]
    bars3 = ax3.barh(results_sorted['Config'], results_sorted['Hit@3'], color=colors)
    ax3.set_xlabel('Hit@3')
    ax3.set_title('Top-3命中率')
    ax3.set_xlim(0, 1.15)
    for bar, val in zip(bars3, results_sorted['Hit@3']):
        ax3.text(val + 0.02, bar.get_y() + bar.get_height()/2, f'{val:.0%}', va='center')

    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#FF6B6B', label='单模态'),
        Patch(facecolor='#45B7D1', label='双模态'),
        Patch(facecolor='#4ECDC4', label='三模态融合')
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.02))

    plt.tight_layout()
    plt.savefig(ABLATION_DIR / 'ablation_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n[OK] 消融实验对比图已保存: {ABLATION_DIR / 'ablation_comparison.png'}")

def plot_modality_contribution(results_df):
    """绘制模态贡献雷达图"""
    # 提取单模态结果
    single_modalities = ['Multi-only', 'Static-only', 'OJIP-only']
    single_df = results_df[results_df['Config'].isin(single_modalities)]

    # 指标
    metrics = ['R2', 'Spearman', 'Pairwise_Acc', 'Hit@3', 'Hit@5']

    # 雷达图
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # 闭合

    colors = ['#FF6B6B', '#45B7D1', '#4ECDC4']
    labels = ['Multi', 'Static', 'OJIP']

    for i, (_, row) in enumerate(single_df.iterrows()):
        values = [row[m] for m in metrics]
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=labels[i], color=colors[i])
        ax.fill(angles, values, alpha=0.25, color=colors[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
    ax.set_title('单模态性能对比', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(ABLATION_DIR / 'modality_radar.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] 模态雷达图已保存: {ABLATION_DIR / 'modality_radar.png'}")

def main():
    # 运行消融实验
    results_df = run_ablation_experiment()

    # 保存结果
    results_df.to_csv(ABLATION_DIR / 'ablation_results.csv', index=False)
    print(f"\n[OK] 消融实验结果已保存: {ABLATION_DIR / 'ablation_results.csv'}")

    # 计算融合增益
    gains_df = calculate_fusion_gain(results_df)
    gains_df.to_csv(ABLATION_DIR / 'fusion_gains.csv', index=False)

    # 绘制图表
    plot_ablation_results(results_df)
    plot_modality_contribution(results_df)

    # 打印汇总表
    print("\n" + "=" * 60)
    print("消融实验结果汇总")
    print("=" * 60)
    print(results_df.to_string(index=False))

    return results_df

if __name__ == "__main__":
    main()
