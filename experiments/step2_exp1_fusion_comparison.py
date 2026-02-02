"""
Step 2: Exp-1 融合有效性验证
目的: 证明多源融合带来性能增益

方法:
1. 固定模型 = CatBoost
2. 5-Fold KFold，获取OOF预测
3. 对比 FS1/FS2/FS3/FS4
4. 统计检验: Friedman + Nemenyi

输出:
- results/tables/exp1_fusion_comparison.csv
- results/figures/fig5a_fusion_comparison.png
"""

import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostRegressor
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# 路径配置
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
RESULTS_DIR = BASE_DIR / "results"
os.makedirs(RESULTS_DIR / "tables", exist_ok=True)
os.makedirs(RESULTS_DIR / "figures", exist_ok=True)

# 随机种子
RANDOM_STATE = 42
N_SPLITS = 5

# 图表风格
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-whitegrid')

def load_data():
    """加载特征数据和特征集定义"""
    df = pd.read_csv(f"{DATA_DIR}/features_40.csv")
    with open(f"{DATA_DIR}/feature_sets.json", 'r', encoding='utf-8') as f:
        feature_sets = json.load(f)
    print(f"加载数据: {len(df)} 样本")
    return df, feature_sets

def get_variety_metrics(y_true_variety, y_pred_variety):
    """
    计算品种层指标

    输入: 品种层的真值和预测值 (长度=13)
    """
    from scipy.stats import spearmanr

    # R²
    ss_res = np.sum((y_true_variety - y_pred_variety) ** 2)
    ss_tot = np.sum((y_true_variety - np.mean(y_true_variety)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    # RMSE
    rmse = np.sqrt(np.mean((y_true_variety - y_pred_variety) ** 2))

    # Spearman相关
    spearman_r, spearman_p = spearmanr(y_true_variety, y_pred_variety)

    # Pairwise Ranking Accuracy
    n = len(y_true_variety)
    correct_pairs = 0
    total_pairs = 0
    for i in range(n):
        for j in range(i+1, n):
            total_pairs += 1
            true_order = y_true_variety[i] > y_true_variety[j]
            pred_order = y_pred_variety[i] > y_pred_variety[j]
            if true_order == pred_order:
                correct_pairs += 1
    pairwise_acc = correct_pairs / total_pairs if total_pairs > 0 else 0

    # Hit@3 (Top-3命中率)
    true_top3 = set(np.argsort(y_true_variety)[-3:])
    pred_top3 = set(np.argsort(y_pred_variety)[-3:])
    hit_at_3 = len(true_top3 & pred_top3) / 3

    # Hit@5
    true_top5 = set(np.argsort(y_true_variety)[-5:])
    pred_top5 = set(np.argsort(y_pred_variety)[-5:])
    hit_at_5 = len(true_top5 & pred_top5) / 5

    return {
        'R2': r2,
        'RMSE': rmse,
        'Spearman': spearman_r,
        'Spearman_p': spearman_p,
        'Pairwise_Acc': pairwise_acc,
        'Hit@3': hit_at_3,
        'Hit@5': hit_at_5
    }

def run_kfold_cv(df, feature_cols, target_col='D_conv', n_splits=5):
    """
    运行5-Fold交叉验证，返回OOF预测
    """
    X = df[feature_cols].values
    y = df[target_col].values
    varieties = df['Variety'].values

    # 存储OOF预测
    oof_predictions = np.zeros(len(y))
    fold_metrics = []

    groups = df['Variety'].values  # 按品种分组
    gkfold = GroupKFold(n_splits=n_splits)

    for fold, (train_idx, test_idx) in enumerate(gkfold.split(X, y, groups=groups)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # 折内标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 训练CatBoost
        model = CatBoostRegressor(
            iterations=500,
            learning_rate=0.05,
            depth=4,
            l2_leaf_reg=5,
            min_data_in_leaf=3,
            loss_function='RMSE',
            random_seed=RANDOM_STATE,
            verbose=False
        )
        model.fit(X_train_scaled, y_train)

        # 预测
        y_pred = model.predict(X_test_scaled)
        oof_predictions[test_idx] = y_pred

    # 聚合到品种层
    df_result = df[['Variety', target_col]].copy()
    df_result['pred'] = oof_predictions

    variety_agg = df_result.groupby('Variety').agg({
        target_col: 'first',  # 真值（每个品种相同）
        'pred': 'mean'        # 预测值取平均
    }).reset_index()

    # 计算品种层指标
    metrics = get_variety_metrics(
        variety_agg[target_col].values,
        variety_agg['pred'].values
    )

    return metrics, oof_predictions, variety_agg

def run_experiment():
    """运行Exp-1融合有效性验证"""
    print("=" * 60)
    print("Step 2: Exp-1 融合有效性验证")
    print("=" * 60)

    # 加载数据
    df, feature_sets = load_data()

    # 存储结果
    results = []
    all_variety_agg = {}

    # 对每个特征集运行实验
    for fs_name in ['FS1', 'FS2', 'FS3', 'FS4']:
        print(f"\n{'='*40}")
        print(f"运行 {fs_name}: {feature_sets[fs_name]['description']}")
        print(f"特征数: {feature_sets[fs_name]['n_features']}")
        print(f"{'='*40}")

        feature_cols = feature_sets[fs_name]['features']

        # 检查特征是否存在
        missing = [f for f in feature_cols if f not in df.columns]
        if missing:
            print(f"[ERROR] 缺失特征: {missing}")
            continue

        # 运行交叉验证
        metrics, oof_pred, variety_agg = run_kfold_cv(df, feature_cols)

        # 保存结果
        metrics['Feature_Set'] = fs_name
        metrics['N_Features'] = feature_sets[fs_name]['n_features']
        results.append(metrics)
        all_variety_agg[fs_name] = variety_agg

        print(f"\nVariety-level metrics:")
        print(f"  R2: {metrics['R2']:.4f}")
        print(f"  RMSE: {metrics['RMSE']:.4f}")
        print(f"  Spearman rho: {metrics['Spearman']:.4f} (p={metrics['Spearman_p']:.4f})")
        print(f"  Pairwise Acc: {metrics['Pairwise_Acc']:.4f}")
        print(f"  Hit@3: {metrics['Hit@3']:.4f}")
        print(f"  Hit@5: {metrics['Hit@5']:.4f}")

    # 汇总结果
    results_df = pd.DataFrame(results)
    results_df = results_df[['Feature_Set', 'N_Features', 'R2', 'RMSE',
                             'Spearman', 'Pairwise_Acc', 'Hit@3', 'Hit@5']]

    print("\n" + "=" * 60)
    print("融合对比结果汇总")
    print("=" * 60)
    print(results_df.to_string(index=False))

    # 计算增益
    print("\n" + "=" * 60)
    print("融合增益分析")
    print("=" * 60)

    fs1_spearman = results_df[results_df['Feature_Set']=='FS1']['Spearman'].values[0]
    fs2_spearman = results_df[results_df['Feature_Set']=='FS2']['Spearman'].values[0]
    fs3_spearman = results_df[results_df['Feature_Set']=='FS3']['Spearman'].values[0]
    fs4_spearman = results_df[results_df['Feature_Set']=='FS4']['Spearman'].values[0]

    print(f"FS3 vs FS1 (双源 vs Multi-only): Δρ = {fs3_spearman - fs1_spearman:+.4f}")
    print(f"FS3 vs FS2 (双源 vs Static-only): Δρ = {fs3_spearman - fs2_spearman:+.4f}")
    print(f"FS4 vs FS3 (三源 vs 双源): Δρ = {fs4_spearman - fs3_spearman:+.4f}")
    print(f"FS4 vs FS1 (三源 vs Multi-only): Δρ = {fs4_spearman - fs1_spearman:+.4f}")

    # 保存结果表格
    results_df.to_csv(f"{RESULTS_DIR}/tables/exp1_fusion_comparison.csv", index=False)
    print(f"\n结果已保存至: {RESULTS_DIR}/tables/exp1_fusion_comparison.csv")

    # 生成可视化
    plot_fusion_comparison(results_df)

    return results_df, all_variety_agg

def plot_fusion_comparison(results_df):
    """生成融合对比可视化"""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # 颜色方案
    colors = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3']

    # 1. Spearman对比
    ax1 = axes[0]
    bars1 = ax1.bar(results_df['Feature_Set'], results_df['Spearman'],
                    color=colors, edgecolor='black', linewidth=1.2)
    ax1.set_ylabel('Spearman ρ', fontsize=12)
    ax1.set_xlabel('Feature Set', fontsize=12)
    ax1.set_title('(a) Spearman Correlation', fontsize=13, fontweight='bold')
    ax1.set_ylim(0, 1)
    for bar, val in zip(bars1, results_df['Spearman']):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # 2. R²对比
    ax2 = axes[1]
    bars2 = ax2.bar(results_df['Feature_Set'], results_df['R2'],
                    color=colors, edgecolor='black', linewidth=1.2)
    ax2.set_ylabel('R²', fontsize=12)
    ax2.set_xlabel('Feature Set', fontsize=12)
    ax2.set_title('(b) Coefficient of Determination', fontsize=13, fontweight='bold')
    ax2.set_ylim(0, 1)
    for bar, val in zip(bars2, results_df['R2']):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # 3. Hit@3 对比
    ax3 = axes[2]
    bars3 = ax3.bar(results_df['Feature_Set'], results_df['Hit@3'],
                    color=colors, edgecolor='black', linewidth=1.2)
    ax3.set_ylabel('Hit@3', fontsize=12)
    ax3.set_xlabel('Feature Set', fontsize=12)
    ax3.set_title('(c) Top-3 Hit Rate', fontsize=13, fontweight='bold')
    ax3.set_ylim(0, 1.2)
    for bar, val in zip(bars3, results_df['Hit@3']):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/figures/fig5a_fusion_comparison.png",
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"图表已保存至: {RESULTS_DIR}/figures/fig5a_fusion_comparison.png")

if __name__ == "__main__":
    results, variety_agg = run_experiment()
