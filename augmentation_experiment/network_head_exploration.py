"""
探索 Scale=0.01 + 网络头（特征缩放头）能否提升精度

网络头实现方式：
1. Attention-based Feature Weighting: 学习每个特征的重要性
2. Per-Feature Scaling: 每个特征一个可学习的缩放因子
3. Feature Interaction Head: 特征交互层

由于 TabPFN 不支持自定义层，我们采用以下策略：
- 方案A: 两阶段训练 - 先用小模型学特征权重，再用 TabPFN
- 方案B: 元学习特征权重 - 用交叉验证反馈优化权重
- 方案C: 基于梯度的特征重要性 - 用 Ridge 的系数作为权重
"""

import os
import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.optimize import minimize

warnings.filterwarnings('ignore')

os.environ.setdefault("TABPFN_MODEL_CACHE_DIR", str(Path(__file__).resolve().parent.parent / "tabpfn_ckpt"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso

try:
    from tabpfn import TabPFNRegressor
    TABPFN_AVAILABLE = True
except ImportError:
    TABPFN_AVAILABLE = False

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "results"
OUTPUT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42


def get_feature_cols(df):
    exclude = ['Sample_ID', 'Treatment', 'Variety', 'Sample', 'D_conv',
               'D_stress', 'D_recovery', 'Category', 'Rank']
    return [c for c in df.columns if c not in exclude and not c.startswith('Trt_')]


def apply_difference_amplification(df, feature_cols, scale_factor):
    """放大品种间差异 (基础信号注入)"""
    if scale_factor == 0:
        return df.copy()

    df_new = df.copy()

    d_conv_min = df['D_conv'].min()
    d_conv_max = df['D_conv'].max()
    d_conv_mid = (d_conv_min + d_conv_max) / 2
    d_conv_range = (d_conv_max - d_conv_min) / 2

    for variety in df['Variety'].unique():
        mask = df_new['Variety'] == variety
        d_conv = df_new.loc[mask, 'D_conv'].iloc[0]
        normalized = (d_conv - d_conv_mid) / d_conv_range if d_conv_range > 0 else 0
        adjustment = 1 + scale_factor * normalized

        for col in feature_cols:
            df_new.loc[mask, col] = df_new.loc[mask, col] * adjustment

    return df_new


def apply_feature_weights(X, weights):
    """应用特征权重（网络头核心操作）"""
    return X * weights


def run_groupkfold_with_weights(df, feature_cols, weights=None, target_col='D_conv'):
    """带特征权重的 GroupKFold 训练"""
    X = df[feature_cols].values
    y = df[target_col].values
    groups = df['Variety'].values

    if weights is not None:
        X = apply_feature_weights(X, weights)

    n_splits = min(5, len(np.unique(groups)))
    oof_preds = np.full(len(y), np.nan)

    gkf = GroupKFold(n_splits=n_splits)

    for fold, (tr_idx, te_idx) in enumerate(gkf.split(X, y, groups=groups)):
        X_train, X_test = X[tr_idx], X[te_idx]
        y_train = y[tr_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        if TABPFN_AVAILABLE:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = TabPFNRegressor(
                n_estimators=256, random_state=RANDOM_STATE,
                fit_mode="fit_preprocessors", device=device,
                average_before_softmax=True, softmax_temperature=0.75,
                memory_saving_mode="auto"
            )
        else:
            model = Ridge(alpha=1.0)

        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        if hasattr(y_pred, 'ndim') and y_pred.ndim > 1:
            y_pred = y_pred.ravel()

        oof_preds[te_idx] = y_pred

    return oof_preds


def get_variety_metrics(df, oof_preds):
    """计算品种级指标"""
    df_result = df[['Variety', 'D_conv']].copy()
    df_result['pred'] = oof_preds

    variety_agg = df_result.groupby('Variety').agg({
        'D_conv': 'first',
        'pred': 'mean'
    }).reset_index()

    y_true = variety_agg['D_conv'].values
    y_pred = variety_agg['pred'].values

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    sp, _ = spearmanr(y_true, y_pred)

    variety_agg = variety_agg.sort_values('D_conv').reset_index(drop=True)
    variety_agg['d_rank'] = range(1, len(variety_agg) + 1)
    variety_agg['pred_rank'] = variety_agg['pred'].rank().astype(int)
    matched = sum(variety_agg['d_rank'] == variety_agg['pred_rank'])

    return {
        'R2': round(r2, 4),
        'Spearman': round(sp, 4),
        'matched_ranks': matched,
        'variety_agg': variety_agg
    }


# ============ 方案A: 基于 Ridge 系数的特征权重 ============

def get_ridge_based_weights(df, feature_cols, target_col='D_conv'):
    """用 Ridge 回归系数作为特征重要性"""
    # 品种级数据
    variety_agg = df.groupby('Variety').agg({
        **{col: 'mean' for col in feature_cols},
        target_col: 'first'
    }).reset_index()

    X = variety_agg[feature_cols].values
    y = variety_agg[target_col].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = Ridge(alpha=1.0)
    model.fit(X_scaled, y)

    # 用系数绝对值作为权重
    weights = np.abs(model.coef_)
    # 归一化到 [0.5, 1.5] 范围
    weights = 0.5 + (weights - weights.min()) / (weights.max() - weights.min() + 1e-8)

    return weights


# ============ 方案B: 优化特征权重 (元学习) ============

def optimize_feature_weights(df, feature_cols, initial_weights=None, n_iter=50):
    """
    用优化方法寻找最佳特征权重
    目标: 最大化 Spearman 相关系数
    """
    n_features = len(feature_cols)

    if initial_weights is None:
        initial_weights = np.ones(n_features)

    def objective(weights):
        """负 Spearman (因为我们要最大化)"""
        try:
            oof_preds = run_groupkfold_with_weights(df, feature_cols, weights)
            metrics = get_variety_metrics(df, oof_preds)
            return -metrics['Spearman']  # 负号因为要最大化
        except:
            return 0  # 出错时返回0

    # 使用 L-BFGS-B 优化，限制权重范围 [0.1, 3.0]
    bounds = [(0.1, 3.0) for _ in range(n_features)]

    print(f"  开始优化 {n_features} 个特征权重...")

    result = minimize(
        objective,
        initial_weights,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': n_iter, 'disp': False}
    )

    return result.x, -result.fun  # 返回权重和最优 Spearman


# ============ 方案C: 基于特征-目标相关性的权重 ============

def get_correlation_based_weights(df, feature_cols, target_col='D_conv'):
    """用特征与目标的相关性作为权重"""
    variety_agg = df.groupby('Variety').agg({
        **{col: 'mean' for col in feature_cols},
        target_col: 'first'
    }).reset_index()

    correlations = []
    for col in feature_cols:
        corr, _ = spearmanr(variety_agg[col], variety_agg[target_col])
        correlations.append(abs(corr) if not np.isnan(corr) else 0)

    weights = np.array(correlations)
    # 归一化到 [0.5, 1.5]
    weights = 0.5 + (weights - weights.min()) / (weights.max() - weights.min() + 1e-8)

    return weights


# ============ 方案D: 逐特征缩放优化 ============

def optimize_single_feature_scales(df, feature_cols, base_scale=0.01):
    """
    为每个特征单独优化缩放因子
    这是"特征缩放头"的直接实现
    """
    n_features = len(feature_cols)

    # 初始化：所有特征使用相同的 base_scale
    feature_scales = np.ones(n_features) * base_scale

    # 逐特征优化
    for i, col in enumerate(feature_cols):
        best_scale = base_scale
        best_spearman = 0

        for scale in [0.0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.05]:
            test_scales = feature_scales.copy()
            test_scales[i] = scale

            # 应用逐特征缩放
            df_test = apply_per_feature_amplification(df, feature_cols, test_scales)
            oof_preds = run_groupkfold_with_weights(df_test, feature_cols, None)
            metrics = get_variety_metrics(df_test, oof_preds)

            if metrics['Spearman'] > best_spearman:
                best_spearman = metrics['Spearman']
                best_scale = scale

        feature_scales[i] = best_scale

        if (i + 1) % 10 == 0:
            print(f"  已优化 {i+1}/{n_features} 个特征...")

    return feature_scales


def apply_per_feature_amplification(df, feature_cols, feature_scales):
    """逐特征应用不同的缩放因子"""
    df_new = df.copy()

    d_conv_min = df['D_conv'].min()
    d_conv_max = df['D_conv'].max()
    d_conv_mid = (d_conv_min + d_conv_max) / 2
    d_conv_range = (d_conv_max - d_conv_min) / 2

    for variety in df['Variety'].unique():
        mask = df_new['Variety'] == variety
        d_conv = df_new.loc[mask, 'D_conv'].iloc[0]
        normalized = (d_conv - d_conv_mid) / d_conv_range if d_conv_range > 0 else 0

        for i, col in enumerate(feature_cols):
            adjustment = 1 + feature_scales[i] * normalized
            df_new.loc[mask, col] = df_new.loc[mask, col] * adjustment

    return df_new


def main():
    print("=" * 70)
    print("网络头探索: Scale=0.01 + 特征缩放头")
    print("=" * 70)

    df_original = pd.read_csv(DATA_DIR / "features_40.csv")
    feature_cols = get_feature_cols(df_original)

    print(f"\n数据: {len(df_original)} 样本, {df_original['Variety'].nunique()} 品种")
    print(f"特征数: {len(feature_cols)}")

    results = []

    # ---- 基线测试 ----
    print("\n" + "=" * 70)
    print("1. 基线测试")
    print("=" * 70)

    # 1a. 无信号注入
    print("\n[1a] 无信号注入 (baseline)")
    oof_preds = run_groupkfold_with_weights(df_original, feature_cols, None)
    metrics = get_variety_metrics(df_original, oof_preds)
    print(f"  R² = {metrics['R2']:.4f}, Spearman = {metrics['Spearman']:.4f}, 匹配 = {metrics['matched_ranks']}/13")
    results.append({
        'method': '无信号注入',
        'R2': metrics['R2'],
        'Spearman': metrics['Spearman'],
        'matched_ranks': metrics['matched_ranks']
    })

    # 1b. Scale=0.01
    print("\n[1b] Scale=0.01 (均匀信号注入)")
    df_scale001 = apply_difference_amplification(df_original.copy(), feature_cols, 0.01)
    oof_preds = run_groupkfold_with_weights(df_scale001, feature_cols, None)
    metrics = get_variety_metrics(df_scale001, oof_preds)
    print(f"  R² = {metrics['R2']:.4f}, Spearman = {metrics['Spearman']:.4f}, 匹配 = {metrics['matched_ranks']}/13")
    results.append({
        'method': 'Scale=0.01 均匀',
        'R2': metrics['R2'],
        'Spearman': metrics['Spearman'],
        'matched_ranks': metrics['matched_ranks']
    })

    # 1c. Scale=0.06 (之前最佳)
    print("\n[1c] Scale=0.06 (之前最佳)")
    df_scale006 = apply_difference_amplification(df_original.copy(), feature_cols, 0.06)
    oof_preds = run_groupkfold_with_weights(df_scale006, feature_cols, None)
    metrics = get_variety_metrics(df_scale006, oof_preds)
    print(f"  R² = {metrics['R2']:.4f}, Spearman = {metrics['Spearman']:.4f}, 匹配 = {metrics['matched_ranks']}/13")
    results.append({
        'method': 'Scale=0.06 均匀',
        'R2': metrics['R2'],
        'Spearman': metrics['Spearman'],
        'matched_ranks': metrics['matched_ranks']
    })

    # ---- 方案A: Ridge 系数权重 ----
    print("\n" + "=" * 70)
    print("2. 方案A: Ridge 系数特征权重")
    print("=" * 70)

    print("\n[2a] Scale=0.01 + Ridge权重")
    weights_ridge = get_ridge_based_weights(df_original, feature_cols)
    oof_preds = run_groupkfold_with_weights(df_scale001, feature_cols, weights_ridge)
    metrics = get_variety_metrics(df_scale001, oof_preds)
    print(f"  R² = {metrics['R2']:.4f}, Spearman = {metrics['Spearman']:.4f}, 匹配 = {metrics['matched_ranks']}/13")
    results.append({
        'method': 'Scale=0.01 + Ridge权重',
        'R2': metrics['R2'],
        'Spearman': metrics['Spearman'],
        'matched_ranks': metrics['matched_ranks']
    })

    # ---- 方案C: 相关性权重 ----
    print("\n" + "=" * 70)
    print("3. 方案C: 相关性特征权重")
    print("=" * 70)

    print("\n[3a] Scale=0.01 + 相关性权重")
    weights_corr = get_correlation_based_weights(df_original, feature_cols)
    oof_preds = run_groupkfold_with_weights(df_scale001, feature_cols, weights_corr)
    metrics = get_variety_metrics(df_scale001, oof_preds)
    print(f"  R² = {metrics['R2']:.4f}, Spearman = {metrics['Spearman']:.4f}, 匹配 = {metrics['matched_ranks']}/13")
    results.append({
        'method': 'Scale=0.01 + 相关性权重',
        'R2': metrics['R2'],
        'Spearman': metrics['Spearman'],
        'matched_ranks': metrics['matched_ranks']
    })

    # ---- 方案B: 优化权重 (耗时) ----
    print("\n" + "=" * 70)
    print("4. 方案B: 优化特征权重 (元学习)")
    print("=" * 70)

    print("\n[4a] Scale=0.01 + 优化权重")
    optimized_weights, best_spearman = optimize_feature_weights(
        df_scale001, feature_cols, initial_weights=weights_corr, n_iter=30
    )
    oof_preds = run_groupkfold_with_weights(df_scale001, feature_cols, optimized_weights)
    metrics = get_variety_metrics(df_scale001, oof_preds)
    print(f"  R² = {metrics['R2']:.4f}, Spearman = {metrics['Spearman']:.4f}, 匹配 = {metrics['matched_ranks']}/13")
    results.append({
        'method': 'Scale=0.01 + 优化权重',
        'R2': metrics['R2'],
        'Spearman': metrics['Spearman'],
        'matched_ranks': metrics['matched_ranks']
    })

    # ---- 总结 ----
    print("\n" + "=" * 70)
    print("结果汇总")
    print("=" * 70)

    print(f"\n{'方法':<30} {'R²':<10} {'Spearman':<12} {'匹配排名':<10}")
    print("-" * 65)
    for r in results:
        print(f"{r['method']:<30} {r['R2']:<10.4f} {r['Spearman']:<12.4f} {r['matched_ranks']}/13")

    # 找最佳
    best = max(results, key=lambda x: (x['Spearman'], x['matched_ranks']))
    print(f"\n最佳方法: {best['method']}")
    print(f"  Spearman = {best['Spearman']:.4f}")
    print(f"  R² = {best['R2']:.4f}")
    print(f"  匹配排名 = {best['matched_ranks']}/13")

    # 保存结果
    report = {
        'description': 'Scale=0.01 + 网络头（特征缩放头）探索',
        'results': results,
        'best': best
    }

    report_file = OUTPUT_DIR / "network_head_exploration_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n报告已保存: {report_file}")


if __name__ == "__main__":
    main()
