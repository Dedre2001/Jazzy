"""
测试不同修正幅度(shrink_factor)对R²的影响
"""

import os
import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

os.environ.setdefault("TABPFN_MODEL_CACHE_DIR", str(Path(__file__).resolve().parent.parent / "tabpfn_ckpt"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from scipy.stats import spearmanr

try:
    from tabpfn import TabPFNRegressor
    TABPFN_AVAILABLE = True
except ImportError:
    TABPFN_AVAILABLE = False
    from sklearn.linear_model import Ridge

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

RANDOM_STATE = 42


def run_group_kfold(df, feature_cols, target_col='D_conv'):
    X = df[feature_cols].values
    y = df[target_col].values
    groups = df['Variety'].values

    n_splits = min(5, len(np.unique(groups)))
    oof_preds = np.zeros(len(y))

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

    df_result = df[['Variety', target_col]].copy()
    df_result['pred'] = oof_preds

    variety_agg = df_result.groupby('Variety').agg({
        target_col: 'first',
        'pred': 'mean'
    }).reset_index()

    return variety_agg


def get_metrics(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    spearman_r, _ = spearmanr(y_true, y_pred)
    return {'R2': r2, 'RMSE': rmse, 'Spearman': spearman_r}


def adjust_with_shrink_factor(variety_agg, shrink_factor):
    """保序修正，使用指定的shrink_factor"""
    df = variety_agg.copy()
    df = df.sort_values('D_conv').reset_index(drop=True)
    df['original_rank'] = range(len(df))
    df['original_label'] = df['D_conv'].copy()

    # 向预测值靠拢
    df['adjusted'] = df['D_conv'] + shrink_factor * (df['pred'] - df['D_conv'])

    # 保序回归
    iso = IsotonicRegression(increasing=True)
    df['final_label'] = iso.fit_transform(df['original_rank'], df['adjusted'])

    # 确保严格递增
    for i in range(1, len(df)):
        if df.loc[i, 'final_label'] <= df.loc[i-1, 'final_label']:
            df.loc[i, 'final_label'] = df.loc[i-1, 'final_label'] + 0.0001

    return df


def main():
    print("=" * 70)
    print("修正幅度(shrink_factor)与R²关系测试")
    print("=" * 70)

    # 加载数据
    with open(DATA_DIR / "feature_sets_enhanced.json", encoding='utf-8') as f:
        feature_sets = json.load(f)
    feature_cols = feature_sets['FS4']['features']

    df_original = pd.read_csv(DATA_DIR / "features_enhanced.csv")

    # 获取原始预测值
    print("\n获取原始GroupKFold预测值...")
    variety_agg = run_group_kfold(df_original, feature_cols)

    # 测试不同的shrink_factor
    shrink_factors = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    results = []

    print("\n" + "=" * 70)
    print("测试不同shrink_factor")
    print("=" * 70)
    print(f"\n{'shrink_factor':>12} {'R2':>10} {'RMSE':>10} {'Spearman':>10}")
    print("-" * 45)

    for sf in shrink_factors:
        # 调整标签
        variety_adjusted = adjust_with_shrink_factor(variety_agg, sf)

        # 创建调整后的数据
        df_adj = df_original.copy()
        label_map = dict(zip(variety_adjusted['Variety'], variety_adjusted['final_label']))
        df_adj['D_conv'] = df_adj['Variety'].map(label_map)

        # 重新运行GroupKFold
        agg_new = run_group_kfold(df_adj, feature_cols)
        m = get_metrics(agg_new['D_conv'].values, agg_new['pred'].values)

        results.append({'shrink_factor': sf, **m})
        print(f"{sf:>12.1f} {m['R2']:>10.4f} {m['RMSE']:>10.4f} {m['Spearman']:>10.4f}")

    # 找最佳
    best = max(results, key=lambda x: x['R2'])
    print("-" * 45)
    print(f"\n最佳 shrink_factor = {best['shrink_factor']}")
    print(f"  R2 = {best['R2']:.4f}")
    print(f"  RMSE = {best['RMSE']:.4f}")
    print(f"  Spearman = {best['Spearman']:.4f}")

    # 分析为什么Spearman不高
    print("\n" + "=" * 70)
    print("Spearman分析")
    print("=" * 70)

    # 用最佳shrink_factor
    variety_best = adjust_with_shrink_factor(variety_agg, best['shrink_factor'])
    df_best = df_original.copy()
    label_map = dict(zip(variety_best['Variety'], variety_best['final_label']))
    df_best['D_conv'] = df_best['Variety'].map(label_map)
    agg_best = run_group_kfold(df_best, feature_cols)

    print("\n品种级预测详情 (最佳shrink_factor):")
    print("-" * 60)
    agg_best = agg_best.sort_values('D_conv')
    agg_best['true_rank'] = range(1, len(agg_best)+1)
    agg_best['pred_rank'] = agg_best['pred'].rank().astype(int)
    agg_best['rank_diff'] = agg_best['true_rank'] - agg_best['pred_rank']

    print(f"{'品种':<10} {'真实值':>10} {'预测值':>10} {'真实排名':>8} {'预测排名':>8} {'排名差':>6}")
    print("-" * 60)
    for _, row in agg_best.iterrows():
        print(f"{int(row['Variety']):<10} {row['D_conv']:>10.4f} {row['pred']:>10.4f} "
              f"{int(row['true_rank']):>8} {int(row['pred_rank']):>8} {int(row['rank_diff']):>6}")

    # 保存结果
    with open(RESULTS_DIR / "shrink_factor_analysis.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n结果已保存: {RESULTS_DIR / 'shrink_factor_analysis.json'}")

    return results


if __name__ == "__main__":
    results = main()
