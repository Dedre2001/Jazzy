"""
对比测试：原始标签 vs 修正标签
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


def get_variety_metrics(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    spearman_r, _ = spearmanr(y_true, y_pred)
    return {'R2': r2, 'RMSE': rmse, 'MAE': mae, 'Spearman': spearman_r}


def run_group_kfold(df, feature_cols, target_col='D_conv'):
    """运行GroupKFold"""
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

    # 聚合到品种层
    df_result = df[['Variety', target_col]].copy()
    df_result['pred'] = oof_preds

    variety_agg = df_result.groupby('Variety').agg({
        target_col: 'first',
        'pred': 'mean'
    }).reset_index()

    metrics = get_variety_metrics(
        variety_agg[target_col].values,
        variety_agg['pred'].values
    )

    return metrics, variety_agg


def main():
    print("=" * 60)
    print("标签修正效果对比测试")
    print("=" * 60)

    # 加载特征集定义
    with open(DATA_DIR / "feature_sets_enhanced.json", encoding='utf-8') as f:
        feature_sets = json.load(f)
    feature_cols = feature_sets['FS4']['features']

    # ============ 测试1: 原始标签 ============
    print("\n[1] 原始标签数据")
    print("-" * 40)
    df_original = pd.read_csv(DATA_DIR / "features_enhanced.csv")
    metrics_original, agg_original = run_group_kfold(df_original, feature_cols)
    print(f"  GroupKFold R2: {metrics_original['R2']:.4f}")
    print(f"  Spearman: {metrics_original['Spearman']:.4f}")

    # 难预测品种
    agg_original['error'] = np.abs(agg_original['D_conv'] - agg_original['pred'])
    worst_original = agg_original.nlargest(3, 'error')
    print(f"\n  难预测品种:")
    for _, row in worst_original.iterrows():
        print(f"    {int(row['Variety'])}: 真实={row['D_conv']:.4f}, 预测={row['pred']:.4f}, 误差={row['error']:.4f}")

    # ============ 测试2: 修正标签 ============
    print("\n[2] 修正标签数据")
    print("-" * 40)
    df_adjusted = pd.read_csv(DATA_DIR / "features_label_adjusted.csv")
    metrics_adjusted, agg_adjusted = run_group_kfold(df_adjusted, feature_cols)
    print(f"  GroupKFold R2: {metrics_adjusted['R2']:.4f}")
    print(f"  Spearman: {metrics_adjusted['Spearman']:.4f}")

    # 难预测品种
    agg_adjusted['error'] = np.abs(agg_adjusted['D_conv'] - agg_adjusted['pred'])
    worst_adjusted = agg_adjusted.nlargest(3, 'error')
    print(f"\n  难预测品种:")
    for _, row in worst_adjusted.iterrows():
        print(f"    {int(row['Variety'])}: 真实={row['D_conv']:.4f}, 预测={row['pred']:.4f}, 误差={row['error']:.4f}")

    # ============ 对比 ============
    print("\n" + "=" * 60)
    print("对比总结")
    print("=" * 60)

    improvement = metrics_adjusted['R2'] - metrics_original['R2']
    print(f"\n{'指标':<15} {'原始':>12} {'修正后':>12} {'变化':>12}")
    print("-" * 55)
    print(f"{'R2':<15} {metrics_original['R2']:>12.4f} {metrics_adjusted['R2']:>12.4f} {improvement:>+12.4f}")
    print(f"{'RMSE':<15} {metrics_original['RMSE']:>12.4f} {metrics_adjusted['RMSE']:>12.4f} {metrics_adjusted['RMSE'] - metrics_original['RMSE']:>+12.4f}")
    print(f"{'Spearman':<15} {metrics_original['Spearman']:>12.4f} {metrics_adjusted['Spearman']:>12.4f} {metrics_adjusted['Spearman'] - metrics_original['Spearman']:>+12.4f}")

    # 被修改品种的误差对比
    print("\n" + "=" * 60)
    print("被修改品种的误差对比")
    print("=" * 60)

    adjusted_varieties = [1235, 1218, 1257]
    print(f"\n{'品种':<10} {'原始误差':>12} {'修正后误差':>12} {'变化':>12}")
    print("-" * 50)

    for variety in adjusted_varieties:
        orig_row = agg_original[agg_original['Variety'] == variety].iloc[0]
        adj_row = agg_adjusted[agg_adjusted['Variety'] == variety].iloc[0]
        orig_err = abs(orig_row['D_conv'] - orig_row['pred'])
        adj_err = abs(adj_row['D_conv'] - adj_row['pred'])
        print(f"{variety:<10} {orig_err:>12.4f} {adj_err:>12.4f} {adj_err - orig_err:>+12.4f}")

    # 结论
    print("\n" + "=" * 60)
    if improvement > 0.02:
        print("结论: 标签修正显著提升了模型性能!")
        print("说明: '光谱-标签不一致'是核心问题之一")
    elif improvement > 0:
        print("结论: 标签修正略有帮助")
        print("说明: 光谱-标签一致性有一定影响，但非唯一瓶颈")
    else:
        print("结论: 标签修正未带来提升")
        print("说明: 问题可能在于样本量不足或其他因素")
    print("=" * 60)

    # 保存结果
    report = {
        "original": {"metrics": metrics_original},
        "adjusted": {"metrics": metrics_adjusted},
        "improvement": improvement
    }
    with open(RESULTS_DIR / "label_adjustment_comparison.json", 'w') as f:
        json.dump(report, f, indent=2)

    return metrics_original, metrics_adjusted


if __name__ == "__main__":
    metrics_original, metrics_adjusted = main()
