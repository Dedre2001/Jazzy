"""
对比测试：原始 vs 部分修正 vs 全部修正
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

    metrics = get_variety_metrics(
        variety_agg[target_col].values,
        variety_agg['pred'].values
    )

    return metrics, variety_agg


def main():
    print("=" * 60)
    print("标签修正效果全面对比")
    print("=" * 60)

    with open(DATA_DIR / "feature_sets_enhanced.json", encoding='utf-8') as f:
        feature_sets = json.load(f)
    feature_cols = feature_sets['FS4']['features']

    results = []

    # 1. 原始标签
    print("\n[1] 原始标签")
    df1 = pd.read_csv(DATA_DIR / "features_enhanced.csv")
    m1, agg1 = run_group_kfold(df1, feature_cols)
    results.append(("原始标签", m1))
    print(f"    R2={m1['R2']:.4f}, Spearman={m1['Spearman']:.4f}")

    # 2. 部分修正 (3个双极端品种)
    print("\n[2] 部分修正 (3个双极端品种)")
    df2 = pd.read_csv(DATA_DIR / "features_label_adjusted.csv")
    m2, agg2 = run_group_kfold(df2, feature_cols)
    results.append(("部分修正(3品种)", m2))
    print(f"    R2={m2['R2']:.4f}, Spearman={m2['Spearman']:.4f}")

    # 3. 全部修正 (7个极端品种)
    print("\n[3] 全部修正 (7个极端品种)")
    df3 = pd.read_csv(DATA_DIR / "features_label_full_adjusted.csv")
    m3, agg3 = run_group_kfold(df3, feature_cols)
    results.append(("全部修正(7品种)", m3))
    print(f"    R2={m3['R2']:.4f}, Spearman={m3['Spearman']:.4f}")

    # 汇总
    print("\n" + "=" * 60)
    print("结果汇总")
    print("=" * 60)
    print(f"\n{'方案':<20} {'R2':>10} {'RMSE':>10} {'Spearman':>10} {'vs原始':>10}")
    print("-" * 65)

    baseline_r2 = results[0][1]['R2']
    for name, m in results:
        delta = m['R2'] - baseline_r2
        print(f"{name:<20} {m['R2']:>10.4f} {m['RMSE']:>10.4f} {m['Spearman']:>10.4f} {delta:>+10.4f}")

    # 保存
    report = {r[0]: r[1] for r in results}
    with open(RESULTS_DIR / "full_comparison.json", 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n报告已保存: {RESULTS_DIR / 'full_comparison.json'}")


if __name__ == "__main__":
    main()
