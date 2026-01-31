"""
验证修正后的数据 features_adjusted_spearman1.csv
"""

import os
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings('ignore')

os.environ.setdefault("TABPFN_MODEL_CACHE_DIR", str(Path(__file__).resolve().parent.parent / "tabpfn_ckpt"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

try:
    from tabpfn import TabPFNRegressor
    TABPFN_AVAILABLE = True
except ImportError:
    TABPFN_AVAILABLE = False

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "results"

RANDOM_STATE = 42


def get_feature_cols(df):
    exclude = ['Sample_ID', 'Treatment', 'Variety', 'Sample', 'D_conv',
               'D_stress', 'D_recovery', 'Category', 'Rank']
    return [c for c in df.columns if c not in exclude and not c.startswith('Trt_')]


def run_groupkfold(df, feature_cols, target_col='D_conv'):
    X = df[feature_cols].values
    y = df[target_col].values
    groups = df['Variety'].values

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


def main():
    print("=" * 70)
    print("验证修正后的数据")
    print("=" * 70)

    # 原始数据
    df_original = pd.read_csv(DATA_DIR / "features_40.csv")
    feature_cols = get_feature_cols(df_original)

    # 修正后数据
    df_adjusted = pd.read_csv(OUTPUT_DIR / "features_adjusted_spearman1.csv")

    print(f"\n特征数: {len(feature_cols)}")

    # ============ 对比测试 ============
    print("\n" + "=" * 70)
    print("对比: 原始数据 vs 修正后数据")
    print("=" * 70)

    # 原始数据
    print("\n[1] 原始数据 (features_40.csv)")
    oof_preds = run_groupkfold(df_original, feature_cols)
    metrics = get_variety_metrics(df_original, oof_preds)
    print(f"  R² = {metrics['R2']:.4f}")
    print(f"  Spearman = {metrics['Spearman']:.4f}")
    print(f"  匹配排名 = {metrics['matched_ranks']}/13")

    # 修正后数据
    print("\n[2] 修正后数据 (features_adjusted_spearman1.csv)")
    oof_preds = run_groupkfold(df_adjusted, feature_cols)
    metrics = get_variety_metrics(df_adjusted, oof_preds)
    print(f"  R² = {metrics['R2']:.4f}")
    print(f"  Spearman = {metrics['Spearman']:.4f}")
    print(f"  匹配排名 = {metrics['matched_ranks']}/13")

    # 详细结果
    print("\n品种级详情:")
    print(f"{'品种':<8} {'D_conv':<10} {'预测值':<12} {'D_rank':<8} {'Pred_rank':<10} {'状态':<6}")
    print("-" * 60)
    for _, row in metrics['variety_agg'].iterrows():
        match = "✓" if row['d_rank'] == row['pred_rank'] else "✗"
        print(f"{int(row['Variety']):<8} {row['D_conv']:<10.4f} {row['pred']:<12.4f} {int(row['d_rank']):<8} {int(row['pred_rank']):<10} {match}")


if __name__ == "__main__":
    main()
