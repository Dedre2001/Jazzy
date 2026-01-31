"""
查看 Scale=0.06 时哪些品种排名不匹配
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

try:
    from tabpfn import TabPFNRegressor
    TABPFN_AVAILABLE = True
except ImportError:
    TABPFN_AVAILABLE = False
    from sklearn.linear_model import Ridge

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data" / "processed"

RANDOM_STATE = 42


def get_feature_cols(df):
    exclude = ['Sample_ID', 'Treatment', 'Variety', 'Sample', 'D_conv',
               'D_stress', 'D_recovery', 'Category', 'Rank']
    return [c for c in df.columns if c not in exclude and not c.startswith('Trt_')]


def apply_difference_amplification(df, feature_cols, scale_factor):
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


def main():
    print("=" * 70)
    print("Scale = 0.06 品种级预测详情")
    print("=" * 70)

    df_original = pd.read_csv(DATA_DIR / "features_40.csv")
    feature_cols = get_feature_cols(df_original)

    scale = 0.06
    df = apply_difference_amplification(df_original.copy(), feature_cols, scale)

    print(f"\nScale = {scale} (变化幅度 ±{scale*100:.0f}%)")

    oof_preds = run_groupkfold(df, feature_cols)

    # 品种级汇总
    df_result = df[['Variety', 'D_conv']].copy()
    df_result['pred'] = oof_preds

    variety_agg = df_result.groupby('Variety').agg({
        'D_conv': 'first',
        'pred': 'mean'
    }).reset_index()

    variety_agg = variety_agg.sort_values('D_conv').reset_index(drop=True)
    variety_agg['d_rank'] = range(1, len(variety_agg) + 1)
    variety_agg['pred_rank'] = variety_agg['pred'].rank().astype(int)

    # 计算指标
    sp, _ = spearmanr(variety_agg['D_conv'], variety_agg['pred'])
    matched = sum(variety_agg['d_rank'] == variety_agg['pred_rank'])

    print(f"\nSpearman = {sp:.4f}")
    print(f"匹配排名 = {matched}/13")

    # 详细列表
    print("\n" + "=" * 70)
    print("品种级预测详情")
    print("=" * 70)

    print(f"\n{'品种':<8} {'D_conv':<10} {'预测值':<12} {'D_rank':<8} {'Pred_rank':<10} {'状态':<8}")
    print("-" * 60)

    for _, row in variety_agg.iterrows():
        match = "OK" if row['d_rank'] == row['pred_rank'] else "X"
        print(f"{int(row['Variety']):<8} {row['D_conv']:<10.4f} {row['pred']:<12.4f} {int(row['d_rank']):<8} {int(row['pred_rank']):<10} {match}")

    # 不匹配的品种
    mismatched = variety_agg[variety_agg['d_rank'] != variety_agg['pred_rank']]
    if len(mismatched) > 0:
        print(f"\n不匹配的品种 ({len(mismatched)}个):")
        for _, row in mismatched.iterrows():
            diff = int(row['pred_rank'] - row['d_rank'])
            print(f"  品种{int(row['Variety'])}: D_rank={int(row['d_rank'])}, Pred_rank={int(row['pred_rank'])} (差{diff:+d})")


if __name__ == "__main__":
    main()
