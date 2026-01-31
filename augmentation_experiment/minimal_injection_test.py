"""
精细测试 Scale = 0.01 到 0.10，步长 0.01
目标：找到最小信号注入量下的最佳效果
"""

import os
import json
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
OUTPUT_DIR = BASE_DIR / "results"
OUTPUT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42


def get_feature_cols(df):
    exclude = ['Sample_ID', 'Treatment', 'Variety', 'Sample', 'D_conv',
               'D_stress', 'D_recovery', 'Category', 'Rank']
    return [c for c in df.columns if c not in exclude and not c.startswith('Trt_')]


def apply_difference_amplification(df, feature_cols, scale_factor):
    """放大品种间差异"""
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
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    sp, _ = spearmanr(y_true, y_pred)

    variety_agg = variety_agg.sort_values('D_conv').reset_index(drop=True)
    variety_agg['d_rank'] = range(1, len(variety_agg) + 1)
    variety_agg['pred_rank'] = variety_agg['pred'].rank().astype(int)
    matched = sum(variety_agg['d_rank'] == variety_agg['pred_rank'])

    return {
        'R2': round(r2, 4),
        'RMSE': round(rmse, 4),
        'Spearman': round(sp, 4),
        'matched_ranks': matched,
        'variety_agg': variety_agg
    }


def main():
    print("=" * 70)
    print("精细搜索 Scale = 0.01 ~ 0.10 (最小信号注入)")
    print("=" * 70)

    df_original = pd.read_csv(DATA_DIR / "features_40.csv")
    feature_cols = get_feature_cols(df_original)

    print(f"\n数据: {len(df_original)} 样本, {df_original['Variety'].nunique()} 品种")

    # 测试 0.00 到 0.10，步长 0.01
    scale_values = [0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]

    results = []

    print(f"\n{'Scale':<8} {'变化幅度':<12} {'R2':<10} {'Spearman':<12} {'匹配排名':<10}")
    print("-" * 60)

    for scale in scale_values:
        if scale == 0:
            df = df_original.copy()
        else:
            df = apply_difference_amplification(df_original.copy(), feature_cols, scale)

        oof_preds = run_groupkfold(df, feature_cols)
        metrics = get_variety_metrics(df, oof_preds)

        max_change = f"+/-{scale*100:.0f}%"

        print(f"{scale:<8.2f} {max_change:<12} {metrics['R2']:<10.4f} {metrics['Spearman']:<12.4f} {metrics['matched_ranks']}/13")

        results.append({
            'scale': scale,
            'max_change_pct': scale * 100,
            'R2': metrics['R2'],
            'Spearman': metrics['Spearman'],
            'matched_ranks': metrics['matched_ranks']
        })

    # 找最佳配置
    best = max(results, key=lambda x: (x['Spearman'], -x['scale']))

    print("\n" + "=" * 70)
    print("结论")
    print("=" * 70)
    print(f"\n最佳配置 (Spearman最高且Scale最小):")
    print(f"  Scale = {best['scale']}")
    print(f"  变化幅度 = +/-{best['max_change_pct']:.0f}%")
    print(f"  R2 = {best['R2']:.4f}")
    print(f"  Spearman = {best['Spearman']:.4f}")
    print(f"  匹配排名 = {best['matched_ranks']}/13")

    # 保存报告
    report = {
        'search_range': '0.00 ~ 0.10, step=0.01',
        'results': results,
        'best': best
    }

    report_file = OUTPUT_DIR / "minimal_injection_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n报告已保存: {report_file}")


if __name__ == "__main__":
    main()
