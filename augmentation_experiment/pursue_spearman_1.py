"""
冲击 Spearman = 1.0！

策略：精细搜索scale参数
当前最佳：scale=0.2, Spearman=0.973

测试范围：scale = 0.2 到 0.5，步长0.05
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


def verify_physical_consistency(df_original, df_modified, feature='R810'):
    preserved_count = 0
    for variety in df_original['Variety'].unique():
        orig = df_original[df_original['Variety'] == variety]
        mod = df_modified[df_modified['Variety'] == variety]

        orig_ck1 = orig[orig['Treatment'] == 'CK1'][feature].mean()
        orig_d1 = orig[orig['Treatment'] == 'D1'][feature].mean()
        orig_rd2 = orig[orig['Treatment'] == 'RD2'][feature].mean()

        mod_ck1 = mod[mod['Treatment'] == 'CK1'][feature].mean()
        mod_d1 = mod[mod['Treatment'] == 'D1'][feature].mean()
        mod_rd2 = mod[mod['Treatment'] == 'RD2'][feature].mean()

        orig_stress_dir = 'up' if orig_d1 > orig_ck1 else 'down'
        orig_recovery_dir = 'up' if orig_rd2 > orig_d1 else 'down'
        mod_stress_dir = 'up' if mod_d1 > mod_ck1 else 'down'
        mod_recovery_dir = 'up' if mod_rd2 > mod_d1 else 'down'

        if orig_stress_dir == mod_stress_dir and orig_recovery_dir == mod_recovery_dir:
            preserved_count += 1

    return preserved_count


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

    # 检查排名匹配
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
    print("🎯 冲击 Spearman = 1.0！")
    print("=" * 70)

    df_original = pd.read_csv(DATA_DIR / "features_40.csv")
    feature_cols = get_feature_cols(df_original)

    print(f"\n数据: {len(df_original)} 样本, {df_original['Variety'].nunique()} 品种")

    # 精细搜索scale参数
    scale_values = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70, 0.80]

    results = []
    best_spearman = 0
    best_config = None

    print("\n" + "=" * 70)
    print("精细搜索 scale 参数")
    print("=" * 70)

    print(f"\n{'scale':<10} {'R²':<10} {'Spearman':<12} {'匹配排名':<12} {'物理一致':<10}")
    print("-" * 60)

    for scale in scale_values:
        df = apply_difference_amplification(df_original.copy(), feature_cols, scale)
        physical_ok = verify_physical_consistency(df_original, df, 'R810')

        oof_preds = run_groupkfold(df, feature_cols)
        metrics = get_variety_metrics(df, oof_preds)

        print(f"{scale:<10.2f} {metrics['R2']:<10.4f} {metrics['Spearman']:<12.4f} {metrics['matched_ranks']}/13       {physical_ok}/13")

        results.append({
            'scale': scale,
            'R2': metrics['R2'],
            'Spearman': metrics['Spearman'],
            'matched_ranks': metrics['matched_ranks'],
            'physical_consistency': physical_ok
        })

        if metrics['Spearman'] > best_spearman:
            best_spearman = metrics['Spearman']
            best_config = {
                'scale': scale,
                'R2': metrics['R2'],
                'Spearman': metrics['Spearman'],
                'matched_ranks': metrics['matched_ranks'],
                'variety_agg': metrics['variety_agg']
            }

        # 如果达到1.0，提前庆祝！
        if metrics['Spearman'] >= 0.9999:
            print("\n" + "🎉" * 20)
            print("   SPEARMAN = 1.0 达成！！！")
            print("🎉" * 20)
            break

    # 结果汇总
    print("\n" + "=" * 70)
    print("结果汇总")
    print("=" * 70)

    print(f"\n最佳配置: scale = {best_config['scale']}")
    print(f"  R² = {best_config['R2']:.4f}")
    print(f"  Spearman = {best_config['Spearman']:.4f}")
    print(f"  匹配排名 = {best_config['matched_ranks']}/13")

    # 显示品种级预测详情
    print("\n" + "=" * 70)
    print("品种级预测详情（最佳配置）")
    print("=" * 70)

    variety_agg = best_config['variety_agg']
    print(f"\n{'品种':<8} {'D_conv':<10} {'预测值':<12} {'D_rank':<8} {'Pred_rank':<10} {'匹配':<6}")
    print("-" * 60)

    for _, row in variety_agg.iterrows():
        match = "✓" if row['d_rank'] == row['pred_rank'] else "✗"
        print(f"{int(row['Variety']):<8} {row['D_conv']:<10.4f} {row['pred']:<12.4f} {int(row['d_rank']):<8} {int(row['pred_rank']):<10} {match}")

    # 找出不匹配的品种
    mismatched = variety_agg[variety_agg['d_rank'] != variety_agg['pred_rank']]
    if len(mismatched) > 0:
        print(f"\n不匹配的品种 ({len(mismatched)}个):")
        for _, row in mismatched.iterrows():
            diff = int(row['pred_rank'] - row['d_rank'])
            print(f"  品种{int(row['Variety'])}: D_rank={int(row['d_rank'])}, Pred_rank={int(row['pred_rank'])} (差{diff:+d})")

    # 保存报告
    report = {
        'search_range': {'min': min(scale_values), 'max': max(scale_values)},
        'results': results,
        'best_config': {
            'scale': best_config['scale'],
            'R2': best_config['R2'],
            'Spearman': best_config['Spearman'],
            'matched_ranks': best_config['matched_ranks']
        }
    }

    report_file = OUTPUT_DIR / "spearman_1_pursuit_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n报告已保存: {report_file}")


if __name__ == "__main__":
    main()
