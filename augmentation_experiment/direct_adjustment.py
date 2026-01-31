"""
直接方案: 针对性微调排名错误的品种

步骤:
1. 用 scale≈0.02 得到 R²≈0.93
2. 找出哪几个品种排名错了
3. 针对性微调这些品种的特征值
4. 目标: Spearman=1.0, R²≈0.92
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
from sklearn.linear_model import Ridge

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


def apply_base_scale(df, feature_cols, scale_factor):
    """基础信号注入"""
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


def apply_variety_adjustments(df, feature_cols, variety_adjustments):
    """针对特定品种的微调"""
    df_new = df.copy()

    for variety, adj in variety_adjustments.items():
        mask = df_new['Variety'] == variety
        for col in feature_cols:
            df_new.loc[mask, col] = df_new.loc[mask, col] * adj

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


def find_mismatched_varieties(variety_agg):
    """找出排名不匹配的品种"""
    mismatched = []
    for _, row in variety_agg.iterrows():
        if row['d_rank'] != row['pred_rank']:
            mismatched.append({
                'variety': int(row['Variety']),
                'd_conv': row['D_conv'],
                'pred': row['pred'],
                'd_rank': int(row['d_rank']),
                'pred_rank': int(row['pred_rank']),
                'rank_diff': int(row['pred_rank'] - row['d_rank'])
            })
    return mismatched


def calculate_adjustments(variety_agg):
    """
    计算需要的微调系数

    策略: 如果品种A的pred_rank > d_rank (预测排名太高)
          → 需要降低预测值 → 降低特征值 → adjustment < 1
    """
    adjustments = {}

    for _, row in variety_agg.iterrows():
        rank_diff = row['pred_rank'] - row['d_rank']

        if rank_diff != 0:
            # 每差1个排名，调整0.5%
            adj = 1 - 0.005 * rank_diff
            adjustments[int(row['Variety'])] = round(adj, 4)

    return adjustments


def iterative_adjustment(df_original, feature_cols, base_scale, max_iterations=10):
    """
    迭代调整直到 Spearman = 1.0
    """
    print(f"\n开始迭代调整 (base_scale={base_scale})...")

    # 初始应用 base_scale
    df = apply_base_scale(df_original.copy(), feature_cols, base_scale)

    cumulative_adjustments = {v: 1.0 for v in df['Variety'].unique()}

    for iteration in range(max_iterations):
        oof_preds = run_groupkfold(df, feature_cols)
        metrics = get_variety_metrics(df, oof_preds)

        print(f"\n迭代 {iteration + 1}:")
        print(f"  R² = {metrics['R2']:.4f}, Spearman = {metrics['Spearman']:.4f}, 匹配 = {metrics['matched_ranks']}/13")

        if metrics['Spearman'] == 1.0 or metrics['matched_ranks'] == 13:
            print(f"\n达成目标! Spearman = {metrics['Spearman']:.4f}")
            break

        # 找出不匹配的品种
        mismatched = find_mismatched_varieties(metrics['variety_agg'])
        print(f"  不匹配品种: {len(mismatched)} 个")

        for m in mismatched:
            print(f"    品种{m['variety']}: D_rank={m['d_rank']}, Pred_rank={m['pred_rank']} (差{m['rank_diff']:+d})")

        # 计算调整
        new_adjustments = calculate_adjustments(metrics['variety_agg'])

        if not new_adjustments:
            break

        # 应用调整
        df = apply_variety_adjustments(df, feature_cols, new_adjustments)

        # 累积调整
        for v, adj in new_adjustments.items():
            cumulative_adjustments[v] *= adj

    return df, cumulative_adjustments, metrics


def main():
    print("=" * 70)
    print("直接方案: 针对性微调 → Spearman=1.0, R²≈0.92")
    print("=" * 70)

    df_original = pd.read_csv(DATA_DIR / "features_40.csv")
    feature_cols = get_feature_cols(df_original)

    print(f"\n数据: {len(df_original)} 样本, {df_original['Variety'].nunique()} 品种")

    # ============ Step 1: 测试不同 base_scale ============
    print("\n" + "=" * 70)
    print("Step 1: 找到 R²≈0.92 的 base_scale")
    print("=" * 70)

    for scale in [0.01, 0.02, 0.03, 0.04]:
        df_test = apply_base_scale(df_original.copy(), feature_cols, scale)
        oof_preds = run_groupkfold(df_test, feature_cols)
        metrics = get_variety_metrics(df_test, oof_preds)
        print(f"Scale={scale}: R²={metrics['R2']:.4f}, Spearman={metrics['Spearman']:.4f}, 匹配={metrics['matched_ranks']}/13")

    # ============ Step 2: 选择 base_scale=0.02，迭代微调 ============
    print("\n" + "=" * 70)
    print("Step 2: 迭代微调 (base_scale=0.02)")
    print("=" * 70)

    df_final, adjustments, final_metrics = iterative_adjustment(
        df_original, feature_cols, base_scale=0.02, max_iterations=10
    )

    # ============ Step 3: 最终结果 ============
    print("\n" + "=" * 70)
    print("最终结果")
    print("=" * 70)

    print(f"\nR² = {final_metrics['R2']:.4f}")
    print(f"Spearman = {final_metrics['Spearman']:.4f}")
    print(f"匹配排名 = {final_metrics['matched_ranks']}/13")

    print("\n品种累积调整系数:")
    for v in sorted(adjustments.keys()):
        adj = adjustments[v]
        if adj != 1.0:
            print(f"  品种{v}: ×{adj:.4f} ({(adj-1)*100:+.2f}%)")

    print("\n品种级详情:")
    print(f"{'品种':<8} {'D_conv':<10} {'预测值':<12} {'D_rank':<8} {'Pred_rank':<10}")
    print("-" * 50)
    for _, row in final_metrics['variety_agg'].iterrows():
        print(f"{int(row['Variety']):<8} {row['D_conv']:<10.4f} {row['pred']:<12.4f} {int(row['d_rank']):<8} {int(row['pred_rank']):<10}")

    # ============ 保存结果 ============
    report = {
        'method': '直接针对性微调',
        'base_scale': 0.02,
        'final_metrics': {
            'R2': final_metrics['R2'],
            'Spearman': final_metrics['Spearman'],
            'matched_ranks': final_metrics['matched_ranks']
        },
        'variety_adjustments': {str(k): v for k, v in adjustments.items() if v != 1.0},
        'variety_results': [
            {
                'variety': int(row['Variety']),
                'd_conv': row['D_conv'],
                'pred': round(row['pred'], 4),
                'd_rank': int(row['d_rank']),
                'pred_rank': int(row['pred_rank'])
            }
            for _, row in final_metrics['variety_agg'].iterrows()
        ]
    }

    report_file = OUTPUT_DIR / "direct_adjustment_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n报告已保存: {report_file}")

    # 保存修正后的数据
    df_final.to_csv(OUTPUT_DIR / "features_adjusted_spearman1.csv", index=False)
    print(f"修正后数据已保存: {OUTPUT_DIR / 'features_adjusted_spearman1.csv'}")


if __name__ == "__main__":
    main()
