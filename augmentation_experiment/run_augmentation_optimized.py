"""
数据增强优化版：基于实验1结果，专注于有效方法的组合优化

有效方法：
  - 处理轨迹插值 (R² +0.070)
  - 品种内Mixup (R² +0.044)

测试组合：
  1. 轨迹插值（密集版，9个中间态）
  2. Mixup + 轨迹插值
  3. 轨迹插值（跨处理，CK1-RD2直接插值）
"""

import os
import json
import warnings
from pathlib import Path
from itertools import combinations
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
DATA_DIR = BASE_DIR.parent / "groupkfold_experiment" / "data"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


# ========== 优化的增强方法 ==========

def augment_mixup(df, feature_cols, n_per_pair=1):
    """品种内Mixup（同处理的重复之间）"""
    new_rows = []
    for (variety, treatment), group in df.groupby(['Variety', 'Treatment']):
        if len(group) < 2:
            continue
        for (i, row_i), (j, row_j) in combinations(group.iterrows(), 2):
            for _ in range(n_per_pair):
                alpha = np.random.uniform(0.3, 0.7)
                new_row = row_i.copy()
                new_row[feature_cols] = alpha * row_i[feature_cols] + (1 - alpha) * row_j[feature_cols]
                new_row['Sample_ID'] = f"MIX_{variety}_{treatment}_{i}_{j}"
                new_rows.append(new_row)
    if not new_rows:
        return df
    return pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)


def augment_trajectory_basic(df, feature_cols, alphas=[0.25, 0.5, 0.75]):
    """处理轨迹插值（基础版：CK1-D1, D1-RD2）"""
    pairs = [('CK1', 'D1'), ('D1', 'RD2')]
    new_rows = []

    for variety, v_group in df.groupby('Variety'):
        for trt_a, trt_b in pairs:
            grp_a = v_group[v_group['Treatment'] == trt_a]
            grp_b = v_group[v_group['Treatment'] == trt_b]
            if len(grp_a) == 0 or len(grp_b) == 0:
                continue
            mean_a = grp_a[feature_cols].mean()
            mean_b = grp_b[feature_cols].mean()

            for alpha in alphas:
                template = grp_a.iloc[0].copy()
                template[feature_cols] = (1 - alpha) * mean_a + alpha * mean_b
                template['Sample_ID'] = f"TRAJ_{variety}_{trt_a}_{trt_b}_{alpha}"
                template['Treatment'] = f"{trt_a}_{trt_b}"
                new_rows.append(template)

    if not new_rows:
        return df
    return pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)


def augment_trajectory_dense(df, feature_cols):
    """处理轨迹插值（密集版：9个中间态）"""
    alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    return augment_trajectory_basic(df, feature_cols, alphas=alphas)


def augment_trajectory_full(df, feature_cols):
    """处理轨迹插值（完整版：包括CK1-RD2直接插值）"""
    pairs = [('CK1', 'D1'), ('D1', 'RD2'), ('CK1', 'RD2')]
    alphas = [0.25, 0.5, 0.75]
    new_rows = []

    for variety, v_group in df.groupby('Variety'):
        for trt_a, trt_b in pairs:
            grp_a = v_group[v_group['Treatment'] == trt_a]
            grp_b = v_group[v_group['Treatment'] == trt_b]
            if len(grp_a) == 0 or len(grp_b) == 0:
                continue
            mean_a = grp_a[feature_cols].mean()
            mean_b = grp_b[feature_cols].mean()

            for alpha in alphas:
                template = grp_a.iloc[0].copy()
                template[feature_cols] = (1 - alpha) * mean_a + alpha * mean_b
                template['Sample_ID'] = f"TRAJ_{variety}_{trt_a}_{trt_b}_{alpha}"
                template['Treatment'] = f"{trt_a}_{trt_b}"
                new_rows.append(template)

    if not new_rows:
        return df
    return pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)


def augment_mixup_trajectory(df, feature_cols):
    """组合：先Mixup，再轨迹插值"""
    df_mix = augment_mixup(df, feature_cols, n_per_pair=1)
    df_combined = augment_trajectory_basic(df_mix, feature_cols, alphas=[0.25, 0.5, 0.75])
    return df_combined


def augment_trajectory_mixup(df, feature_cols):
    """组合：先轨迹插值，再Mixup"""
    df_traj = augment_trajectory_basic(df, feature_cols, alphas=[0.25, 0.5, 0.75])
    df_combined = augment_mixup(df_traj, feature_cols, n_per_pair=1)
    return df_combined


# ========== 评估函数 ==========

def run_group_kfold(df, feature_cols, target_col='D_conv'):
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
    sp, _ = spearmanr(y_true, y_pred)
    return {'R2': round(r2, 4), 'RMSE': round(rmse, 4), 'Spearman': round(sp, 4)}


def get_variety_errors(variety_agg):
    hard_varieties = [1252, 1235, 1099]
    errors = {}
    for v in hard_varieties:
        row = variety_agg[variety_agg['Variety'] == v]
        if len(row) > 0:
            errors[str(v)] = round(abs(row['D_conv'].values[0] - row['pred'].values[0]), 4)
    return errors


# ========== 主流程 ==========

def main():
    print("=" * 70)
    print("数据增强优化版实验")
    print("=" * 70)

    with open(DATA_DIR / "feature_sets_enhanced.json", encoding='utf-8') as f:
        feature_sets = json.load(f)
    feature_cols = feature_sets['FS4']['features']

    df_original = pd.read_csv(DATA_DIR / "features_enhanced.csv")
    print(f"原始数据: {len(df_original)} 样本, {len(df_original['Variety'].unique())} 品种")

    methods = {
        'baseline': ('无增强（基线）', lambda df: df),
        'traj_basic': ('轨迹插值(3点)', lambda df: augment_trajectory_basic(df, feature_cols)),
        'traj_dense': ('轨迹插值(9点)', lambda df: augment_trajectory_dense(df, feature_cols)),
        'traj_full': ('轨迹插值+跨处理', lambda df: augment_trajectory_full(df, feature_cols)),
        'mixup': ('Mixup', lambda df: augment_mixup(df, feature_cols)),
        'mixup_traj': ('Mixup→轨迹', lambda df: augment_mixup_trajectory(df, feature_cols)),
        'traj_mixup': ('轨迹→Mixup', lambda df: augment_trajectory_mixup(df, feature_cols)),
    }

    results = {}

    for method_key, (method_name, aug_fn) in methods.items():
        print(f"\n--- {method_name} ---")

        np.random.seed(RANDOM_STATE)
        df_aug = aug_fn(df_original.copy())
        n_samples = len(df_aug)
        n_per_variety = df_aug.groupby('Variety').size().mean()
        print(f"  样本量: {n_samples} (每品种平均 {n_per_variety:.0f})")

        variety_agg = run_group_kfold(df_aug, feature_cols)
        metrics = get_metrics(variety_agg['D_conv'].values, variety_agg['pred'].values)
        var_errors = get_variety_errors(variety_agg)

        results[method_key] = {
            'name': method_name,
            'n_samples': n_samples,
            'n_per_variety': round(n_per_variety, 1),
            'metrics': metrics,
            'hard_variety_errors': var_errors
        }

        print(f"  R² = {metrics['R2']:.4f}  RMSE = {metrics['RMSE']:.4f}  Spearman = {metrics['Spearman']:.4f}")
        print(f"  难预测品种: 1252={var_errors.get('1252', 'N/A')}, 1235={var_errors.get('1235', 'N/A')}, 1099={var_errors.get('1099', 'N/A')}")

    # 汇总
    print("\n" + "=" * 70)
    print("汇总对比（按R²排序）")
    print("=" * 70)

    sorted_results = sorted(results.items(), key=lambda x: x[1]['metrics']['R2'], reverse=True)

    print(f"\n{'方法':<18} {'样本':>5} {'R²':>8} {'ΔR²':>8} {'RMSE':>7} {'Spearman':>9}")
    print("-" * 60)

    baseline_r2 = results['baseline']['metrics']['R2']
    for key, res in sorted_results:
        m = res['metrics']
        delta = m['R2'] - baseline_r2
        delta_str = f"{delta:+.4f}" if key != 'baseline' else "-"
        print(f"{res['name']:<18} {res['n_samples']:>5} {m['R2']:>8.4f} {delta_str:>8} {m['RMSE']:>7.4f} {m['Spearman']:>9.4f}")

    # 难预测品种对比
    print("\n" + "-" * 60)
    print("难预测品种误差对比（越小越好）")
    print("-" * 60)
    print(f"{'方法':<18} {'1252':>10} {'1235':>10} {'1099':>10} {'平均':>10}")
    print("-" * 60)

    for key, res in sorted_results:
        e = res['hard_variety_errors']
        e1252 = e.get('1252', 999)
        e1235 = e.get('1235', 999)
        e1099 = e.get('1099', 999)
        avg = (e1252 + e1235 + e1099) / 3
        print(f"{res['name']:<18} {e1252:>10.4f} {e1235:>10.4f} {e1099:>10.4f} {avg:>10.4f}")

    # 保存
    with open(RESULTS_DIR / "augmentation_optimized_report.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {RESULTS_DIR / 'augmentation_optimized_report.json'}")

    # 推荐
    best_key, best_res = sorted_results[0]
    if best_key != 'baseline':
        print(f"\n{'='*70}")
        print(f"推荐方法: {best_res['name']}")
        print(f"  R²: {baseline_r2:.4f} → {best_res['metrics']['R2']:.4f} ({best_res['metrics']['R2']-baseline_r2:+.4f})")
        print(f"  样本量: 117 → {best_res['n_samples']}")

    return results


if __name__ == "__main__":
    results = main()
