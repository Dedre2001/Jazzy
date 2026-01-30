"""
数据增强实验：测试不同增强方法对GroupKFold R²的影响

增强在CV外部进行（GroupKFold按品种分组，同品种样本不会跨fold，无泄露风险）
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

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


# ========== 数据增强方法 ==========

def augment_mixup(df, feature_cols, n_per_pair=1):
    """方法A: 品种内光谱混合 (同品种同处理的重复之间线性插值)"""
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
    aug_df = pd.DataFrame(new_rows)
    return pd.concat([df, aug_df], ignore_index=True)


def augment_noise(df, feature_cols, n_copies=4, noise_scale=0.02):
    """方法B: 高斯噪声注入"""
    feature_stds = df[feature_cols].std()
    new_rows = []
    for _, row in df.iterrows():
        for k in range(n_copies):
            new_row = row.copy()
            noise = np.random.normal(0, noise_scale, len(feature_cols)) * feature_stds.values
            new_row[feature_cols] = row[feature_cols].values + noise
            new_row['Sample_ID'] = f"NOISE_{row['Sample_ID']}_{k}"
            new_rows.append(new_row)
    aug_df = pd.DataFrame(new_rows)
    return pd.concat([df, aug_df], ignore_index=True)


def augment_trajectory(df, feature_cols):
    """方法C: 处理轨迹插值 (CK1-D1, D1-RD2 之间生成中间态)"""
    treatments = ['CK1', 'D1', 'RD2']
    pairs = [('CK1', 'D1'), ('D1', 'RD2')]
    new_rows = []

    for variety, v_group in df.groupby('Variety'):
        for trt_a, trt_b in pairs:
            grp_a = v_group[v_group['Treatment'] == trt_a]
            grp_b = v_group[v_group['Treatment'] == trt_b]
            if len(grp_a) == 0 or len(grp_b) == 0:
                continue
            # 取每个处理的均值作为代表
            mean_a = grp_a[feature_cols].mean()
            mean_b = grp_b[feature_cols].mean()

            # 生成3个中间态 (alpha=0.25, 0.5, 0.75)
            for alpha in [0.25, 0.5, 0.75]:
                template = grp_a.iloc[0].copy()
                template[feature_cols] = (1 - alpha) * mean_a + alpha * mean_b
                template['Sample_ID'] = f"TRAJ_{variety}_{trt_a}_{trt_b}_{alpha}"
                template['Treatment'] = f"{trt_a}_{trt_b}_{alpha}"
                new_rows.append(template)

    if not new_rows:
        return df
    aug_df = pd.DataFrame(new_rows)
    return pd.concat([df, aug_df], ignore_index=True)


def augment_combined(df, feature_cols):
    """方法D: Mixup + Noise 组合"""
    df_mix = augment_mixup(df, feature_cols, n_per_pair=1)
    df_combined = augment_noise(df_mix, feature_cols, n_copies=2, noise_scale=0.015)
    return df_combined


# ========== 评估函数 ==========

def run_group_kfold(df, feature_cols, target_col='D_conv'):
    """运行GroupKFold交叉验证，返回品种级聚合结果"""
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

    # 品种级聚合
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
    """获取难预测品种的误差"""
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
    print("数据增强实验")
    print("=" * 70)

    # 加载数据和特征集
    with open(DATA_DIR / "feature_sets_enhanced.json", encoding='utf-8') as f:
        feature_sets = json.load(f)
    feature_cols = feature_sets['FS4']['features']

    df_original = pd.read_csv(DATA_DIR / "features_enhanced.csv")
    print(f"原始数据: {len(df_original)} 样本, {len(df_original['Variety'].unique())} 品种")

    # 定义增强方法
    methods = {
        'baseline': ('无增强（基线）', lambda df: df),
        'mixup': ('品种内Mixup', lambda df: augment_mixup(df, feature_cols, n_per_pair=1)),
        'noise_2pct': ('高斯噪声(2%)', lambda df: augment_noise(df, feature_cols, n_copies=4, noise_scale=0.02)),
        'noise_5pct': ('高斯噪声(5%)', lambda df: augment_noise(df, feature_cols, n_copies=4, noise_scale=0.05)),
        'trajectory': ('处理轨迹插值', lambda df: augment_trajectory(df, feature_cols)),
        'mixup_noise': ('Mixup+Noise组合', lambda df: augment_combined(df, feature_cols)),
    }

    results = {}

    for method_key, (method_name, aug_fn) in methods.items():
        print(f"\n--- {method_name} ---")

        # 增强数据
        np.random.seed(RANDOM_STATE)
        df_aug = aug_fn(df_original.copy())
        n_samples = len(df_aug)
        n_per_variety = df_aug.groupby('Variety').size().mean()
        print(f"  样本量: {n_samples} (每品种平均 {n_per_variety:.0f})")

        # 运行GroupKFold
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
        if var_errors:
            print(f"  难预测品种误差: {var_errors}")

    # 汇总表
    print("\n" + "=" * 70)
    print("汇总对比")
    print("=" * 70)
    print(f"\n{'方法':<20} {'样本量':>6} {'R²':>8} {'RMSE':>8} {'Spearman':>8}")
    print("-" * 55)
    baseline_r2 = results['baseline']['metrics']['R2']
    for key, res in results.items():
        m = res['metrics']
        delta = m['R2'] - baseline_r2
        delta_str = f"({delta:+.4f})" if key != 'baseline' else ""
        print(f"{res['name']:<20} {res['n_samples']:>6} {m['R2']:>8.4f} {m['RMSE']:>8.4f} {m['Spearman']:>8.4f}  {delta_str}")

    # 保存结果
    with open(RESULTS_DIR / "augmentation_report.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {RESULTS_DIR / 'augmentation_report.json'}")

    return results


if __name__ == "__main__":
    results = main()
