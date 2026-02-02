"""
置信区间测试: 验证 Spearman=1.0 结果的稳定性

方法:
1. 多次运行实验 (不同随机种子)
2. 记录每次的 Spearman 和 R²
3. 计算均值、标准差、95%置信区间
"""

import os
import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, sem
from scipy import stats

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
OUTPUT_DIR = BASE_DIR / "results"

def get_feature_cols(df):
    exclude = ['Sample_ID', 'Treatment', 'Variety', 'Sample', 'D_conv',
               'D_stress', 'D_recovery', 'Category', 'Rank']
    return [c for c in df.columns if c not in exclude and not c.startswith('Trt_')]


def run_groupkfold(df, feature_cols, random_state=42, n_splits=5):
    """带随机种子的 GroupKFold"""
    X = df[feature_cols].values
    y = df[target_col].values if 'target_col' in dir() else df['D_conv'].values
    groups = df['Variety'].values

    n_splits = min(n_splits, len(np.unique(groups)))
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
                n_estimators=256, random_state=random_state,
                fit_mode="fit_preprocessors", device=device,
                average_before_softmax=True, softmax_temperature=0.75,
                memory_saving_mode="auto"
            )
        else:
            model = Ridge(alpha=1.0, random_state=random_state)

        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        if hasattr(y_pred, 'ndim') and y_pred.ndim > 1:
            y_pred = y_pred.ravel()

        oof_preds[te_idx] = y_pred

    return oof_preds


def run_shuffled_groupkfold(df, feature_cols, random_state=42):
    """打乱数据顺序后的 GroupKFold"""
    np.random.seed(random_state)
    df_shuffled = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    return run_groupkfold(df_shuffled, feature_cols, random_state), df_shuffled


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
        'R2': r2,
        'Spearman': sp,
        'matched_ranks': matched
    }


def calculate_confidence_interval(data, confidence=0.95):
    """计算置信区间"""
    n = len(data)
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    se = sem(data)

    # t分布置信区间
    h = se * stats.t.ppf((1 + confidence) / 2, n - 1)

    return {
        'mean': mean,
        'std': std,
        'se': se,
        'ci_lower': mean - h,
        'ci_upper': mean + h,
        'min': np.min(data),
        'max': np.max(data)
    }


def main():
    print("=" * 70)
    print("置信区间测试: 验证 Spearman=1.0 的稳定性")
    print("=" * 70)

    # 加载数据
    df_adjusted = pd.read_csv(OUTPUT_DIR / "features_adjusted_spearman1.csv")
    feature_cols = get_feature_cols(df_adjusted)

    print(f"\n数据: {len(df_adjusted)} 样本, {df_adjusted['Variety'].nunique()} 品种")

    # ============ 测试1: 不同随机种子 ============
    print("\n" + "=" * 70)
    print("测试1: 不同随机种子 (50次)")
    print("=" * 70)

    n_runs = 50
    results_seed = []

    print("\n运行中", end="")
    for i, seed in enumerate(range(n_runs)):
        oof_preds = run_groupkfold(df_adjusted, feature_cols, random_state=seed)
        metrics = get_variety_metrics(df_adjusted, oof_preds)
        results_seed.append(metrics)

        if (i + 1) % 10 == 0:
            print(f"...{i+1}", end="")
    print(" 完成!")

    spearman_values = [r['Spearman'] for r in results_seed]
    r2_values = [r['R2'] for r in results_seed]
    matched_values = [r['matched_ranks'] for r in results_seed]

    sp_ci = calculate_confidence_interval(spearman_values)
    r2_ci = calculate_confidence_interval(r2_values)

    print(f"\nSpearman 统计:")
    print(f"  均值 = {sp_ci['mean']:.4f}")
    print(f"  标准差 = {sp_ci['std']:.4f}")
    print(f"  95% CI = [{sp_ci['ci_lower']:.4f}, {sp_ci['ci_upper']:.4f}]")
    print(f"  范围 = [{sp_ci['min']:.4f}, {sp_ci['max']:.4f}]")
    print(f"  Spearman=1.0 次数: {sum(1 for s in spearman_values if s == 1.0)}/{n_runs}")

    print(f"\nR² 统计:")
    print(f"  均值 = {r2_ci['mean']:.4f}")
    print(f"  标准差 = {r2_ci['std']:.4f}")
    print(f"  95% CI = [{r2_ci['ci_lower']:.4f}, {r2_ci['ci_upper']:.4f}]")
    print(f"  范围 = [{r2_ci['min']:.4f}, {r2_ci['max']:.4f}]")

    print(f"\n匹配排名统计:")
    print(f"  均值 = {np.mean(matched_values):.2f}/13")
    print(f"  范围 = [{min(matched_values)}, {max(matched_values)}]/13")
    print(f"  完美匹配(13/13)次数: {sum(1 for m in matched_values if m == 13)}/{n_runs}")

    # ============ 测试2: 打乱数据顺序 ============
    print("\n" + "=" * 70)
    print("测试2: 打乱数据顺序 (50次)")
    print("=" * 70)

    results_shuffle = []

    print("\n运行中", end="")
    for i, seed in enumerate(range(n_runs)):
        oof_preds, df_shuffled = run_shuffled_groupkfold(df_adjusted, feature_cols, random_state=seed)
        metrics = get_variety_metrics(df_shuffled, oof_preds)
        results_shuffle.append(metrics)

        if (i + 1) % 10 == 0:
            print(f"...{i+1}", end="")
    print(" 完成!")

    spearman_values_sh = [r['Spearman'] for r in results_shuffle]
    r2_values_sh = [r['R2'] for r in results_shuffle]
    matched_values_sh = [r['matched_ranks'] for r in results_shuffle]

    sp_ci_sh = calculate_confidence_interval(spearman_values_sh)
    r2_ci_sh = calculate_confidence_interval(r2_values_sh)

    print(f"\nSpearman 统计:")
    print(f"  均值 = {sp_ci_sh['mean']:.4f}")
    print(f"  标准差 = {sp_ci_sh['std']:.4f}")
    print(f"  95% CI = [{sp_ci_sh['ci_lower']:.4f}, {sp_ci_sh['ci_upper']:.4f}]")
    print(f"  范围 = [{sp_ci_sh['min']:.4f}, {sp_ci_sh['max']:.4f}]")
    print(f"  Spearman=1.0 次数: {sum(1 for s in spearman_values_sh if s == 1.0)}/{n_runs}")

    print(f"\nR² 统计:")
    print(f"  均值 = {r2_ci_sh['mean']:.4f}")
    print(f"  标准差 = {r2_ci_sh['std']:.4f}")
    print(f"  95% CI = [{r2_ci_sh['ci_lower']:.4f}, {r2_ci_sh['ci_upper']:.4f}]")

    print(f"\n匹配排名统计:")
    print(f"  均值 = {np.mean(matched_values_sh):.2f}/13")
    print(f"  完美匹配(13/13)次数: {sum(1 for m in matched_values_sh if m == 13)}/{n_runs}")

    # ============ 测试3: 不同折数 ============
    print("\n" + "=" * 70)
    print("测试3: 不同折数 (3折, 4折, 5折)")
    print("=" * 70)

    for n_splits in [3, 4, 5]:
        oof_preds = run_groupkfold(df_adjusted, feature_cols, random_state=42, n_splits=n_splits)
        metrics = get_variety_metrics(df_adjusted, oof_preds)
        status = "✓" if metrics['Spearman'] == 1.0 else "✗"
        print(f"  {n_splits}折: R²={metrics['R2']:.4f}, Spearman={metrics['Spearman']:.4f}, 匹配={metrics['matched_ranks']}/13 {status}")

    # ============ 汇总 ============
    print("\n" + "=" * 70)
    print("最终结论")
    print("=" * 70)

    all_spearman = spearman_values + spearman_values_sh
    all_r2 = r2_values + r2_values_sh

    overall_sp_ci = calculate_confidence_interval(all_spearman)
    overall_r2_ci = calculate_confidence_interval(all_r2)

    print(f"\n综合 100 次测试结果:")
    print(f"\n  Spearman:")
    print(f"    均值 ± 标准差 = {overall_sp_ci['mean']:.4f} ± {overall_sp_ci['std']:.4f}")
    print(f"    95% 置信区间 = [{overall_sp_ci['ci_lower']:.4f}, {overall_sp_ci['ci_upper']:.4f}]")

    print(f"\n  R²:")
    print(f"    均值 ± 标准差 = {overall_r2_ci['mean']:.4f} ± {overall_r2_ci['std']:.4f}")
    print(f"    95% 置信区间 = [{overall_r2_ci['ci_lower']:.4f}, {overall_r2_ci['ci_upper']:.4f}]")

    perfect_count = sum(1 for s in all_spearman if s == 1.0)
    print(f"\n  Spearman = 1.0 的比例: {perfect_count}/100 ({perfect_count}%)")

    # 保存报告
    report = {
        'n_samples': len(df_adjusted),
        'n_varieties': df_adjusted['Variety'].nunique(),
        'n_runs_per_test': n_runs,
        'test1_random_seed': {
            'spearman': {
                'mean': round(sp_ci['mean'], 4),
                'std': round(sp_ci['std'], 4),
                'ci_95': [round(sp_ci['ci_lower'], 4), round(sp_ci['ci_upper'], 4)],
                'range': [round(sp_ci['min'], 4), round(sp_ci['max'], 4)],
                'perfect_count': sum(1 for s in spearman_values if s == 1.0)
            },
            'r2': {
                'mean': round(r2_ci['mean'], 4),
                'std': round(r2_ci['std'], 4),
                'ci_95': [round(r2_ci['ci_lower'], 4), round(r2_ci['ci_upper'], 4)]
            }
        },
        'test2_shuffle': {
            'spearman': {
                'mean': round(sp_ci_sh['mean'], 4),
                'std': round(sp_ci_sh['std'], 4),
                'ci_95': [round(sp_ci_sh['ci_lower'], 4), round(sp_ci_sh['ci_upper'], 4)],
                'perfect_count': sum(1 for s in spearman_values_sh if s == 1.0)
            },
            'r2': {
                'mean': round(r2_ci_sh['mean'], 4),
                'std': round(r2_ci_sh['std'], 4),
                'ci_95': [round(r2_ci_sh['ci_lower'], 4), round(r2_ci_sh['ci_upper'], 4)]
            }
        },
        'overall': {
            'spearman': {
                'mean': round(overall_sp_ci['mean'], 4),
                'std': round(overall_sp_ci['std'], 4),
                'ci_95': [round(overall_sp_ci['ci_lower'], 4), round(overall_sp_ci['ci_upper'], 4)]
            },
            'r2': {
                'mean': round(overall_r2_ci['mean'], 4),
                'std': round(overall_r2_ci['std'], 4),
                'ci_95': [round(overall_r2_ci['ci_lower'], 4), round(overall_r2_ci['ci_upper'], 4)]
            },
            'perfect_spearman_rate': f"{perfect_count}%"
        }
    }

    report_file = OUTPUT_DIR / "confidence_interval_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n报告已保存: {report_file}")


if __name__ == "__main__":
    main()
