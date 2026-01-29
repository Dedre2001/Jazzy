"""
标签变换实验
测试不同标签处理策略对GroupKFold性能的影响
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
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_data():
    df = pd.read_csv(DATA_DIR / "features_enhanced.csv")
    with open(DATA_DIR / "feature_sets_enhanced.json", encoding='utf-8') as f:
        feature_sets = json.load(f)
    return df, feature_sets


def get_variety_metrics(y_true, y_pred):
    """计算品种层指标"""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    spearman_r, _ = spearmanr(y_true, y_pred)
    return {'R2': float(r2), 'RMSE': float(rmse), 'MAE': float(mae), 'Spearman': float(spearman_r)}


# ============ 标签变换策略 ============

def transform_none(y):
    """原始标签"""
    return y.copy(), lambda x: x

def transform_log(y):
    """Log变换"""
    y_trans = np.log1p(y)
    inverse = lambda x: np.expm1(x)
    return y_trans, inverse

def transform_sqrt(y):
    """平方根变换"""
    y_trans = np.sqrt(y)
    inverse = lambda x: x ** 2
    return y_trans, inverse

def transform_rank(y):
    """秩变换 (转为0-1均匀分布)"""
    from scipy.stats import rankdata
    ranks = rankdata(y)
    y_trans = (ranks - 1) / (len(ranks) - 1)
    # 逆变换需要原始值映射
    return y_trans, None  # 无法直接逆变换

def transform_winsorize(y, lower=0.05, upper=0.95):
    """Winsorize (截断极端值)"""
    low_val = np.quantile(y, lower)
    high_val = np.quantile(y, upper)
    y_trans = np.clip(y, low_val, high_val)
    return y_trans, lambda x: x  # 逆变换保持原样

def transform_zscore(y):
    """Z-score标准化"""
    mean, std = y.mean(), y.std()
    y_trans = (y - mean) / std
    inverse = lambda x: x * std + mean
    return y_trans, inverse

def transform_minmax(y):
    """Min-Max归一化到[0,1]"""
    ymin, ymax = y.min(), y.max()
    y_trans = (y - ymin) / (ymax - ymin)
    inverse = lambda x: x * (ymax - ymin) + ymin
    return y_trans, inverse

def transform_shrink_extreme(y, shrink_factor=0.5):
    """收缩极端值 (向均值收缩边缘10%)"""
    mean = y.mean()
    q10, q90 = np.quantile(y, 0.1), np.quantile(y, 0.9)

    y_trans = y.copy()
    # 低端收缩
    low_mask = y < q10
    y_trans[low_mask] = mean - (mean - y[low_mask]) * shrink_factor
    # 高端收缩
    high_mask = y > q90
    y_trans[high_mask] = mean + (y[high_mask] - mean) * shrink_factor

    return y_trans, None


# ============ 运行实验 ============

def run_group_kfold(df, feature_cols, y_transformed, inverse_fn=None):
    """运行GroupKFold"""
    X = df[feature_cols].values
    groups = df['Variety'].values
    y_original = df['D_conv'].values

    n_splits = min(5, len(np.unique(groups)))
    oof_preds = np.zeros(len(y_transformed))

    gkf = GroupKFold(n_splits=n_splits)

    for fold, (tr_idx, te_idx) in enumerate(gkf.split(X, y_transformed, groups=groups)):
        X_train, X_test = X[tr_idx], X[te_idx]
        y_train = y_transformed[tr_idx]

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

    # 逆变换预测值 (如果有)
    if inverse_fn is not None:
        oof_preds_original = inverse_fn(oof_preds)
    else:
        oof_preds_original = oof_preds

    # 聚合到品种层
    df_result = df[['Variety', 'D_conv']].copy()
    df_result['pred'] = oof_preds_original

    variety_agg = df_result.groupby('Variety').agg({
        'D_conv': 'first',
        'pred': 'mean'
    }).reset_index()

    metrics = get_variety_metrics(
        variety_agg['D_conv'].values,
        variety_agg['pred'].values
    )

    return metrics, variety_agg


def run_experiment(name, df, feature_cols, transform_fn, **kwargs):
    """运行单个实验"""
    print(f"\n  [{name}]")

    y_original = df['D_conv'].values

    # 应用变换
    if kwargs:
        y_trans, inverse_fn = transform_fn(y_original, **kwargs)
    else:
        y_trans, inverse_fn = transform_fn(y_original)

    # 显示变换后的分布
    print(f"    变换后范围: [{y_trans.min():.4f}, {y_trans.max():.4f}]")

    # 运行GroupKFold
    metrics, variety_agg = run_group_kfold(df, feature_cols, y_trans, inverse_fn)

    print(f"    GroupKFold R2: {metrics['R2']:.4f}")
    print(f"    Spearman: {metrics['Spearman']:.4f}")

    # 分析难预测品种
    variety_agg['error'] = np.abs(variety_agg['D_conv'] - variety_agg['pred'])
    worst = variety_agg.nlargest(3, 'error')

    return {
        'name': name,
        'metrics': metrics,
        'worst_varieties': worst.to_dict('records'),
        'variety_agg': variety_agg
    }


def main():
    print("=" * 60)
    print("标签变换实验")
    print("=" * 60)

    df, feature_sets = load_data()
    feature_cols = feature_sets['FS4']['features']

    print(f"\n样本数: {len(df)}, 品种数: {df['Variety'].nunique()}")
    print(f"原始D_conv范围: [{df['D_conv'].min():.4f}, {df['D_conv'].max():.4f}]")

    results = []

    # 1. 基线 (无变换)
    results.append(run_experiment("Baseline", df, feature_cols, transform_none))

    # 2. Log变换
    results.append(run_experiment("Log", df, feature_cols, transform_log))

    # 3. 平方根变换
    results.append(run_experiment("Sqrt", df, feature_cols, transform_sqrt))

    # 4. Z-score
    results.append(run_experiment("Z-score", df, feature_cols, transform_zscore))

    # 5. Min-Max
    results.append(run_experiment("MinMax", df, feature_cols, transform_minmax))

    # 6. Winsorize (截断极端值)
    results.append(run_experiment("Winsorize(5%)", df, feature_cols, transform_winsorize, lower=0.05, upper=0.95))

    # 7. 收缩极端值
    results.append(run_experiment("Shrink(0.5)", df, feature_cols, transform_shrink_extreme, shrink_factor=0.5))
    results.append(run_experiment("Shrink(0.7)", df, feature_cols, transform_shrink_extreme, shrink_factor=0.7))

    # 8. 秩变换
    results.append(run_experiment("Rank", df, feature_cols, transform_rank))

    # 汇总
    print("\n" + "=" * 60)
    print("实验结果汇总")
    print("=" * 60)

    print(f"\n{'策略':<20} {'R2':>10} {'RMSE':>10} {'Spearman':>10}")
    print("-" * 52)

    for r in results:
        print(f"{r['name']:<20} {r['metrics']['R2']:>10.4f} {r['metrics']['RMSE']:>10.4f} {r['metrics']['Spearman']:>10.4f}")

    # 最佳策略
    best = max(results, key=lambda x: x['metrics']['R2'])
    baseline_r2 = results[0]['metrics']['R2']

    print("-" * 52)
    print(f"\n最佳策略: {best['name']}")
    print(f"  R2: {best['metrics']['R2']:.4f} (vs Baseline {baseline_r2:.4f}, 提升 {best['metrics']['R2'] - baseline_r2:+.4f})")

    # 难预测品种对比
    print("\n" + "=" * 60)
    print("难预测品种误差对比")
    print("=" * 60)

    print(f"\n{'品种':<10} {'真实值':>8} ", end="")
    for r in results[:5]:
        print(f"{r['name'][:8]:>10}", end="")
    print()
    print("-" * 60)

    # 收集所有品种
    all_varieties = df.groupby('Variety')['D_conv'].first().sort_values()
    for variety in [all_varieties.idxmin(), all_varieties.idxmax()]:  # 最小和最大
        true_val = all_varieties[variety]
        print(f"{int(variety):<10} {true_val:>8.4f} ", end="")
        for r in results[:5]:
            va = r['variety_agg']
            pred = va[va['Variety'] == variety]['pred'].values[0]
            err = abs(true_val - pred)
            print(f"{err:>10.4f}", end="")
        print()

    # 保存结果
    report = {
        "experiments": [
            {"name": r['name'], "metrics": r['metrics'], "worst": r['worst_varieties']}
            for r in results
        ],
        "best": best['name'],
        "best_r2": best['metrics']['R2'],
        "baseline_r2": baseline_r2
    }

    with open(RESULTS_DIR / "label_transform_report.json", 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n报告已保存: {RESULTS_DIR / 'label_transform_report.json'}")

    return results


if __name__ == "__main__":
    results = main()
