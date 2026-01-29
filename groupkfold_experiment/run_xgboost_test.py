"""
XGBoost A/B测试
XGBoost 原生支持 sample_weight，可真正测试加权效果
"""

import os
import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

from xgboost import XGBRegressor
from sklearn.model_selection import KFold, GroupKFold
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

RANDOM_STATE = 42

os.makedirs(RESULTS_DIR, exist_ok=True)


def load_data(use_enhanced=True):
    """加载特征数据"""
    if use_enhanced:
        features_path = DATA_DIR / "features_enhanced.csv"
        fs_path = DATA_DIR / "feature_sets_enhanced.json"
    else:
        features_path = DATA_DIR / "features_40.csv"
        fs_path = DATA_DIR / "feature_sets.json"

    df = pd.read_csv(features_path)
    with open(fs_path, 'r', encoding='utf-8') as f:
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

    n = len(y_true)
    correct_pairs = 0
    total_pairs = 0
    for i in range(n):
        for j in range(i+1, n):
            total_pairs += 1
            if (y_true[i] > y_true[j]) == (y_pred[i] > y_pred[j]):
                correct_pairs += 1
    pairwise_acc = correct_pairs / total_pairs if total_pairs > 0 else 0

    return {
        'R2': float(r2),
        'RMSE': float(rmse),
        'MAE': float(mae),
        'Spearman': float(spearman_r),
        'Pairwise_Acc': float(pairwise_acc)
    }


def compute_sample_weights(y, q_low=0.1, q_high=0.9, boost=2.0):
    """计算样本权重"""
    low_threshold = np.quantile(y, q_low)
    high_threshold = np.quantile(y, q_high)

    weights = np.ones(len(y))
    weights[(y <= low_threshold) | (y >= high_threshold)] = boost

    return weights


def sample_level_normalize(df, band_cols, static_cols):
    """样本级归一化"""
    X = df.copy()

    if band_cols:
        mean_band = X[band_cols].mean(axis=1)
        std_band = X[band_cols].std(axis=1) + 1e-9
        X[band_cols] = X[band_cols].sub(mean_band, axis=0).div(std_band, axis=0)

    if static_cols:
        mean_static = X[static_cols].mean(axis=1)
        std_static = X[static_cols].std(axis=1) + 1e-9
        X[static_cols] = X[static_cols].sub(mean_static, axis=0).div(std_static, axis=0)

    return X


def run_group_kfold_xgb(df, feature_cols, xgb_params, sample_weight=None,
                        use_sample_norm=True, n_splits=5):
    """
    运行GroupKFold + XGBoost
    """
    band_cols = [c for c in feature_cols if c.startswith('R') and not c.startswith('RF')]
    static_cols = [c for c in feature_cols if c in ['BF(F440)', 'GF(F520)', 'RF(F690)', 'FrF(f740)']]

    if use_sample_norm and (band_cols or static_cols):
        df_norm = sample_level_normalize(df[feature_cols], band_cols, static_cols)
        X = df_norm.values
    else:
        X = df[feature_cols].values

    y = df['D_conv'].values
    groups = df['Variety'].values

    n_unique = len(np.unique(groups))
    n_splits = min(n_splits, n_unique)

    oof_preds = np.zeros(len(y))

    gkf = GroupKFold(n_splits=n_splits)

    for fold, (tr_idx, te_idx) in enumerate(gkf.split(X, y, groups=groups)):
        X_train, X_test = X[tr_idx], X[te_idx]
        y_train, y_test = y[tr_idx], y[te_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = XGBRegressor(**xgb_params)

        # XGBoost 原生支持 sample_weight
        if sample_weight is not None:
            model.fit(X_train_scaled, y_train, sample_weight=sample_weight[tr_idx])
        else:
            model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)
        oof_preds[te_idx] = y_pred

    # 聚合到品种层
    df_result = df[['Variety', 'D_conv']].copy()
    df_result['pred'] = oof_preds

    variety_agg = df_result.groupby('Variety').agg({
        'D_conv': 'first',
        'pred': 'mean'
    }).reset_index()

    metrics = get_variety_metrics(
        variety_agg['D_conv'].values,
        variety_agg['pred'].values
    )

    return metrics, variety_agg


def run_random_kfold_xgb(df, feature_cols, xgb_params, use_sample_norm=True, n_splits=5):
    """运行Random KFold + XGBoost"""
    band_cols = [c for c in feature_cols if c.startswith('R') and not c.startswith('RF')]
    static_cols = [c for c in feature_cols if c in ['BF(F440)', 'GF(F520)', 'RF(F690)', 'FrF(f740)']]

    if use_sample_norm and (band_cols or static_cols):
        df_norm = sample_level_normalize(df[feature_cols], band_cols, static_cols)
        X = df_norm.values
    else:
        X = df[feature_cols].values

    y = df['D_conv'].values

    oof_preds = np.zeros(len(y))

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    for fold, (tr_idx, te_idx) in enumerate(kf.split(X)):
        X_train, X_test = X[tr_idx], X[te_idx]
        y_train, y_test = y[tr_idx], y[te_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = XGBRegressor(**xgb_params)
        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)
        oof_preds[te_idx] = y_pred

    df_result = df[['Variety', 'D_conv']].copy()
    df_result['pred'] = oof_preds

    variety_agg = df_result.groupby('Variety').agg({
        'D_conv': 'first',
        'pred': 'mean'
    }).reset_index()

    metrics = get_variety_metrics(
        variety_agg['D_conv'].values,
        variety_agg['pred'].values
    )

    return metrics, variety_agg


def run_experiment(name, df, feature_cols, xgb_params, sample_weight=None):
    """运行单个实验"""
    print(f"\n  [{name}]")

    group_metrics, group_agg = run_group_kfold_xgb(
        df, feature_cols, xgb_params,
        sample_weight=sample_weight
    )

    random_metrics, random_agg = run_random_kfold_xgb(
        df, feature_cols, xgb_params
    )

    print(f"    GroupKFold R²: {group_metrics['R2']:.4f}")
    print(f"    Random KFold R²: {random_metrics['R2']:.4f}")
    print(f"    Gap: {random_metrics['R2'] - group_metrics['R2']:.4f}")

    group_agg['error'] = np.abs(group_agg['D_conv'] - group_agg['pred'])
    worst = group_agg.nlargest(3, 'error')

    return {
        'name': name,
        'n_features': len(feature_cols),
        'group_metrics': group_metrics,
        'random_metrics': random_metrics,
        'gap': random_metrics['R2'] - group_metrics['R2'],
        'worst_varieties': worst[['Variety', 'D_conv', 'pred', 'error']].to_dict('records'),
        'variety_results': group_agg
    }


def main():
    print("=" * 60)
    print("XGBoost GroupKFold 性能测试")
    print("=" * 60)

    # 检查增强特征是否存在
    enhanced_path = DATA_DIR / "features_enhanced.csv"
    if not enhanced_path.exists():
        print("\n[Step 1] 生成增强特征...")
        import feature_engineering_enhanced
        feature_engineering_enhanced.main()

    # 加载数据
    print("\n[Step 1] 加载数据...")
    df, feature_sets = load_data(use_enhanced=True)
    print(f"  样本数: {len(df)}")
    print(f"  品种数: {df['Variety'].nunique()}")

    # XGBoost 参数
    xgb_params = {
        'n_estimators': 200,
        'max_depth': 4,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': RANDOM_STATE,
        'n_jobs': -1,
        'verbosity': 0
    }

    print(f"\nXGBoost 参数: max_depth={xgb_params['max_depth']}, "
          f"n_estimators={xgb_params['n_estimators']}, lr={xgb_params['learning_rate']}")

    # 运行实验
    print("\n[Step 2] 运行A/B测试...")
    print("=" * 60)

    results = []

    # Baseline
    result_baseline = run_experiment(
        "Baseline",
        df,
        feature_sets['FS4']['features'],
        xgb_params,
        sample_weight=None
    )
    results.append(result_baseline)

    # A: +导数
    result_a = run_experiment(
        "A: +导数",
        df,
        feature_sets['FS4_enhanced']['features'],
        xgb_params,
        sample_weight=None
    )
    results.append(result_a)

    # B: +加权 (boost=2.0)
    weights_2 = compute_sample_weights(df['D_conv'].values, q_low=0.1, q_high=0.9, boost=2.0)
    result_b = run_experiment(
        "B: +加权(2x)",
        df,
        feature_sets['FS4']['features'],
        xgb_params,
        sample_weight=weights_2
    )
    results.append(result_b)

    # B2: +加权 (boost=3.0)
    weights_3 = compute_sample_weights(df['D_conv'].values, q_low=0.1, q_high=0.9, boost=3.0)
    result_b2 = run_experiment(
        "B2: +加权(3x)",
        df,
        feature_sets['FS4']['features'],
        xgb_params,
        sample_weight=weights_3
    )
    results.append(result_b2)

    # C: 导数替代
    result_c = run_experiment(
        "C: 导数替代",
        df,
        feature_sets['FS4_derivatives']['features'],
        xgb_params,
        sample_weight=None
    )
    results.append(result_c)

    # D: 导数+加权
    result_d = run_experiment(
        "D: 导数+加权(2x)",
        df,
        feature_sets['FS4_enhanced']['features'],
        xgb_params,
        sample_weight=weights_2
    )
    results.append(result_d)

    # E: 导数+加权(3x)
    result_e = run_experiment(
        "E: 导数+加权(3x)",
        df,
        feature_sets['FS4_enhanced']['features'],
        xgb_params,
        sample_weight=weights_3
    )
    results.append(result_e)

    # 汇总
    print("\n" + "=" * 60)
    print("实验结果汇总")
    print("=" * 60)

    print("\n方案对比:")
    print("-" * 70)
    print(f"{'方案':<18} {'特征数':>6} {'GroupKFold R²':>14} {'Random R²':>11} {'Gap':>8}")
    print("-" * 70)

    for r in results:
        print(f"{r['name']:<18} {r['n_features']:>6} {r['group_metrics']['R2']:>14.4f} "
              f"{r['random_metrics']['R2']:>11.4f} {r['gap']:>8.4f}")

    # 找最佳
    best = max(results, key=lambda x: x['group_metrics']['R2'])
    baseline_r2 = results[0]['group_metrics']['R2']
    improvement = best['group_metrics']['R2'] - baseline_r2

    print("-" * 70)
    print(f"\n最佳方案: {best['name']}")
    print(f"  GroupKFold R²: {best['group_metrics']['R2']:.4f}")
    print(f"  相比Baseline提升: {improvement:+.4f}")

    # 难预测品种分析
    print("\n" + "=" * 60)
    print("难预测品种误差对比")
    print("=" * 60)
    print(f"\n{'品种':<10} {'真实值':>8} ", end="")
    for r in results[:4]:  # 显示前4个方案
        print(f"{r['name'][:10]:>12}", end="")
    print()
    print("-" * 60)

    # 收集所有品种误差
    all_varieties = set()
    for r in results:
        for v in r['worst_varieties']:
            all_varieties.add(v['Variety'])

    for variety in sorted(all_varieties):
        row = results[0]['variety_results']
        true_val = row[row['Variety'] == variety]['D_conv'].values[0]
        print(f"{variety:<10} {true_val:>8.4f} ", end="")
        for r in results[:4]:
            vr = r['variety_results']
            pred = vr[vr['Variety'] == variety]['pred'].values[0]
            err = abs(true_val - pred)
            print(f"{err:>12.4f}", end="")
        print()

    # 保存结果
    report = {
        "model": "XGBoost",
        "params": xgb_params,
        "experiments": [
            {
                "name": r['name'],
                "n_features": r['n_features'],
                "group_r2": r['group_metrics']['R2'],
                "random_r2": r['random_metrics']['R2'],
                "gap": r['gap'],
                "group_metrics": r['group_metrics'],
                "worst_varieties": r['worst_varieties']
            }
            for r in results
        ],
        "best_experiment": best['name'],
        "baseline_r2": baseline_r2,
        "best_r2": best['group_metrics']['R2'],
        "improvement": improvement
    }

    report_path = RESULTS_DIR / "xgboost_ab_test_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n报告已保存: {report_path}")

    return results


if __name__ == "__main__":
    results = main()
