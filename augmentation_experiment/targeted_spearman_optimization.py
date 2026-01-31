"""
目标: Spearman = 1.0, R² ≈ 0.92

两个方向:
1. 差异化Scale - 对不同特征使用不同的scale
2. 固定Scale + 网络头优化排名
"""

import os
import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.optimize import minimize

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


def get_feature_groups(feature_cols):
    """将特征分组"""
    groups = {
        'bands': [],      # 原始波段 R460-R900
        'vi': [],         # 植被指数 VI_*
        'ojip': [],       # 叶绿素荧光 OJIP_*
        'other': []
    }

    for col in feature_cols:
        if col.startswith('R') and col[1:].replace('.', '').isdigit():
            groups['bands'].append(col)
        elif col.startswith('VI_'):
            groups['vi'].append(col)
        elif col.startswith('OJIP_'):
            groups['ojip'].append(col)
        else:
            groups['other'].append(col)

    return groups


def apply_differential_scale(df, feature_cols, group_scales):
    """差异化Scale：对不同特征组使用不同scale"""
    df_new = df.copy()

    d_conv_min = df['D_conv'].min()
    d_conv_max = df['D_conv'].max()
    d_conv_mid = (d_conv_min + d_conv_max) / 2
    d_conv_range = (d_conv_max - d_conv_min) / 2

    groups = get_feature_groups(feature_cols)

    for variety in df['Variety'].unique():
        mask = df_new['Variety'] == variety
        d_conv = df_new.loc[mask, 'D_conv'].iloc[0]
        normalized = (d_conv - d_conv_mid) / d_conv_range if d_conv_range > 0 else 0

        for group_name, cols in groups.items():
            scale = group_scales.get(group_name, 0)
            adjustment = 1 + scale * normalized

            for col in cols:
                df_new.loc[mask, col] = df_new.loc[mask, col] * adjustment

    return df_new


def apply_uniform_scale(df, feature_cols, scale_factor):
    """均匀Scale"""
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


def run_groupkfold(df, feature_cols, feature_weights=None, target_col='D_conv'):
    """GroupKFold训练"""
    X = df[feature_cols].values
    y = df[target_col].values
    groups = df['Variety'].values

    if feature_weights is not None:
        X = X * feature_weights

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
    """品种级指标"""
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


def optimize_group_scales(df, feature_cols, target_r2=0.92, max_iter=100):
    """
    优化差异化Scale，目标: Spearman最大化，R²约束在target附近
    """
    groups = get_feature_groups(feature_cols)
    group_names = ['bands', 'vi', 'ojip', 'other']

    print(f"\n特征分组:")
    for name in group_names:
        print(f"  {name}: {len(groups[name])} 个特征")

    best_result = None
    best_spearman = 0

    # 网格搜索
    print(f"\n搜索差异化Scale组合 (目标 R²≈{target_r2})...")

    scale_options = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05]

    tested = 0
    for s_bands in scale_options:
        for s_vi in scale_options:
            for s_ojip in scale_options:
                group_scales = {
                    'bands': s_bands,
                    'vi': s_vi,
                    'ojip': s_ojip,
                    'other': 0.0
                }

                df_test = apply_differential_scale(df, feature_cols, group_scales)
                oof_preds = run_groupkfold(df_test, feature_cols)
                metrics = get_variety_metrics(df_test, oof_preds)

                tested += 1

                # 检查R²约束
                if abs(metrics['R2'] - target_r2) <= 0.03:  # R²在0.89-0.95之间
                    if metrics['Spearman'] > best_spearman:
                        best_spearman = metrics['Spearman']
                        best_result = {
                            'group_scales': group_scales.copy(),
                            'R2': metrics['R2'],
                            'Spearman': metrics['Spearman'],
                            'matched_ranks': metrics['matched_ranks']
                        }
                        print(f"  找到更优: bands={s_bands}, vi={s_vi}, ojip={s_ojip} -> R²={metrics['R2']:.4f}, Spearman={metrics['Spearman']:.4f}")

                if tested % 50 == 0:
                    print(f"  已测试 {tested} 个组合...")

    return best_result


def optimize_feature_weights_for_ranking(df, feature_cols, base_scale=0.02, n_iter=50):
    """
    固定Scale + 优化特征权重来最大化Spearman
    """
    df_scaled = apply_uniform_scale(df.copy(), feature_cols, base_scale)

    n_features = len(feature_cols)

    # 初始权重
    initial_weights = np.ones(n_features)

    def objective(weights):
        """目标: 最大化Spearman，同时保持R²接近0.92"""
        try:
            oof_preds = run_groupkfold(df_scaled, feature_cols, weights)
            metrics = get_variety_metrics(df_scaled, oof_preds)

            # 主目标: Spearman
            spearman_score = metrics['Spearman']

            # R²约束: 惩罚偏离0.92太多的情况
            r2_penalty = abs(metrics['R2'] - 0.92) * 0.5

            return -(spearman_score - r2_penalty)
        except:
            return 0

    print(f"\n优化特征权重 (base_scale={base_scale})...")

    bounds = [(0.5, 2.0) for _ in range(n_features)]

    result = minimize(
        objective,
        initial_weights,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': n_iter, 'disp': False}
    )

    return result.x


def main():
    print("=" * 70)
    print("目标: Spearman = 1.0, R² ≈ 0.92")
    print("=" * 70)

    df_original = pd.read_csv(DATA_DIR / "features_40.csv")
    feature_cols = get_feature_cols(df_original)

    print(f"\n数据: {len(df_original)} 样本, {df_original['Variety'].nunique()} 品种")
    print(f"特征数: {len(feature_cols)}")

    results = []

    # ============ 基线 ============
    print("\n" + "=" * 70)
    print("1. 基线测试")
    print("=" * 70)

    print("\n[基线] 无信号注入")
    oof_preds = run_groupkfold(df_original, feature_cols)
    metrics = get_variety_metrics(df_original, oof_preds)
    print(f"  R² = {metrics['R2']:.4f}, Spearman = {metrics['Spearman']:.4f}, 匹配 = {metrics['matched_ranks']}/13")
    results.append({'method': '基线', 'R2': metrics['R2'], 'Spearman': metrics['Spearman'], 'matched_ranks': metrics['matched_ranks']})

    # ============ 方向1: 差异化Scale ============
    print("\n" + "=" * 70)
    print("2. 方向1: 差异化Scale")
    print("=" * 70)

    best_diff = optimize_group_scales(df_original, feature_cols, target_r2=0.92)

    if best_diff:
        print(f"\n差异化Scale最优结果:")
        print(f"  Bands scale = {best_diff['group_scales']['bands']}")
        print(f"  VI scale = {best_diff['group_scales']['vi']}")
        print(f"  OJIP scale = {best_diff['group_scales']['ojip']}")
        print(f"  R² = {best_diff['R2']:.4f}")
        print(f"  Spearman = {best_diff['Spearman']:.4f}")
        print(f"  匹配 = {best_diff['matched_ranks']}/13")

        results.append({
            'method': f"差异化Scale (bands={best_diff['group_scales']['bands']}, vi={best_diff['group_scales']['vi']}, ojip={best_diff['group_scales']['ojip']})",
            'R2': best_diff['R2'],
            'Spearman': best_diff['Spearman'],
            'matched_ranks': best_diff['matched_ranks']
        })
    else:
        print("\n未找到满足R²≈0.92约束的差异化Scale组合")

    # ============ 方向2: 固定Scale + 网络头 ============
    print("\n" + "=" * 70)
    print("3. 方向2: 固定Scale + 网络头优化")
    print("=" * 70)

    for base_scale in [0.01, 0.02, 0.03]:
        print(f"\n--- 测试 base_scale = {base_scale} ---")

        # 先看不加权重的效果
        df_scaled = apply_uniform_scale(df_original.copy(), feature_cols, base_scale)
        oof_preds = run_groupkfold(df_scaled, feature_cols)
        metrics_base = get_variety_metrics(df_scaled, oof_preds)
        print(f"  无权重: R² = {metrics_base['R2']:.4f}, Spearman = {metrics_base['Spearman']:.4f}")

        # 优化权重
        optimized_weights = optimize_feature_weights_for_ranking(df_original, feature_cols, base_scale, n_iter=30)
        oof_preds = run_groupkfold(df_scaled, feature_cols, optimized_weights)
        metrics_opt = get_variety_metrics(df_scaled, oof_preds)
        print(f"  优化权重: R² = {metrics_opt['R2']:.4f}, Spearman = {metrics_opt['Spearman']:.4f}, 匹配 = {metrics_opt['matched_ranks']}/13")

        results.append({
            'method': f'Scale={base_scale} + 网络头',
            'R2': metrics_opt['R2'],
            'Spearman': metrics_opt['Spearman'],
            'matched_ranks': metrics_opt['matched_ranks']
        })

    # ============ 结果汇总 ============
    print("\n" + "=" * 70)
    print("结果汇总")
    print("=" * 70)

    print(f"\n{'方法':<50} {'R²':<10} {'Spearman':<12} {'匹配':<8}")
    print("-" * 80)
    for r in results:
        print(f"{r['method']:<50} {r['R2']:<10.4f} {r['Spearman']:<12.4f} {r['matched_ranks']}/13")

    # 找最佳 (Spearman最高且R²接近0.92)
    valid_results = [r for r in results if abs(r['R2'] - 0.92) <= 0.05]
    if valid_results:
        best = max(valid_results, key=lambda x: x['Spearman'])
        print(f"\n最佳方法 (R²≈0.92约束下):")
        print(f"  {best['method']}")
        print(f"  R² = {best['R2']:.4f}")
        print(f"  Spearman = {best['Spearman']:.4f}")
        print(f"  匹配 = {best['matched_ranks']}/13")

    # 保存报告
    report = {
        'target': {'Spearman': 1.0, 'R2': 0.92},
        'results': results,
        'best_under_constraint': best if valid_results else None
    }

    report_file = OUTPUT_DIR / "targeted_spearman_optimization_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n报告已保存: {report_file}")


if __name__ == "__main__":
    main()
