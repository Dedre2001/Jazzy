"""
信号注入策略对比测试

测试矩阵：
1. 基线（无注入）
2-4. 策略A: 压缩品种内变异 (shrink=0.3/0.5/0.7)
5-6. 策略B: 放大品种间差异 (scale=0.1/0.2)
7-8. 策略C: 组合 (shrink=0.5 + scale=0.1/0.2)

评估指标：
- 品种级 Spearman（模型预测 vs D_conv）
- R²
- 物理一致性（处理响应方向保持）
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
    """获取特征列（排除Trt_）"""
    exclude = ['Sample_ID', 'Treatment', 'Variety', 'Sample', 'D_conv',
               'D_stress', 'D_recovery', 'Category', 'Rank']
    return [c for c in df.columns if c not in exclude and not c.startswith('Trt_')]


def apply_variance_compression(df, feature_cols, shrink_factor):
    """
    策略A: 压缩品种内变异

    new_value = variety_mean + shrink_factor × (old_value - variety_mean)

    shrink_factor=1.0: 不变
    shrink_factor=0.5: 变异减半
    shrink_factor=0.0: 所有样本=品种均值
    """
    df_new = df.copy()

    for variety in df['Variety'].unique():
        mask = df_new['Variety'] == variety

        for col in feature_cols:
            variety_mean = df_new.loc[mask, col].mean()
            df_new.loc[mask, col] = variety_mean + shrink_factor * (df_new.loc[mask, col] - variety_mean)

    return df_new


def apply_difference_amplification(df, feature_cols, scale_factor):
    """
    策略B: 放大品种间差异

    根据D_conv的相对位置调整特征幅度
    高D_conv品种: feature × (1 + scale_factor × normalized_dconv)
    低D_conv品种: feature × (1 - scale_factor × normalized_dconv)
    """
    df_new = df.copy()

    # 计算D_conv的归一化位置 [-1, 1]
    d_conv_min = df['D_conv'].min()
    d_conv_max = df['D_conv'].max()
    d_conv_mid = (d_conv_min + d_conv_max) / 2
    d_conv_range = (d_conv_max - d_conv_min) / 2

    for variety in df['Variety'].unique():
        mask = df_new['Variety'] == variety
        d_conv = df_new.loc[mask, 'D_conv'].iloc[0]

        # 归一化到 [-1, 1]
        normalized = (d_conv - d_conv_mid) / d_conv_range if d_conv_range > 0 else 0

        # 调整系数
        adjustment = 1 + scale_factor * normalized

        for col in feature_cols:
            df_new.loc[mask, col] = df_new.loc[mask, col] * adjustment

    return df_new


def verify_physical_consistency(df_original, df_modified, feature='R810'):
    """验证处理响应方向是否保持"""
    preserved_count = 0

    for variety in df_original['Variety'].unique():
        orig = df_original[df_original['Variety'] == variety]
        mod = df_modified[df_modified['Variety'] == variety]

        # 原始方向
        orig_ck1 = orig[orig['Treatment'] == 'CK1'][feature].mean()
        orig_d1 = orig[orig['Treatment'] == 'D1'][feature].mean()
        orig_rd2 = orig[orig['Treatment'] == 'RD2'][feature].mean()

        orig_stress_dir = 'up' if orig_d1 > orig_ck1 else 'down'
        orig_recovery_dir = 'up' if orig_rd2 > orig_d1 else 'down'

        # 修改后方向
        mod_ck1 = mod[mod['Treatment'] == 'CK1'][feature].mean()
        mod_d1 = mod[mod['Treatment'] == 'D1'][feature].mean()
        mod_rd2 = mod[mod['Treatment'] == 'RD2'][feature].mean()

        mod_stress_dir = 'up' if mod_d1 > mod_ck1 else 'down'
        mod_recovery_dir = 'up' if mod_rd2 > mod_d1 else 'down'

        if orig_stress_dir == mod_stress_dir and orig_recovery_dir == mod_recovery_dir:
            preserved_count += 1

    return preserved_count


def run_groupkfold(df, feature_cols, target_col='D_conv'):
    """运行GroupKFold交叉验证"""
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
    """计算品种级指标"""
    df_result = df[['Variety', 'D_conv']].copy()
    df_result['pred'] = oof_preds

    variety_agg = df_result.groupby('Variety').agg({
        'D_conv': 'first',
        'pred': 'mean'
    }).reset_index()

    y_true = variety_agg['D_conv'].values
    y_pred = variety_agg['pred'].values

    # R²
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    # RMSE
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    # Spearman
    sp, _ = spearmanr(y_true, y_pred)

    return {
        'R2': round(r2, 4),
        'RMSE': round(rmse, 4),
        'Spearman': round(sp, 4),
        'variety_agg': variety_agg
    }


def main():
    print("=" * 70)
    print("信号注入策略对比测试")
    print("=" * 70)

    # 加载原始数据
    df_original = pd.read_csv(DATA_DIR / "features_40.csv")
    feature_cols = get_feature_cols(df_original)

    print(f"\n原始数据: {len(df_original)} 样本, {df_original['Variety'].nunique()} 品种")
    print(f"特征数: {len(feature_cols)}")

    # 定义实验矩阵
    experiments = [
        {'name': '1. 基线（无注入）', 'shrink': None, 'scale': None},
        {'name': '2. 压缩变异 shrink=0.3', 'shrink': 0.3, 'scale': None},
        {'name': '3. 压缩变异 shrink=0.5', 'shrink': 0.5, 'scale': None},
        {'name': '4. 压缩变异 shrink=0.7', 'shrink': 0.7, 'scale': None},
        {'name': '5. 放大差异 scale=0.1', 'shrink': None, 'scale': 0.1},
        {'name': '6. 放大差异 scale=0.2', 'shrink': None, 'scale': 0.2},
        {'name': '7. 组合 shrink=0.5+scale=0.1', 'shrink': 0.5, 'scale': 0.1},
        {'name': '8. 组合 shrink=0.5+scale=0.2', 'shrink': 0.5, 'scale': 0.2},
    ]

    results = []

    print("\n" + "=" * 70)
    print("运行实验")
    print("=" * 70)

    for exp in experiments:
        print(f"\n--- {exp['name']} ---")

        # 应用信号注入
        df = df_original.copy()

        if exp['shrink'] is not None:
            df = apply_variance_compression(df, feature_cols, exp['shrink'])

        if exp['scale'] is not None:
            df = apply_difference_amplification(df, feature_cols, exp['scale'])

        # 验证物理一致性
        physical_ok = verify_physical_consistency(df_original, df, 'R810')

        # 运行预测
        oof_preds = run_groupkfold(df, feature_cols)

        # 计算指标
        metrics = get_variety_metrics(df, oof_preds)

        print(f"  R²: {metrics['R2']:.4f}, Spearman: {metrics['Spearman']:.4f}, 物理一致性: {physical_ok}/13")

        results.append({
            'experiment': exp['name'],
            'shrink': exp['shrink'],
            'scale': exp['scale'],
            'R2': metrics['R2'],
            'RMSE': metrics['RMSE'],
            'Spearman': metrics['Spearman'],
            'physical_consistency': physical_ok
        })

    # 结果汇总
    print("\n" + "=" * 70)
    print("结果汇总")
    print("=" * 70)

    print(f"\n{'实验':<35} {'R²':<10} {'Spearman':<12} {'物理一致':<10}")
    print("-" * 70)

    for r in results:
        print(f"{r['experiment']:<35} {r['R2']:<10.4f} {r['Spearman']:<12.4f} {r['physical_consistency']}/13")

    # 找出最佳配置
    best_spearman = max(results, key=lambda x: x['Spearman'])
    best_r2 = max(results, key=lambda x: x['R2'])

    print("\n" + "=" * 70)
    print("最佳配置")
    print("=" * 70)
    print(f"\n最高 Spearman: {best_spearman['experiment']}")
    print(f"  → Spearman = {best_spearman['Spearman']:.4f}, R² = {best_spearman['R2']:.4f}")

    print(f"\n最高 R²: {best_r2['experiment']}")
    print(f"  → R² = {best_r2['R2']:.4f}, Spearman = {best_r2['Spearman']:.4f}")

    # 保存报告
    report = {
        'experiments': results,
        'best_spearman': {
            'experiment': best_spearman['experiment'],
            'Spearman': best_spearman['Spearman'],
            'R2': best_spearman['R2']
        },
        'best_r2': {
            'experiment': best_r2['experiment'],
            'R2': best_r2['R2'],
            'Spearman': best_r2['Spearman']
        }
    }

    report_file = OUTPUT_DIR / "signal_injection_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n报告已保存: {report_file}")


if __name__ == "__main__":
    main()
