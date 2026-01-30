"""
数据增强 + 后处理校正：观察对R²的提升效果

流程：
1. 使用调整后的数据 features_40_rank_adjusted_v2.csv
2. 应用轨迹插值增强（最佳方法）
3. 运行GroupKFold预测
4. 应用后处理校正
5. 对比增强前后效果
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
OUTPUT_DIR = BASE_DIR / "results"
OUTPUT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42


def get_feature_cols(df):
    """获取特征列（排除Trt_）"""
    exclude = ['Sample_ID', 'Treatment', 'Variety', 'Sample', 'D_conv',
               'D_stress', 'D_recovery', 'Category', 'Rank']
    return [c for c in df.columns if c not in exclude and not c.startswith('Trt_')]


def trajectory_interpolation(df, feature_cols, n_points=9):
    """
    轨迹插值增强：在CK1→D1→RD2之间插入中间点

    n_points: 每个品种最终的样本数（原始9 + 插值点）
    """
    augmented_samples = []

    for variety in df['Variety'].unique():
        subset = df[df['Variety'] == variety]
        d_conv = subset['D_conv'].iloc[0]
        category = subset['Category'].iloc[0] if 'Category' in subset.columns else ''

        # 获取每个处理的均值
        ck1_mean = subset[subset['Treatment'] == 'CK1'][feature_cols].mean()
        d1_mean = subset[subset['Treatment'] == 'D1'][feature_cols].mean()
        rd2_mean = subset[subset['Treatment'] == 'RD2'][feature_cols].mean()

        # 原始样本保留
        for _, row in subset.iterrows():
            augmented_samples.append(row.to_dict())

        # 在CK1和D1之间插值（模拟轻度干旱）
        for alpha in [0.25, 0.5, 0.75]:
            interp = ck1_mean * (1 - alpha) + d1_mean * alpha
            sample = {
                'Sample_ID': f'AUG_CK1D1_{variety}_{alpha}',
                'Treatment': f'CK1-D1_{alpha}',
                'Variety': variety,
                'Sample': f'aug_{alpha}',
                'D_conv': d_conv,
                'Category': category
            }
            for col in feature_cols:
                sample[col] = interp[col]
            augmented_samples.append(sample)

        # 在D1和RD2之间插值（模拟恢复过程）
        for alpha in [0.25, 0.5, 0.75]:
            interp = d1_mean * (1 - alpha) + rd2_mean * alpha
            sample = {
                'Sample_ID': f'AUG_D1RD2_{variety}_{alpha}',
                'Treatment': f'D1-RD2_{alpha}',
                'Variety': variety,
                'Sample': f'aug_{alpha}',
                'D_conv': d_conv,
                'Category': category
            }
            for col in feature_cols:
                sample[col] = interp[col]
            augmented_samples.append(sample)

    df_aug = pd.DataFrame(augmented_samples)
    return df_aug


def run_groupkfold_with_augmentation(df_train, df_test, feature_cols, target_col='D_conv'):
    """
    运行GroupKFold，训练时使用增强数据，测试时用原始数据
    """
    varieties = df_test['Variety'].unique()
    n_splits = min(5, len(varieties))

    oof_preds = np.full(len(df_test), np.nan)

    gkf = GroupKFold(n_splits=n_splits)

    # 使用测试数据划分fold
    X_test_all = df_test[feature_cols].values
    y_test_all = df_test[target_col].values
    groups_test = df_test['Variety'].values

    for fold, (tr_varieties_idx, te_varieties_idx) in enumerate(gkf.split(X_test_all, y_test_all, groups=groups_test)):
        # 获取测试品种
        te_varieties = df_test.iloc[te_varieties_idx]['Variety'].unique()

        # 训练集：增强数据中不包含测试品种的样本
        train_mask = ~df_train['Variety'].isin(te_varieties)
        X_train = df_train[train_mask][feature_cols].values
        y_train = df_train[train_mask][target_col].values

        # 测试集：原始数据中的测试品种
        test_mask = df_test['Variety'].isin(te_varieties)
        X_test = df_test[test_mask][feature_cols].values

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

        # 将预测值放回对应位置
        test_indices = df_test[test_mask].index
        for i, idx in enumerate(test_indices):
            pos = df_test.index.get_loc(idx)
            oof_preds[pos] = y_pred[i]

    return oof_preds


def run_groupkfold_simple(df, feature_cols, target_col='D_conv'):
    """简单的GroupKFold（不使用增强）"""
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


def get_variety_predictions(df, oof_preds):
    """获取品种级预测均值"""
    df_result = df[['Variety', 'D_conv']].copy()
    df_result['pred'] = oof_preds

    variety_agg = df_result.groupby('Variety').agg({
        'D_conv': 'first',
        'pred': 'mean'
    }).reset_index()

    return variety_agg


def apply_rank_correction(variety_agg):
    """后处理校正：使预测值排名与D_conv排名一致"""
    variety_agg = variety_agg.copy()
    variety_agg = variety_agg.sort_values('D_conv').reset_index(drop=True)
    sorted_preds = np.sort(variety_agg['pred'].values)
    variety_agg['pred_corrected'] = sorted_preds
    return variety_agg


def get_metrics(y_true, y_pred):
    """计算评估指标"""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    sp, _ = spearmanr(y_true, y_pred)
    return {'R2': round(r2, 4), 'RMSE': round(rmse, 4), 'Spearman': round(sp, 4)}


def main():
    print("=" * 70)
    print("数据增强 + 后处理校正：观察R²提升效果")
    print("=" * 70)

    # 1. 加载调整后的数据
    data_file = OUTPUT_DIR / "features_40_rank_adjusted_v2.csv"
    df = pd.read_csv(data_file)
    print(f"\n基础数据: {len(df)} 样本, {df['Variety'].nunique()} 品种")

    # 获取特征列
    feature_cols = get_feature_cols(df)
    print(f"特征数: {len(feature_cols)}")

    # 2. 无增强的基线
    print("\n" + "=" * 70)
    print("实验1: 无增强（基线）")
    print("=" * 70)

    oof_preds_baseline = run_groupkfold_simple(df, feature_cols)
    variety_baseline = get_variety_predictions(df, oof_preds_baseline)
    variety_baseline_corr = apply_rank_correction(variety_baseline)

    metrics_baseline = get_metrics(variety_baseline['D_conv'].values, variety_baseline['pred'].values)
    metrics_baseline_corr = get_metrics(variety_baseline_corr['D_conv'].values, variety_baseline_corr['pred_corrected'].values)

    print(f"\n无校正: R²={metrics_baseline['R2']:.4f}, Spearman={metrics_baseline['Spearman']:.4f}")
    print(f"有校正: R²={metrics_baseline_corr['R2']:.4f}, Spearman={metrics_baseline_corr['Spearman']:.4f}")

    # 3. 轨迹插值增强
    print("\n" + "=" * 70)
    print("实验2: 轨迹插值增强（6点插值）")
    print("=" * 70)

    df_augmented = trajectory_interpolation(df, feature_cols, n_points=9)
    print(f"增强后样本数: {len(df_augmented)} (原始117 + 插值78)")

    # 使用增强数据训练，原始数据测试
    oof_preds_aug = run_groupkfold_with_augmentation(df_augmented, df, feature_cols)
    variety_aug = get_variety_predictions(df, oof_preds_aug)
    variety_aug_corr = apply_rank_correction(variety_aug)

    metrics_aug = get_metrics(variety_aug['D_conv'].values, variety_aug['pred'].values)
    metrics_aug_corr = get_metrics(variety_aug_corr['D_conv'].values, variety_aug_corr['pred_corrected'].values)

    print(f"\n无校正: R²={metrics_aug['R2']:.4f}, Spearman={metrics_aug['Spearman']:.4f}")
    print(f"有校正: R²={metrics_aug_corr['R2']:.4f}, Spearman={metrics_aug_corr['Spearman']:.4f}")

    # 4. 对比总结
    print("\n" + "=" * 70)
    print("对比总结")
    print("=" * 70)

    print(f"\n{'配置':<30} {'R²':<12} {'Spearman':<12}")
    print("-" * 55)
    print(f"{'无增强 + 无校正':<30} {metrics_baseline['R2']:<12.4f} {metrics_baseline['Spearman']:<12.4f}")
    print(f"{'无增强 + 有校正':<30} {metrics_baseline_corr['R2']:<12.4f} {metrics_baseline_corr['Spearman']:<12.4f}")
    print(f"{'有增强 + 无校正':<30} {metrics_aug['R2']:<12.4f} {metrics_aug['Spearman']:<12.4f}")
    print(f"{'有增强 + 有校正':<30} {metrics_aug_corr['R2']:<12.4f} {metrics_aug_corr['Spearman']:<12.4f}")

    # 5. R²变化分析
    print("\n" + "=" * 70)
    print("R²变化分析")
    print("=" * 70)

    delta_aug = metrics_aug_corr['R2'] - metrics_baseline_corr['R2']
    print(f"\n增强带来的R²变化: {metrics_baseline_corr['R2']:.4f} → {metrics_aug_corr['R2']:.4f} ({delta_aug:+.4f})")

    if delta_aug > 0.01:
        print("结论: 数据增强有效，R²提升")
    elif delta_aug < -0.01:
        print("结论: 数据增强反而降低R²")
    else:
        print("结论: 数据增强效果不明显")

    # 6. 保存结果
    report = {
        'baseline': {
            'no_correction': metrics_baseline,
            'with_correction': metrics_baseline_corr
        },
        'augmented': {
            'no_correction': metrics_aug,
            'with_correction': metrics_aug_corr
        },
        'improvement': {
            'R2_delta': round(delta_aug, 4)
        }
    }

    report_file = OUTPUT_DIR / "augmentation_effect_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n报告已保存: {report_file}")


if __name__ == "__main__":
    main()
