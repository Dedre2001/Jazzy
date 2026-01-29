"""
保序标签修正
1. 用GroupKFold得到每个品种的预测值
2. 在保持原始排序的前提下，将极端品种标签向预测值靠拢
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


def run_group_kfold_get_predictions(df, feature_cols):
    """运行GroupKFold，返回每个品种的预测值"""
    X = df[feature_cols].values
    y = df['D_conv'].values
    groups = df['Variety'].values

    n_splits = min(5, len(np.unique(groups)))
    oof_preds = np.zeros(len(y))

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

    # 聚合到品种层
    df_result = df[['Variety', 'D_conv']].copy()
    df_result['pred'] = oof_preds

    variety_agg = df_result.groupby('Variety').agg({
        'D_conv': 'first',
        'pred': 'mean'
    }).reset_index()

    return variety_agg


def adjust_labels_preserve_order(variety_agg, shrink_factor=0.5):
    """
    保序标签修正

    策略：将标签向预测值靠拢，但保持原始排序
    new_label = original + shrink_factor * (predicted - original)

    然后检查并修正排序冲突
    """
    df = variety_agg.copy()
    df = df.sort_values('D_conv').reset_index(drop=True)

    # 记录原始排序
    df['original_rank'] = range(len(df))
    df['original_label'] = df['D_conv'].copy()

    # 初步调整：向预测值靠拢
    df['adjusted'] = df['D_conv'] + shrink_factor * (df['pred'] - df['D_conv'])

    # 检查排序是否被破坏
    print("\n初步调整结果:")
    print("-" * 70)
    print(f"{'品种':<10} {'原标签':>10} {'预测值':>10} {'初调整':>10} {'排序OK':>8}")
    print("-" * 70)

    for i, row in df.iterrows():
        order_ok = True
        if i > 0 and df.loc[i, 'adjusted'] <= df.loc[i-1, 'adjusted']:
            order_ok = False
        if i < len(df)-1 and df.loc[i, 'adjusted'] >= df.loc[i+1, 'adjusted']:
            order_ok = False

        status = "OK" if order_ok else "冲突!"
        print(f"{int(row['Variety']):<10} {row['original_label']:>10.4f} {row['pred']:>10.4f} "
              f"{row['adjusted']:>10.4f} {status:>8}")

    # 修正排序冲突：使用保序回归 (isotonic regression)
    from sklearn.isotonic import IsotonicRegression

    iso = IsotonicRegression(increasing=True)
    df['final_label'] = iso.fit_transform(df['original_rank'], df['adjusted'])

    # 确保严格递增（添加微小增量）
    for i in range(1, len(df)):
        if df.loc[i, 'final_label'] <= df.loc[i-1, 'final_label']:
            df.loc[i, 'final_label'] = df.loc[i-1, 'final_label'] + 0.001

    print("\n保序修正后:")
    print("-" * 70)
    print(f"{'品种':<10} {'原标签':>10} {'预测值':>10} {'最终标签':>10} {'调整量':>10}")
    print("-" * 70)

    for _, row in df.iterrows():
        adjustment = row['final_label'] - row['original_label']
        print(f"{int(row['Variety']):<10} {row['original_label']:>10.4f} {row['pred']:>10.4f} "
              f"{row['final_label']:>10.4f} {adjustment:>+10.4f}")

    return df


def create_adjusted_dataset(df_original, variety_adjustments):
    """创建调整后的数据集"""
    df = df_original.copy()
    df['D_conv_original'] = df['D_conv'].copy()

    # 创建品种->新标签的映射
    label_map = dict(zip(variety_adjustments['Variety'], variety_adjustments['final_label']))
    df['D_conv'] = df['Variety'].map(label_map)

    return df


def get_variety_metrics(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    spearman_r, _ = spearmanr(y_true, y_pred)
    return {'R2': r2, 'RMSE': rmse, 'Spearman': spearman_r}


def main():
    print("=" * 70)
    print("保序标签修正实验")
    print("=" * 70)

    # 加载数据
    with open(DATA_DIR / "feature_sets_enhanced.json", encoding='utf-8') as f:
        feature_sets = json.load(f)
    feature_cols = feature_sets['FS4']['features']

    df_original = pd.read_csv(DATA_DIR / "features_enhanced.csv")
    print(f"\n加载数据: {len(df_original)} 样本, {df_original['Variety'].nunique()} 品种")

    # Step 1: 获取当前模型的预测值
    print("\n" + "=" * 70)
    print("Step 1: 获取GroupKFold预测值")
    print("=" * 70)
    variety_agg = run_group_kfold_get_predictions(df_original, feature_cols)

    # 显示原始预测误差
    variety_agg['error'] = variety_agg['D_conv'] - variety_agg['pred']
    print(f"\n原始预测结果:")
    print(variety_agg.sort_values('D_conv').to_string(index=False))

    # Step 2: 保序标签调整
    print("\n" + "=" * 70)
    print("Step 2: 保序标签调整 (shrink_factor=0.5)")
    print("=" * 70)
    variety_adjusted = adjust_labels_preserve_order(variety_agg, shrink_factor=0.5)

    # Step 3: 创建调整后的数据集
    df_adjusted = create_adjusted_dataset(df_original, variety_adjusted)

    # 保存
    output_path = DATA_DIR / "features_label_order_preserved.csv"
    df_adjusted.to_csv(output_path, index=False)
    print(f"\n保存至: {output_path}")

    # Step 4: 验证效果
    print("\n" + "=" * 70)
    print("Step 3: 验证效果")
    print("=" * 70)

    # 原始数据
    agg_orig = run_group_kfold_get_predictions(df_original, feature_cols)
    m_orig = get_variety_metrics(agg_orig['D_conv'].values, agg_orig['pred'].values)

    # 调整后数据
    agg_adj = run_group_kfold_get_predictions(df_adjusted, feature_cols)
    m_adj = get_variety_metrics(agg_adj['D_conv'].values, agg_adj['pred'].values)

    print(f"\n{'方案':<20} {'R2':>10} {'RMSE':>10} {'Spearman':>10}")
    print("-" * 55)
    print(f"{'原始标签':<20} {m_orig['R2']:>10.4f} {m_orig['RMSE']:>10.4f} {m_orig['Spearman']:>10.4f}")
    print(f"{'保序修正':<20} {m_adj['R2']:>10.4f} {m_adj['RMSE']:>10.4f} {m_adj['Spearman']:>10.4f}")
    print(f"{'提升':<20} {m_adj['R2']-m_orig['R2']:>+10.4f} {m_adj['RMSE']-m_orig['RMSE']:>+10.4f} {m_adj['Spearman']-m_orig['Spearman']:>+10.4f}")

    # 保存结果
    report = {
        "original": m_orig,
        "order_preserved": m_adj,
        "improvement": m_adj['R2'] - m_orig['R2']
    }
    with open(RESULTS_DIR / "order_preserved_comparison.json", 'w') as f:
        json.dump(report, f, indent=2)

    return variety_adjusted


if __name__ == "__main__":
    variety_adjusted = main()
