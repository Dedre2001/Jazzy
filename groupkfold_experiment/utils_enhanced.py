"""
增强版工具函数
支持样本加权和GroupKFold
"""

import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
from sklearn.model_selection import KFold, GroupKFold
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

RANDOM_STATE = 42
N_SPLITS = 5

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
    # R2
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    # RMSE
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    # MAE
    mae = np.mean(np.abs(y_true - y_pred))

    # Spearman
    spearman_r, spearman_p = spearmanr(y_true, y_pred)

    # Pairwise Ranking Accuracy
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
    """
    计算样本权重，对边缘标签样本增加权重
    """
    low_threshold = np.quantile(y, q_low)
    high_threshold = np.quantile(y, q_high)

    weights = np.ones(len(y))
    weights[(y <= low_threshold) | (y >= high_threshold)] = boost

    return weights


def sample_level_normalize(df, band_cols, static_cols):
    """按样本对光谱/静态荧光做幅值归一化（z-score per sample）"""
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


def run_group_kfold_cv(df, feature_cols, model_class, model_params,
                       target_col='D_conv', n_splits=5,
                       sample_weight=None, use_sample_norm=True):
    """
    运行GroupKFold交叉验证（按品种划分）

    参数:
    - sample_weight: 可选的样本权重数组
    - use_sample_norm: 是否使用样本级归一化
    """
    # 推断特征类型
    band_cols = [c for c in feature_cols if c.startswith('R') and not c.startswith('RF')]
    static_cols = [c for c in feature_cols if c in ['BF(F440)', 'GF(F520)', 'RF(F690)', 'FrF(f740)']]

    # 样本级归一化
    if use_sample_norm and (band_cols or static_cols):
        df_norm = sample_level_normalize(df[feature_cols], band_cols, static_cols)
        X = df_norm.values
    else:
        X = df[feature_cols].values

    y = df[target_col].values
    groups = df['Variety'].values

    n_unique = len(np.unique(groups))
    n_splits = min(n_splits, n_unique)

    oof_preds = np.zeros(len(y))
    fold_metrics = []

    gkf = GroupKFold(n_splits=n_splits)

    for fold, (tr_idx, te_idx) in enumerate(gkf.split(X, y, groups=groups)):
        X_train, X_test = X[tr_idx], X[te_idx]
        y_train, y_test = y[tr_idx], y[te_idx]

        # 折内标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 训练模型
        model = model_class(**model_params)

        # 如果模型支持样本权重
        if sample_weight is not None and hasattr(model, 'fit'):
            try:
                model.fit(X_train_scaled, y_train, sample_weight=sample_weight[tr_idx])
            except TypeError:
                model.fit(X_train_scaled, y_train)
        else:
            model.fit(X_train_scaled, y_train)

        # 预测
        y_pred = model.predict(X_test_scaled)
        if hasattr(y_pred, 'ndim') and y_pred.ndim > 1:
            y_pred = y_pred.ravel()

        oof_preds[te_idx] = y_pred

    # 聚合到品种层
    df_result = df[['Variety', target_col]].copy()
    df_result['pred'] = oof_preds

    variety_agg = df_result.groupby('Variety').agg({
        target_col: 'first',
        'pred': 'mean'
    }).reset_index()

    # 计算品种层指标
    metrics = get_variety_metrics(
        variety_agg[target_col].values,
        variety_agg['pred'].values
    )

    return metrics, variety_agg, oof_preds


def run_random_kfold_cv(df, feature_cols, model_class, model_params,
                        target_col='D_conv', n_splits=5, use_sample_norm=True):
    """
    运行随机KFold交叉验证（可能存在数据泄露）
    """
    band_cols = [c for c in feature_cols if c.startswith('R') and not c.startswith('RF')]
    static_cols = [c for c in feature_cols if c in ['BF(F440)', 'GF(F520)', 'RF(F690)', 'FrF(f740)']]

    if use_sample_norm and (band_cols or static_cols):
        df_norm = sample_level_normalize(df[feature_cols], band_cols, static_cols)
        X = df_norm.values
    else:
        X = df[feature_cols].values

    y = df[target_col].values

    oof_preds = np.zeros(len(y))

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    for fold, (tr_idx, te_idx) in enumerate(kf.split(X)):
        X_train, X_test = X[tr_idx], X[te_idx]
        y_train, y_test = y[tr_idx], y[te_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = model_class(**model_params)
        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)
        if hasattr(y_pred, 'ndim') and y_pred.ndim > 1:
            y_pred = y_pred.ravel()

        oof_preds[te_idx] = y_pred

    # 聚合到品种层
    df_result = df[['Variety', target_col]].copy()
    df_result['pred'] = oof_preds

    variety_agg = df_result.groupby('Variety').agg({
        target_col: 'first',
        'pred': 'mean'
    }).reset_index()

    metrics = get_variety_metrics(
        variety_agg[target_col].values,
        variety_agg['pred'].values
    )

    return metrics, variety_agg, oof_preds
