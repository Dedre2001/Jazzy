"""
Step 4 共享工具函数
复用 step2_exp1_fusion_comparison.py 中已验证的代码
"""

import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

# 路径配置
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
RESULTS_DIR = BASE_DIR / "results"
EXP6_DIR = RESULTS_DIR / "exp6"

# 随机种子
RANDOM_STATE = 42
N_SPLITS = 5

# 确保目录存在
os.makedirs(EXP6_DIR, exist_ok=True)


def load_data():
    """加载特征数据和特征集定义"""
    # 允许通过环境变量覆盖特征文件路径，便于增强/对比实验
    features_path = Path(os.environ.get("FEATURES_PATH", DATA_DIR / "features_40.csv"))
    df = pd.read_csv(features_path)
    with open(DATA_DIR / "feature_sets.json", 'r', encoding='utf-8') as f:
        feature_sets = json.load(f)
    return df, feature_sets


def get_variety_metrics(y_true_variety, y_pred_variety):
    """
    计算品种层指标（复用step2验证过的代码）

    输入: 品种层的真值和预测值 (长度=13)
    """
    # R2
    ss_res = np.sum((y_true_variety - y_pred_variety) ** 2)
    ss_tot = np.sum((y_true_variety - np.mean(y_true_variety)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    # RMSE
    rmse = np.sqrt(np.mean((y_true_variety - y_pred_variety) ** 2))

    # MAE
    mae = np.mean(np.abs(y_true_variety - y_pred_variety))

    # Spearman相关
    spearman_r, spearman_p = spearmanr(y_true_variety, y_pred_variety)

    # Pairwise Ranking Accuracy
    n = len(y_true_variety)
    correct_pairs = 0
    total_pairs = 0
    for i in range(n):
        for j in range(i+1, n):
            total_pairs += 1
            true_order = y_true_variety[i] > y_true_variety[j]
            pred_order = y_pred_variety[i] > y_pred_variety[j]
            if true_order == pred_order:
                correct_pairs += 1
    pairwise_acc = correct_pairs / total_pairs if total_pairs > 0 else 0

    # Hit@3 (Top-3命中率)
    true_top3 = set(np.argsort(y_true_variety)[-3:])
    pred_top3 = set(np.argsort(y_pred_variety)[-3:])
    hit_at_3 = len(true_top3 & pred_top3) / 3

    # Hit@5
    true_top5 = set(np.argsort(y_true_variety)[-5:])
    pred_top5 = set(np.argsort(y_pred_variety)[-5:])
    hit_at_5 = len(true_top5 & pred_top5) / 5

    return {
        'R2': float(r2),
        'RMSE': float(rmse),
        'MAE': float(mae),
        'Spearman': float(spearman_r),
        'Spearman_p': float(spearman_p),
        'Pairwise_Acc': float(pairwise_acc),
        'Hit@3': float(hit_at_3),
        'Hit@5': float(hit_at_5)
    }


def run_kfold_cv(df, feature_cols, model_class, model_params, target_col='D_conv', n_splits=5):
    """
    运行5-Fold GroupKFold交叉验证，返回OOF预测和品种层指标

    参数:
    - df: 数据框
    - feature_cols: 特征列名列表
    - model_class: 模型类
    - model_params: 模型参数字典
    - target_col: 目标列名
    - n_splits: 折数

    返回:
    - metrics: 品种层指标字典
    - variety_agg: 品种层聚合结果DataFrame
    """
    X = df[feature_cols].values
    y = df[target_col].values
    groups = df['Variety'].values  # 按品种分组

    # 存储OOF预测
    oof_predictions = np.zeros(len(y))

    gkfold = GroupKFold(n_splits=n_splits)

    for fold, (train_idx, test_idx) in enumerate(gkfold.split(X, y, groups=groups)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # 折内标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 训练模型
        model = model_class(**model_params)
        model.fit(X_train_scaled, y_train)

        # 预测
        y_pred = model.predict(X_test_scaled)

        # 处理可能的多维输出
        if hasattr(y_pred, 'ndim') and y_pred.ndim > 1:
            y_pred = y_pred.ravel()

        oof_predictions[test_idx] = y_pred

    # 聚合到品种层
    df_result = df[['Variety', target_col]].copy()
    df_result['pred'] = oof_predictions

    variety_agg = df_result.groupby('Variety').agg({
        target_col: 'first',  # 真值（每个品种相同）
        'pred': 'mean'        # 预测值取平均
    }).reset_index()

    # 计算品种层指标
    metrics = get_variety_metrics(
        variety_agg[target_col].values,
        variety_agg['pred'].values
    )

    return metrics, variety_agg


def save_model_results(model_name, layer, metrics, variety_agg):
    """保存模型结果到JSON和CSV"""
    # 保存指标
    result = {
        'model': model_name,
        'layer': layer,
        'metrics': metrics
    }

    json_path = EXP6_DIR / f"model_{model_name.lower()}_results.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    # 保存品种层预测
    csv_path = EXP6_DIR / f"model_{model_name.lower()}_variety_pred.csv"
    variety_agg.to_csv(csv_path, index=False)

    print(f"[OK] {model_name} 结果已保存:")
    print(f"     - {json_path}")
    print(f"     - {csv_path}")

    return json_path, csv_path
