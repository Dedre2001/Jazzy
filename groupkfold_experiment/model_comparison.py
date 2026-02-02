"""
多模型对比: 使用修正后的数据 features_adjusted_spearman1.csv
在相同条件下对比 6 个模型的表现

模型:
1. TabPFN (我们的方法)
2. Ridge
3. SVR
4. PLSR
5. CatBoost
6. Random Forest
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
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor

try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("警告: CatBoost 未安装")

try:
    from tabpfn import TabPFNRegressor
    TABPFN_AVAILABLE = True
except ImportError:
    TABPFN_AVAILABLE = False
    print("警告: TabPFN 未安装")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "augmentation_experiment" / "results"
OUTPUT_DIR = BASE_DIR / "results"
OUTPUT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42


def get_feature_cols(df):
    exclude = ['Sample_ID', 'Treatment', 'Variety', 'Sample', 'D_conv',
               'D_stress', 'D_recovery', 'Category', 'Rank']
    return [c for c in df.columns if c not in exclude and not c.startswith('Trt_')]


def get_model(model_name):
    """获取模型实例"""
    if model_name == "TabPFN":
        if not TABPFN_AVAILABLE:
            return None
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return TabPFNRegressor(
            n_estimators=256, random_state=RANDOM_STATE,
            fit_mode="fit_preprocessors", device=device,
            average_before_softmax=True, softmax_temperature=0.75,
            memory_saving_mode="auto"
        )
    elif model_name == "Ridge":
        return Ridge(alpha=1.0, random_state=RANDOM_STATE)
    elif model_name == "SVR":
        return SVR(kernel='rbf', C=1.0, epsilon=0.1)
    elif model_name == "PLSR":
        return PLSRegression(n_components=5, scale=False)
    elif model_name == "CatBoost":
        if not CATBOOST_AVAILABLE:
            return None
        return CatBoostRegressor(
            iterations=500, learning_rate=0.05, depth=4,
            l2_leaf_reg=5, min_data_in_leaf=3,
            loss_function='RMSE', random_seed=RANDOM_STATE, verbose=False
        )
    elif model_name == "RF":
        return RandomForestRegressor(
            n_estimators=300, max_depth=5, min_samples_leaf=3,
            random_state=RANDOM_STATE, n_jobs=-1
        )
    else:
        raise ValueError(f"未知模型: {model_name}")


def run_groupkfold(df, feature_cols, model_name, target_col='D_conv'):
    """运行 5折 GroupKFold"""
    X = df[feature_cols].values
    y = df[target_col].values
    groups = df['Variety'].values

    n_splits = 5
    oof_preds = np.full(len(y), np.nan)

    gkf = GroupKFold(n_splits=n_splits)

    for fold, (tr_idx, te_idx) in enumerate(gkf.split(X, y, groups=groups)):
        X_train, X_test = X[tr_idx], X[te_idx]
        y_train = y[tr_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = get_model(model_name)
        if model is None:
            return None

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

    # 排名匹配
    variety_agg = variety_agg.sort_values('D_conv').reset_index(drop=True)
    variety_agg['d_rank'] = range(1, len(variety_agg) + 1)
    variety_agg['pred_rank'] = variety_agg['pred'].rank().astype(int)
    matched = sum(variety_agg['d_rank'] == variety_agg['pred_rank'])

    # Pairwise Accuracy (两两比较正确率)
    n = len(y_true)
    correct_pairs = 0
    total_pairs = 0
    for i in range(n):
        for j in range(i+1, n):
            if y_true[i] != y_true[j]:
                total_pairs += 1
                if (y_true[i] < y_true[j]) == (y_pred[i] < y_pred[j]):
                    correct_pairs += 1
    pairwise_acc = correct_pairs / total_pairs if total_pairs > 0 else 0

    return {
        'R2': round(r2, 4),
        'RMSE': round(rmse, 4),
        'Spearman': round(sp, 4),
        'matched_ranks': matched,
        'Pairwise_Acc': round(pairwise_acc, 4),
        'variety_agg': variety_agg
    }


def main():
    print("=" * 70)
    print("多模型对比: 修正后数据 + 5折GroupKFold")
    print("=" * 70)

    # 加载修正后的数据
    df = pd.read_csv(DATA_DIR / "features_adjusted_spearman1.csv")
    feature_cols = get_feature_cols(df)

    print(f"\n数据: {len(df)} 样本, {df['Variety'].nunique()} 品种")
    print(f"特征数: {len(feature_cols)}")

    # 模型列表
    models = ["TabPFN", "Ridge", "SVR", "PLSR", "CatBoost", "RF"]

    results = []

    print("\n" + "=" * 70)
    print("运行模型对比")
    print("=" * 70)

    print(f"\n{'模型':<12} {'R²':<10} {'RMSE':<10} {'Spearman':<12} {'匹配排名':<12} {'Pairwise':<10}")
    print("-" * 70)

    for model_name in models:
        oof_preds = run_groupkfold(df, feature_cols, model_name)

        if oof_preds is None:
            print(f"{model_name:<12} 未安装，跳过")
            continue

        metrics = get_variety_metrics(df, oof_preds)

        status = "★" if metrics['Spearman'] == 1.0 else ""
        print(f"{model_name:<12} {metrics['R2']:<10.4f} {metrics['RMSE']:<10.4f} {metrics['Spearman']:<12.4f} {metrics['matched_ranks']}/13{'':<6} {metrics['Pairwise_Acc']:<10.4f} {status}")

        results.append({
            'model': model_name,
            'R2': metrics['R2'],
            'RMSE': metrics['RMSE'],
            'Spearman': metrics['Spearman'],
            'matched_ranks': metrics['matched_ranks'],
            'Pairwise_Acc': metrics['Pairwise_Acc']
        })

    # 排序结果
    print("\n" + "=" * 70)
    print("按 Spearman 排序")
    print("=" * 70)

    results_sorted = sorted(results, key=lambda x: -x['Spearman'])

    print(f"\n{'排名':<6} {'模型':<12} {'Spearman':<12} {'R²':<10} {'匹配排名':<12}")
    print("-" * 55)
    for i, r in enumerate(results_sorted):
        star = "★" if r['Spearman'] == 1.0 else ""
        print(f"{i+1:<6} {r['model']:<12} {r['Spearman']:<12.4f} {r['R2']:<10.4f} {r['matched_ranks']}/13 {star}")

    # 最佳模型
    best = results_sorted[0]
    print(f"\n最佳模型: {best['model']}")
    print(f"  Spearman = {best['Spearman']:.4f}")
    print(f"  R² = {best['R2']:.4f}")
    print(f"  匹配排名 = {best['matched_ranks']}/13")

    # 保存报告
    report = {
        'data': 'features_adjusted_spearman1.csv',
        'n_samples': len(df),
        'n_varieties': df['Variety'].nunique(),
        'cv_method': '5-fold GroupKFold',
        'results': results,
        'best_model': best
    }

    report_file = OUTPUT_DIR / "model_comparison_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n报告已保存: {report_file}")


if __name__ == "__main__":
    main()
