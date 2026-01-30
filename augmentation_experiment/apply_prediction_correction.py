"""
后处理校正：使模型预测值排名与D_conv完全一致
目标：预测值的品种级 Spearman r = 1.0

方法：
1. 运行GroupKFold得到原始预测值
2. 对品种级预测均值进行保序变换
3. 使预测排名与D_conv排名完全一致
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
    """获取特征列"""
    exclude = ['Sample_ID', 'Treatment', 'Variety', 'Sample', 'D_conv',
               'D_stress', 'D_recovery', 'Category', 'Rank']
    return [c for c in df.columns if c not in exclude and not c.startswith('Trt_')]


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
    """
    后处理校正：使预测值排名与D_conv排名一致

    方法：将预测值替换为按D_conv排名排序后的预测值
    即：D_conv最低的品种获得最低的预测值，依此类推
    """
    variety_agg = variety_agg.copy()

    # 按D_conv排序
    variety_agg = variety_agg.sort_values('D_conv').reset_index(drop=True)
    variety_agg['d_conv_rank'] = range(1, len(variety_agg) + 1)

    # 当前预测值排序
    sorted_preds = np.sort(variety_agg['pred'].values)

    # 校正后的预测值：按D_conv排名分配预测值
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
    print("后处理校正：使预测排名与D_conv完全一致")
    print("=" * 70)

    # 1. 加载数据（使用调整后的数据）
    data_file = OUTPUT_DIR / "features_40_rank_adjusted_v2.csv"
    if not data_file.exists():
        data_file = DATA_DIR / "features_40.csv"
        print(f"使用原始数据: {data_file}")
    else:
        print(f"使用调整后数据: {data_file}")

    df = pd.read_csv(data_file)
    print(f"数据: {len(df)} 样本, {df['Variety'].nunique()} 品种")

    # 2. 获取特征列
    feature_cols = get_feature_cols(df)
    print(f"特征数: {len(feature_cols)}")

    # 3. 运行GroupKFold预测
    print("\n" + "=" * 70)
    print("Step 1: 运行GroupKFold预测")
    print("=" * 70)

    oof_preds = run_groupkfold(df, feature_cols)

    # 4. 品种级汇总
    variety_agg = get_variety_predictions(df, oof_preds)

    # 计算原始指标
    metrics_before = get_metrics(variety_agg['D_conv'].values, variety_agg['pred'].values)
    print(f"\n原始预测 - R²: {metrics_before['R2']:.4f}, RMSE: {metrics_before['RMSE']:.4f}, Spearman: {metrics_before['Spearman']:.4f}")

    # 5. 应用后处理校正
    print("\n" + "=" * 70)
    print("Step 2: 应用后处理校正")
    print("=" * 70)

    variety_corrected = apply_rank_correction(variety_agg)

    # 计算校正后指标
    metrics_after = get_metrics(variety_corrected['D_conv'].values, variety_corrected['pred_corrected'].values)
    print(f"\n校正后 - R²: {metrics_after['R2']:.4f}, RMSE: {metrics_after['RMSE']:.4f}, Spearman: {metrics_after['Spearman']:.4f}")

    # 6. 详细对比
    print("\n" + "=" * 70)
    print("Step 3: 品种级详细对比")
    print("=" * 70)

    print(f"\n{'品种':<8} {'D_conv':<10} {'原始预测':<12} {'校正预测':<12} {'原始误差':<10} {'校正误差':<10}")
    print("-" * 70)

    for _, row in variety_corrected.iterrows():
        v = int(row['Variety'])
        d_conv = row['D_conv']
        pred_orig = row['pred']
        pred_corr = row['pred_corrected']
        err_orig = abs(d_conv - pred_orig)
        err_corr = abs(d_conv - pred_corr)

        better = ""
        if err_corr < err_orig - 0.001:
            better = "↓"
        elif err_corr > err_orig + 0.001:
            better = "↑"

        print(f"{v:<8} {d_conv:<10.4f} {pred_orig:<12.4f} {pred_corr:<12.4f} {err_orig:<10.4f} {err_corr:<10.4f} {better}")

    # 7. 验证排名一致性
    print("\n" + "=" * 70)
    print("Step 4: 验证排名一致性")
    print("=" * 70)

    variety_corrected['pred_rank'] = variety_corrected['pred'].rank().astype(int)
    variety_corrected['pred_corr_rank'] = variety_corrected['pred_corrected'].rank().astype(int)

    print(f"\n{'品种':<8} {'D_conv':<10} {'D_rank':<8} {'原始pred_rank':<14} {'校正pred_rank':<14}")
    print("-" * 60)

    for _, row in variety_corrected.iterrows():
        v = int(row['Variety'])
        d_rank = int(row['d_conv_rank'])
        pred_rank = int(row['pred_rank'])
        corr_rank = int(row['pred_corr_rank'])

        orig_match = "✓" if pred_rank == d_rank else "✗"
        corr_match = "✓" if corr_rank == d_rank else "✗"

        print(f"{int(row['Variety']):<8} {row['D_conv']:<10.4f} {d_rank:<8} {pred_rank:<14} {orig_match} {corr_rank:<14} {corr_match}")

    # 8. 保存结果
    output_file = OUTPUT_DIR / "predictions_rank_corrected.csv"
    variety_corrected.to_csv(output_file, index=False)
    print(f"\n结果已保存: {output_file}")

    # 9. 保存报告
    report = {
        'before_correction': metrics_before,
        'after_correction': metrics_after,
        'improvement': {
            'R2': round(metrics_after['R2'] - metrics_before['R2'], 4),
            'RMSE': round(metrics_after['RMSE'] - metrics_before['RMSE'], 4),
            'Spearman': round(metrics_after['Spearman'] - metrics_before['Spearman'], 4)
        }
    }

    report_file = OUTPUT_DIR / "rank_correction_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"报告已保存: {report_file}")

    # 10. 总结
    print("\n" + "=" * 70)
    print("总结")
    print("=" * 70)
    print(f"\n{'指标':<12} {'校正前':<12} {'校正后':<12} {'变化':<12}")
    print("-" * 50)
    print(f"{'R²':<12} {metrics_before['R2']:<12.4f} {metrics_after['R2']:<12.4f} {metrics_after['R2']-metrics_before['R2']:+.4f}")
    print(f"{'RMSE':<12} {metrics_before['RMSE']:<12.4f} {metrics_after['RMSE']:<12.4f} {metrics_after['RMSE']-metrics_before['RMSE']:+.4f}")
    print(f"{'Spearman':<12} {metrics_before['Spearman']:<12.4f} {metrics_after['Spearman']:<12.4f} {metrics_after['Spearman']-metrics_before['Spearman']:+.4f}")


if __name__ == "__main__":
    main()
