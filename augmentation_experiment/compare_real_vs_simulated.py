"""
用真实光谱数据运行实验，对比模拟数据的效果
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
MAIN_DATA_DIR = BASE_DIR.parent / "groupkfold_experiment" / "data"

RANDOM_STATE = 42


def compute_vegetation_indices(df):
    """计算植被指数"""
    eps = 1e-10
    df['VI_NDVI'] = (df['R810'] - df['R660']) / (df['R810'] + df['R660'] + eps)
    df['VI_NDRE'] = (df['R810'] - df['R710']) / (df['R810'] + df['R710'] + eps)
    df['VI_EVI'] = 2.5 * (df['R810'] - df['R660']) / (df['R810'] + 6*df['R660'] - 7.5*df['R460'] + 1 + eps)
    df['VI_SIPI'] = (df['R810'] - df['R460']) / (df['R810'] - df['R660'] + eps)
    df['VI_PRI'] = (df['R520'] - df['R580']) / (df['R520'] + df['R580'] + eps)
    df['VI_MTCI'] = (df['R810'] - df['R710']) / (df['R710'] - df['R660'] + eps)
    df['VI_GNDVI'] = (df['R810'] - df['R520']) / (df['R810'] + df['R520'] + eps)
    df['VI_NDWI'] = (df['R810'] - df['R900']) / (df['R810'] + df['R900'] + eps)
    return df


def compute_static_ratios(df):
    """计算静态荧光比值"""
    eps = 1e-10
    df['SR_F690_F740'] = df['RF(F690)'] / (df['FrF(f740)'] + eps)
    df['SR_F440_F690'] = df['BF(F440)'] / (df['RF(F690)'] + eps)
    df['SR_F440_F520'] = df['BF(F440)'] / (df['GF(F520)'] + eps)
    df['SR_F520_F690'] = df['GF(F520)'] / (df['RF(F690)'] + eps)
    df['SR_F440_F740'] = df['BF(F440)'] / (df['FrF(f740)'] + eps)
    df['SR_F520_F740'] = df['GF(F520)'] / (df['FrF(f740)'] + eps)
    return df


def run_group_kfold(df, feature_cols, target_col='D_conv'):
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

    df_result = df[['Variety', target_col]].copy()
    df_result['pred'] = oof_preds
    variety_agg = df_result.groupby('Variety').agg({
        target_col: 'first',
        'pred': 'mean'
    }).reset_index()

    return variety_agg


def get_metrics(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    sp, _ = spearmanr(y_true, y_pred)
    return {'R2': round(r2, 4), 'RMSE': round(rmse, 4), 'Spearman': round(sp, 4)}


def main():
    print("=" * 70)
    print("真实数据 vs 模拟数据 对比实验")
    print("=" * 70)

    # 1. 加载模拟数据
    df_sim = pd.read_csv(MAIN_DATA_DIR / "features_enhanced.csv")
    print(f"\n模拟数据: {len(df_sim)} 样本")

    # 2. 加载真实数据
    df_real = pd.read_csv(BASE_DIR / "raw_bands_cleaned.csv")
    print(f"真实数据: {len(df_real)} 样本")

    # 3. 为真实数据添加D_conv标签（从模拟数据获取）
    d_conv_map = df_sim.groupby('Variety')['D_conv'].first().to_dict()
    df_real['D_conv'] = df_real['Variety'].map(d_conv_map)

    # 4. 为真实数据计算植被指数和荧光比值
    df_real = compute_vegetation_indices(df_real)
    df_real = compute_static_ratios(df_real)

    # 5. 添加Treatment编码
    df_real['Trt_CK1'] = (df_real['Treatment'] == 'CK1').astype(int)
    df_real['Trt_D1'] = (df_real['Treatment'] == 'D1').astype(int)
    df_real['Trt_RD2'] = (df_real['Treatment'] == 'RD2').astype(int)

    # 6. 定义特征列（不含OJIP，因为真实数据可能没有）
    band_cols = ['R460', 'R520', 'R580', 'R660', 'R710', 'R730', 'R760', 'R780', 'R810', 'R850', 'R900']
    vi_cols = ['VI_NDVI', 'VI_NDRE', 'VI_EVI', 'VI_SIPI', 'VI_PRI', 'VI_MTCI', 'VI_GNDVI', 'VI_NDWI']
    static_cols = ['BF(F440)', 'GF(F520)', 'RF(F690)', 'FrF(f740)']
    sr_cols = ['SR_F690_F740', 'SR_F440_F690', 'SR_F440_F520', 'SR_F520_F690', 'SR_F440_F740', 'SR_F520_F740']
    trt_cols = ['Trt_CK1', 'Trt_D1', 'Trt_RD2']

    feature_cols_real = band_cols + vi_cols + static_cols + sr_cols + trt_cols
    feature_cols_real = [c for c in feature_cols_real if c in df_real.columns]

    # 模拟数据使用相同特征（排除OJIP以便公平比较）
    feature_cols_sim = [c for c in feature_cols_real if c in df_sim.columns]

    print(f"\n使用特征数: {len(feature_cols_real)}")

    # 7. 运行实验
    print("\n" + "=" * 70)
    print("GroupKFold 实验结果")
    print("=" * 70)

    # 模拟数据
    print("\n--- 模拟数据 (features_enhanced.csv) ---")
    agg_sim = run_group_kfold(df_sim, feature_cols_sim)
    m_sim = get_metrics(agg_sim['D_conv'].values, agg_sim['pred'].values)
    print(f"  R² = {m_sim['R2']:.4f}  RMSE = {m_sim['RMSE']:.4f}  Spearman = {m_sim['Spearman']:.4f}")

    # 真实数据
    print("\n--- 真实数据 (raw_bands_cleaned.csv) ---")
    agg_real = run_group_kfold(df_real, feature_cols_real)
    m_real = get_metrics(agg_real['D_conv'].values, agg_real['pred'].values)
    print(f"  R² = {m_real['R2']:.4f}  RMSE = {m_real['RMSE']:.4f}  Spearman = {m_real['Spearman']:.4f}")

    # 对比
    print("\n" + "=" * 70)
    print("对比")
    print("=" * 70)
    delta_r2 = m_real['R2'] - m_sim['R2']
    print(f"\nR² 变化: 模拟={m_sim['R2']:.4f} → 真实={m_real['R2']:.4f} ({delta_r2:+.4f})")

    # 品种级对比
    print("\n品种级预测对比:")
    print(f"{'品种':<8} {'D_conv':<10} {'模拟Pred':<12} {'真实Pred':<12} {'模拟误差':<10} {'真实误差':<10}")
    print("-" * 70)

    agg_sim = agg_sim.sort_values('D_conv')
    for _, row in agg_sim.iterrows():
        v = row['Variety']
        d_conv = row['D_conv']
        pred_sim = row['pred']
        err_sim = abs(d_conv - pred_sim)

        real_row = agg_real[agg_real['Variety'] == v]
        if len(real_row) > 0:
            pred_real = real_row['pred'].values[0]
            err_real = abs(d_conv - pred_real)
        else:
            pred_real = np.nan
            err_real = np.nan

        better = ""
        if not np.isnan(err_real):
            if err_real < err_sim - 0.01:
                better = "真实更好"
            elif err_sim < err_real - 0.01:
                better = "模拟更好"

        print(f"{int(v):<8} {d_conv:<10.4f} {pred_sim:<12.4f} {pred_real:<12.4f} {err_sim:<10.4f} {err_real:<10.4f} {better}")


if __name__ == "__main__":
    main()
