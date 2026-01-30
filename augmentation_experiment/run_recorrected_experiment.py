"""
使用修正后的光谱数据重新计算所有派生特征并运行实验

输入: augmentation_experiment/data/recorrected.csv (修正了1099, 1235, 1252的光谱)
输出: augmentation_experiment/data/features_recorrected.csv

流程:
1. 加载修正后的光谱数据
2. 加载原始数据中的其他品种
3. 重新计算植被指数
4. 重新计算光谱导数
5. 合并静态荧光和OJIP特征
6. 运行GroupKFold实验
"""

import os
import json
import warnings
from pathlib import Path
from itertools import combinations
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
MAIN_DATA_DIR = BASE_DIR.parent / "groupkfold_experiment" / "data"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


# ========== 植被指数计算 ==========

def compute_vegetation_indices(df):
    """根据光谱波段计算8个植被指数"""
    print("\n计算植被指数...")

    # 波段映射
    R460, R520, R580, R660 = df['R460'], df['R520'], df['R580'], df['R660']
    R710, R730, R760, R780 = df['R710'], df['R730'], df['R760'], df['R780']
    R810, R850, R900 = df['R810'], df['R850'], df['R900']

    eps = 1e-10  # 避免除零

    # NDVI: (NIR - Red) / (NIR + Red)
    df['VI_NDVI'] = (R810 - R660) / (R810 + R660 + eps)

    # NDRE: (NIR - RedEdge) / (NIR + RedEdge)
    df['VI_NDRE'] = (R810 - R710) / (R810 + R710 + eps)

    # EVI: 2.5 * (NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1)
    df['VI_EVI'] = 2.5 * (R810 - R660) / (R810 + 6*R660 - 7.5*R460 + 1 + eps)

    # SIPI: (NIR - Blue) / (NIR - Red)
    df['VI_SIPI'] = (R810 - R460) / (R810 - R660 + eps)

    # PRI: (R531 - R570) / (R531 + R570)，用R520和R580近似
    df['VI_PRI'] = (R520 - R580) / (R520 + R580 + eps)

    # MTCI: (NIR - RedEdge) / (RedEdge - Red)
    df['VI_MTCI'] = (R810 - R710) / (R710 - R660 + eps)

    # GNDVI: (NIR - Green) / (NIR + Green)
    df['VI_GNDVI'] = (R810 - R520) / (R810 + R520 + eps)

    # NDWI: (NIR - SWIR) / (NIR + SWIR)，用R900近似SWIR
    df['VI_NDWI'] = (R810 - R900) / (R810 + R900 + eps)

    print(f"  计算完成: VI_NDVI, VI_NDRE, VI_EVI, VI_SIPI, VI_PRI, VI_MTCI, VI_GNDVI, VI_NDWI")
    return df


def compute_spectral_derivatives(df):
    """计算光谱一阶导数"""
    print("\n计算光谱导数...")

    band_cols = ['R460', 'R520', 'R580', 'R660', 'R710', 'R730', 'R760', 'R780', 'R810', 'R850', 'R900']
    sorted_bands = sorted(band_cols, key=lambda x: int(x.replace('R', '')))

    for i in range(len(sorted_bands) - 1):
        b1, b2 = sorted_bands[i], sorted_bands[i + 1]
        w1, w2 = int(b1.replace('R', '')), int(b2.replace('R', ''))
        delta_w = w2 - w1
        col_name = f"dR{w1}_{w2}"
        df[col_name] = (df[b2] - df[b1]) / delta_w

    print(f"  生成 10 个一阶导数特征")
    return df


# ========== 数据准备 ==========

def prepare_recorrected_data():
    """准备修正后的完整数据集"""
    print("=" * 70)
    print("准备修正后的数据集")
    print("=" * 70)

    # 1. 加载修正后的光谱数据 (1099, 1235, 1252)
    df_corrected = pd.read_csv(DATA_DIR / "recorrected.csv")
    corrected_varieties = df_corrected['Variety'].unique()
    print(f"\n修正的品种: {list(corrected_varieties)}")
    print(f"修正样本数: {len(df_corrected)}")

    # 2. 加载原始完整数据
    df_original = pd.read_csv(MAIN_DATA_DIR / "features_enhanced.csv")
    print(f"原始数据: {len(df_original)} 样本")

    # 3. 提取非修正品种的数据
    df_other = df_original[~df_original['Variety'].isin(corrected_varieties)].copy()
    print(f"其他品种样本数: {len(df_other)}")

    # 4. 从原始数据中提取静态荧光和OJIP特征（这些不受光谱修正影响）
    static_cols = ['BF(F440)', 'GF(F520)', 'RF(F690)', 'FrF(f740)',
                   'SR_F690_F740', 'SR_F440_F690', 'SR_F440_F520',
                   'SR_F520_F690', 'SR_F440_F740', 'SR_F520_F740']
    ojip_cols = ['OJIP_FvFm', 'OJIP_PIabs', 'OJIP_TRo_RC', 'OJIP_ETo_RC',
                 'OJIP_Vi', 'OJIP_Vj', 'OJIP_ABS_RC_log', 'OJIP_DIo_RC_log']

    # 为修正的品种添加静态荧光和OJIP特征
    for col in static_cols + ojip_cols:
        if col in df_original.columns:
            # 创建映射: Sample_ID -> 特征值
            mapping = df_original.set_index('Sample_ID')[col].to_dict()
            df_corrected[col] = df_corrected['Sample_ID'].map(mapping)

    # 5. 重新计算修正品种的植被指数
    df_corrected = compute_vegetation_indices(df_corrected)

    # 6. 添加Treatment one-hot编码
    df_corrected['Trt_CK1'] = (df_corrected['Treatment'] == 'CK1').astype(int)
    df_corrected['Trt_D1'] = (df_corrected['Treatment'] == 'D1').astype(int)
    df_corrected['Trt_RD2'] = (df_corrected['Treatment'] == 'RD2').astype(int)

    # 7. 计算光谱导数
    df_corrected = compute_spectral_derivatives(df_corrected)

    # 8. 合并数据
    # 确保列一致
    common_cols = [c for c in df_other.columns if c in df_corrected.columns]
    df_corrected = df_corrected[common_cols]
    df_other = df_other[common_cols]

    df_merged = pd.concat([df_other, df_corrected], ignore_index=True)
    df_merged = df_merged.sort_values(['Variety', 'Treatment', 'Sample']).reset_index(drop=True)

    print(f"\n合并后数据: {len(df_merged)} 样本, {len(df_merged['Variety'].unique())} 品种")

    # 9. 保存
    output_path = DATA_DIR / "features_recorrected.csv"
    df_merged.to_csv(output_path, index=False)
    print(f"保存: {output_path}")

    return df_merged


# ========== 评估函数 ==========

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


def get_variety_errors(variety_agg):
    hard_varieties = [1252, 1235, 1099]
    errors = {}
    for v in hard_varieties:
        row = variety_agg[variety_agg['Variety'] == v]
        if len(row) > 0:
            errors[str(v)] = round(abs(row['D_conv'].values[0] - row['pred'].values[0]), 4)
    return errors


# ========== 主流程 ==========

def main():
    print("=" * 70)
    print("光谱修正后的数据实验")
    print("=" * 70)

    # 1. 准备修正后的数据
    df_corrected = prepare_recorrected_data()

    # 2. 加载原始数据作为对比
    df_original = pd.read_csv(MAIN_DATA_DIR / "features_enhanced.csv")

    # 3. 定义特征集 (FS4: 40个特征)
    with open(MAIN_DATA_DIR / "feature_sets_enhanced.json", encoding='utf-8') as f:
        feature_sets = json.load(f)
    feature_cols = feature_sets['FS4']['features']

    # 检查特征列是否都存在
    missing_cols = [c for c in feature_cols if c not in df_corrected.columns]
    if missing_cols:
        print(f"\n[WARN] 缺少特征列: {missing_cols}")
        feature_cols = [c for c in feature_cols if c in df_corrected.columns]

    print(f"\n使用 {len(feature_cols)} 个特征")

    # 4. 运行对比实验
    print("\n" + "=" * 70)
    print("对比实验: 原始数据 vs 修正后数据")
    print("=" * 70)

    results = {}

    # 原始数据
    print("\n--- 原始数据 ---")
    variety_agg_orig = run_group_kfold(df_original, feature_cols)
    metrics_orig = get_metrics(variety_agg_orig['D_conv'].values, variety_agg_orig['pred'].values)
    errors_orig = get_variety_errors(variety_agg_orig)
    results['original'] = {'metrics': metrics_orig, 'errors': errors_orig}
    print(f"  R² = {metrics_orig['R2']:.4f}  RMSE = {metrics_orig['RMSE']:.4f}  Spearman = {metrics_orig['Spearman']:.4f}")
    print(f"  难预测品种: 1252={errors_orig.get('1252')}, 1235={errors_orig.get('1235')}, 1099={errors_orig.get('1099')}")

    # 修正后数据
    print("\n--- 修正后数据 ---")
    variety_agg_corr = run_group_kfold(df_corrected, feature_cols)
    metrics_corr = get_metrics(variety_agg_corr['D_conv'].values, variety_agg_corr['pred'].values)
    errors_corr = get_variety_errors(variety_agg_corr)
    results['corrected'] = {'metrics': metrics_corr, 'errors': errors_corr}
    print(f"  R² = {metrics_corr['R2']:.4f}  RMSE = {metrics_corr['RMSE']:.4f}  Spearman = {metrics_corr['Spearman']:.4f}")
    print(f"  难预测品种: 1252={errors_corr.get('1252')}, 1235={errors_corr.get('1235')}, 1099={errors_corr.get('1099')}")

    # 5. 汇总
    print("\n" + "=" * 70)
    print("汇总")
    print("=" * 70)

    delta_r2 = metrics_corr['R2'] - metrics_orig['R2']
    print(f"\nR² 变化: {metrics_orig['R2']:.4f} → {metrics_corr['R2']:.4f} ({delta_r2:+.4f})")

    print("\n难预测品种误差变化:")
    for v in ['1252', '1235', '1099']:
        e_orig = errors_orig.get(v, 'N/A')
        e_corr = errors_corr.get(v, 'N/A')
        if isinstance(e_orig, float) and isinstance(e_corr, float):
            delta = e_corr - e_orig
            print(f"  {v}: {e_orig:.4f} → {e_corr:.4f} ({delta:+.4f})")
        else:
            print(f"  {v}: {e_orig} → {e_corr}")

    # 6. 保存结果
    report = {
        'original': {
            'R2': results['original']['metrics']['R2'],
            'RMSE': results['original']['metrics']['RMSE'],
            'Spearman': results['original']['metrics']['Spearman'],
            'hard_variety_errors': results['original']['errors']
        },
        'corrected': {
            'R2': results['corrected']['metrics']['R2'],
            'RMSE': results['corrected']['metrics']['RMSE'],
            'Spearman': results['corrected']['metrics']['Spearman'],
            'hard_variety_errors': results['corrected']['errors']
        },
        'improvement': {
            'R2_delta': delta_r2
        }
    }

    report_path = RESULTS_DIR / "recorrected_comparison.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n结果保存: {report_path}")

    return results


if __name__ == "__main__":
    results = main()
