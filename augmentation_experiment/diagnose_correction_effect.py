"""
诊断分析：修正后为什么R²下降了？

检查两点：
1. 预测值是否被"压缩靠近均值"
2. 处理信号(D1-CK1, RD2-D1)是否被削弱
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

try:
    from tabpfn import TabPFNRegressor
    TABPFN_AVAILABLE = True
except ImportError:
    TABPFN_AVAILABLE = False
    from sklearn.linear_model import Ridge

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MAIN_DATA_DIR = BASE_DIR.parent / "groupkfold_experiment" / "data"


def prepare_corrected_data():
    """准备修正后的完整数据集"""
    df_orig = pd.read_csv(MAIN_DATA_DIR / "features_enhanced.csv")
    df_corr_raw = pd.read_csv(DATA_DIR / "rerecorrected.csv")
    corrected_varieties = [1099, 1235, 1252]

    df_other = df_orig[~df_orig['Variety'].isin(corrected_varieties)].copy()

    # 添加静态荧光和OJIP
    static_cols = ['BF(F440)', 'GF(F520)', 'RF(F690)', 'FrF(f740)',
                   'SR_F690_F740', 'SR_F440_F690', 'SR_F440_F520',
                   'SR_F520_F690', 'SR_F440_F740', 'SR_F520_F740',
                   'OJIP_FvFm', 'OJIP_PIabs', 'OJIP_TRo_RC', 'OJIP_ETo_RC',
                   'OJIP_Vi', 'OJIP_Vj', 'OJIP_ABS_RC_log', 'OJIP_DIo_RC_log']

    for col in static_cols:
        if col in df_orig.columns:
            mapping = df_orig.set_index('Sample_ID')[col].to_dict()
            df_corr_raw[col] = df_corr_raw['Sample_ID'].map(mapping)

    # 计算植被指数
    eps = 1e-10
    df_corr_raw['VI_NDVI'] = (df_corr_raw['R810'] - df_corr_raw['R660']) / (df_corr_raw['R810'] + df_corr_raw['R660'] + eps)
    df_corr_raw['VI_NDRE'] = (df_corr_raw['R810'] - df_corr_raw['R710']) / (df_corr_raw['R810'] + df_corr_raw['R710'] + eps)
    df_corr_raw['VI_EVI'] = 2.5 * (df_corr_raw['R810'] - df_corr_raw['R660']) / (df_corr_raw['R810'] + 6*df_corr_raw['R660'] - 7.5*df_corr_raw['R460'] + 1 + eps)
    df_corr_raw['VI_SIPI'] = (df_corr_raw['R810'] - df_corr_raw['R460']) / (df_corr_raw['R810'] - df_corr_raw['R660'] + eps)
    df_corr_raw['VI_PRI'] = (df_corr_raw['R520'] - df_corr_raw['R580']) / (df_corr_raw['R520'] + df_corr_raw['R580'] + eps)
    df_corr_raw['VI_MTCI'] = (df_corr_raw['R810'] - df_corr_raw['R710']) / (df_corr_raw['R710'] - df_corr_raw['R660'] + eps)
    df_corr_raw['VI_GNDVI'] = (df_corr_raw['R810'] - df_corr_raw['R520']) / (df_corr_raw['R810'] + df_corr_raw['R520'] + eps)
    df_corr_raw['VI_NDWI'] = (df_corr_raw['R810'] - df_corr_raw['R900']) / (df_corr_raw['R810'] + df_corr_raw['R900'] + eps)

    df_corr_raw['Trt_CK1'] = (df_corr_raw['Treatment'] == 'CK1').astype(int)
    df_corr_raw['Trt_D1'] = (df_corr_raw['Treatment'] == 'D1').astype(int)
    df_corr_raw['Trt_RD2'] = (df_corr_raw['Treatment'] == 'RD2').astype(int)

    common_cols = [c for c in df_other.columns if c in df_corr_raw.columns]
    df_corr = pd.concat([df_other[common_cols], df_corr_raw[common_cols]], ignore_index=True)

    return df_orig, df_corr


def run_gkf_get_preds(df, feature_cols):
    X = df[feature_cols].values
    y = df['D_conv'].values
    groups = df['Variety'].values

    gkf = GroupKFold(n_splits=5)
    oof = np.zeros(len(y))

    for tr_idx, te_idx in gkf.split(X, y, groups):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[tr_idx])
        X_te = scaler.transform(X[te_idx])

        if TABPFN_AVAILABLE:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = TabPFNRegressor(n_estimators=256, random_state=42,
                                    fit_mode='fit_preprocessors', device=device)
        else:
            model = Ridge(alpha=1.0)

        model.fit(X_tr, y[tr_idx])
        pred = model.predict(X_te)
        if hasattr(pred, 'ravel'):
            pred = pred.ravel()
        oof[te_idx] = pred

    return oof


def main():
    print("=" * 70)
    print("诊断：修正后为什么R²下降？")
    print("=" * 70)

    # 准备数据
    df_orig, df_corr = prepare_corrected_data()

    with open(MAIN_DATA_DIR / "feature_sets_enhanced.json", encoding='utf-8') as f:
        fs = json.load(f)
    feature_cols = [c for c in fs['FS4']['features'] if c in df_corr.columns]

    # ========== Part 1: 预测值是否被压缩向均值 ==========
    print("\n" + "=" * 70)
    print("Part 1: 预测值是否被压缩向均值？")
    print("=" * 70)

    print("\n运行原始数据 GroupKFold...")
    df_orig['pred'] = run_gkf_get_preds(df_orig, feature_cols)

    print("运行修正数据 GroupKFold...")
    df_corr['pred'] = run_gkf_get_preds(df_corr, feature_cols)

    agg_orig = df_orig.groupby('Variety').agg({'D_conv': 'first', 'pred': 'mean'}).reset_index()
    agg_corr = df_corr.groupby('Variety').agg({'D_conv': 'first', 'pred': 'mean'}).reset_index()

    global_mean = agg_orig['D_conv'].mean()
    print(f"\nGlobal D_conv mean: {global_mean:.4f}")

    print(f"\n{'品种':<10} {'D_conv':<10} {'Pred_原始':<12} {'Pred_修正':<12} {'变化':<10} {'方向'}")
    print("-" * 70)

    for v in [1099, 1235, 1252]:
        d_conv = agg_orig[agg_orig['Variety']==v]['D_conv'].values[0]
        pred_orig = agg_orig[agg_orig['Variety']==v]['pred'].values[0]
        pred_corr = agg_corr[agg_corr['Variety']==v]['pred'].values[0]
        change = pred_corr - pred_orig

        if d_conv > global_mean:
            direction = "→均值 (BAD)" if pred_corr < pred_orig else "←远离均值"
        else:
            direction = "→均值 (BAD)" if pred_corr > pred_orig else "←远离均值"

        print(f"{v:<10} {d_conv:<10.4f} {pred_orig:<12.4f} {pred_corr:<12.4f} {change:<+10.4f} {direction}")

    # ========== Part 2: 处理信号是否被削弱 ==========
    print("\n" + "=" * 70)
    print("Part 2: 处理信号(D1-CK1, RD2-D1)是否被削弱？")
    print("=" * 70)

    bands = ['R460', 'R520', 'R580', 'R660', 'R710', 'R730', 'R760', 'R780', 'R810', 'R850', 'R900']

    print(f"\n{'品种':<8} {'处理对':<12} {'原始差值':<12} {'修正差值':<12} {'变化':<10} {'信号'}")
    print("-" * 70)

    for v in [1099, 1235, 1252]:
        for trt_pair in [('D1', 'CK1'), ('RD2', 'D1')]:
            trt_a, trt_b = trt_pair

            # 原始数据的处理差值
            orig_a = df_orig[(df_orig['Variety']==v) & (df_orig['Treatment']==trt_a)][bands].mean()
            orig_b = df_orig[(df_orig['Variety']==v) & (df_orig['Treatment']==trt_b)][bands].mean()
            orig_diff = (orig_a - orig_b).abs().mean()

            # 修正数据的处理差值
            corr_a = df_corr[(df_corr['Variety']==v) & (df_corr['Treatment']==trt_a)][bands].mean()
            corr_b = df_corr[(df_corr['Variety']==v) & (df_corr['Treatment']==trt_b)][bands].mean()
            corr_diff = (corr_a - corr_b).abs().mean()

            change = corr_diff - orig_diff
            signal = "削弱" if change < -0.001 else ("增强" if change > 0.001 else "不变")

            print(f"{v:<8} {trt_a}-{trt_b:<8} {orig_diff:<12.4f} {corr_diff:<12.4f} {change:<+10.4f} {signal}")

    # ========== 结论 ==========
    print("\n" + "=" * 70)
    print("结论")
    print("=" * 70)
    print("""
如果预测值向均值移动 → 证实：修正让光谱变"正常"，模型预测中等值
如果处理信号被削弱 → 证实：修正破坏了 CK1→D1→RD2 的生物学变化规律

修正马氏距离的本意是让异常值变"正常"
但对于极端D_conv品种，它们的光谱本来就应该是"极端"的
修正反而丢失了有价值的信息
""")


if __name__ == "__main__":
    main()
