"""
测试真实数据的分类效果
将D_conv分为高/中/低三类，用光谱+特征工程预测
"""

import os
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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

try:
    from tabpfn import TabPFNClassifier
    TABPFN_AVAILABLE = True
except ImportError:
    TABPFN_AVAILABLE = False
    from sklearn.ensemble import RandomForestClassifier

BASE_DIR = Path(__file__).resolve().parent
MAIN_DATA_DIR = BASE_DIR.parent / "groupkfold_experiment" / "data"


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


def assign_class(d_conv, thresholds):
    """根据D_conv分配类别"""
    if d_conv <= thresholds[0]:
        return 0  # 低抗
    elif d_conv <= thresholds[1]:
        return 1  # 中抗
    else:
        return 2  # 高抗


def run_classification(df, feature_cols, target_col='class'):
    """运行GroupKFold分类"""
    X = df[feature_cols].values
    y = df[target_col].values
    groups = df['Variety'].values

    n_splits = min(5, len(np.unique(groups)))
    oof_preds = np.full(len(y), -1)

    gkf = GroupKFold(n_splits=n_splits)

    for fold, (tr_idx, te_idx) in enumerate(gkf.split(X, y, groups=groups)):
        X_train, X_test = X[tr_idx], X[te_idx]
        y_train = y[tr_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        if TABPFN_AVAILABLE:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = TabPFNClassifier(device=device, N_ensemble_configurations=32)
        else:
            model = RandomForestClassifier(n_estimators=100, random_state=42)

        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        oof_preds[te_idx] = y_pred

    return oof_preds


def main():
    print("=" * 70)
    print("真实数据分类效果测试")
    print("=" * 70)

    # 加载模拟数据（用于获取D_conv标签）
    df_sim = pd.read_csv(MAIN_DATA_DIR / "features_enhanced.csv")

    # 加载真实数据
    df_real = pd.read_csv(BASE_DIR / "raw_bands_cleaned.csv")
    print(f"\n真实数据样本数: {len(df_real)}")
    print(f"真实数据品种数: {df_real['Variety'].nunique()}")

    # 添加D_conv标签
    d_conv_map = df_sim.groupby('Variety')['D_conv'].first().to_dict()
    df_real['D_conv'] = df_real['Variety'].map(d_conv_map)

    # 删除没有标签的样本
    df_real = df_real.dropna(subset=['D_conv'])
    print(f"有标签的样本数: {len(df_real)}")

    # 计算特征工程
    df_real = compute_vegetation_indices(df_real)
    df_real = compute_static_ratios(df_real)

    # 添加Treatment编码
    df_real['Trt_CK1'] = (df_real['Treatment'] == 'CK1').astype(int)
    df_real['Trt_D1'] = (df_real['Treatment'] == 'D1').astype(int)
    df_real['Trt_RD2'] = (df_real['Treatment'] == 'RD2').astype(int)

    # 定义特征
    band_cols = ['R460', 'R520', 'R580', 'R660', 'R710', 'R730', 'R760', 'R780', 'R810', 'R850', 'R900']
    vi_cols = ['VI_NDVI', 'VI_NDRE', 'VI_EVI', 'VI_SIPI', 'VI_PRI', 'VI_MTCI', 'VI_GNDVI', 'VI_NDWI']
    static_cols = ['BF(F440)', 'GF(F520)', 'RF(F690)', 'FrF(f740)']
    sr_cols = ['SR_F690_F740', 'SR_F440_F690', 'SR_F440_F520', 'SR_F520_F690', 'SR_F440_F740', 'SR_F520_F740']
    trt_cols = ['Trt_CK1', 'Trt_D1', 'Trt_RD2']

    feature_cols = band_cols + vi_cols + static_cols + sr_cols + trt_cols
    feature_cols = [c for c in feature_cols if c in df_real.columns]
    print(f"特征数: {len(feature_cols)}")

    # D_conv分布
    print("\n" + "=" * 70)
    print("D_conv 分布（按品种）")
    print("=" * 70)
    d_conv_values = df_real.groupby('Variety')['D_conv'].first().sort_values()
    for v, d in d_conv_values.items():
        print(f"  品种 {int(v)}: D_conv = {d:.4f}")

    # 测试不同的分类方案
    print("\n" + "=" * 70)
    print("分类效果测试")
    print("=" * 70)

    # 方案1: 三分类（低/中/高）
    print("\n--- 方案1: 三分类 (三等分) ---")
    q33, q67 = np.percentile(d_conv_values.values, [33, 67])
    print(f"阈值: 低抗 <= {q33:.3f} < 中抗 <= {q67:.3f} < 高抗")

    df_real['class'] = df_real['D_conv'].apply(lambda x: assign_class(x, [q33, q67]))
    class_dist = df_real.groupby('class').size()
    print(f"类别分布: 低抗={class_dist.get(0,0)}, 中抗={class_dist.get(1,0)}, 高抗={class_dist.get(2,0)}")

    preds = run_classification(df_real, feature_cols)
    acc = accuracy_score(df_real['class'], preds)
    print(f"\nGroupKFold Accuracy: {acc:.4f} ({acc*100:.1f}%)")
    print("\n混淆矩阵:")
    print(confusion_matrix(df_real['class'], preds))

    # 方案2: 二分类（低 vs 高，去掉中间）
    print("\n--- 方案2: 二分类 (去掉中间品种) ---")
    q25, q75 = np.percentile(d_conv_values.values, [25, 75])
    print(f"阈值: 低抗 <= {q25:.3f}, 高抗 >= {q75:.3f}")

    df_binary = df_real[(df_real['D_conv'] <= q25) | (df_real['D_conv'] >= q75)].copy()
    df_binary['class'] = (df_binary['D_conv'] >= q75).astype(int)

    class_dist = df_binary.groupby('class').size()
    print(f"类别分布: 低抗={class_dist.get(0,0)}, 高抗={class_dist.get(1,0)}")
    print(f"品种数: {df_binary['Variety'].nunique()}")

    if df_binary['Variety'].nunique() >= 2:
        preds = run_classification(df_binary, feature_cols)
        acc = accuracy_score(df_binary['class'], preds)
        print(f"\nGroupKFold Accuracy: {acc:.4f} ({acc*100:.1f}%)")
        print("\n混淆矩阵:")
        print(confusion_matrix(df_binary['class'], preds))
    else:
        print("品种数不足，无法进行GroupKFold")

    # 方案3: 极端二分类（最低 vs 最高）
    print("\n--- 方案3: 极端二分类 (最低3个 vs 最高3个品种) ---")
    sorted_varieties = d_conv_values.index.tolist()
    low_varieties = sorted_varieties[:3]
    high_varieties = sorted_varieties[-3:]
    print(f"低抗品种: {low_varieties}")
    print(f"高抗品种: {high_varieties}")

    df_extreme = df_real[df_real['Variety'].isin(low_varieties + high_varieties)].copy()
    df_extreme['class'] = df_extreme['Variety'].isin(high_varieties).astype(int)

    class_dist = df_extreme.groupby('class').size()
    print(f"类别分布: 低抗={class_dist.get(0,0)}, 高抗={class_dist.get(1,0)}")

    preds = run_classification(df_extreme, feature_cols)
    acc = accuracy_score(df_extreme['class'], preds)
    print(f"\nGroupKFold Accuracy: {acc:.4f} ({acc*100:.1f}%)")
    print("\n混淆矩阵:")
    print(confusion_matrix(df_extreme['class'], preds))

    # 随机基线
    print("\n--- 随机基线 ---")
    print(f"三分类随机: 33.3%")
    print(f"二分类随机: 50.0%")

    print("\n" + "=" * 70)
    print("结论")
    print("=" * 70)


if __name__ == "__main__":
    main()
