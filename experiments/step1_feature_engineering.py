"""
Step 1: 特征工程
目的: 生成40个特征（文献先验筛选）

特征结构（40个）：
├── Multi原始波段（11个）        → R460-R900
├── 植被指数（8个）              → NDVI, NDRE, EVI, SIPI, PRI, MTCI, GNDVI, NDWI
├── Static原始波段（4个）        → F440, F520, F690, F740
├── Static比值（6个）            → F690/F740, F440/F690, F440/F520, F520/F690, F440/F740, F520/F740
├── OJIP参数（8个）              → FvFm, PIabs, ABS_RC, TRo_RC, ETo_RC, DIo_RC, Vi, Vj
└── Treatment one-hot（3个）     → CK1, D1, RD2

输出:
- data/processed/features_40.csv (40个特征 + 元数据)
- data/processed/feature_sets.json (FS1/FS2/FS3/FS4定义)
"""

import pandas as pd
import numpy as np
import json
import os
from pathlib import Path

# 路径配置（相对项目根目录）
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "data" / "processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data():
    """加载融合数据"""
    df = pd.read_csv(f"{DATA_DIR}/fusion_all.csv")
    print(f"加载数据: {len(df)} 样本, {len(df.columns)} 列")
    return df

def compute_vegetation_indices(df):
    """
    计算8个植被指数（文献先验）

    参考文献：
    - NDVI: Rouse et al. (1973)
    - NDRE: Gitelson & Merzlyak (1994)
    - EVI: Huete et al. (2002)
    - SIPI: Peñuelas et al. (1995)
    - PRI: Gamon et al. (1992)
    - MTCI: Dash & Curran (2004)
    - GNDVI: Gitelson et al. (1996)
    - NDWI: Gao (1996)
    """
    print("\n计算植被指数...")

    # 提取波段（使用最接近的可用波段）
    R460 = df['R460']
    R520 = df['R520']
    R580 = df['R580']
    R660 = df['R660']
    R710 = df['R710']
    R760 = df['R760']
    R850 = df['R850']
    R900 = df['R900']

    # 计算植被指数
    indices = pd.DataFrame()

    # 1. NDVI = (R850 - R660) / (R850 + R660)
    indices['VI_NDVI'] = (R850 - R660) / (R850 + R660 + 1e-10)

    # 2. NDRE = (R850 - R710) / (R850 + R710)
    indices['VI_NDRE'] = (R850 - R710) / (R850 + R710 + 1e-10)

    # 3. EVI = 2.5 * (R850 - R660) / (R850 + 6*R660 - 7.5*R460 + 1)
    indices['VI_EVI'] = 2.5 * (R850 - R660) / (R850 + 6*R660 - 7.5*R460 + 1 + 1e-10)

    # 4. SIPI = (R850 - R460) / (R850 + R660)
    indices['VI_SIPI'] = (R850 - R460) / (R850 + R660 + 1e-10)

    # 5. PRI = (R520 - R580) / (R520 + R580)
    indices['VI_PRI'] = (R520 - R580) / (R520 + R580 + 1e-10)

    # 6. MTCI = (R760 - R710) / (R710 - R660)
    indices['VI_MTCI'] = (R760 - R710) / (R710 - R660 + 1e-10)

    # 7. GNDVI = (R850 - R520) / (R850 + R520)
    indices['VI_GNDVI'] = (R850 - R520) / (R850 + R520 + 1e-10)

    # 8. NDWI = (R850 - R900) / (R850 + R900)
    indices['VI_NDWI'] = (R850 - R900) / (R850 + R900 + 1e-10)

    print(f"  计算完成: {len(indices.columns)} 个植被指数")
    for col in indices.columns:
        print(f"    {col}: mean={indices[col].mean():.4f}, std={indices[col].std():.4f}")

    return indices

def compute_static_ratios(df):
    """
    计算6个Static荧光比值（文献先验）

    参考文献：
    - Buschmann (2007)
    - Lichtenthaler (1996)
    """
    print("\n计算Static荧光比值...")

    # 提取荧光波段
    F440 = df['BF(F440)']
    F520 = df['GF(F520)']
    F690 = df['RF(F690)']
    F740 = df['FrF(f740)']

    ratios = pd.DataFrame()

    # 1. F690/F740 - 叶绿素荧光比，重吸收效应
    ratios['SR_F690_F740'] = F690 / (F740 + 1e-10)

    # 2. F440/F690 - 蓝/红荧光比，胁迫敏感
    ratios['SR_F440_F690'] = F440 / (F690 + 1e-10)

    # 3. F440/F520 - 蓝/绿荧光比
    ratios['SR_F440_F520'] = F440 / (F520 + 1e-10)

    # 4. F520/F690 - 绿/红荧光比
    ratios['SR_F520_F690'] = F520 / (F690 + 1e-10)

    # 5. F440/F740 - 蓝/远红比
    ratios['SR_F440_F740'] = F440 / (F740 + 1e-10)

    # 6. F520/F740 - 绿/远红比
    ratios['SR_F520_F740'] = F520 / (F740 + 1e-10)

    print(f"  计算完成: {len(ratios.columns)} 个荧光比值")
    for col in ratios.columns:
        print(f"    {col}: mean={ratios[col].mean():.4f}, std={ratios[col].std():.4f}")

    return ratios

def select_ojip_features(df):
    """
    筛选8个核心OJIP参数（文献先验）

    参考文献：
    - Strasser, R.J. et al. (2004) Analysis of the chlorophyll a fluorescence transient.

    保留参数：
    - OJIP_FvFm: PSII最大光化学效率
    - OJIP_PIabs: 综合性能指数
    - OJIP_ABS_RC: 每RC吸收光能 (需要log变换)
    - OJIP_TRo_RC: 每RC捕获能量
    - OJIP_ETo_RC: 每RC电子传递
    - OJIP_DIo_RC: 每RC热耗散 (需要log变换)
    - OJIP_Vi: I相相对荧光
    - OJIP_Vj: J相相对荧光

    删除参数：
    - OJIP_phiPo (与FvFm冗余, r≈1.0)
    - OJIP_Fm, OJIP_Fo, OJIP_Fv (原始荧光值)
    - OJIP_phiEo, OJIP_psiEo (与ETo_RC冗余)
    - OJIP_N, OJIP_Sm (次要参数)
    """
    print("\n筛选OJIP参数...")

    ojip_selected = pd.DataFrame()

    # 直接使用的参数
    ojip_selected['OJIP_FvFm'] = df['OJIP_FvFm']
    ojip_selected['OJIP_PIabs'] = df['OJIP_PIabs']
    ojip_selected['OJIP_TRo_RC'] = df['OJIP_TRo_RC']
    ojip_selected['OJIP_ETo_RC'] = df['OJIP_ETo_RC']
    ojip_selected['OJIP_Vi'] = df['OJIP_Vi']
    ojip_selected['OJIP_Vj'] = df['OJIP_Vj']

    # 需要log变换的参数（高偏度）
    ojip_selected['OJIP_ABS_RC_log'] = np.log1p(df['OJIP_ABS_RC'])
    ojip_selected['OJIP_DIo_RC_log'] = np.log1p(df['OJIP_DIo_RC'])

    print(f"  筛选完成: {len(ojip_selected.columns)} 个OJIP参数")
    for col in ojip_selected.columns:
        print(f"    {col}: mean={ojip_selected[col].mean():.4f}, std={ojip_selected[col].std():.4f}")

    return ojip_selected

def encode_treatment(df):
    """
    Treatment one-hot编码
    """
    print("\n编码Treatment...")

    treatment_dummies = pd.get_dummies(df['Treatment'], prefix='Trt')

    print(f"  编码完成: {len(treatment_dummies.columns)} 个Treatment特征")
    print(f"    列名: {list(treatment_dummies.columns)}")

    return treatment_dummies

def create_feature_sets():
    """
    定义FS1/FS2/FS3/FS4特征集
    """
    feature_sets = {
        "FS1": {
            "description": "Multi-only (Multi原始波段 + 植被指数 + Treatment)",
            "features": [
                # Multi原始波段 (11个)
                "R460", "R520", "R580", "R660", "R710", "R730", "R760", "R780", "R810", "R850", "R900",
                # 植被指数 (8个)
                "VI_NDVI", "VI_NDRE", "VI_EVI", "VI_SIPI", "VI_PRI", "VI_MTCI", "VI_GNDVI", "VI_NDWI",
                # Treatment (3个)
                "Trt_CK1", "Trt_D1", "Trt_RD2"
            ],
            "n_features": 22
        },
        "FS2": {
            "description": "Static-only (Static原始波段 + Static比值 + Treatment)",
            "features": [
                # Static原始波段 (4个)
                "BF(F440)", "GF(F520)", "RF(F690)", "FrF(f740)",
                # Static比值 (6个)
                "SR_F690_F740", "SR_F440_F690", "SR_F440_F520", "SR_F520_F690", "SR_F440_F740", "SR_F520_F740",
                # Treatment (3个)
                "Trt_CK1", "Trt_D1", "Trt_RD2"
            ],
            "n_features": 13
        },
        "FS3": {
            "description": "双源融合 (Multi + Static + Treatment，不含OJIP)",
            "features": [
                # Multi原始波段 (11个)
                "R460", "R520", "R580", "R660", "R710", "R730", "R760", "R780", "R810", "R850", "R900",
                # 植被指数 (8个)
                "VI_NDVI", "VI_NDRE", "VI_EVI", "VI_SIPI", "VI_PRI", "VI_MTCI", "VI_GNDVI", "VI_NDWI",
                # Static原始波段 (4个)
                "BF(F440)", "GF(F520)", "RF(F690)", "FrF(f740)",
                # Static比值 (6个)
                "SR_F690_F740", "SR_F440_F690", "SR_F440_F520", "SR_F520_F690", "SR_F440_F740", "SR_F520_F740",
                # Treatment (3个)
                "Trt_CK1", "Trt_D1", "Trt_RD2"
            ],
            "n_features": 32
        },
        "FS4": {
            "description": "三源融合 (Multi + Static + OJIP + Treatment)",
            "features": [
                # Multi原始波段 (11个)
                "R460", "R520", "R580", "R660", "R710", "R730", "R760", "R780", "R810", "R850", "R900",
                # 植被指数 (8个)
                "VI_NDVI", "VI_NDRE", "VI_EVI", "VI_SIPI", "VI_PRI", "VI_MTCI", "VI_GNDVI", "VI_NDWI",
                # Static原始波段 (4个)
                "BF(F440)", "GF(F520)", "RF(F690)", "FrF(f740)",
                # Static比值 (6个)
                "SR_F690_F740", "SR_F440_F690", "SR_F440_F520", "SR_F520_F690", "SR_F440_F740", "SR_F520_F740",
                # OJIP参数 (8个)
                "OJIP_FvFm", "OJIP_PIabs", "OJIP_ABS_RC_log", "OJIP_TRo_RC",
                "OJIP_ETo_RC", "OJIP_DIo_RC_log", "OJIP_Vi", "OJIP_Vj",
                # Treatment (3个)
                "Trt_CK1", "Trt_D1", "Trt_RD2"
            ],
            "n_features": 40
        }
    }

    return feature_sets

def main():
    print("=" * 60)
    print("Step 1: 特征工程")
    print("=" * 60)

    # 1. 加载数据
    df = load_data()

    # 2. 提取元数据
    metadata_cols = ['Sample_ID', 'Treatment', 'Variety', 'Sample',
                     'D_conv', 'D_stress', 'D_recovery', 'Category', 'Rank']
    metadata = df[metadata_cols].copy()

    # 3. 提取Multi原始波段
    multi_cols = ['R460', 'R520', 'R580', 'R660', 'R710', 'R730',
                  'R760', 'R780', 'R810', 'R850', 'R900']
    multi_features = df[multi_cols].copy()
    print(f"\nMulti原始波段: {len(multi_cols)} 个")

    # 4. 计算植被指数
    vegetation_indices = compute_vegetation_indices(df)

    # 5. 提取Static原始波段
    static_cols = ['BF(F440)', 'GF(F520)', 'RF(F690)', 'FrF(f740)']
    static_features = df[static_cols].copy()
    print(f"\nStatic原始波段: {len(static_cols)} 个")

    # 6. 计算Static比值
    static_ratios = compute_static_ratios(df)

    # 7. 筛选OJIP参数
    ojip_features = select_ojip_features(df)

    # 8. Treatment编码
    treatment_features = encode_treatment(df)

    # 9. 合并所有特征
    print("\n" + "=" * 60)
    print("合并特征...")

    features_all = pd.concat([
        metadata,
        multi_features,
        vegetation_indices,
        static_features,
        static_ratios,
        ojip_features,
        treatment_features
    ], axis=1)

    print(f"\n最终特征矩阵: {features_all.shape}")
    print(f"  样本数: {len(features_all)}")
    print(f"  总列数: {len(features_all.columns)}")
    print(f"  元数据列: {len(metadata_cols)}")
    print(f"  特征列: {len(features_all.columns) - len(metadata_cols)}")

    # 10. 特征汇总
    print("\n" + "=" * 60)
    print("特征汇总")
    print("=" * 60)
    feature_summary = {
        "Multi原始波段": len(multi_cols),
        "植被指数": len(vegetation_indices.columns),
        "Static原始波段": len(static_cols),
        "Static比值": len(static_ratios.columns),
        "OJIP参数": len(ojip_features.columns),
        "Treatment": len(treatment_features.columns)
    }
    total_features = sum(feature_summary.values())
    for name, count in feature_summary.items():
        print(f"  {name}: {count}")
    print(f"  ---------------------")
    print(f"  总计: {total_features}")

    # 11. 创建特征集定义
    feature_sets = create_feature_sets()

    print("\n特征集定义:")
    for fs_name, fs_info in feature_sets.items():
        print(f"  {fs_name}: {fs_info['n_features']} 个特征 - {fs_info['description']}")

    # 12. 保存结果
    print("\n" + "=" * 60)
    print("保存结果...")

    # 保存特征矩阵
    output_path = OUTPUT_DIR / "features_40.csv"
    features_all.to_csv(output_path, index=False)
    print(f"  特征矩阵已保存至: {output_path}")

    # 保存特征集定义
    fs_path = OUTPUT_DIR / "feature_sets.json"
    with open(fs_path, 'w', encoding='utf-8') as f:
        json.dump(feature_sets, f, indent=2, ensure_ascii=False)
    print(f"  特征集定义已保存至: {fs_path}")

    # 13. 验证
    print("\n" + "=" * 60)
    print("验证")
    print("=" * 60)

    # 检查FS4特征是否都存在
    fs4_features = feature_sets['FS4']['features']
    missing_features = [f for f in fs4_features if f not in features_all.columns]
    if missing_features:
        print(f"  [ERROR] 缺失特征: {missing_features}")
    else:
        print(f"  [OK] FS4所有 {len(fs4_features)} 个特征均存在")

    # 检查缺失值
    missing_count = features_all[fs4_features].isnull().sum().sum()
    if missing_count > 0:
        print(f"  [WARNING] FS4特征存在 {missing_count} 个缺失值")
    else:
        print(f"  [OK] FS4特征无缺失值")

    print("\n特征工程完成!")

    return features_all, feature_sets

if __name__ == "__main__":
    features, feature_sets = main()
