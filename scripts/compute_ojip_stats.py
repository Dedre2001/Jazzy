"""
Step 1: 从ojip.csv计算OJIP群体平均值和与D_conv的相关系数
Step 2: 输出品种级数据用于corrected_ojip_metadata.csv
"""
import pandas as pd
import numpy as np
from scipy import stats

# 读取数据
ojip = pd.read_csv(r'F:\all_exp\data\raw\ojip.csv')

# 只保留CK1和D1处理
ojip_cd = ojip[ojip['Treatment'].isin(['CK1', 'D1'])].copy()

# 关键参数列
params = ['OJIP_FvFm', 'OJIP_PIabs', 'OJIP_Vi', 'OJIP_Vj', 'OJIP_TRo_RC', 'OJIP_ETo_RC']
extra = ['OJIP_Fo', 'OJIP_Fm']

print("=" * 70)
print("品种列表和样本量")
print("=" * 70)
varieties = sorted(ojip_cd['Variety'].unique())
print(f"品种数: {len(varieties)}")
print(f"品种: {varieties}")
for v in varieties:
    ck = ojip_cd[(ojip_cd['Variety'] == v) & (ojip_cd['Treatment'] == 'CK1')]
    d1 = ojip_cd[(ojip_cd['Variety'] == v) & (ojip_cd['Treatment'] == 'D1')]
    cat = ck['Category'].iloc[0] if len(ck) > 0 else 'N/A'
    print(f"  品种{v}: CK1={len(ck)}样本, D1={len(d1)}样本, Category={cat}")

print("\n" + "=" * 70)
print("Step 1: 群体平均值 (所有品种)")
print("=" * 70)

# 群体平均值
for trt in ['CK1', 'D1']:
    subset = ojip_cd[ojip_cd['Treatment'] == trt]
    print(f"\n{trt} (n={len(subset)}):")
    for p in params + extra:
        print(f"  {p}: {subset[p].mean():.4f} (SD={subset[p].std():.4f})")

# 变化幅度
ck_means = ojip_cd[ojip_cd['Treatment'] == 'CK1'][params].mean()
d1_means = ojip_cd[ojip_cd['Treatment'] == 'D1'][params].mean()
print("\n变化幅度 (%):")
for p in params:
    pct = (d1_means[p] - ck_means[p]) / ck_means[p] * 100
    print(f"  {p}: {pct:+.1f}%")

# Pearson相关系数 (仅D1数据)
print("\n" + "=" * 70)
print("与D_conv的Pearson相关系数 (仅D1处理)")
print("=" * 70)
d1_data = ojip_cd[ojip_cd['Treatment'] == 'D1'].copy()
for p in params:
    r, pval = stats.pearsonr(d1_data[p], d1_data['D_conv'])
    sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else 'ns'
    print(f"  {p}: r={r:+.3f}, p={pval:.4f} {sig}")

# Spearman相关系数
print("\n与D_conv的Spearman相关系数 (仅D1处理):")
for p in params:
    rho, pval = stats.spearmanr(d1_data[p], d1_data['D_conv'])
    sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else 'ns'
    print(f"  {p}: ρ={rho:+.3f}, p={pval:.4f} {sig}")

print("\n" + "=" * 70)
print("Step 2: 品种级数据 (1252, 1228, 1235)")
print("=" * 70)

target_varieties = [1252, 1228, 1235]
all_params = extra + params

for v in target_varieties:
    for trt in ['CK1', 'D1']:
        subset = ojip_cd[(ojip_cd['Variety'] == v) & (ojip_cd['Treatment'] == trt)]
        if len(subset) == 0:
            print(f"\n品种{v} {trt}: 无数据!")
            continue
        cat = subset['Category'].iloc[0]
        print(f"\n品种{v} ({cat}) {trt} (n={len(subset)}):")
        for p in all_params:
            print(f"  {p}: {subset[p].mean():.4f}")

# 品种级变化幅度
print("\n品种级变化幅度 (%):")
for v in target_varieties:
    ck = ojip_cd[(ojip_cd['Variety'] == v) & (ojip_cd['Treatment'] == 'CK1')]
    d1 = ojip_cd[(ojip_cd['Variety'] == v) & (ojip_cd['Treatment'] == 'D1')]
    if len(ck) == 0 or len(d1) == 0:
        print(f"\n品种{v}: 数据不完整")
        continue
    cat = ck['Category'].iloc[0]
    print(f"\n品种{v} ({cat}):")
    for p in all_params:
        ck_val = ck[p].mean()
        d1_val = d1[p].mean()
        pct = (d1_val - ck_val) / ck_val * 100
        print(f"  {p}: {ck_val:.4f} → {d1_val:.4f} ({pct:+.1f}%)")
