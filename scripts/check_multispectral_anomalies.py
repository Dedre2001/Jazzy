# -*- coding: utf-8 -*-
"""
多光谱数据异常检查
1. 反射率基本范围（是否在0~1之间，是否有负值）
2. VI值域检查（各指数理论范围）
3. 处理间变化方向一致性（跨品种）
4. 极端值/离群点检测（均值±3SD）
5. 品种×处理均值矩阵（辅助人工判断）
"""
import pandas as pd
import numpy as np

df = pd.read_csv(r'F:/all_exp/Thesis/论文图片/Figure_3-5/features_40_nir_corrected.csv')

bands = ['R460','R520','R590','R660','R680','R710','R730','R780','R820','R850','R910']
vi_cols = ['VI_NDVI','VI_NDRE','VI_EVI','VI_SIPI','VI_PRI','VI_MTCI','VI_GNDVI','VI_NDWI']

varieties  = sorted(df['Variety'].astype(str).unique())
treatments = ['CK1','D1','RD2']

# ============================================================
# 1. 反射率值域检查（理论：0~1，植被通常0.01~0.7）
# ============================================================
print("="*70)
print("【1】原始波段反射率值域检查")
print("="*70)
for band in bands:
    vmin = df[band].min(); vmax = df[band].max()
    neg_n = (df[band] < 0).sum()
    gt1_n = (df[band] > 1).sum()
    flag = ''
    if neg_n > 0: flag += f' *** {neg_n}个负值'
    if gt1_n > 0: flag += f' *** {gt1_n}个>1'
    if vmax > 0.8: flag += f' *** 最大值偏高({vmax:.3f})'
    print(f"  {band}: min={vmin:.4f}, max={vmax:.4f}, mean={df[band].mean():.4f}{flag}")

# ============================================================
# 2. VI值域检查
# ============================================================
print()
print("="*70)
print("【2】植被指数值域检查（理论范围 vs 实际范围）")
print("="*70)
vi_ranges = {
    'VI_NDVI':  (-1, 1,   '正常植被>0.1，高植被>0.5'),
    'VI_NDRE':  (-1, 1,   '正常植被>0.1'),
    'VI_EVI':   (-1, 1,   '通常0~0.8，无明确上限'),
    'VI_SIPI':  (0,  5,   '通常1~2，>2可能异常'),
    'VI_PRI':   (-1, 1,   'CK1通常>0，重度胁迫可<0'),
    'VI_MTCI':  (-5, 10,  '通常0~5，分母接近0时可极端'),
    'VI_GNDVI': (-1, 1,   '正常植被>0.2'),
    'VI_NDWI':  (-1, 1,   '通常-0.2~0.2'),
}
for vi, (lo, hi, note) in vi_ranges.items():
    vmin = df[vi].min(); vmax = df[vi].max()
    out_lo = (df[vi] < lo).sum()
    out_hi = (df[vi] > hi).sum()
    neg_n  = (df[vi] < 0).sum()
    flag = ''
    if out_lo > 0: flag += f' *** {out_lo}个<{lo}'
    if out_hi > 0: flag += f' *** {out_hi}个>{hi}'
    if neg_n > 0 and vi not in ['VI_PRI','VI_NDWI']:
        # PRI/NDWI允许负值
        flag += f' *** {neg_n}个负值（需关注）'
    print(f"  {vi:<12}: [{vmin:+.4f}, {vmax:+.4f}]  {note}{flag}")

# ============================================================
# 3. 各VI负值样本明细（重点关注）
# ============================================================
print()
print("="*70)
print("【3】负值样本明细（排除PRI/NDWI，允许负值的指数除外）")
print("="*70)
for vi in ['VI_NDVI','VI_NDRE','VI_EVI','VI_SIPI','VI_MTCI','VI_GNDVI']:
    neg = df[df[vi] < 0][['Sample_ID','Treatment','Variety',vi]]
    if len(neg) > 0:
        print(f"\n  {vi} 负值 ({len(neg)}个):")
        print(neg.to_string(index=False))
    else:
        print(f"  {vi}: 无负值 ✓")

print()
print("  PRI负值（胁迫下允许，但列出供参考）:")
neg_pri = df[df['VI_PRI'] < 0][['Sample_ID','Treatment','Variety','VI_PRI']]
print(f"  共{len(neg_pri)}个，Treatment分布: {neg_pri['Treatment'].value_counts().to_dict()}")
if len(neg_pri) <= 15:
    print(neg_pri.to_string(index=False))

# ============================================================
# 4. 极端值检测（品种×处理内部，>3SD）
# ============================================================
print()
print("="*70)
print("【4】极端值检测（同处理内 >3SD 的样本）")
print("="*70)
all_anomalies = []
for treat in treatments:
    sub = df[df['Treatment'] == treat]
    for col in bands + vi_cols:
        mu = sub[col].mean(); sd = sub[col].std()
        if sd < 1e-9: continue
        outliers = sub[np.abs(sub[col] - mu) > 3*sd]
        for _, row in outliers.iterrows():
            z = (row[col] - mu) / sd
            all_anomalies.append({
                'Treatment': treat, 'Sample_ID': row['Sample_ID'],
                'Variety': str(row['Variety']), 'Column': col,
                'Value': row[col], 'Mean': mu, 'SD': sd, 'Z': z
            })

if all_anomalies:
    adf = pd.DataFrame(all_anomalies).sort_values('Z', key=abs, ascending=False)
    # 只显示波段异常（VI是派生的，波段异常更根本）
    band_anom = adf[adf['Column'].isin(bands)]
    vi_anom   = adf[adf['Column'].isin(vi_cols)]
    print(f"\n  波段极端值（共{len(band_anom)}个）:")
    if len(band_anom) > 0:
        print(band_anom[['Treatment','Variety','Sample_ID','Column','Value','Mean','Z']].to_string(index=False))
    print(f"\n  VI极端值（共{len(vi_anom)}个，前20）:")
    if len(vi_anom) > 0:
        print(vi_anom[['Treatment','Variety','Sample_ID','Column','Value','Mean','Z']].head(20).to_string(index=False))
else:
    print("  无>3SD的极端值 ✓")

# ============================================================
# 5. 变化方向一致性检查（CK1→D1跨品种）
# ============================================================
print()
print("="*70)
print("【5】CK1→D1 变化方向（各品种方向是否一致）")
print("="*70)
pivot = {}
for var in varieties:
    sub = df[df['Variety'].astype(str) == var]
    row = {}
    for col in bands + vi_cols:
        ck = sub[sub['Treatment']=='CK1'][col].mean()
        d1 = sub[sub['Treatment']=='D1'][col].mean()
        if ck != 0:
            row[col] = (d1 - ck) / abs(ck) * 100
        else:
            row[col] = float('nan')
    pivot[var] = row

print("\n  各品种CK1→D1变化%（正=升高，负=降低）")
print(f"  {'波段/VI':<12}", end='')
for var in varieties: print(f"  {var:>8}", end='')
print()

for col in bands + vi_cols:
    name = col.replace('VI_','')
    vals = [pivot[v][col] for v in varieties]
    # 判断方向一致性
    pos_count = sum(1 for v in vals if not np.isnan(v) and v > 0)
    neg_count = sum(1 for v in vals if not np.isnan(v) and v < 0)
    if pos_count > 0 and neg_count > 0:
        direction_flag = f'  *** 方向分化 ({pos_count}升/{neg_count}降)'
    else:
        direction_flag = ''
    # 检查极端变化（>100%）
    extreme = [v for v in varieties if not np.isnan(pivot[v][col]) and abs(pivot[v][col]) > 100]
    if extreme:
        direction_flag += f'  *** 极端变化:{extreme}'

    print(f"  {name:<12}", end='')
    for var in varieties:
        v = pivot[var][col]
        print(f"  {v:>+8.1f}", end='')
    print(direction_flag)

# ============================================================
# 6. MTCI 专项检查（分母接近0时会爆炸）
# ============================================================
print()
print("="*70)
print("【6】MTCI分母检查（R710-R680接近0时异常）")
print("="*70)
df['MTCI_denom'] = df['R710'] - df['R680']
print(f"  R710-R680 范围: [{df['MTCI_denom'].min():.4f}, {df['MTCI_denom'].max():.4f}]")
small_denom = df[df['MTCI_denom'].abs() < 0.01]
if len(small_denom) > 0:
    print(f"  *** 分母<0.01的样本 ({len(small_denom)}个):")
    print(small_denom[['Sample_ID','Treatment','Variety','MTCI_denom','VI_MTCI']].to_string(index=False))
else:
    print("  分母均>0.01，MTCI计算稳定 ✓")

extreme_mtci = df[df['VI_MTCI'].abs() > 5]
if len(extreme_mtci) > 0:
    print(f"\n  MTCI绝对值>5的样本 ({len(extreme_mtci)}个):")
    print(extreme_mtci[['Sample_ID','Treatment','Variety','VI_MTCI','MTCI_denom']].to_string(index=False))

# ============================================================
# 7. SIPI分母检查（R820-R680接近0时异常）
# ============================================================
print()
print("="*70)
print("【7】SIPI分母检查（R820-R680接近0时异常）")
print("="*70)
df['SIPI_denom'] = df['R820'] - df['R680']
print(f"  R820-R680 范围: [{df['SIPI_denom'].min():.4f}, {df['SIPI_denom'].max():.4f}]")
small_sipi = df[df['SIPI_denom'].abs() < 0.01]
if len(small_sipi) > 0:
    print(f"  *** 分母<0.01的样本 ({len(small_sipi)}个):")
    print(small_sipi[['Sample_ID','Treatment','Variety','SIPI_denom','VI_SIPI']].to_string(index=False))
else:
    print("  分母均>0.01，SIPI计算稳定 ✓")

print()
print("="*70)
print("【8】品种×处理 NDVI均值矩阵（快速总览）")
print("="*70)
matrix = df.pivot_table(values='VI_NDVI', index='Variety', columns='Treatment', aggfunc='mean')
matrix = matrix[['CK1','D1','RD2']]
matrix['D1变化%'] = (matrix['D1'] - matrix['CK1']) / matrix['CK1'].abs() * 100
matrix['RD2/D1变化%'] = (matrix['RD2'] - matrix['D1']) / matrix['D1'].abs() * 100
matrix['RD2/CK1变化%'] = (matrix['RD2'] - matrix['CK1']) / matrix['CK1'].abs() * 100
print(matrix.round(3).to_string())
