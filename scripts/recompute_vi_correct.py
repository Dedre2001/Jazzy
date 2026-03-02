# -*- coding: utf-8 -*-
"""
使用backup版正确公式，基于current版原始波段数据，重新计算所有VI
正确公式：
  NDVI  = (R820 - R680) / (R820 + R680)        [Rouse 1974]
  NDRE  = (R820 - R710) / (R820 + R710)        [Gitelson 1994]
  EVI   = 2.5*(R820-R680)/(R820+6*R680-7.5*R460+1)  [Huete 2002]
  SIPI  = (R820 - R460) / (R820 - R680)        [Penuelas 1995]
  PRI   = (R520 - R590) / (R520 + R590)        [Gamon 1992]
  MTCI  = (R730 - R710) / (R710 - R680)        [Dash 2004]
  GNDVI = (R820 - R520) / (R820 + R520)        [Gitelson 1996]
  NDWI  = (R850 - R910) / (R850 + R910)        [backup近似, Gao 1996替代]
"""
import pandas as pd
import numpy as np

curr_path = r'F:/all_exp/Thesis/论文图片/Figure_3-5/features_40_nir_corrected.csv'
df = pd.read_csv(curr_path)

# ============================================================
# 用正确公式重新计算VI
# ============================================================
df['VI_NDVI']  = (df['R820'] - df['R680']) / (df['R820'] + df['R680'])
df['VI_NDRE']  = (df['R820'] - df['R710']) / (df['R820'] + df['R710'])
df['VI_EVI']   = 2.5 * (df['R820'] - df['R680']) / (df['R820'] + 6*df['R680'] - 7.5*df['R460'] + 1)
df['VI_SIPI']  = (df['R820'] - df['R460']) / (df['R820'] - df['R680'])
df['VI_PRI']   = (df['R520'] - df['R590']) / (df['R520'] + df['R590'])
df['VI_MTCI']  = (df['R730'] - df['R710']) / (df['R710'] - df['R680'])
df['VI_GNDVI'] = (df['R820'] - df['R520']) / (df['R820'] + df['R520'])
df['VI_NDWI']  = (df['R850'] - df['R910']) / (df['R850'] + df['R910'])

# ============================================================
# 保存重新计算后的数据
# ============================================================
out_path = r'F:/all_exp/Thesis/论文图片/Figure_3-5/features_40_nir_corrected.csv'
df.to_csv(out_path, index=False)
print("已保存更新后的数据到:", out_path)

# ============================================================
# 输出论文所需的统计数据
# ============================================================
vi_cols = ['VI_NDVI','VI_NDRE','VI_GNDVI','VI_SIPI','VI_PRI','VI_EVI','VI_MTCI','VI_NDWI']
treatments = ['CK1','D1','RD2']
varieties  = ['1252','1228','1235']

print()
print("="*70)
print("Table 3-11: 群体处理均值（三个处理，全部13个品种）")
print("="*70)
grp = df.groupby('Treatment')[vi_cols].mean()
ck1 = grp.loc['CK1']
d1  = grp.loc['D1']
rd2 = grp.loc['RD2']

print(f"{'指数':<10} {'CK1':>10} {'D1':>10} {'D1变化%':>10} {'RD2':>10} {'RD2变化%':>10}")
print("-"*60)
for vi in vi_cols:
    name = vi.replace('VI_','')
    c = ck1[vi]; d = d1[vi]; r = rd2[vi]
    dp = (d-c)/abs(c)*100 if c != 0 else float('nan')
    rp = (r-d)/abs(d)*100 if d != 0 else float('nan')   # RD2%相对D1
    print(f"{name:<10} {c:>10.3f} {d:>10.3f} {dp:>+10.1f} {r:>10.3f} {rp:>+10.1f}")

print()
print("注：RD2变化%相对于D1期计算")

print()
print("="*70)
print("Table 3-11 附：RD2相对CK1的总变化%")
print("="*70)
print(f"{'指数':<10} {'CK1':>10} {'RD2':>10} {'RD2/CK1变化%':>14}")
for vi in vi_cols:
    name = vi.replace('VI_','')
    c = ck1[vi]; r = rd2[vi]
    rp = (r-c)/abs(c)*100 if c != 0 else float('nan')
    print(f"{name:<10} {c:>10.3f} {r:>10.3f} {rp:>+14.1f}")

print()
print("="*70)
print("Table 3-12: 三个代表品种各VI（CK1/D1/RD2）")
print("="*70)
vi_show = ['VI_NDVI','VI_NDRE','VI_EVI','VI_PRI']  # 论文中涉及的主要VI

for vi in vi_show:
    name = vi.replace('VI_','')
    print(f"\n  [{name}]")
    print(f"  {'品种':<6} {'CK1':>10} {'D1':>10} {'D1变化%':>10} {'RD2':>10} {'RD2/D1%':>10}")
    for var in varieties:
        sub = df[df['Variety'].astype(str) == var]
        ck = sub[sub['Treatment']=='CK1'][vi].mean()
        d1v = sub[sub['Treatment']=='D1'][vi].mean()
        rd2v = sub[sub['Treatment']=='RD2'][vi].mean()
        dp = (d1v - ck) / abs(ck) * 100 if ck != 0 else float('nan')
        rp = (rd2v - d1v) / abs(d1v) * 100 if d1v != 0 else float('nan')
        print(f"  {var:<6} {ck:>10.3f} {d1v:>10.3f} {dp:>+10.1f} {rd2v:>10.3f} {rp:>+10.1f}")

# 也输出原始波段R520/R660/R710/R820的品种均值（Table 3-12上半部分）
print()
print("="*70)
print("Table 3-12: 原始波段（R520/R660/R710/R820）三品种均值")
print("="*70)
band_show = ['R520','R660','R710','R820']
for band in band_show:
    print(f"\n  [{band}]")
    print(f"  {'品种':<6} {'CK1':>10} {'D1':>10} {'D1变化%':>10} {'RD2':>10} {'RD2/D1%':>10}")
    for var in varieties:
        sub = df[df['Variety'].astype(str) == var]
        ck = sub[sub['Treatment']=='CK1'][band].mean()
        d1v = sub[sub['Treatment']=='D1'][band].mean()
        rd2v = sub[sub['Treatment']=='RD2'][band].mean()
        dp = (d1v - ck) / abs(ck) * 100 if ck != 0 else float('nan')
        rp = (rd2v - d1v) / abs(d1v) * 100 if d1v != 0 else float('nan')
        print(f"  {var:<6} {ck:>10.3f} {d1v:>10.3f} {dp:>+10.1f} {rd2v:>10.3f} {rp:>+10.1f}")

# GNDVI单独输出（Table 3-11用到）
print()
print("="*70)
print("GNDVI 三品种详细（用于论文文本）")
print("="*70)
vi = 'VI_GNDVI'
print(f"  {'品种':<6} {'CK1':>10} {'D1':>10} {'D1变化%':>10} {'RD2':>10} {'RD2/D1%':>10} {'RD2/CK1%':>10}")
for var in varieties:
    sub = df[df['Variety'].astype(str) == var]
    ck = sub[sub['Treatment']=='CK1'][vi].mean()
    d1v = sub[sub['Treatment']=='D1'][vi].mean()
    rd2v = sub[sub['Treatment']=='RD2'][vi].mean()
    dp = (d1v - ck) / abs(ck) * 100 if ck != 0 else float('nan')
    rp = (rd2v - d1v) / abs(d1v) * 100 if d1v != 0 else float('nan')
    rck = (rd2v - ck) / abs(ck) * 100 if ck != 0 else float('nan')
    print(f"  {var:<6} {ck:>10.3f} {d1v:>10.3f} {dp:>+10.1f} {rd2v:>10.3f} {rp:>+10.1f} {rck:>+10.1f}")

print()
print("="*70)
print("SIPI、MTCI、NDWI 群体均值（Table 3-11其余行核验）")
print("="*70)
for vi in ['VI_SIPI','VI_MTCI','VI_NDWI']:
    name = vi.replace('VI_','')
    c = ck1[vi]; d = d1[vi]; r = rd2[vi]
    dp = (d-c)/abs(c)*100
    rp = (r-d)/abs(d)*100
    rck = (r-c)/abs(c)*100
    print(f"{name}: CK1={c:.3f}, D1={d:.3f}({dp:+.1f}%), RD2={r:.3f}({rp:+.1f}%/D1, {rck:+.1f}%/CK1)")
