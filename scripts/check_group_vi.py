# -*- coding: utf-8 -*-
import pandas as pd, numpy as np

curr_path = r'F:\all_exp\Thesis\论文图片\Figure_3-5\features_40_nir_corrected.csv'
df = pd.read_csv(curr_path)

def calc_idx(d):
    ndvi = (d['R820']-d['R680'])/(d['R820']+d['R680'])
    ndre = (d['R820']-d['R710'])/(d['R820']+d['R710'])
    evi  = 2.5*(d['R820']-d['R680'])/(d['R820']+6*d['R680']-7.5*d['R460']+1)
    pri  = (d['R520']-d['R590'])/(d['R520']+d['R590'])
    return ndvi, ndre, evi, pri

sub_ck  = df[df['Treatment']=='CK1'].mean(numeric_only=True)
sub_d1  = df[df['Treatment']=='D1'].mean(numeric_only=True)
sub_rd2 = df[df['Treatment']=='RD2'].mean(numeric_only=True)

ndvi_ck,  ndre_ck,  evi_ck,  pri_ck  = calc_idx(sub_ck)
ndvi_d1,  ndre_d1,  evi_d1,  pri_d1  = calc_idx(sub_d1)
ndvi_rd2, ndre_rd2, evi_rd2, pri_rd2 = calc_idx(sub_rd2)

print('Table 3-11 验证:')
print(f'NDVI: CK1={ndvi_ck:.3f}, D1={ndvi_d1:.3f}({(ndvi_d1-ndvi_ck)/abs(ndvi_ck)*100:+.1f}%), RD2={ndvi_rd2:.3f}({(ndvi_rd2-ndvi_ck)/abs(ndvi_ck)*100:+.1f}%)')
print(f'NDRE: CK1={ndre_ck:.3f}, D1={ndre_d1:.3f}({(ndre_d1-ndre_ck)/abs(ndre_ck)*100:+.1f}%), RD2={ndre_rd2:.3f}({(ndre_rd2-ndre_ck)/abs(ndre_ck)*100:+.1f}%)')
print(f'EVI:  CK1={evi_ck:.3f},  D1={evi_d1:.3f}({(evi_d1-evi_ck)/abs(evi_ck)*100:+.1f}%),  RD2={evi_rd2:.3f}({(evi_rd2-evi_ck)/abs(evi_ck)*100:+.1f}%)')
print(f'PRI:  CK1={pri_ck:.3f},  D1={pri_d1:.3f}({(pri_d1-pri_ck)/abs(pri_ck)*100:+.1f}%),  RD2={pri_rd2:.3f}({(pri_rd2-pri_ck)/abs(pri_ck)*100:+.1f}%)')
print()
print('Table 3-11 文本值 vs 实际值:')
text = {
    'NDVI': (0.601, 0.493, -18.0, 0.324, -46.0),
    'NDRE': (0.333, 0.252, -24.2, 0.113, -66.0),
    'EVI':  (0.384, 0.318, -17.2, 0.195, -49.2),
    'PRI':  (0.128, 0.032, -74.9, 0.058, -54.5),
}
actual = {
    'NDVI': (ndvi_ck, ndvi_d1, (ndvi_d1-ndvi_ck)/abs(ndvi_ck)*100, ndvi_rd2, (ndvi_rd2-ndvi_ck)/abs(ndvi_ck)*100),
    'NDRE': (ndre_ck, ndre_d1, (ndre_d1-ndre_ck)/abs(ndre_ck)*100, ndre_rd2, (ndre_rd2-ndre_ck)/abs(ndre_ck)*100),
    'EVI':  (evi_ck,  evi_d1,  (evi_d1-evi_ck)/abs(evi_ck)*100,   evi_rd2,  (evi_rd2-evi_ck)/abs(evi_ck)*100),
    'PRI':  (pri_ck,  pri_d1,  (pri_d1-pri_ck)/abs(pri_ck)*100,   pri_rd2,  (pri_rd2-pri_ck)/abs(pri_ck)*100),
}
for name in ['NDVI','NDRE','EVI','PRI']:
    t = text[name]
    a = actual[name]
    print(f'{name}: text=({t[0]:.3f},{t[1]:.3f},{t[2]:+.1f}%,{t[3]:.3f},{t[4]:+.1f}%) actual=({a[0]:.3f},{a[1]:.3f},{a[2]:+.1f}%,{a[3]:.3f},{a[4]:+.1f}%)')
