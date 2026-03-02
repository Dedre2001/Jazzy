# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

curr_path = r'F:/all_exp/Thesis/论文图片/Figure_3-5/features_40_nir_corrected.csv'
bak_path  = r'F:/all_exp/Thesis/论文图片/Figure_3-5/features_40_nir_corrected_backup.csv'
df_c = pd.read_csv(curr_path)
df_b = pd.read_csv(bak_path)

def R(row, band):
    return row[band]

def check_formula(df, vi_col, func, n=20):
    errs = []
    for i in range(min(n, len(df))):
        r = df.iloc[i]
        try:
            val = func(r)
            errs.append(abs(val - r[vi_col]))
        except Exception as e:
            errs.append(999)
    return np.mean(errs)

print("=== VI公式验证汇总 ===\n")

# NDVI
print("[VI_NDVI]")
print("  curr  (R820-R660)/(R820+R660):", check_formula(df_c, 'VI_NDVI', lambda r: (r['R820']-r['R660'])/(r['R820']+r['R660'])))
print("  curr  (R820-R680)/(R820+R680):", check_formula(df_c, 'VI_NDVI', lambda r: (r['R820']-r['R680'])/(r['R820']+r['R680'])))
print("  backup(R820-R680)/(R820+R680):", check_formula(df_b, 'VI_NDVI', lambda r: (r['R820']-r['R680'])/(r['R820']+r['R680'])))
print("  backup(R820-R660)/(R820+R660):", check_formula(df_b, 'VI_NDVI', lambda r: (r['R820']-r['R660'])/(r['R820']+r['R660'])))

print("\n[VI_NDRE]")
print("  curr  (R820-R710)/(R820+R710):", check_formula(df_c, 'VI_NDRE', lambda r: (r['R820']-r['R710'])/(r['R820']+r['R710'])))
print("  backup(R820-R710)/(R820+R710):", check_formula(df_b, 'VI_NDRE', lambda r: (r['R820']-r['R710'])/(r['R820']+r['R710'])))

print("\n[VI_EVI]  -- 标准: 2.5*(NIR-Red)/(NIR+6*Red-7.5*Blue+1)")
print("  curr  2.5*(R820-R660)/(R820+6*R660-7.5*R460+1):",
      check_formula(df_c, 'VI_EVI', lambda r: 2.5*(r['R820']-r['R660'])/(r['R820']+6*r['R660']-7.5*r['R460']+1)))
print("  curr  2.5*(R820-R680)/(R820+6*R680-7.5*R460+1):",
      check_formula(df_c, 'VI_EVI', lambda r: 2.5*(r['R820']-r['R680'])/(r['R820']+6*r['R680']-7.5*r['R460']+1)))
print("  backup2.5*(R820-R680)/(R820+6*R680-7.5*R460+1):",
      check_formula(df_b, 'VI_EVI', lambda r: 2.5*(r['R820']-r['R680'])/(r['R820']+6*r['R680']-7.5*r['R460']+1)))
print("  backup2.5*(R820-R660)/(R820+6*R660-7.5*R460+1):",
      check_formula(df_b, 'VI_EVI', lambda r: 2.5*(r['R820']-r['R660'])/(r['R820']+6*r['R660']-7.5*r['R460']+1)))

print("\n[VI_SIPI]  -- 原始Penuelas 1995: (R800-R445)/(R800-R680)")
print("  curr  (R820-R460)/(R820-R680):",
      check_formula(df_c, 'VI_SIPI', lambda r: (r['R820']-r['R460'])/(r['R820']-r['R680'])))
print("  backup(R820-R460)/(R820-R680):",
      check_formula(df_b, 'VI_SIPI', lambda r: (r['R820']-r['R460'])/(r['R820']-r['R680'])))

print("\n[VI_PRI]  -- 标准: (R531-R570)/(R531+R570)")
print("  curr  (R520-R590)/(R520+R590):",
      check_formula(df_c, 'VI_PRI', lambda r: (r['R520']-r['R590'])/(r['R520']+r['R590'])))
print("  backup(R520-R590)/(R520+R590):",
      check_formula(df_b, 'VI_PRI', lambda r: (r['R520']-r['R590'])/(r['R520']+r['R590'])))

print("\n[VI_MTCI]  -- 原始Dash 2004: (R754-R709)/(R709-R681)")
print("  curr  (R730-R710)/(R710-R680):",
      check_formula(df_c, 'VI_MTCI', lambda r: (r['R730']-r['R710'])/(r['R710']-r['R680'])))
print("  backup(R730-R710)/(R710-R680):",
      check_formula(df_b, 'VI_MTCI', lambda r: (r['R730']-r['R710'])/(r['R710']-r['R680'])))

print("\n[VI_GNDVI]  -- 标准Gitelson 1996: (R820-R520)/(R820+R520)")
print("  curr  (R820-R520)/(R820+R520):",
      check_formula(df_c, 'VI_GNDVI', lambda r: (r['R820']-r['R520'])/(r['R820']+r['R520'])))
print("  backup(R820-R520)/(R820+R520):",
      check_formula(df_b, 'VI_GNDVI', lambda r: (r['R820']-r['R520'])/(r['R820']+r['R520'])))

print("\n[VI_NDWI]  -- Gao 1996: (R860-R1240)/(R860+R1240); backup替代: (R850-R910)/(R850+R910)")
print("  curr  (R850-R680)/(R850+R680):",
      check_formula(df_c, 'VI_NDWI', lambda r: (r['R850']-r['R680'])/(r['R850']+r['R680'])))
print("  backup(R850-R910)/(R850+R910):",
      check_formula(df_b, 'VI_NDWI', lambda r: (r['R850']-r['R910'])/(r['R850']+r['R910'])))
print("  Note: Gao NDWI需要1240nm(不可用); Penuelas WI = R900/R970(比值非差值)")
print("        McFeeters NDWI(水体) = (R520-R820)/(R520+R820) -- 不适用于叶片水分")
