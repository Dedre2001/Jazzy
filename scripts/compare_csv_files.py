# -*- coding: utf-8 -*-
"""
对比 features_40_nir_corrected.csv 和 features_40_nir_corrected_backup.csv
找出两个文件之间的数据差异
"""
import pandas as pd
import numpy as np

curr_path = r'F:\all_exp\Thesis\论文图片\Figure_3-5\features_40_nir_corrected.csv'
bak_path  = r'F:\all_exp\Thesis\论文图片\Figure_3-5\features_40_nir_corrected_backup.csv'

df_curr = pd.read_csv(curr_path)
df_bak  = pd.read_csv(bak_path)

print(f"当前文件 (curr): {len(df_curr)} 行, {len(df_curr.columns)} 列")
print(f"备份文件 (backup): {len(df_bak)} 行, {len(df_bak.columns)} 列")

# 只比较数值列
num_cols = df_curr.select_dtypes(include='number').columns.tolist()
print(f"\n数值列数: {len(num_cols)}")

# 找出有差异的列
print("\n=== 有差异的列和行 ===")
diff_rows_all = []
for col in num_cols:
    diff_mask = ~np.isclose(df_curr[col].fillna(0), df_bak[col].fillna(0), rtol=1e-6, atol=1e-10)
    if diff_mask.any():
        n_diff = diff_mask.sum()
        print(f"\n列 [{col}]: {n_diff} 行有差异")
        diff_idx = df_curr[diff_mask].index
        for i in diff_idx:
            row_id = df_curr.loc[i, 'Sample_ID']
            v_curr = df_curr.loc[i, col]
            v_bak  = df_bak.loc[i, col]
            pct = (v_curr - v_bak) / abs(v_bak) * 100 if v_bak != 0 else float('nan')
            print(f"  {row_id}: backup={v_bak:.6f} -> curr={v_curr:.6f} ({pct:+.2f}%)")
