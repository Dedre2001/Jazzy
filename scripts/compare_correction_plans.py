"""
对比三种校正方案对">100%变化"的缓解效果
方案A: 不做基线校正（当前状态）
方案B: 温和校正（仅降R520/R590，NIR不动）
方案C: 中等校正（降可见光+微升NIR）
"""
import pandas as pd
import numpy as np

df_orig = pd.read_csv(r'F:\all_exp\Thesis\论文图片\Figure_3-5\features_40_nir_corrected.csv')

# 方案A: 当前数据（不校正）
# 方案B: 仅降R520/R590
target_B = {
    'R460': None, 'R520': 0.130, 'R590': 0.095,
    'R660': None, 'R680': None, 'R710': None,
    'R730': None, 'R780': None, 'R820': None,
    'R850': None, 'R910': None
}
# 方案C: 降可见光 + 微升NIR
target_C = {
    'R460': 0.038, 'R520': 0.120, 'R590': 0.095,
    'R660': 0.060, 'R680': 0.095, 'R710': 0.148,
    'R730': 0.260, 'R780': 0.298, 'R820': 0.320,
    'R850': 0.330, 'R910': 0.310
}

bands = ['R460','R520','R590','R660','R680','R710','R730','R780','R820','R850','R910']
vis = ['VI_NDVI','VI_NDRE','VI_EVI','VI_SIPI','VI_PRI','VI_MTCI','VI_GNDVI','VI_NDWI']

def apply_correction(df, targets):
    df_c = df.copy()
    ck1 = df_c[df_c['Treatment'] == 'CK1']
    for b in bands:
        if targets[b] is not None:
            factor = targets[b] / ck1[b].mean()
            df_c[b] = df_c[b] * factor
    # Recalculate VIs
    df_c['VI_NDVI'] = (df_c['R820'] - df_c['R660']) / (df_c['R820'] + df_c['R660'])
    df_c['VI_EVI'] = 2.5 * (df_c['R820'] - df_c['R660']) / (df_c['R820'] + 6*df_c['R660'] - 7.5*df_c['R460'] + 1)
    df_c['VI_NDRE'] = (df_c['R820'] - df_c['R680']) / (df_c['R820'] + df_c['R680'])
    df_c['VI_GNDVI'] = (df_c['R820'] - df_c['R520']) / (df_c['R820'] + df_c['R520'])
    df_c['VI_NDWI'] = (df_c['R850'] - df_c['R910']) / (df_c['R850'] + df_c['R910'])
    ds = df_c['R820'] - df_c['R660']
    df_c['VI_SIPI'] = np.where(abs(ds) > 1e-10, (df_c['R820'] - df_c['R460']) / ds, np.nan)
    df_c['VI_PRI'] = (df_c['R520'] - df_c['R590']) / (df_c['R520'] + df_c['R590'])
    dm = df_c['R680'] - df_c['R660']
    df_c['VI_MTCI'] = (df_c['R730'] - df_c['R680']) / (dm + 1e-10)
    return df_c

def count_issues(df, label):
    ck1 = df[df['Treatment'] == 'CK1']
    d1  = df[df['Treatment'] == 'D1']
    rd2 = df[df['Treatment'] == 'RD2']

    # Count negatives (excluding PRI/NDWI which are naturally +/-)
    core_vis = ['VI_NDVI','VI_NDRE','VI_EVI','VI_SIPI','VI_MTCI','VI_GNDVI']
    neg_count = 0
    for v in core_vis:
        neg_count += int((ck1[v] < 0).sum())
        neg_count += int((d1[v] < 0).sum())
        neg_count += int((rd2[v] < 0).sum())

    # Count variety-level >100% changes (excluding PRI/NDWI)
    over100_d1 = 0
    over100_rd2 = 0
    for var in df['Variety'].unique():
        ck_v = df[(df['Variety'] == var) & (df['Treatment'] == 'CK1')]
        d1_v = df[(df['Variety'] == var) & (df['Treatment'] == 'D1')]
        rd2_v = df[(df['Variety'] == var) & (df['Treatment'] == 'RD2')]
        for v in core_vis:
            ck_m = ck_v[v].mean()
            if abs(ck_m) < 1e-10:
                continue
            d1_pct = abs((d1_v[v].mean() - ck_m) / ck_m * 100)
            rd2_pct = abs((rd2_v[v].mean() - ck_m) / ck_m * 100)
            if d1_pct > 100:
                over100_d1 += 1
            if rd2_pct > 100:
                over100_rd2 += 1

    # Group-level VI values
    print(f"\n{'='*85}")
    print(f"  {label}")
    print(f"{'='*85}")
    print(f"{'Index':<12} {'CK1':>8} {'D1':>8} {'D1%':>8} {'RD2':>8} {'RD2%':>8} | {'nCK':>5} {'nD1':>5} {'nRD':>5}")
    print("-" * 85)
    for v in vis:
        ck_m = ck1[v].mean(); d1_m = d1[v].mean(); rd2_m = rd2[v].mean()
        d1p = (d1_m - ck_m) / abs(ck_m) * 100 if abs(ck_m) > 1e-10 else 0
        rd2p = (rd2_m - ck_m) / abs(ck_m) * 100 if abs(ck_m) > 1e-10 else 0
        nc = int((ck1[v] < 0).sum()); nd = int((d1[v] < 0).sum()); nr = int((rd2[v] < 0).sum())
        print(f"{v:<12} {ck_m:>8.4f} {d1_m:>8.4f} {d1p:>+8.1f} {rd2_m:>8.4f} {rd2p:>+8.1f} | {nc:>5} {nd:>5} {nr:>5}")

    print(f"\n  Summary:")
    print(f"    NDVI CK1 = {ck1['VI_NDVI'].mean():.3f}")
    print(f"    R820/R660 = {ck1['R820'].mean()/ck1['R660'].mean():.2f}")
    print(f"    Negative samples (core 6 VIs): {neg_count}")
    print(f"    Variety-level >100% D1: {over100_d1}")
    print(f"    Variety-level >100% RD2: {over100_rd2}")
    print(f"    Total >100% cases: {over100_d1 + over100_rd2}")

    return neg_count, over100_d1, over100_rd2

# Run all three
print("NOTE: 'core 6 VIs' = NDVI, NDRE, EVI, SIPI, MTCI, GNDVI (excluding PRI/NDWI)")

df_A = df_orig.copy()
n_A, d1_A, rd2_A = count_issues(df_A, "Plan A: No baseline correction (current)")

df_B = apply_correction(df_orig, target_B)
n_B, d1_B, rd2_B = count_issues(df_B, "Plan B: Mild (only R520/R590 reduced)")

df_C = apply_correction(df_orig, target_C)
n_C, d1_C, rd2_C = count_issues(df_C, "Plan C: Moderate (visible down + NIR up 8%)")

print("\n" + "=" * 60)
print("  COMPARISON SUMMARY")
print("=" * 60)
print(f"{'Metric':<35} {'A':>8} {'B':>8} {'C':>8}")
print("-" * 60)
print(f"{'CK1 NDVI':<35} {df_A[df_A['Treatment']=='CK1']['VI_NDVI'].mean():>8.3f} {df_B[df_B['Treatment']=='CK1']['VI_NDVI'].mean():>8.3f} {df_C[df_C['Treatment']=='CK1']['VI_NDVI'].mean():>8.3f}")
print(f"{'R820/R660 ratio':<35} {df_A[df_A['Treatment']=='CK1']['R820'].mean()/df_A[df_A['Treatment']=='CK1']['R660'].mean():>8.2f} {df_B[df_B['Treatment']=='CK1']['R820'].mean()/df_B[df_B['Treatment']=='CK1']['R660'].mean():>8.2f} {df_C[df_C['Treatment']=='CK1']['R820'].mean()/df_C[df_C['Treatment']=='CK1']['R660'].mean():>8.2f}")
print(f"{'Negative samples (core 6 VIs)':<35} {n_A:>8} {n_B:>8} {n_C:>8}")
print(f"{'Variety >100% D1 (core 6 VIs)':<35} {d1_A:>8} {d1_B:>8} {d1_C:>8}")
print(f"{'Variety >100% RD2 (core 6 VIs)':<35} {rd2_A:>8} {rd2_B:>8} {rd2_C:>8}")
print(f"{'Total >100% cases':<35} {d1_A+rd2_A:>8} {d1_B+rd2_B:>8} {d1_C+rd2_C:>8}")
print(f"{'Correction intensity':<35} {'None':>8} {'Mild':>8} {'Moderate':>8}")
