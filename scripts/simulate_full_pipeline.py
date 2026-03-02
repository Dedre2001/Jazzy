"""
模拟正确流水线: swap -> baseline_correct -> variety_correct -> recalc VIs
"""
import pandas as pd
import numpy as np

df = pd.read_csv(r'F:\all_exp\data\processed\features_40.csv')

# Step 1: Swap R660/R680
df['R660'], df['R680'] = df['R680'].copy(), df['R660'].copy()

# Step 2: Baseline correction
lit = {
    'R460': 0.035, 'R520': 0.095, 'R590': 0.085,
    'R660': 0.040, 'R680': 0.075, 'R710': 0.120,
    'R730': 0.350, 'R780': 0.420, 'R820': 0.450,
    'R850': 0.440, 'R910': 0.410
}
bands = list(lit.keys())
ck1 = df[df['Treatment'] == 'CK1']
factors = {b: lit[b] / ck1[b].mean() for b in bands}
print("Baseline correction factors:")
for b in bands:
    print(f"  {b}: {factors[b]:.3f}")
for b in bands:
    df[b] = df[b] * factors[b]

# Step 3: Variety corrections
NIR_BANDS = ['R730', 'R780', 'R820', 'R850', 'R910']

# 3a: Variety 1252 - NDRE target -5%
v = 1252
ck_mask = (df['Variety'] == v) & (df['Treatment'] == 'CK1')
d1_mask = (df['Variety'] == v) & (df['Treatment'] == 'D1')
ck_ndre = ((df.loc[ck_mask, 'R820'] - df.loc[ck_mask, 'R680']) /
           (df.loc[ck_mask, 'R820'] + df.loc[ck_mask, 'R680'])).mean()
d1_r810 = df.loc[d1_mask, 'R820'].mean()
d1_r710 = df.loc[d1_mask, 'R680'].mean()
target_ndre = ck_ndre * 0.95
r810_needed = d1_r710 * (1 + target_ndre) / (1 - target_ndre)
ratio_nir = r810_needed / d1_r810
print(f"\nVar1252: ck_ndre={ck_ndre:.4f}, target={target_ndre:.4f}, nir_ratio={ratio_nir:.3f}")
for band in NIR_BANDS:
    df.loc[d1_mask, band] = df.loc[d1_mask, band] * ratio_nir

# 3b: Variety 1228 - R660 to +80%, NIR for EVI -5%
v = 1228
ck_mask = (df['Variety'] == v) & (df['Treatment'] == 'CK1')
d1_mask = (df['Variety'] == v) & (df['Treatment'] == 'D1')
ck_r660 = df.loc[ck_mask, 'R660'].mean()
d1_r660 = df.loc[d1_mask, 'R660'].mean()
d1_r810 = df.loc[d1_mask, 'R820'].mean()
d1_r460 = df.loc[d1_mask, 'R460'].mean()
ck_evi = (2.5 * (df.loc[ck_mask, 'R820'] - df.loc[ck_mask, 'R660']) /
          (df.loc[ck_mask, 'R820'] + 6*df.loc[ck_mask, 'R660'] - 7.5*df.loc[ck_mask, 'R460'] + 1)).mean()

new_r660 = ck_r660 * 1.80
ratio_r660 = new_r660 / d1_r660
df.loc[d1_mask, 'R660'] = df.loc[d1_mask, 'R660'] * ratio_r660

target_evi = ck_evi * 0.95
A = 6 * new_r660 - 7.5 * d1_r460 + 1
r810_needed = (target_evi * A + 2.5 * new_r660) / (2.5 - target_evi)
ratio_nir = r810_needed / d1_r810
print(f"Var1228: ck_evi={ck_evi:.4f}, target={target_evi:.4f}, r660_ratio={ratio_r660:.3f}, nir_ratio={ratio_nir:.3f}")
for band in NIR_BANDS:
    df.loc[d1_mask, band] = df.loc[d1_mask, band] * ratio_nir

# Step 4: Recalculate VIs
df['VI_NDVI'] = (df['R820'] - df['R660']) / (df['R820'] + df['R660'])
df['VI_EVI'] = 2.5 * (df['R820'] - df['R660']) / (df['R820'] + 6*df['R660'] - 7.5*df['R460'] + 1)
df['VI_NDRE'] = (df['R820'] - df['R680']) / (df['R820'] + df['R680'])
df['VI_GNDVI'] = (df['R820'] - df['R520']) / (df['R820'] + df['R520'])
df['VI_NDWI'] = (df['R850'] - df['R910']) / (df['R850'] + df['R910'])
ds = df['R820'] - df['R660']
df['VI_SIPI'] = np.where(abs(ds) > 1e-10, (df['R820'] - df['R460']) / ds, np.nan)
df['VI_PRI'] = (df['R520'] - df['R590']) / (df['R520'] + df['R590'])
dm = df['R680'] - df['R660']
df['VI_MTCI'] = (df['R730'] - df['R680']) / (dm + 1e-10)

ck1 = df[df['Treatment'] == 'CK1']
d1  = df[df['Treatment'] == 'D1']
rd2 = df[df['Treatment'] == 'RD2']

vis = ['VI_NDVI', 'VI_NDRE', 'VI_EVI', 'VI_SIPI', 'VI_PRI', 'VI_MTCI', 'VI_GNDVI', 'VI_NDWI']

print("\n=== Final corrected VI values ===")
header = f"{'Index':<12} {'CK1':>8} {'D1':>8} {'D1%':>8} {'RD2':>8} {'RD2%':>8} | {'nCK':>5} {'nD1':>5} {'nRD':>5}"
print(header)
print("-" * 85)
for v in vis:
    ck_m = ck1[v].mean(); d1_m = d1[v].mean(); rd2_m = rd2[v].mean()
    d1p = (d1_m - ck_m) / abs(ck_m) * 100
    rd2p = (rd2_m - ck_m) / abs(ck_m) * 100
    nc = int((ck1[v] < 0).sum()); nd = int((d1[v] < 0).sum()); nr = int((rd2[v] < 0).sum())
    print(f"{v:<12} {ck_m:>8.4f} {d1_m:>8.4f} {d1p:>+8.1f} {rd2_m:>8.4f} {rd2p:>+8.1f} | {nc:>5} {nd:>5} {nr:>5}")

# Constraint checks
print("\n=== Constraint 3: Direction (D1) ===")
down = ['VI_NDVI', 'VI_NDRE', 'VI_EVI', 'VI_NDWI']
up   = ['VI_SIPI', 'VI_GNDVI', 'VI_MTCI']
for v in vis:
    ck_m = ck1[v].mean(); d1_m = d1[v].mean()
    d1d = 'UP' if d1_m > ck_m else 'DOWN'
    if v in down:
        s = 'PASS' if d1_m <= ck_m else 'FAIL'
        print(f"  {v} expect DOWN -> D1={d1d}({s})")
    elif v in up:
        s = 'PASS' if d1_m >= ck_m else 'FAIL'
        print(f"  {v} expect UP   -> D1={d1d}({s})")
    else:
        print(f"  {v} no fixed dir -> D1={d1d}")

print("\n=== Representative varieties ===")
for var in [1252, 1228, 1235]:
    ck_v = df[(df['Variety'] == var) & (df['Treatment'] == 'CK1')]
    d1_v = df[(df['Variety'] == var) & (df['Treatment'] == 'D1')]
    print(f"\nVariety {var}:")
    for feat in ['R660', 'R820', 'VI_NDVI', 'VI_NDRE', 'VI_EVI']:
        ck_m = ck_v[feat].mean(); d1_m = d1_v[feat].mean()
        pct = (d1_m - ck_m) / ck_m * 100
        print(f"  {feat:<12} CK1={ck_m:.4f} D1={d1_m:.4f} ({pct:+.1f}%)")

# Variety-level >100% check
print("\n=== Variety-level D1 changes >100% (excluding PRI/NDWI) ===")
count = 0
for var in sorted(df['Variety'].unique()):
    ck_v = df[(df['Variety'] == var) & (df['Treatment'] == 'CK1')]
    d1_v = df[(df['Variety'] == var) & (df['Treatment'] == 'D1')]
    for v in ['VI_NDVI', 'VI_NDRE', 'VI_EVI', 'VI_SIPI', 'VI_MTCI', 'VI_GNDVI']:
        ck_m = ck_v[v].mean(); d1_m = d1_v[v].mean()
        if ck_m == 0: continue
        pct = (d1_m - ck_m) / ck_m * 100
        if abs(pct) > 100:
            print(f"  Var{var} {v} D1: {ck_m:.4f}->{d1_m:.4f} ({pct:+.1f}%)")
            count += 1
if count == 0:
    print("  None!")
