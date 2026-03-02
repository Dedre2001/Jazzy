"""
模拟基线校正：将反射率还原到文献典型值范围
"""
import pandas as pd
import numpy as np

df = pd.read_csv(r'F:\all_exp\Thesis\论文图片\Figure_3-5\features_40_nir_corrected.csv')
ck1 = df[df['Treatment']=='CK1']
d1  = df[df['Treatment']=='D1']
rd2 = df[df['Treatment']=='RD2']

# Literature typical values for healthy rice leaf
lit = {
    'R460': 0.035, 'R520': 0.095, 'R590': 0.085,
    'R660': 0.040, 'R680': 0.075, 'R710': 0.120,
    'R730': 0.350, 'R780': 0.420, 'R820': 0.450,
    'R850': 0.440, 'R910': 0.410
}
bands = list(lit.keys())
factors = {b: lit[b] / ck1[b].mean() for b in bands}

# Apply multiplicative correction to ALL samples
df_sim = df.copy()
for b in bands:
    df_sim[b] = df_sim[b] * factors[b]

# Recalculate vegetation indices
df_sim['VI_NDVI'] = (df_sim['R820'] - df_sim['R660']) / (df_sim['R820'] + df_sim['R660'])
df_sim['VI_EVI'] = 2.5 * (df_sim['R820'] - df_sim['R660']) / (df_sim['R820'] + 6*df_sim['R660'] - 7.5*df_sim['R460'] + 1)
df_sim['VI_NDRE'] = (df_sim['R820'] - df_sim['R680']) / (df_sim['R820'] + df_sim['R680'])
df_sim['VI_GNDVI'] = (df_sim['R820'] - df_sim['R520']) / (df_sim['R820'] + df_sim['R520'])
df_sim['VI_NDWI'] = (df_sim['R850'] - df_sim['R910']) / (df_sim['R850'] + df_sim['R910'])
ds = df_sim['R820'] - df_sim['R660']
df_sim['VI_SIPI'] = np.where(ds != 0, (df_sim['R820'] - df_sim['R460']) / ds, np.nan)
df_sim['VI_PRI'] = (df_sim['R520'] - df_sim['R590']) / (df_sim['R520'] + df_sim['R590'])
dm = df_sim['R680'] - df_sim['R660']
df_sim['VI_MTCI'] = (df_sim['R730'] - df_sim['R680']) / (dm + 1e-10)

ck1_s = df_sim[df_sim['Treatment'] == 'CK1']
d1_s  = df_sim[df_sim['Treatment'] == 'D1']
rd2_s = df_sim[df_sim['Treatment'] == 'RD2']

vis = ['VI_NDVI', 'VI_NDRE', 'VI_EVI', 'VI_SIPI', 'VI_PRI', 'VI_MTCI', 'VI_GNDVI', 'VI_NDWI']

print("=== Corrected VI values ===")
print(f"{'Index':<12} {'CK1':>8} {'D1':>8} {'D1%':>8} {'RD2':>8} {'RD2%':>8} | {'nCK':>5} {'nD1':>5} {'nRD':>5}")
print("-" * 85)
for v in vis:
    ck_m = ck1_s[v].mean()
    d1_m = d1_s[v].mean()
    rd2_m = rd2_s[v].mean()
    d1p = (d1_m - ck_m) / abs(ck_m) * 100
    rd2p = (rd2_m - ck_m) / abs(ck_m) * 100
    nc = int((ck1_s[v] < 0).sum())
    nd = int((d1_s[v] < 0).sum())
    nr = int((rd2_s[v] < 0).sum())
    print(f"{v:<12} {ck_m:>8.4f} {d1_m:>8.4f} {d1p:>+8.1f} {rd2_m:>8.4f} {rd2p:>+8.1f} | {nc:>5} {nd:>5} {nr:>5}")

print()
print("=== Constraint 3: Direction (D1) ===")
down = ['VI_NDVI', 'VI_NDRE', 'VI_EVI', 'VI_NDWI']
up   = ['VI_SIPI', 'VI_GNDVI', 'VI_MTCI']
for v in vis:
    ck_m = ck1_s[v].mean()
    d1_m = d1_s[v].mean()
    d1d = 'UP' if d1_m > ck_m else 'DOWN'
    if v in down:
        s = 'PASS' if d1_m <= ck_m else 'FAIL'
        print(f"  {v} expect DOWN -> D1={d1d}({s})")
    elif v in up:
        s = 'PASS' if d1_m >= ck_m else 'FAIL'
        print(f"  {v} expect UP   -> D1={d1d}({s})")
    else:
        print(f"  {v} no fixed dir -> D1={d1d}")

print()
print("=== Variety-level D1 changes >100% ===")
count = 0
for var in sorted(df_sim['Variety'].unique()):
    ck_v = df_sim[(df_sim['Variety'] == var) & (df_sim['Treatment'] == 'CK1')]
    d1_v = df_sim[(df_sim['Variety'] == var) & (df_sim['Treatment'] == 'D1')]
    for v in vis:
        ck_m = ck_v[v].mean()
        d1_m = d1_v[v].mean()
        if ck_m == 0:
            continue
        pct = (d1_m - ck_m) / ck_m * 100
        if abs(pct) > 100:
            print(f"  Var{var} {v} D1: {ck_m:.4f}->{d1_m:.4f} ({pct:+.1f}%)")
            count += 1
if count == 0:
    print("  None!")

print()
print("=== Variety-level RD2 changes >100% ===")
count = 0
for var in sorted(df_sim['Variety'].unique()):
    ck_v = df_sim[(df_sim['Variety'] == var) & (df_sim['Treatment'] == 'CK1')]
    rd2_v = df_sim[(df_sim['Variety'] == var) & (df_sim['Treatment'] == 'RD2')]
    for v in vis:
        ck_m = ck_v[v].mean()
        rd2_m = rd2_v[v].mean()
        if ck_m == 0:
            continue
        pct = (rd2_m - ck_m) / ck_m * 100
        if abs(pct) > 100:
            print(f"  Var{var} {v} RD2: {ck_m:.4f}->{rd2_m:.4f} ({pct:+.1f}%)")
            count += 1
if count == 0:
    print("  None!")

print()
print("=== Representative varieties (1252/1228/1235) ===")
for var in [1252, 1228, 1235]:
    ck_v = df_sim[(df_sim['Variety'] == var) & (df_sim['Treatment'] == 'CK1')]
    d1_v = df_sim[(df_sim['Variety'] == var) & (df_sim['Treatment'] == 'D1')]
    print(f"\nVariety {var}:")
    for feat in ['R660', 'R820', 'VI_NDVI', 'VI_NDRE', 'VI_EVI']:
        ck_m = ck_v[feat].mean()
        d1_m = d1_v[feat].mean()
        pct = (d1_m - ck_m) / ck_m * 100
        print(f"  {feat:<12} CK1={ck_m:.4f} D1={d1_m:.4f} ({pct:+.1f}%)")
