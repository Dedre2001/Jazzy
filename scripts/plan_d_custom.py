"""
Plan D: 定制方案 - 精准校正偏离萌发期范围的波段
原则:
1. R680 不动 (保护 MTCI 分母 R680-R660)
2. R520/R590 降至萌发期中值
3. R660 微降至萌发期中值
4. NIR 微升 ~5% (萌发期中值偏上)
5. R460/R710/R730 不动 (已在范围内)
"""
import pandas as pd
import numpy as np

df = pd.read_csv(r'F:\all_exp\Thesis\论文图片\Figure_3-5\features_40_nir_corrected.csv')
ck1_ref = df[df['Treatment'] == 'CK1']

# Current CK1 means
print("=== Current CK1 values ===")
bands = ['R460','R520','R590','R660','R680','R710','R730','R780','R820','R850','R910']
for b in bands:
    print(f"  {b}: {ck1_ref[b].mean():.4f}")

# Plan D targets (None = no change)
plans = {
    'D1: Conservative': {
        'R460': None,       # 0.038 OK
        'R520': 0.140,      # 0.171 -> 0.140 (seedling mid-upper)
        'R590': 0.105,      # 0.135 -> 0.105 (seedling mid)
        'R660': 0.065,      # 0.073 -> 0.065 (seedling mid)
        'R680': None,       # 0.122 KEEP (protect MTCI)
        'R710': None,       # 0.148 OK
        'R730': None,       # 0.260 OK
        'R780': None,       # 0.298 OK
        'R820': 0.315,      # 0.297 -> 0.315 (+6%)
        'R850': 0.330,      # 0.312 -> 0.330 (+6%)
        'R910': 0.310,      # 0.295 -> 0.310 (+5%)
    },
    'D2: Moderate': {
        'R460': None,
        'R520': 0.130,      # -> 0.130 (seedling mid)
        'R590': 0.095,      # -> 0.095 (seedling mid)
        'R660': 0.058,      # -> 0.058 (seedling mid-lower)
        'R680': None,       # KEEP
        'R710': None,
        'R730': None,
        'R780': None,
        'R820': 0.330,      # +11%
        'R850': 0.345,      # +11%
        'R910': 0.325,      # +10%
    },
    'D3: Aggressive': {
        'R460': None,
        'R520': 0.120,      # -> 0.120 (seedling mid-lower)
        'R590': 0.085,      # -> 0.085 (seedling lower)
        'R660': 0.050,      # -> 0.050 (seedling lower)
        'R680': None,       # KEEP
        'R710': None,
        'R730': None,
        'R780': None,
        'R820': 0.350,      # +18%
        'R850': 0.365,      # +17%
        'R910': 0.340,      # +15%
    },
}

vis = ['VI_NDVI','VI_NDRE','VI_EVI','VI_SIPI','VI_PRI','VI_MTCI','VI_GNDVI','VI_NDWI']
core_vis = ['VI_NDVI','VI_NDRE','VI_EVI','VI_SIPI','VI_MTCI','VI_GNDVI']

def apply_and_analyze(df_in, targets, label):
    df_c = df_in.copy()
    ck1 = df_c[df_c['Treatment'] == 'CK1']
    for b in bands:
        if targets[b] is not None:
            factor = targets[b] / ck1[b].mean()
            df_c[b] = df_c[b] * factor

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

    ck1_s = df_c[df_c['Treatment'] == 'CK1']
    d1_s  = df_c[df_c['Treatment'] == 'D1']
    rd2_s = df_c[df_c['Treatment'] == 'RD2']

    print(f"\n{'='*90}")
    print(f"  {label}")
    print(f"{'='*90}")

    # Correction factors
    print("  Correction factors:")
    for b in bands:
        if targets[b] is not None:
            ck_orig = ck1_ref[b].mean()
            f = targets[b] / ck_orig
            print(f"    {b}: {ck_orig:.4f} -> {targets[b]:.4f} (x{f:.3f}, {(f-1)*100:+.1f}%)")

    print(f"\n  {'Index':<12} {'CK1':>8} {'D1':>8} {'D1%':>8} {'RD2':>8} {'RD2%':>8} | {'nCK':>4} {'nD1':>4} {'nRD':>4}")
    print("  " + "-" * 80)
    for v in vis:
        ck_m = ck1_s[v].mean(); d1_m = d1_s[v].mean(); rd2_m = rd2_s[v].mean()
        d1p = (d1_m - ck_m) / abs(ck_m) * 100 if abs(ck_m) > 1e-10 else 0
        rd2p = (rd2_m - ck_m) / abs(ck_m) * 100 if abs(ck_m) > 1e-10 else 0
        nc = int((ck1_s[v] < 0).sum()); nd = int((d1_s[v] < 0).sum()); nr = int((rd2_s[v] < 0).sum())
        print(f"  {v:<12} {ck_m:>8.4f} {d1_m:>8.4f} {d1p:>+8.1f} {rd2_m:>8.4f} {rd2p:>+8.1f} | {nc:>4} {nd:>4} {nr:>4}")

    # Counts
    neg_count = 0
    for v in core_vis:
        neg_count += int((ck1_s[v] < 0).sum()) + int((d1_s[v] < 0).sum()) + int((rd2_s[v] < 0).sum())

    over100_d1 = 0; over100_rd2 = 0
    over100_details_d1 = []; over100_details_rd2 = []
    for var in sorted(df_c['Variety'].unique()):
        ck_v = df_c[(df_c['Variety'] == var) & (df_c['Treatment'] == 'CK1')]
        d1_v = df_c[(df_c['Variety'] == var) & (df_c['Treatment'] == 'D1')]
        rd2_v = df_c[(df_c['Variety'] == var) & (df_c['Treatment'] == 'RD2')]
        for v in core_vis:
            ck_m = ck_v[v].mean()
            if abs(ck_m) < 1e-10: continue
            d1_pct = (d1_v[v].mean() - ck_m) / ck_m * 100
            rd2_pct = (rd2_v[v].mean() - ck_m) / ck_m * 100
            if abs(d1_pct) > 100:
                over100_d1 += 1
                over100_details_d1.append(f"    Var{var} {v}: {d1_pct:+.1f}%")
            if abs(rd2_pct) > 100:
                over100_rd2 += 1
                over100_details_rd2.append(f"    Var{var} {v}: {rd2_pct:+.1f}%")

    # Direction check
    down = ['VI_NDVI','VI_NDRE','VI_EVI','VI_NDWI']
    up   = ['VI_SIPI','VI_GNDVI','VI_MTCI']
    dir_fail = 0
    for v in vis:
        ck_m = ck1_s[v].mean(); d1_m = d1_s[v].mean()
        if v in down and d1_m > ck_m: dir_fail += 1
        if v in up and d1_m < ck_m: dir_fail += 1

    # Representative varieties
    print(f"\n  Representative varieties (D1):")
    for var in [1252, 1228, 1235]:
        ck_v = df_c[(df_c['Variety'] == var) & (df_c['Treatment'] == 'CK1')]
        d1_v = df_c[(df_c['Variety'] == var) & (df_c['Treatment'] == 'D1')]
        ndvi_ck = ck_v['VI_NDVI'].mean(); ndvi_d1 = d1_v['VI_NDVI'].mean()
        evi_ck = ck_v['VI_EVI'].mean(); evi_d1 = d1_v['VI_EVI'].mean()
        ndre_ck = ck_v['VI_NDRE'].mean(); ndre_d1 = d1_v['VI_NDRE'].mean()
        print(f"    Var{var}: NDVI {ndvi_ck:.3f}->{ndvi_d1:.3f}({(ndvi_d1-ndvi_ck)/ndvi_ck*100:+.1f}%) "
              f"NDRE {ndre_ck:.3f}->{ndre_d1:.3f}({(ndre_d1-ndre_ck)/ndre_ck*100:+.1f}%) "
              f"EVI {evi_ck:.3f}->{evi_d1:.3f}({(evi_d1-evi_ck)/evi_ck*100:+.1f}%)")

    print(f"\n  Summary:")
    print(f"    NDVI CK1 = {ck1_s['VI_NDVI'].mean():.3f}")
    print(f"    R820/R660 = {ck1_s['R820'].mean()/ck1_s['R660'].mean():.2f}")
    print(f"    Negatives (core 6): {neg_count}")
    print(f"    Direction fails (D1): {dir_fail}")
    print(f"    >100% D1: {over100_d1}")
    if over100_details_d1:
        for d in over100_details_d1: print(d)
    print(f"    >100% RD2: {over100_rd2}")
    if over100_details_rd2:
        for d in over100_details_rd2: print(d)
    print(f"    Total >100%: {over100_d1 + over100_rd2}")

    return neg_count, over100_d1 + over100_rd2, dir_fail

# Run Plan A (current) for reference
results = {}
n, o, d = apply_and_analyze(df, {b: None for b in bands}, "Plan A: No correction (current)")
results['A'] = (n, o, d)

# Run all D plans
for name, targets in plans.items():
    n, o, d = apply_and_analyze(df, targets, f"Plan {name}")
    results[name] = (n, o, d)

# Final comparison
print("\n" + "=" * 70)
print("  FINAL COMPARISON")
print("=" * 70)
print(f"{'Plan':<25} {'Negatives':>10} {'>100%':>10} {'Dir Fail':>10}")
print("-" * 55)
for name, (n, o, d) in results.items():
    print(f"{name:<25} {n:>10} {o:>10} {d:>10}")
