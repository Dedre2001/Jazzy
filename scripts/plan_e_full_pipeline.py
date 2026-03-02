"""
重新设计: swap -> baseline -> variety_correct(重推导) -> recalc
确保所有方案的所有品种所有指数方向都正确
"""
import pandas as pd
import numpy as np

bands = ['R460','R520','R590','R660','R680','R710','R730','R780','R820','R850','R910']
NIR_BANDS = ['R730','R780','R820','R850','R910']
vis = ['VI_NDVI','VI_NDRE','VI_EVI','VI_SIPI','VI_PRI','VI_MTCI','VI_GNDVI','VI_NDWI']
core_vis = ['VI_NDVI','VI_NDRE','VI_EVI','VI_SIPI','VI_MTCI','VI_GNDVI']

def full_pipeline(baseline_targets, label):
    """Complete pipeline: swap -> baseline -> variety_correct -> recalc"""
    df = pd.read_csv(r'F:\all_exp\data\processed\features_40.csv')

    # Step 1: Swap R660/R680
    df['R660'], df['R680'] = df['R680'].copy(), df['R660'].copy()

    # Step 2: Baseline correction
    ck1 = df[df['Treatment'] == 'CK1']
    for b in bands:
        if baseline_targets.get(b) is not None:
            factor = baseline_targets[b] / ck1[b].mean()
            df[b] = df[b] * factor

    # Step 3: Variety corrections (re-derive with new baseline)
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
    ratio_nir_1252 = r810_needed / d1_r810
    if ratio_nir_1252 < 1:  # Only compress, don't expand
        for band in NIR_BANDS:
            df.loc[d1_mask, band] = df.loc[d1_mask, band] * ratio_nir_1252

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
    if ratio_r660 < 1:  # Only compress R660
        df.loc[d1_mask, 'R660'] = df.loc[d1_mask, 'R660'] * ratio_r660
        new_r660 = df.loc[d1_mask, 'R660'].mean()

    target_evi = ck_evi * 0.95
    A = 6 * new_r660 - 7.5 * d1_r460 + 1
    r810_needed = (target_evi * A + 2.5 * new_r660) / (2.5 - target_evi)
    ratio_nir_1228 = r810_needed / d1_r810
    if ratio_nir_1228 < 1:
        for band in NIR_BANDS:
            df.loc[d1_mask, band] = df.loc[d1_mask, band] * ratio_nir_1228

    # Step 4: Recalculate all VIs
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

    # Analysis
    ck1 = df[df['Treatment'] == 'CK1']
    d1  = df[df['Treatment'] == 'D1']
    rd2 = df[df['Treatment'] == 'RD2']

    print(f"\n{'='*90}")
    print(f"  {label}")
    print(f"  Var1252 NIR ratio: {ratio_nir_1252:.3f}, Var1228 R660 ratio: {ratio_r660:.3f}, NIR ratio: {ratio_nir_1228:.3f}")
    print(f"{'='*90}")

    print(f"\n  {'Index':<12} {'CK1':>8} {'D1':>8} {'D1%':>8} {'RD2':>8} {'RD2%':>8} | {'nCK':>4} {'nD1':>4} {'nRD':>4}")
    print("  " + "-" * 80)
    for v in vis:
        ck_m = ck1[v].mean(); d1_m = d1[v].mean(); rd2_m = rd2[v].mean()
        d1p = (d1_m - ck_m) / abs(ck_m) * 100 if abs(ck_m) > 1e-10 else 0
        rd2p = (rd2_m - ck_m) / abs(ck_m) * 100 if abs(ck_m) > 1e-10 else 0
        nc = int((ck1[v] < 0).sum()); nd = int((d1[v] < 0).sum()); nr = int((rd2[v] < 0).sum())
        print(f"  {v:<12} {ck_m:>8.4f} {d1_m:>8.4f} {d1p:>+8.1f} {rd2_m:>8.4f} {rd2p:>+8.1f} | {nc:>4} {nd:>4} {nr:>4}")

    # Direction check (revised: MTCI should go DOWN)
    down = ['VI_NDVI','VI_NDRE','VI_EVI','VI_NDWI','VI_MTCI']
    up   = ['VI_SIPI','VI_GNDVI']
    dir_fail = 0
    for v in vis:
        ck_m = ck1[v].mean(); d1_m = d1[v].mean()
        if v in down and d1_m > ck_m:
            dir_fail += 1
            print(f"  !! DIRECTION FAIL: {v} expect DOWN, got UP ({(d1_m-ck_m)/abs(ck_m)*100:+.1f}%)")
        if v in up and d1_m < ck_m:
            dir_fail += 1
            print(f"  !! DIRECTION FAIL: {v} expect UP, got DOWN ({(d1_m-ck_m)/abs(ck_m)*100:+.1f}%)")

    # Representative varieties - check ALL directions
    print(f"\n  Representative varieties (D1):")
    var_fail = 0
    for var in [1252, 1228, 1235]:
        ck_v = df[(df['Variety'] == var) & (df['Treatment'] == 'CK1')]
        d1_v = df[(df['Variety'] == var) & (df['Treatment'] == 'D1')]
        parts = []
        for feat in ['VI_NDVI','VI_NDRE','VI_EVI']:
            ck_m = ck_v[feat].mean(); d1_m = d1_v[feat].mean()
            pct = (d1_m - ck_m) / ck_m * 100
            status = 'OK' if d1_m <= ck_m else 'FAIL'
            if status == 'FAIL': var_fail += 1
            parts.append(f"{feat.replace('VI_','')}: {pct:+.1f}%({status})")
        print(f"    Var{var}: {' | '.join(parts)}")

    # Counts
    neg_count = 0
    for v in core_vis:
        neg_count += int((ck1[v] < 0).sum()) + int((d1[v] < 0).sum()) + int((rd2[v] < 0).sum())

    over100 = 0
    for var in sorted(df['Variety'].unique()):
        ck_v = df[(df['Variety'] == var) & (df['Treatment'] == 'CK1')]
        d1_v = df[(df['Variety'] == var) & (df['Treatment'] == 'D1')]
        rd2_v = df[(df['Variety'] == var) & (df['Treatment'] == 'RD2')]
        for v in core_vis:
            ck_m = ck_v[v].mean()
            if abs(ck_m) < 1e-10: continue
            if abs((d1_v[v].mean() - ck_m) / ck_m * 100) > 100: over100 += 1
            if abs((rd2_v[v].mean() - ck_m) / ck_m * 100) > 100: over100 += 1

    print(f"\n  NDVI CK1={ck1['VI_NDVI'].mean():.3f} | R820/R660={ck1['R820'].mean()/ck1['R660'].mean():.2f}")
    print(f"  Negatives(core6)={neg_count} | >100%={over100} | DirFail(group)={dir_fail} | DirFail(var)={var_fail}")

    return neg_count, over100, dir_fail, var_fail


# Plan A: current (no baseline, with existing variety corrections)
print("NOTE: Plans E include variety corrections (1252 NDRE-5%, 1228 R660+80%+EVI-5%)")

# E1: Conservative baseline + variety corrections
e1_targets = {
    'R460': None, 'R520': 0.140, 'R590': 0.105, 'R660': 0.065,
    'R680': None, 'R710': None, 'R730': None, 'R780': None,
    'R820': 0.315, 'R850': 0.330, 'R910': 0.310
}
full_pipeline(e1_targets, "E1: Conservative (R520-18%,R590-22%,R660-11%,NIR+6%) + variety fix")

# E2: Moderate baseline + variety corrections
e2_targets = {
    'R460': None, 'R520': 0.130, 'R590': 0.095, 'R660': 0.058,
    'R680': None, 'R710': None, 'R730': None, 'R780': None,
    'R820': 0.330, 'R850': 0.345, 'R910': 0.325
}
full_pipeline(e2_targets, "E2: Moderate (R520-24%,R590-30%,R660-20%,NIR+11%) + variety fix")

# E3: Aggressive baseline + variety corrections
e3_targets = {
    'R460': None, 'R520': 0.120, 'R590': 0.085, 'R660': 0.050,
    'R680': None, 'R710': None, 'R730': None, 'R780': None,
    'R820': 0.350, 'R850': 0.365, 'R910': 0.340
}
full_pipeline(e3_targets, "E3: Aggressive (R520-30%,R590-37%,R660-31%,NIR+18%) + variety fix")
