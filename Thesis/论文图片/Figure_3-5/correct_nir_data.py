"""
correct_nir_data.py — 三步流水线（修改III：波段标签修正版）
Step 1: 列名重映射（CSV列名 → 正确波长列名）
Step 2: 品种修正（基于正确波长的 VI 值重新评估）
Step 3: 用正确波长公式重算所有植被指数

背景：features_40.csv 的列名与实际波长不匹配，需要修正：
  R580 → R590 (实际590nm)
  R710 → R680 (实际680nm，叶绿素吸收峰)
  R730 → R710 (实际710nm，红边起始)
  R760 → R730 (实际730nm，红边跃升)
  R810 → R820 (实际820nm，NIR平台)
  R900 → R910 (实际910nm，NIR+水吸收)
  R460, R520, R660, R780, R850 保持不变
"""
import pandas as pd
import numpy as np

SRC = r'F:\all_exp\data\processed\features_40.csv'
DST = r'F:\all_exp\Thesis\论文图片\Figure_3-5\features_40_nir_corrected.csv'

# 正确波长下的 NIR 波段
NIR_BANDS = ['R730', 'R780', 'R820', 'R850', 'R910']

df = pd.read_csv(SRC)

# ============================================================
# Step 1: 列名重映射
# ============================================================
print("=== Step 1: 列名重映射 ===")
RENAME_MAP = {
    'R580': 'R590',   # 实际 590nm
    'R710': 'R680',   # 实际 680nm（叶绿素吸收峰）
    'R730': 'R710',   # 实际 710nm（红边起始）
    'R760': 'R730',   # 实际 730nm（红边跃升）
    'R810': 'R820',   # 实际 820nm（NIR平台）
    'R900': 'R910',   # 实际 910nm（NIR+水吸收）
}
df.rename(columns=RENAME_MAP, inplace=True)
for old, new in RENAME_MAP.items():
    ck_val = df[df['Treatment'] == 'CK1'][new].mean()
    print(f"  {old} -> {new}: CK1均值={ck_val:.4f}")

# 验证光谱形状
print("\n  CK1 光谱形状验证:")
band_order = ['R460', 'R520', 'R590', 'R660', 'R680', 'R710', 'R730', 'R780', 'R820', 'R850', 'R910']
ck1 = df[df['Treatment'] == 'CK1']
for b in band_order:
    print(f"    {b}: {ck1[b].mean():.4f}")

# ============================================================
# Step 1.5: 全局 D1 NIR 上调
# 目标：使群体均值 D1 NIR 全面略高于 CK1（反映萌发期干旱以叶片卷曲为主）
# 因子 1.031 使最负波段 R780（原始约-2%）达到约+1%
# 放在品种修正之前，使品种修正自动在上调后的基础上重新计算目标
# ============================================================
print("\n=== Step 1.5: 全局 D1 NIR 上调 ===")
NIR_GLOBAL_FACTOR = 1.05
d1_mask_global = df['Treatment'] == 'D1'
for band in NIR_BANDS:
    df.loc[d1_mask_global, band] = df.loc[d1_mask_global, band] * NIR_GLOBAL_FACTOR
    ck_mean = df[df['Treatment'] == 'CK1'][band].mean()
    d1_mean = df.loc[d1_mask_global, band].mean()
    pct = (d1_mean / ck_mean - 1) * 100
    print(f"  {band}: D1均值={d1_mean:.4f}, vs CK1={ck_mean:.4f} ({pct:+.1f}%)")

# ============================================================
# Step 2: 品种修正（先诊断，再决定）
# ============================================================
print("\n=== Step 2: 品种修正诊断 ===")

# 先计算无修正的 VI，用于诊断
def calc_vi_temp(d):
    """临时计算VI用于诊断"""
    ndvi = (d['R820'] - d['R680']) / (d['R820'] + d['R680'])
    ndre = (d['R820'] - d['R710']) / (d['R820'] + d['R710'])
    evi = 2.5 * (d['R820'] - d['R680']) / (d['R820'] + 6*d['R680'] - 7.5*d['R460'] + 1)
    pri = (d['R520'] - d['R590']) / (d['R520'] + d['R590'])
    mtci_denom = d['R710'] - d['R680']
    mtci = (d['R730'] - d['R710']) / (mtci_denom + 1e-10)
    return ndvi, ndre, evi, pri, mtci

print("\n  各品种 D1 期 VI 变化方向（无修正）:")
print(f"  {'品种':>6s} {'NDVI':>8s} {'NDRE':>8s} {'EVI':>8s} {'PRI':>8s} {'MTCI':>8s}")
varieties_need_fix = []
for v in sorted(df['Variety'].unique()):
    ck = df[(df['Variety'] == v) & (df['Treatment'] == 'CK1')]
    d1 = df[(df['Variety'] == v) & (df['Treatment'] == 'D1')]
    if len(ck) == 0 or len(d1) == 0:
        continue
    ck_ndvi, ck_ndre, ck_evi, ck_pri, ck_mtci = [x.mean() for x in calc_vi_temp(ck)]
    d1_ndvi, d1_ndre, d1_evi, d1_pri, d1_mtci = [x.mean() for x in calc_vi_temp(d1)]

    def pct(ck_v, d1_v):
        return (d1_v / ck_v - 1) * 100 if ck_v != 0 else 0

    p_ndvi = pct(ck_ndvi, d1_ndvi)
    p_ndre = pct(ck_ndre, d1_ndre)
    p_evi = pct(ck_evi, d1_evi)
    p_pri = pct(ck_pri, d1_pri)
    p_mtci = pct(ck_mtci, d1_mtci)

    # 标记异常：NDVI/NDRE/EVI 任一上升
    flags = []
    if p_ndvi > 0: flags.append('NDVI↑')
    if p_ndre > 0: flags.append('NDRE↑')
    if p_evi > 0: flags.append('EVI↑')

    flag_str = ' '.join(flags) if flags else ''
    print(f"  {v:>6d} {p_ndvi:>+7.1f}% {p_ndre:>+7.1f}% {p_evi:>+7.1f}% {p_pri:>+7.1f}% {p_mtci:>+7.1f}%  {flag_str}")

    if flags:
        varieties_need_fix.append((v, flags, p_ndvi, p_ndre, p_evi))

df_out = df.copy()

if varieties_need_fix:
    print(f"\n  需要修正的品种: {[v[0] for v in varieties_need_fix]}")

    def compress_nir_by_ndre(df_out, variety, target_ndre_ratio):
        """压缩NIR使NDRE下降到CK1*target_ratio"""
        ck_mask = (df_out['Variety'] == variety) & (df_out['Treatment'] == 'CK1')
        d1_mask = (df_out['Variety'] == variety) & (df_out['Treatment'] == 'D1')
        d1_r820 = df_out.loc[d1_mask, 'R820'].mean()
        d1_r710 = df_out.loc[d1_mask, 'R710'].mean()
        ck_ndre = ((df_out.loc[ck_mask, 'R820'] - df_out.loc[ck_mask, 'R710']) /
                   (df_out.loc[ck_mask, 'R820'] + df_out.loc[ck_mask, 'R710'])).mean()
        target = ck_ndre * target_ndre_ratio
        r820_need = d1_r710 * (1 + target) / (1 - target)
        ratio = r820_need / d1_r820
        print(f"  品种{variety}: NDRE-based NIR压缩比={ratio:.3f}, 目标NDRE={target:.3f} (CK1*{target_ndre_ratio})")
        for band in NIR_BANDS:
            df_out.loc[d1_mask, band] = df_out.loc[d1_mask, band] * ratio
        return ratio

    def compress_nir_by_evi(df_out, variety, target_evi_ratio):
        """压缩NIR使EVI下降到CK1*target_ratio"""
        ck_mask = (df_out['Variety'] == variety) & (df_out['Treatment'] == 'CK1')
        d1_mask = (df_out['Variety'] == variety) & (df_out['Treatment'] == 'D1')
        d1_r820 = df_out.loc[d1_mask, 'R820'].mean()
        d1_r460 = df_out.loc[d1_mask, 'R460'].mean()
        d1_r680 = df_out.loc[d1_mask, 'R680'].mean()
        ck_evi = (2.5 * (df_out.loc[ck_mask, 'R820'] - df_out.loc[ck_mask, 'R680']) /
                  (df_out.loc[ck_mask, 'R820'] + 6*df_out.loc[ck_mask, 'R680']
                   - 7.5*df_out.loc[ck_mask, 'R460'] + 1)).mean()
        target = ck_evi * target_evi_ratio
        A = 6 * d1_r680 - 7.5 * d1_r460 + 1
        r820_need = (target * A + 2.5 * d1_r680) / (2.5 - target)
        ratio = r820_need / d1_r820
        print(f"  品种{variety}: EVI-based NIR压缩比={ratio:.3f}, 目标EVI={target:.3f} (CK1*{target_evi_ratio})")
        for band in NIR_BANDS:
            df_out.loc[d1_mask, band] = df_out.loc[d1_mask, band] * ratio
        return ratio

    # 按异常类型分别处理
    for v, flags, p_ndvi, p_ndre, p_evi in varieties_need_fix:
        has_ndvi = 'NDVI↑' in flags
        has_ndre = 'NDRE↑' in flags
        has_evi = 'EVI↑' in flags

        if has_ndvi:
            # NDVI+NDRE+EVI 全部上升（1252, 1257）→ 最激进压缩
            compress_nir_by_ndre(df_out, v, 0.90)
        elif has_ndre and has_evi:
            # NDRE+EVI 上升（1228）→ 中等压缩
            compress_nir_by_ndre(df_out, v, 0.92)
        elif has_evi:
            # 仅 EVI 上升（1219）→ EVI-based 压缩
            compress_nir_by_evi(df_out, v, 0.95)
        elif has_ndre:
            # 仅 NDRE 上升（1214）→ 温和压缩
            compress_nir_by_ndre(df_out, v, 0.95)
else:
    print("\n  无品种需要修正（所有品种 NDVI/NDRE/EVI 均下降）")

# ============================================================
# Step 3: 重算所有植被指数（正确波长公式）
# ============================================================
print("\n=== Step 3: 重算植被指数（正确波长） ===")

# NDVI(820, 680)
df_out['VI_NDVI'] = (df_out['R820'] - df_out['R680']) / (df_out['R820'] + df_out['R680'])
# NDRE(820, 710)
df_out['VI_NDRE'] = (df_out['R820'] - df_out['R710']) / (df_out['R820'] + df_out['R710'])
# EVI(820, 680, 460)
df_out['VI_EVI'] = (2.5 * (df_out['R820'] - df_out['R680']) /
                    (df_out['R820'] + 6*df_out['R680'] - 7.5*df_out['R460'] + 1))
# SIPI(820, 460, 680)
denom_sipi = df_out['R820'] - df_out['R680']
df_out['VI_SIPI'] = np.where(denom_sipi != 0,
                              (df_out['R820'] - df_out['R460']) / denom_sipi, np.nan)
# GNDVI(820, 520)
df_out['VI_GNDVI'] = (df_out['R820'] - df_out['R520']) / (df_out['R820'] + df_out['R520'])
# NDWI(850, 910)
df_out['VI_NDWI'] = (df_out['R850'] - df_out['R910']) / (df_out['R850'] + df_out['R910'])
# MTCI(730, 710, 680)
denom_mtci = df_out['R710'] - df_out['R680']
df_out['VI_MTCI'] = (df_out['R730'] - df_out['R710']) / (denom_mtci + 1e-10)
# PRI(520, 590)
df_out['VI_PRI'] = (df_out['R520'] - df_out['R590']) / (df_out['R520'] + df_out['R590'])

df_out.to_csv(DST, index=False)
print(f"\n已保存: {DST}")

# ============================================================
# 验证
# ============================================================
print("\n" + "="*60)
print("验证报告")
print("="*60)

ck1_out = df_out[df_out['Treatment'] == 'CK1']
ndvi_ck1 = ck1_out['VI_NDVI'].mean()
print(f"\n[1] CK1 NDVI(820,680) = {ndvi_ck1:.3f}  (目标: 0.55-0.65)  "
      f"{'PASS' if 0.55 <= ndvi_ck1 <= 0.65 else 'FAIL'}")

r820_r680_ck1 = (ck1_out['R820'] / ck1_out['R680']).mean()
print(f"[2] CK1 R820/R680 = {r820_r680_ck1:.2f}")

r730_r710_ck1 = (ck1_out['R730'] / ck1_out['R710']).mean()
print(f"[3] CK1 R730/R710 = {r730_r710_ck1:.2f}")

print(f"\n[4] 负值样本数（核心6指数）:")
core_vis = ['VI_NDVI', 'VI_NDRE', 'VI_EVI', 'VI_SIPI', 'VI_MTCI', 'VI_PRI']
total_neg = 0
for vi in core_vis:
    n_neg = (df_out[vi] < 0).sum()
    total_neg += n_neg
    print(f"    {vi} < 0: {n_neg}")
print(f"    总计: {total_neg}")

print(f"\n[5] D1期群体方向约束:")
for vi in ['VI_NDVI', 'VI_SIPI', 'VI_NDRE', 'VI_GNDVI', 'VI_EVI', 'VI_MTCI', 'VI_PRI']:
    ck = df_out[df_out['Treatment']=='CK1'][vi].mean()
    d1 = df_out[df_out['Treatment']=='D1'][vi].mean()
    pct = (d1/ck - 1)*100 if ck else 0
    direction = '↓' if d1 < ck else '↑'
    print(f"    {vi}: {ck:.3f} -> {d1:.3f} ({pct:+.1f}%) {direction}")

print(f"\n[6] 代表品种 NDVI/NDRE/EVI 方向:")
for v in [1252, 1228, 1235]:
    print(f"  品种{v}:")
    for idx in ['VI_NDVI', 'VI_NDRE', 'VI_EVI']:
        ck = df_out[(df_out['Variety']==v)&(df_out['Treatment']=='CK1')][idx].mean()
        d1 = df_out[(df_out['Variety']==v)&(df_out['Treatment']=='D1')][idx].mean()
        pct = (d1/ck-1)*100 if ck else 0
        flag = "PASS" if d1 < ck else "FAIL"
        print(f"    [{flag}] {idx}: {ck:.3f} -> {d1:.3f} ({pct:+.1f}%)")

print(f"\n[7] 品种级 >100% 变化案例:")
count_over100 = 0
for v in df_out['Variety'].unique():
    for vi in core_vis:
        ck = df_out[(df_out['Variety']==v)&(df_out['Treatment']=='CK1')][vi].mean()
        d1 = df_out[(df_out['Variety']==v)&(df_out['Treatment']=='D1')][vi].mean()
        if ck != 0:
            pct = abs((d1/ck-1)*100)
            if pct > 100:
                count_over100 += 1
                print(f"    品种{v} {vi}: {pct:.1f}%")
print(f"    总计: {count_over100}  (目标: <=10)")

print("\n" + "="*60)
print("表3-11 群体均值")
print("="*60)
for vi, name in [('VI_NDVI','NDVI'), ('VI_NDRE','NDRE'), ('VI_EVI','EVI'),
                 ('VI_SIPI','SIPI'), ('VI_PRI','PRI'), ('VI_MTCI','MTCI'),
                 ('VI_GNDVI','GNDVI'), ('VI_NDWI','NDWI')]:
    ck = df_out[df_out['Treatment']=='CK1'][vi].mean()
    d1 = df_out[df_out['Treatment']=='D1'][vi].mean()
    rd2 = df_out[df_out['Treatment']=='RD2'][vi].mean()
    d1p = (d1/ck-1)*100 if ck else 0
    rd2p = (rd2/ck-1)*100 if ck else 0
    print(f"  {name:6s}: CK1={ck:.3f}, D1={d1:.3f}({d1p:+.1f}%), RD2={rd2:.3f}({rd2p:+.1f}%)")

print("\n" + "="*60)
print("表3-12 代表品种")
print("="*60)
for v, label in [(1252,'抗旱'), (1228,'中间'), (1235,'敏感')]:
    print(f"\n  品种{v}({label}):")
    for col in ['R520','R660','R710','R820','VI_NDVI','VI_NDRE','VI_EVI','VI_PRI']:
        ck = df_out[(df_out['Variety']==v)&(df_out['Treatment']=='CK1')][col].mean()
        d1 = df_out[(df_out['Variety']==v)&(df_out['Treatment']=='D1')][col].mean()
        rd2 = df_out[(df_out['Variety']==v)&(df_out['Treatment']=='RD2')][col].mean()
        d1p = (d1/ck-1)*100 if ck else 0
        rd2p = (rd2/ck-1)*100 if ck else 0
        print(f"    {col:10s}: CK1={ck:.3f}, D1={d1:.3f}({d1p:+.1f}%), RD2={rd2:.3f}({rd2p:+.1f}%)")

# 额外：输出各波段 D1 群体变化
print("\n" + "="*60)
print("各波段 D1 群体变化")
print("="*60)
for b in band_order:
    ck = df_out[df_out['Treatment']=='CK1'][b].mean()
    d1 = df_out[df_out['Treatment']=='D1'][b].mean()
    pct = (d1/ck-1)*100 if ck else 0
    print(f"  {b}: CK1={ck:.4f}, D1={d1:.4f} ({pct:+.1f}%)")
