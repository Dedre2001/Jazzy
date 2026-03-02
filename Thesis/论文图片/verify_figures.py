"""
验证脚本：逐图检查数据与论文第三章正文描述的一致性
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

print("=" * 70)
print("第三章 图文一致性验证报告")
print("=" * 70)

# ── 加载数据 ──
physio = pd.read_csv(r'F:\all_exp\data\physio_combined.csv')
multi = pd.read_csv(r'F:\all_exp\data\raw\multi.csv')
static = pd.read_csv(r'F:\all_exp\data\raw\static.csv')
ojip = pd.read_csv(r'F:\all_exp\data\raw\ojip.csv')
features = pd.read_csv(r'F:\all_exp\data\processed\features_40.csv')

indicators = ['plant_height', 'leaf_area', 'leaf_length', 'leaf_width', 'SPAD']

# ── Figure 3-1: LC_stress 箱线图 ──
print("\n【Figure 3-1】LC_stress 箱线图")
print("-" * 50)
ck1 = physio[physio['treatment'] == 'CK1'].groupby('variety')[indicators].mean()
d1 = physio[physio['treatment'] == 'D1'].groupby('variety')[indicators].mean()
lc_stress = d1 / ck1

for col, cn in zip(indicators, ['株高', '叶面积', '叶长', '叶宽', 'SPAD']):
    vals = lc_stress[col]
    print(f"  {cn}: 均值={vals.mean():.3f}, CV={vals.std()/vals.mean()*100:.1f}%")

thesis_vals = {'plant_height': 0.741, 'leaf_area': 0.452, 'leaf_length': 0.716,
               'leaf_width': 0.622, 'SPAD': 0.921}
print("\n  论文表3-3对照:")
for col in indicators:
    actual = lc_stress[col].mean()
    expected = thesis_vals[col]
    match = "OK" if abs(actual - expected) < 0.01 else "MISMATCH"
    print(f"    {col}: actual={actual:.3f}, thesis={expected:.3f} [{match}]")

# ── Figure 3-2: PCA 双标图 ──
print("\n【Figure 3-2】PCA 双标图")
print("-" * 50)
rd2 = physio[physio['treatment'] == 'RD2'].groupby('variety')[indicators].mean()
lc_recovery = rd2 / d1
lc_all = pd.concat([lc_stress.add_suffix('_s'), lc_recovery.add_suffix('_r')], axis=1)
scaler = StandardScaler()
Z = scaler.fit_transform(lc_all)
pca = PCA(n_components=3)
pca.fit(Z)
var_ratio = pca.explained_variance_ratio_ * 100
print(f"  实际方差贡献率: PC1={var_ratio[0]:.2f}%, PC2={var_ratio[1]:.2f}%, PC3={var_ratio[2]:.2f}%")
print(f"  论文表3-5:      PC1=46.83%, PC2=22.53%, PC3=17.92%")
for i, (actual, expected) in enumerate(zip(var_ratio, [46.83, 22.53, 17.92])):
    match = "OK" if abs(actual - expected) < 2.0 else "MISMATCH!"
    print(f"    PC{i+1}: 实际={actual:.2f}%, 论文={expected:.2f}% {match}")

# ── Figure 3-4: D_stress vs D_recovery ──
print("\n【Figure 3-4】D_stress vs D_recovery 散点图")
print("-" * 50)
gt = pd.read_csv(r'F:\all_exp\data\raw\static.csv')  # 用static.csv中的D_stress/D_recovery
gt_variety = gt.drop_duplicates('Variety')[['Variety', 'D_conv', 'D_stress', 'D_recovery', 'Category']]
gt_variety = gt_variety.sort_values('D_conv', ascending=False)
print("  品种  D_stress  D_recovery  Category")
for _, row in gt_variety.iterrows():
    print(f"  {int(row['Variety']):>5}  {row['D_stress']:.4f}    {row['D_recovery']:.4f}    {row['Category']}")

# 对照论文表3-7关键品种
print("\n  论文表3-7关键品种对照:")
key_check = {1252: (0.647, 0.502), 1099: (0.348, 0.709), 1235: (0.235, 0.111)}
for v, (exp_s, exp_r) in key_check.items():
    row = gt_variety[gt_variety['Variety'] == v].iloc[0]
    ms = "OK" if abs(row['D_stress'] - exp_s) < 0.01 else "MISMATCH"
    mr = "OK" if abs(row['D_recovery'] - exp_r) < 0.01 else "MISMATCH"
    print(f"    品种{v}: D_stress={row['D_stress']:.4f}(论文{exp_s}) {ms}, "
          f"D_recovery={row['D_recovery']:.4f}(论文{exp_r}) {mr}")

# ── Figure 3-5: 反射率曲线 ──
print("\n【Figure 3-5】CK1 vs D1 平均反射率曲线")
print("-" * 50)
bands = ['R460','R520','R580','R660','R710','R730','R760','R780','R810','R850','R900']
ck1_ref = multi[multi['Treatment'] == 'CK1'][bands].mean()
d1_ref = multi[multi['Treatment'] == 'D1'][bands].mean()
print("  论文描述: 可见光区反射率升高, 红边区斜率降低, 近红外区反射率下降")
print(f"  R520: CK1={ck1_ref['R520']:.3f} → D1={d1_ref['R520']:.3f} (变化{(d1_ref['R520']/ck1_ref['R520']-1)*100:+.1f}%)")
print(f"  R660: CK1={ck1_ref['R660']:.3f} → D1={d1_ref['R660']:.3f} (变化{(d1_ref['R660']/ck1_ref['R660']-1)*100:+.1f}%)")
print(f"  R850: CK1={ck1_ref['R850']:.3f} → D1={d1_ref['R850']:.3f} (变化{(d1_ref['R850']/ck1_ref['R850']-1)*100:+.1f}%)")
vis_up = d1_ref['R520'] > ck1_ref['R520'] and d1_ref['R660'] > ck1_ref['R660']
nir_down = d1_ref['R850'] < ck1_ref['R850']
print(f"  visible up: {'OK' if vis_up else 'MISMATCH'}")
print(f"  NIR down:   {'OK' if nir_down else 'MISMATCH'}")
# 论文说近红外下降12.7%和15.2%
r850_change = (d1_ref['R850']/ck1_ref['R850']-1)*100
r900_change = (d1_ref['R900']/ck1_ref['R900']-1)*100
print(f"  R850变化: {r850_change:+.1f}% (论文: -12.7%)")
print(f"  R900变化: {r900_change:+.1f}% (论文: -15.2%)")

# ── Figure 3-5b: 三品种反射率 ──
print("\n【Figure 3-5b】三类品种反射率曲线")
print("-" * 50)
for v_id, v_name in [(1252, '抗旱型'), (1228, '中间型'), (1235, '敏感型')]:
    ck = multi[(multi['Variety'] == v_id) & (multi['Treatment'] == 'CK1')][bands].mean()
    d = multi[(multi['Variety'] == v_id) & (multi['Treatment'] == 'D1')][bands].mean()
    ndvi_ck = (ck['R810'] - ck['R660']) / (ck['R810'] + ck['R660'])
    ndvi_d = (d['R810'] - d['R660']) / (d['R810'] + d['R660'])
    print(f"  品种{v_id}({v_name}): NDVI CK1={ndvi_ck:.2f} → D1={ndvi_d:.2f} ({(ndvi_d/ndvi_ck-1)*100:+.1f}%)")

# ── Figure 3-5c: 静态荧光雷达图 ──
print("\n【Figure 3-5c】稳态荧光雷达图")
print("-" * 50)
channels = ['BF(F440)', 'GF(F520)', 'RF(F690)', 'FrF(f740)']
print("  论文描述: 抗旱型BF升高最大(+47.4%), 敏感型RF下降最大(-24.0%)")
for v_id, v_name in [(1252, '抗旱型'), (1228, '中间型'), (1235, '敏感型')]:
    ck = static[(static['Variety'] == v_id) & (static['Treatment'] == 'CK1')][channels].mean()
    d = static[(static['Variety'] == v_id) & (static['Treatment'] == 'D1')][channels].mean()
    bf_change = (d['BF(F440)'] / ck['BF(F440)'] - 1) * 100
    rf_change = (d['RF(F690)'] / ck['RF(F690)'] - 1) * 100
    print(f"  品种{v_id}({v_name}): BF变化={bf_change:+.1f}%, RF变化={rf_change:+.1f}%")

# ── Figure 3-6: OJIP曲线 ──
print("\n【Figure 3-6】OJIP曲线对比")
print("-" * 50)
print("  论文描述: 1252 Fv/Fm CK1=0.80→D1=0.76(-5.0%), 1235 Fv/Fm CK1=0.79→D1=0.63(-20.3%)")
for v_id, v_name in [(1252, '抗旱型'), (1235, '敏感型')]:
    for trt in ['CK1', 'D1']:
        sub = ojip[(ojip['Variety'] == v_id) & (ojip['Treatment'] == trt)]
        Fo = sub['OJIP_Fo'].mean()
        Fm = sub['OJIP_Fm'].mean()
        Vj = sub['OJIP_Vj'].mean()
        Vi = sub['OJIP_Vi'].mean()
        FvFm = sub['OJIP_FvFm'].mean()
        print(f"  品种{v_id} {trt}: Fo={Fo:.0f}, Fm={Fm:.0f}, Fv/Fm={FvFm:.3f}, Vj={Vj:.3f}, Vi={Vi:.3f}")

# ── Figure 3-6b: OJIP雷达图 ──
print("\n【Figure 3-6b】OJIP参数雷达图 (D1/CK1 %)")
print("-" * 50)
params = ['OJIP_FvFm', 'OJIP_PIabs', 'OJIP_Vi', 'OJIP_Vj', 'OJIP_TRo_RC', 'OJIP_ETo_RC']
param_names = ['Fv/Fm', 'PIabs', 'Vi', 'Vj', 'TRo/RC', 'ETo/RC']
print("  论文表3-12b关键数值:")
for v_id, v_name in [(1252, '抗旱型'), (1228, '中间型'), (1235, '敏感型')]:
    ck = ojip[(ojip['Variety'] == v_id) & (ojip['Treatment'] == 'CK1')][params].mean()
    d = ojip[(ojip['Variety'] == v_id) & (ojip['Treatment'] == 'D1')][params].mean()
    ratio = d / ck * 100
    fvfm_r = ratio['OJIP_FvFm']
    piabs_r = ratio['OJIP_PIabs']
    vi_r = ratio['OJIP_Vi']
    print(f"  品种{v_id}({v_name}): Fv/Fm={fvfm_r:.1f}%, PIabs={piabs_r:.1f}%, Vi={vi_r:.1f}%")

# 论文说1235 PIabs下降69.4% (即D1/CK1=30.6%), Fv/Fm下降20.3% (即79.7%)
print("  论文: 1235 PIabs D1/CK1≈30.6%, Fv/Fm D1/CK1≈79.7%")

# ── Figure 3-7: 相关性热力图 ──
print("\n【Figure 3-7】37特征相关性热力图")
print("-" * 50)
multi_cols = ['R460','R520','R580','R660','R710','R730','R760','R780','R810','R850','R900',
              'VI_NDVI','VI_NDRE','VI_EVI','VI_SIPI','VI_PRI','VI_MTCI','VI_GNDVI','VI_NDWI']
static_cols = ['BF(F440)','GF(F520)','RF(F690)','FrF(f740)',
               'SR_F690_F740','SR_F440_F690','SR_F440_F520','SR_F520_F690','SR_F440_F740','SR_F520_F740']
ojip_cols = ['OJIP_FvFm','OJIP_PIabs','OJIP_TRo_RC','OJIP_ETo_RC','OJIP_Vi','OJIP_Vj',
             'OJIP_ABS_RC_log','OJIP_DIo_RC_log']
all_feats = multi_cols + static_cols + ojip_cols
corr = features[all_feats].corr()

# 模态间平均相关性
ms_corr = corr.loc[multi_cols, static_cols].abs().mean().mean()
mo_corr = corr.loc[multi_cols, ojip_cols].abs().mean().mean()
so_corr = corr.loc[static_cols, ojip_cols].abs().mean().mean()
print(f"  Multi-Static 平均|r|: {ms_corr:.2f} (论文: 0.27)")
print(f"  Multi-OJIP   平均|r|: {mo_corr:.2f} (论文: 0.29)")
print(f"  Static-OJIP  平均|r|: {so_corr:.2f} (论文: 0.25)")

print("\n" + "=" * 70)
print("验证完成")
print("=" * 70)
