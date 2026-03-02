"""Figure 4-2: 样本量敏感性分析曲线（学习曲线）"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'Times New Roman'

# ── 数据加载 ──────────────────────────────────────────────
df = pd.read_csv('F:/all_exp/data/processed/features_40.csv')
d1 = df[df['Treatment'] == 'D1'].copy()

FEAT_COLS = [
    'R460','R520','R580','R660','R710','R730','R760','R780','R810','R850','R900',
    'VI_NDVI','VI_NDRE','VI_EVI','VI_SIPI','VI_PRI','VI_MTCI','VI_GNDVI','VI_NDWI',
    'BF(F440)','GF(F520)','RF(F690)','FrF(f740)',
    'SR_F690_F740','SR_F440_F690','SR_F440_F520','SR_F520_F690','SR_F440_F740','SR_F520_F740',
    'OJIP_FvFm','OJIP_PIabs','OJIP_TRo_RC','OJIP_ETo_RC','OJIP_Vi','OJIP_Vj',
    'OJIP_ABS_RC_log','OJIP_DIo_RC_log'
]

varieties = sorted(d1['Variety'].unique())  # 13 varieties
n_total = len(varieties)  # 13

X_all = d1[FEAT_COLS].values
y_all = d1['D_conv'].values
var_all = d1['Variety'].values

# 品种级真实D_conv（每品种3个重复取均值）
var_true = d1.groupby('Variety')['D_conv'].mean().to_dict()

# ── 模型定义 ──────────────────────────────────────────────
def make_models():
    return {
        'TabPFN':   None,   # 单独处理
        'CatBoost': None,   # 单独处理
        'Ridge':    Pipeline([('sc', StandardScaler()), ('m', Ridge(alpha=1.0))]),
        'PLSR':     Pipeline([('sc', StandardScaler()), ('m', PLSRegression(n_components=3))]),
        'RF':       RandomForestRegressor(n_estimators=300, max_depth=5, random_state=42),
        'SVR':      Pipeline([('sc', StandardScaler()), ('m', SVR(kernel='rbf', C=1.0))]),
    }

try:
    from tabpfn import TabPFNRegressor
    HAS_TABPFN = True
except ImportError:
    HAS_TABPFN = False

try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

# ── 学习曲线实验 ──────────────────────────────────────────
# 训练品种数从3到12（步长3），测试剩余品种
train_sizes = [3, 6, 9, 12]
N_REPEAT = 20
rng = np.random.default_rng(42)

model_names = ['TabPFN', 'CatBoost', 'Ridge', 'PLSR', 'RF', 'SVR']
results = {m: {k: [] for k in train_sizes} for m in model_names}

for n_train in train_sizes:
    n_test = n_total - n_train
    if n_test < 2:
        continue
    for _ in range(N_REPEAT):
        train_vars = rng.choice(varieties, size=n_train, replace=False).tolist()
        test_vars  = [v for v in varieties if v not in train_vars]

        train_mask = np.isin(var_all, train_vars)
        test_mask  = np.isin(var_all, test_vars)

        X_tr, y_tr = X_all[train_mask], y_all[train_mask]
        X_te, y_te = X_all[test_mask],  y_all[test_mask]
        var_te = var_all[test_mask]

        # 品种级预测（取重复均值）
        def variety_spearman(preds, var_ids):
            pred_df = pd.DataFrame({'var': var_ids, 'pred': preds})
            pred_mean = pred_df.groupby('var')['pred'].mean()
            true_mean = pd.Series({v: var_true[v] for v in pred_mean.index})
            rho, _ = spearmanr(pred_mean.values, true_mean[pred_mean.index].values)
            return rho

        for mname in model_names:
            try:
                if mname == 'TabPFN':
                    if not HAS_TABPFN: continue
                    m = TabPFNRegressor(n_estimators=32, random_state=42)
                elif mname == 'CatBoost':
                    if not HAS_CATBOOST: continue
                    m = CatBoostRegressor(iterations=500, learning_rate=0.05,
                                         random_seed=42, verbose=0)
                else:
                    models = make_models()
                    m = models[mname]

                sc = StandardScaler()
                Xtr_s = sc.fit_transform(X_tr) if mname in ['TabPFN','CatBoost','RF'] else X_tr
                Xte_s = sc.transform(X_te)     if mname in ['TabPFN','CatBoost','RF'] else X_te

                if mname in ['TabPFN','CatBoost','RF']:
                    m.fit(Xtr_s, y_tr)
                    preds = m.predict(Xte_s)
                else:
                    m.fit(X_tr, y_tr)
                    preds = m.predict(X_te)
                    if preds.ndim > 1: preds = preds.ravel()

                rho = variety_spearman(preds, var_te)
                results[mname][n_train].append(rho)
            except Exception:
                pass

# ── 绘图 ──────────────────────────────────────────────────
colors_map = {
    'TabPFN':   '#2166AC',
    'CatBoost': '#4DAC26',
    'Ridge':    '#762A83',
    'PLSR':     '#E08214',
    'RF':       '#1A9850',
    'SVR':      '#D73027',
}
markers = {'TabPFN':'o','CatBoost':'s','Ridge':'^','PLSR':'D','RF':'v','SVR':'x'}

fig, ax = plt.subplots(figsize=(7, 4.5))

for mname in model_names:
    means, stds, xs = [], [], []
    for k in train_sizes:
        vals = results[mname][k]
        if len(vals) >= 3:
            means.append(np.mean(vals))
            stds.append(np.std(vals))
            xs.append(k * 3)   # 转换为样本数

    if not means: continue
    xs, means, stds = np.array(xs), np.array(means), np.array(stds)
    ax.plot(xs, means, marker=markers[mname], color=colors_map[mname],
            linewidth=1.8, markersize=6, label=mname, zorder=3)
    ax.fill_between(xs, means-stds, means+stds,
                    color=colors_map[mname], alpha=0.12, zorder=2)

ax.set_xlabel('Number of training samples', fontsize=11)
ax.set_ylabel('Spearman $\\rho$ (variety-level)', fontsize=11)
ax.set_xticks([9, 18, 27, 36])
ax.set_xticklabels(['9\n(3 var)', '18\n(6 var)', '27\n(9 var)', '36\n(12 var)'])
ax.set_ylim(0.3, 1.05)
ax.yaxis.grid(True, linestyle='--', alpha=0.4)
ax.set_axisbelow(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(loc='lower right', fontsize=9, ncol=2, framealpha=0.9)
ax.set_title('Figure 4-2  Learning curves: Spearman ρ vs. training sample size',
             fontsize=11, pad=10)

plt.tight_layout()
plt.savefig('F:/all_exp/Thesis/论文图片/Figure_4-2/Figure_4-2_learning_curve.png',
            dpi=300, bbox_inches='tight')
plt.close()
print("Figure 4-2 saved.")
