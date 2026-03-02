# -*- coding: utf-8 -*-
"""
第5章全部图片生成脚本 (v2 - 对标第4章顶刊风格)
Figure 5-1: SHAP蜂群图 (Top-10特征贡献方向)
Figure 5-2: 37×37 SHAP交互热力图
Figure 5-3: 跨模态交互网络图
Figure 5-4: 三个典型品种SHAP瀑布图
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np
import os, warnings
warnings.filterwarnings('ignore')

from catboost import CatBoostRegressor
from sklearn.preprocessing import StandardScaler
import shap

import scienceplots                          # 第4章同款科学绘图库
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
import seaborn as sns

# ── 风格：完全对标第4章 ──────────────────────────────────────────────────────
plt.style.use(['science', 'nature', 'no-latex'])
plt.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["Times New Roman"],
    "mathtext.fontset":  "stix",           # ρ、R²等数学符号专业化
    "font.size":         11,
    "axes.labelsize":    12,
    "axes.linewidth":    1.25,
    "figure.dpi":        150,              # 渲染用低DPI；保存用600
    # 中文支持（覆盖 scienceplots 的 serif 设置）
    "font.sans-serif":   ["SimHei", "Microsoft YaHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
})

# ── 路径 ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
OUT_DIR  = BASE_DIR / "Thesis" / "论文图片"
os.makedirs(OUT_DIR, exist_ok=True)

# ── 配色：Paul Tol 科学配色（与第4章协调） ──────────────────────────────────
MODALITY_COLORS = {
    'Multi':   '#4477AA',   # 科技蓝（与第4章 R² 色一致）
    'Static':  '#EE7733',   # 温暖橙
    'OJIP':    '#228833',   # 森林绿
}
VARIETY_COLORS = {
    '抗旱型': '#228833',
    '中间型': '#EE7733',
    '敏感型': '#CC3311',
}

# ── 特征分组（37个，不含 Treatment 哑变量） ──────────────────────────────────
FEATURE_GROUPS = {
    'Multi': [
        'R460','R520','R590','R660','R680','R710','R730','R780','R820','R850','R910',
        'VI_NDVI','VI_NDRE','VI_EVI','VI_SIPI','VI_PRI','VI_MTCI','VI_GNDVI','VI_NDWI'
    ],
    'Static': [
        'BF(F440)','GF(F520)','RF(F690)','FrF(f740)',
        'SR_F690_F740','SR_F440_F690','SR_F440_F520',
        'SR_F520_F690','SR_F440_F740','SR_F520_F740'
    ],
    'OJIP': [
        'OJIP_FvFm','OJIP_PIabs','OJIP_TRo_RC','OJIP_ETo_RC',
        'OJIP_Vi','OJIP_Vj','OJIP_ABS_RC_log','OJIP_DIo_RC_log'
    ]
}
ALL_FEATURES = FEATURE_GROUPS['Multi'] + FEATURE_GROUPS['Static'] + FEATURE_GROUPS['OJIP']

def get_modality(feat):
    for mod, feats in FEATURE_GROUPS.items():
        if feat in feats:
            return mod
    return 'Unknown'

# ── 特征显示名映射（论文用名） ────────────────────────────────────────────────
DISPLAY_NAMES = {
    'OJIP_Vi': 'OJIP_Vi', 'OJIP_FvFm': 'OJIP_FvFm', 'BF(F440)': 'BF(F440)',
    'SR_F440_F520': 'SR_F440_F520', 'OJIP_ETo_RC': 'OJIP_ETo_RC',
    'SR_F440_F690': 'SR_F440_F690', 'VI_MTCI': 'VI_MTCI', 'VI_SIPI': 'VI_SIPI',
    'FrF(f740)': 'FrF(F740)', 'R780': 'R780',
}

# ── 统一保存函数（对标第4章：PNG 600 DPI + PDF 矢量双格式） ──────────────────
def save_fig(fig, name, dpi=600):
    fig.savefig(OUT_DIR / f'{name}.png', dpi=dpi, bbox_inches='tight', facecolor='white')
    fig.savefig(OUT_DIR / f'{name}.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f'[OK] {name} saved (PNG + PDF)')

# =============================================================================
# 数据加载与模型训练
# =============================================================================

def load_all_data():
    """加载全量117个样本（含CK1/D1/RD2），与step7保持一致"""
    df = pd.read_csv(DATA_DIR / "features_40.csv")
    return df.reset_index(drop=True)

def train_catboost(df):
    X    = df[ALL_FEATURES].values
    y    = df['D_conv'].values
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)
    model = CatBoostRegressor(
        iterations=500, learning_rate=0.05, depth=4,
        l2_leaf_reg=5, min_data_in_leaf=3,
        random_seed=42, verbose=False
    )
    model.fit(X_sc, y)
    return model, scaler, X_sc, y

def compute_shap(model, X_sc):
    explainer  = shap.TreeExplainer(model)
    shap_vals  = explainer.shap_values(X_sc)
    shap_inter = explainer.shap_interaction_values(X_sc)
    return explainer, shap_vals, shap_inter

# =============================================================================
# Figure 5-1: SHAP 蜂群图
# =============================================================================

def plot_beeswarm(shap_vals, X_sc, df_d1, out_name):
    feat_importance = np.abs(shap_vals).mean(axis=0)
    top10_idx  = np.argsort(feat_importance)[::-1][:10]
    top10_feats = [ALL_FEATURES[i] for i in top10_idx]
    top10_shap  = shap_vals[:, top10_idx]
    top10_X     = X_sc[:, top10_idx]

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    cmap     = plt.cm.RdBu_r
    n_feats  = len(top10_feats)

    for rank, (feat, shap_col, x_col) in enumerate(
            zip(top10_feats[::-1], top10_shap.T[::-1], top10_X.T[::-1])):
        np.random.seed(42)
        jitter = np.random.uniform(-0.25, 0.25, len(shap_col))
        norm   = Normalize(vmin=x_col.min(), vmax=x_col.max())
        colors = cmap(norm(x_col))
        ax.scatter(shap_col, rank + jitter, c=colors, s=20,
                   alpha=0.85, linewidths=0, zorder=3)

    # y轴：特征名，颜色区分模态
    feat_labels = [DISPLAY_NAMES.get(f, f) for f in top10_feats[::-1]]
    ax.set_yticks(range(n_feats))
    ax.set_yticklabels(feat_labels, fontsize=10.5)
    for rank, feat in enumerate(top10_feats[::-1]):
        ax.get_yticklabels()[rank].set_color(MODALITY_COLORS[get_modality(feat)])

    ax.axvline(0, color='#444444', linewidth=0.9, linestyle='--', alpha=0.6)
    ax.set_xlabel('SHAP Value  (contribution to predicted $D_{\\mathrm{conv}}$)',
                  fontsize=11)
    # 内向刻度（对标第4章）
    ax.tick_params(direction='in', top=True, right=True,
                   which='both', length=5, width=1.0)
    ax.spines['top'].set_linewidth(0.8)
    ax.spines['right'].set_linewidth(0.8)

    # 颜色条
    sm   = ScalarMappable(cmap=cmap, norm=Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.55, pad=0.02, aspect=20)
    cbar.set_label('Feature value  (low \u2192 high)', fontsize=9.5)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(['Low', 'Mid', 'High'])
    cbar.ax.tick_params(labelsize=9)

    # 图例（frameon=False，与第4章一致）
    patches = [mpatches.Patch(color=c, label=m, linewidth=0)
               for m, c in MODALITY_COLORS.items()]
    ax.legend(handles=patches, loc='lower right', fontsize=9.5,
              title='Modality', title_fontsize=9.5, frameon=False)
    ax.grid(axis='x', alpha=0.25, linewidth=0.6)

    plt.tight_layout()
    save_fig(fig, out_name)

# =============================================================================
# Figure 5-2: SHAP 交互热力图（37×37）
# =============================================================================

def plot_interaction_heatmap(shap_inter, out_name):
    mean_abs = np.mean(np.abs(shap_inter), axis=0)
    inter_df = pd.DataFrame(mean_abs, index=ALL_FEATURES, columns=ALL_FEATURES)

    ordered  = FEATURE_GROUPS['OJIP'] + FEATURE_GROUPS['Static'] + FEATURE_GROUPS['Multi']
    inter_df = inter_df.loc[ordered, ordered]

    fig, ax  = plt.subplots(figsize=(13, 11))
    cmap_custom = LinearSegmentedColormap.from_list(
        'white_blue', ['#FFFFFF', '#C6DBEF', '#6BAED6', '#2171B5', '#08306B'])

    sns.heatmap(
        inter_df, ax=ax,
        cmap=cmap_custom, vmin=0,
        xticklabels=True, yticklabels=True,
        linewidths=0, linecolor='none',
        cbar_kws={'label': 'Mean |SHAP Interaction Value|', 'shrink': 0.65}
    )

    # 白色模态分隔线
    n_ojip   = len(FEATURE_GROUPS['OJIP'])
    n_static = len(FEATURE_GROUPS['Static'])
    for pos in [n_ojip, n_ojip + n_static]:
        ax.axhline(pos, color='white', linewidth=2.0)
        ax.axvline(pos, color='white', linewidth=2.0)

    # 模态区域标签
    mid_ojip   = n_ojip / 2
    mid_static = n_ojip + n_static / 2
    mid_multi  = n_ojip + n_static + len(FEATURE_GROUPS['Multi']) / 2
    for mid, label, color in [
        (mid_ojip,   'OJIP',   MODALITY_COLORS['OJIP']),
        (mid_static, 'Static', MODALITY_COLORS['Static']),
        (mid_multi,  'Multi',  MODALITY_COLORS['Multi']),
    ]:
        ax.text(-2.0, mid, label, ha='right', va='center',
                fontsize=10.5, fontweight='bold', color=color, rotation=90)
        ax.text(mid, len(ordered) + 1.5, label, ha='center', va='bottom',
                fontsize=10.5, fontweight='bold', color=color)

    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=6.5)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0,  fontsize=6.5)
    for tick in ax.get_xticklabels():
        tick.set_color(MODALITY_COLORS.get(get_modality(tick.get_text()), 'black'))
    for tick in ax.get_yticklabels():
        tick.set_color(MODALITY_COLORS.get(get_modality(tick.get_text()), 'black'))

    plt.tight_layout()
    save_fig(fig, out_name)

# =============================================================================
# Figure 5-3: 跨模态交互网络图
# =============================================================================

def plot_interaction_network(shap_inter, out_name):
    mean_abs = np.mean(np.abs(shap_inter), axis=0)
    inter_df = pd.DataFrame(mean_abs, index=ALL_FEATURES, columns=ALL_FEATURES)

    pairs = []
    for i, f1 in enumerate(ALL_FEATURES):
        for j, f2 in enumerate(ALL_FEATURES):
            if j <= i:
                continue
            m1, m2 = get_modality(f1), get_modality(f2)
            if m1 != m2:
                pairs.append((f1, m1, f2, m2, inter_df.loc[f1, f2]))
    pairs_df = pd.DataFrame(pairs, columns=['F1','M1','F2','M2','Strength'])
    pairs_df = pairs_df.sort_values('Strength', ascending=False)
    top_pairs = pairs_df.head(15)

    nodes    = list(set(top_pairs['F1'].tolist() + top_pairs['F2'].tolist()))
    node_mod = {n: get_modality(n) for n in nodes}

    ojip_nodes   = [n for n in nodes if node_mod[n] == 'OJIP']
    static_nodes = [n for n in nodes if node_mod[n] == 'Static']
    multi_nodes  = [n for n in nodes if node_mod[n] == 'Multi']

    pos = {}
    for i, n in enumerate(ojip_nodes):
        pos[n] = (0.12, 0.9 - i * 0.85 / max(len(ojip_nodes)-1, 1))
    for i, n in enumerate(static_nodes):
        pos[n] = (0.50, 0.9 - i * 0.85 / max(len(static_nodes)-1, 1))
    for i, n in enumerate(multi_nodes):
        pos[n] = (0.88, 0.9 - i * 0.85 / max(len(multi_nodes)-1, 1))

    fig, ax = plt.subplots(figsize=(10.5, 7.5))
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.10, 1.05)
    ax.axis('off')

    max_str = top_pairs['Strength'].max()
    for _, row in top_pairs.iterrows():
        x1, y1 = pos[row['F1']]
        x2, y2 = pos[row['F2']]
        lw    = 0.8 + 6.0 * (row['Strength'] / max_str)
        alpha = 0.25 + 0.65 * (row['Strength'] / max_str)
        ax.plot([x1, x2], [y1, y2], color='#888888', lw=lw, alpha=alpha, zorder=1)

    node_importance = np.abs(mean_abs).sum(axis=1)
    imp_dict = dict(zip(ALL_FEATURES, node_importance))
    max_imp  = max(imp_dict[n] for n in nodes)

    for n in nodes:
        x, y = pos[n]
        size  = 250 + 1100 * (imp_dict[n] / max_imp)
        ax.scatter(x, y, s=size, c=MODALITY_COLORS[node_mod[n]],
                   zorder=3, edgecolors='white', linewidths=1.5)
        label = DISPLAY_NAMES.get(n, n)
        if node_mod[n] == 'OJIP':
            ax.text(x - 0.04, y, label, ha='right', va='center', fontsize=8.5, zorder=4)
        elif node_mod[n] == 'Multi':
            ax.text(x + 0.04, y, label, ha='left',  va='center', fontsize=8.5, zorder=4)
        else:
            ax.text(x, y - 0.05, label, ha='center', va='top',   fontsize=8.5, zorder=4)

    for xp, label, mod in [(0.12, 'OJIP', 'OJIP'), (0.50, 'Static', 'Static'), (0.88, 'Multi', 'Multi')]:
        ax.text(xp, 0.98, label, ha='center', va='top', fontsize=13,
                fontweight='bold', color=MODALITY_COLORS[mod])

    for val, lbl in [(max_str, 'Strongest'), (max_str*0.5, 'Moderate'), (max_str*0.2, 'Weak')]:
        lw = 0.8 + 6.0 * (val / max_str)
        ax.plot([], [], color='#888888', lw=lw, label=lbl)
    ax.legend(title='Interaction strength', loc='lower left',
              fontsize=9, title_fontsize=9, frameon=False)

    plt.tight_layout()
    save_fig(fig, out_name)

# =============================================================================
# Figure 5-4: 三个典型品种 SHAP 瀑布图
# =============================================================================

def plot_waterfall_trio(shap_vals, X_sc, df_d1, explainer, out_name):
    varieties = {'1252': '抗旱型', '1228': '中间型', '1235': '敏感型'}
    C_POS, C_NEG = '#CC3311', '#4477AA'   # 正贡献=暖红，负贡献=科技蓝

    fig, axes = plt.subplots(1, 3, figsize=(15, 6.8), sharey=False)
    base_val  = explainer.expected_value

    for ax, (var_id, var_type) in zip(axes, varieties.items()):
        mask = df_d1['Variety'].astype(str) == var_id
        idx  = df_d1[mask].index.tolist()
        if not idx:
            ax.text(0.5, 0.5, f'Var {var_id}\nno data', ha='center', va='center')
            continue

        sv_mean  = shap_vals[idx].mean(axis=0)
        d_conv   = df_d1.loc[idx, 'D_conv'].mean()

        top_idx   = np.argsort(np.abs(sv_mean))[::-1][:8]
        top_feats = [ALL_FEATURES[i] for i in top_idx]
        top_sv    = sv_mean[top_idx]
        other_sv  = sv_mean.sum() - top_sv.sum()

        feats_plot = top_feats[::-1] + ['Others']
        sv_plot    = list(top_sv[::-1]) + [other_sv]
        colors     = [C_POS if sv > 0 else C_NEG for sv in sv_plot]

        y_pos = range(len(feats_plot))
        bars  = ax.barh(list(y_pos), sv_plot, color=colors, height=0.65,
                        edgecolor='white', linewidth=0.4)

        for bar, sv in zip(bars, sv_plot):
            x  = bar.get_width()
            ha = 'left' if x >= 0 else 'right'
            offset = 0.0008 if x >= 0 else -0.0008
            ax.text(x + offset, bar.get_y() + bar.get_height()/2,
                    f'{sv:+.3f}', va='center', ha=ha, fontsize=7.5)

        labels = [DISPLAY_NAMES.get(f, f) for f in feats_plot]
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(labels, fontsize=8.5)
        for tick, feat in zip(ax.get_yticklabels(), feats_plot):
            if feat != 'Others':
                tick.set_color(MODALITY_COLORS.get(get_modality(feat), 'black'))

        ax.axvline(0, color='#444444', linewidth=0.9)
        ax.set_xlabel('SHAP Value', fontsize=10)
        ax.set_title(
            f'Var {var_id}  ({var_type})\n$D_{{\\mathrm{{conv}}}}$ = {d_conv:.3f}',
            fontsize=10.5, fontweight='bold',
            color=VARIETY_COLORS.get(var_type, 'black'), pad=7
        )
        ax.tick_params(direction='in', top=True, right=True,
                       which='both', length=4, width=0.9)
        ax.grid(axis='x', alpha=0.25, linewidth=0.6)
        ax.text(0.98, 0.02, f'Base: {base_val:.3f}',
                transform=ax.transAxes, ha='right', va='bottom',
                fontsize=7.5, color='gray')

    patches = [mpatches.Patch(color=c, label=m, linewidth=0)
               for m, c in MODALITY_COLORS.items()]
    fig.legend(handles=patches, loc='upper center', ncol=3,
               fontsize=9.5, title='Modality', title_fontsize=9.5,
               bbox_to_anchor=(0.5, 1.02), frameon=False)

    plt.tight_layout()
    save_fig(fig, out_name)

# =============================================================================
# 主流程
# =============================================================================

if __name__ == '__main__':
    print('=== 加载数据 ===')
    df_d1 = load_all_data()
    print(f'全量样本数: {len(df_d1)}，品种数: {df_d1["Variety"].nunique()}')

    print('=== 训练CatBoost代理模型 (iterations=300) ===')
    model, scaler, X_sc, y = train_catboost(df_d1)
    print('模型训练完成')

    print('=== 计算SHAP值和交互值（约1-2分钟）===')
    explainer, shap_vals, shap_inter = compute_shap(model, X_sc)
    print(f'SHAP值: {shap_vals.shape}, 交互值: {shap_inter.shape}')

    print('=== Figure 5-1: SHAP蜂群图 ===')
    plot_beeswarm(shap_vals, X_sc, df_d1,
                  'Figure5-1_SHAP_beeswarm')

    print('=== Figure 5-2: 交互热力图 ===')
    plot_interaction_heatmap(shap_inter,
                  'Figure5-2_SHAP_interaction_heatmap')

    print('=== Figure 5-3: 跨模态交互网络图 ===')
    plot_interaction_network(shap_inter,
                  'Figure5-3_cross_modal_network')

    print('=== Figure 5-4: 典型品种瀑布图 ===')
    plot_waterfall_trio(shap_vals, X_sc, df_d1, explainer,
                  'Figure5-4_SHAP_waterfall')

    print(f'\n全部完成！图片保存至: {OUT_DIR}')
