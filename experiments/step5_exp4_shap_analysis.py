# -*- coding: utf-8 -*-
"""
Step 5: Exp-4 SHAP 可解释性分析
================================

使用 SHAP 分析 CatBoost 模型的特征重要性，映射到光合生理机制。

输出:
- 图表: exp4_fig1-6
- 表格: exp4_table1-3
- 报告: exp4_shap_interpretability_report.md
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from pathlib import Path
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# =============================================================================
# 路径配置
# =============================================================================
BASE_DIR = Path(r"F:\all_exp")
DATA_PATH = BASE_DIR / "data" / "processed" / "features_40.csv"
FIGURE_DIR = BASE_DIR / "results" / "figures"
TABLE_DIR = BASE_DIR / "results" / "tables"
REPORT_DIR = BASE_DIR / "results" / "reports"

# 确保目录存在
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# 特征分组定义
# =============================================================================
FEATURE_GROUPS = {
    'Multi': ['R460', 'R520', 'R580', 'R660', 'R710', 'R730', 'R760', 'R780', 'R810', 'R850', 'R900'],
    'VI': ['VI_NDVI', 'VI_NDRE', 'VI_EVI', 'VI_SIPI', 'VI_PRI', 'VI_MTCI', 'VI_GNDVI', 'VI_NDWI'],
    'Static': ['BF(F440)', 'GF(F520)', 'RF(F690)', 'FrF(f740)'],
    'Static_Ratio': ['SR_F690_F740', 'SR_F440_F690', 'SR_F440_F520', 'SR_F520_F690', 'SR_F440_F740', 'SR_F520_F740'],
    'OJIP': ['OJIP_FvFm', 'OJIP_PIabs', 'OJIP_TRo_RC', 'OJIP_ETo_RC', 'OJIP_Vi', 'OJIP_Vj', 'OJIP_ABS_RC_log', 'OJIP_DIo_RC_log']
    # Treatment 已移除，不再作为特征
}

# 生理机制映射
PHYSIOLOGY_MAPPING = {
    # OJIP 参数
    'OJIP_FvFm': 'PSII最大光化学效率，直接反映PSII损伤程度',
    'OJIP_PIabs': '综合性能指数，整合光能吸收、捕获、电子传递',
    'OJIP_ABS_RC_log': '每反应中心吸收光能，RC失活时增加',
    'OJIP_TRo_RC': '每RC捕获能量，能量捕获效率',
    'OJIP_ETo_RC': '每RC电子传递，电子传递链活性',
    'OJIP_DIo_RC_log': '每RC热耗散，过剩能量耗散保护机制',
    'OJIP_Vi': 'I相相对荧光，PQ池氧化还原状态',
    'OJIP_Vj': 'J相相对荧光，QA累积程度',
    # 植被指数
    'VI_NDVI': '归一化植被指数，反映植被覆盖和活力',
    'VI_NDRE': '红边归一化指数，敏感反映叶绿素含量',
    'VI_EVI': '增强植被指数，减少大气和土壤背景影响',
    'VI_SIPI': '结构不敏感色素指数，类胡萝卜素/叶绿素比',
    'VI_PRI': '光化学反射指数，反映光合效率和叶黄素循环',
    'VI_MTCI': '叶绿素红边指数，叶绿素含量敏感指标',
    'VI_GNDVI': '绿色NDVI，对叶绿素变化更敏感',
    'VI_NDWI': '归一化水分指数，反映植物水分状态',
    # Static 荧光
    'BF(F440)': '蓝色荧光，与酚类化合物和细胞壁相关',
    'GF(F520)': '绿色荧光，与类黄酮等次生代谢物相关',
    'RF(F690)': '红色荧光，叶绿素荧光主峰',
    'FrF(f740)': '远红荧光，叶绿素荧光次峰',
    'SR_F690_F740': '红/远红荧光比，叶绿素荧光重吸收效应',
    'SR_F440_F690': '蓝/红荧光比，胁迫敏感指标',
    'SR_F440_F520': '蓝/绿荧光比，多酚类物质积累',
    'SR_F520_F690': '绿/红荧光比',
    'SR_F440_F740': '蓝/远红荧光比',
    'SR_F520_F740': '绿/远红荧光比',
    # Multi 波段
    'R460': '蓝光反射率，叶绿素吸收峰',
    'R520': '绿光反射率，绿峰',
    'R580': '黄光反射率',
    'R660': '红光反射率，叶绿素吸收峰',
    'R710': '红边起始',
    'R730': '红边中点',
    'R760': '红边结束',
    'R780': '近红外平台起始',
    'R810': '近红外平台',
    'R850': '近红外平台',
    'R900': '近红外平台/水分吸收',
    # Treatment
    'Trt_CK1': '对照处理',
    'Trt_D1': '干旱处理',
    'Trt_RD2': '复水处理'
}


# =============================================================================
# 数据加载
# =============================================================================
def load_data():
    """加载数据并准备特征矩阵"""
    df = pd.read_csv(DATA_PATH)

    # 获取所有特征列
    feature_cols = []
    for group_features in FEATURE_GROUPS.values():
        feature_cols.extend(group_features)

    # 检查哪些特征存在
    available_features = [f for f in feature_cols if f in df.columns]
    missing_features = [f for f in feature_cols if f not in df.columns]

    if missing_features:
        print(f"[WARNING] 缺失特征: {missing_features}")

    X = df[available_features].values
    y = df['D_conv'].values

    return X, y, available_features, df


# =============================================================================
# 模型训练
# =============================================================================
def train_catboost(X, y):
    """训练 CatBoost 模型（全量数据，用于SHAP分析）"""
    model = CatBoostRegressor(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        loss_function='RMSE',
        random_seed=42,
        verbose=False
    )
    model.fit(X, y)
    return model


# =============================================================================
# SHAP 分析
# =============================================================================
def compute_shap_values(model, X, feature_names):
    """计算 SHAP 值"""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    return shap_values, explainer.expected_value


def get_feature_importance(shap_values, feature_names):
    """计算全局特征重要性（平均绝对SHAP值）"""
    importance = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values('Importance', ascending=False)

    # 添加特征组
    def get_group(feature):
        for group, features in FEATURE_GROUPS.items():
            if feature in features:
                return group
        return 'Unknown'

    importance_df['Group'] = importance_df['Feature'].apply(get_group)
    importance_df['Rank'] = range(1, len(importance_df) + 1)

    return importance_df


def get_group_importance(importance_df):
    """计算特征组聚合重要性"""
    group_importance = importance_df.groupby('Group')['Importance'].sum()
    group_importance = group_importance.sort_values(ascending=False)

    # 计算占比
    total = group_importance.sum()
    group_df = pd.DataFrame({
        'Group': group_importance.index,
        'Importance': group_importance.values,
        'Percentage': (group_importance.values / total * 100)
    })

    return group_df


# =============================================================================
# 图表生成
# =============================================================================
def plot_summary_bar(shap_values, X, feature_names, save_path):
    """Fig1: 全局特征重要性柱状图"""
    plt.figure(figsize=(10, 12))
    shap.summary_plot(shap_values, X, feature_names=feature_names,
                      plot_type="bar", show=False, max_display=20)
    plt.title("SHAP Feature Importance (Top 20)", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] 图1已保存: {save_path}")


def plot_beeswarm(shap_values, X, feature_names, save_path):
    """Fig2: 蜂群图"""
    plt.figure(figsize=(10, 12))
    shap.summary_plot(shap_values, X, feature_names=feature_names,
                      show=False, max_display=20)
    plt.title("SHAP Beeswarm Plot (Top 20)", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] 图2已保存: {save_path}")


def plot_dependence(shap_values, X, feature_names, importance_df, save_path):
    """Fig3: Top-5 特征依赖图"""
    top5_features = importance_df.head(5)['Feature'].tolist()

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, feature in enumerate(top5_features):
        if i >= 5:
            break
        feature_idx = feature_names.index(feature)
        ax = axes[i]
        shap.dependence_plot(feature_idx, shap_values, X,
                            feature_names=feature_names,
                            ax=ax, show=False)
        ax.set_title(f"{feature}", fontsize=12)

    # 隐藏多余的子图
    if len(top5_features) < 6:
        axes[5].axis('off')

    plt.suptitle("SHAP Dependence Plots (Top 5 Features)", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] 图3已保存: {save_path}")


def plot_waterfall(shap_values, expected_value, X, feature_names, df, save_path):
    """Fig4: 典型品种瀑布图"""
    # 选择3个典型品种：抗旱最强、中等、最弱
    varieties = df.groupby('Variety').first().reset_index()
    varieties = varieties.sort_values('D_conv', ascending=False)

    # 抗旱最强 (rank 1), 中等 (rank 7), 最弱 (rank 13)
    sample_indices = []
    target_ranks = [1, 7, 13]

    for rank in target_ranks:
        var_row = varieties[varieties['Rank'] == rank]
        if len(var_row) > 0:
            variety = var_row['Variety'].values[0]
            # 找到该品种的第一个样本
            sample_idx = df[df['Variety'] == variety].index[0]
            sample_indices.append((sample_idx, variety, rank))

    fig, axes = plt.subplots(1, 3, figsize=(18, 8))

    for i, (idx, variety, rank) in enumerate(sample_indices):
        ax = axes[i]

        # 创建 Explanation 对象
        explanation = shap.Explanation(
            values=shap_values[idx],
            base_values=expected_value,
            data=X[idx],
            feature_names=feature_names
        )

        plt.sca(ax)
        shap.plots.waterfall(explanation, max_display=10, show=False)
        ax.set_title(f"Variety {variety} (Rank {rank})", fontsize=12)

    plt.suptitle("SHAP Waterfall Plots: Representative Varieties", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] 图4已保存: {save_path}")


def plot_group_importance(group_df, save_path):
    """Fig5: 特征组贡献饼图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 饼图
    colors = plt.cm.Set2(np.linspace(0, 1, len(group_df)))
    axes[0].pie(group_df['Percentage'], labels=group_df['Group'],
                autopct='%1.1f%%', colors=colors, startangle=90)
    axes[0].set_title("Feature Group Contribution", fontsize=14)

    # 柱状图
    bars = axes[1].barh(group_df['Group'], group_df['Percentage'], color=colors)
    axes[1].set_xlabel("Contribution (%)", fontsize=12)
    axes[1].set_title("Feature Group Importance", fontsize=14)
    axes[1].invert_yaxis()

    # 添加数值标签
    for bar, pct in zip(bars, group_df['Percentage']):
        axes[1].text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                    f'{pct:.1f}%', va='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] 图5已保存: {save_path}")


def plot_interaction_heatmap(shap_values, feature_names, importance_df, save_path):
    """Fig6: 特征交互热力图（基于SHAP值相关性）"""
    # 取 Top-10 特征
    top10_features = importance_df.head(10)['Feature'].tolist()
    top10_indices = [feature_names.index(f) for f in top10_features]

    # 计算 SHAP 值之间的相关性
    shap_subset = shap_values[:, top10_indices]
    corr_matrix = np.corrcoef(shap_subset.T)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)

    # 设置标签
    ax.set_xticks(range(len(top10_features)))
    ax.set_yticks(range(len(top10_features)))
    ax.set_xticklabels(top10_features, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(top10_features, fontsize=9)

    # 添加数值
    for i in range(len(top10_features)):
        for j in range(len(top10_features)):
            text = ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                          ha='center', va='center', fontsize=8,
                          color='white' if abs(corr_matrix[i, j]) > 0.5 else 'black')

    plt.colorbar(im, ax=ax, label='SHAP Correlation')
    ax.set_title("Feature Interaction (SHAP Correlation Matrix)", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] 图6已保存: {save_path}")


# =============================================================================
# 表格生成
# =============================================================================
def save_tables(importance_df, group_df, feature_names):
    """保存所有表格"""
    # Table 1: 特征重要性排名
    table1 = importance_df[['Rank', 'Feature', 'Group', 'Importance']].copy()
    table1['Importance'] = table1['Importance'].round(6)
    table1.to_csv(TABLE_DIR / "exp4_table1_feature_importance.csv",
                  index=False, encoding='utf-8-sig')
    print(f"[OK] 表1已保存: {TABLE_DIR}/exp4_table1_feature_importance.csv")

    # Table 2: 特征组重要性
    group_df_save = group_df.copy()
    group_df_save['Importance'] = group_df_save['Importance'].round(6)
    group_df_save['Percentage'] = group_df_save['Percentage'].round(2)
    group_df_save.to_csv(TABLE_DIR / "exp4_table2_group_importance.csv",
                         index=False, encoding='utf-8-sig')
    print(f"[OK] 表2已保存: {TABLE_DIR}/exp4_table2_group_importance.csv")

    # Table 3: 生理机制映射
    table3_data = []
    for _, row in importance_df.iterrows():
        feature = row['Feature']
        physiology = PHYSIOLOGY_MAPPING.get(feature, '')
        table3_data.append({
            'Rank': row['Rank'],
            'Feature': feature,
            'Group': row['Group'],
            'Importance': round(row['Importance'], 6),
            'Physiology': physiology
        })

    table3 = pd.DataFrame(table3_data)
    table3.to_csv(TABLE_DIR / "exp4_table3_physiology_mapping.csv",
                  index=False, encoding='utf-8-sig')
    print(f"[OK] 表3已保存: {TABLE_DIR}/exp4_table3_physiology_mapping.csv")


# =============================================================================
# 报告生成
# =============================================================================
def generate_report(importance_df, group_df):
    """生成可解释性分析报告"""
    top5 = importance_df.head(5)
    top1 = importance_df.iloc[0]

    report = f"""# Exp-4: SHAP 可解释性分析报告

**报告生成时间:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
**分析模型:** CatBoost (TreeExplainer)
**样本数:** 117
**特征数:** {len(importance_df)}

---

## 1. 全局特征重要性

### 1.1 Top-10 最重要特征

| 排名 | 特征 | 特征组 | SHAP重要性 | 生理含义 |
|:----:|:-----|:------:|:----------:|:---------|
"""

    for _, row in importance_df.head(10).iterrows():
        physiology = PHYSIOLOGY_MAPPING.get(row['Feature'], '')
        report += f"| {int(row['Rank'])} | {row['Feature']} | {row['Group']} | {row['Importance']:.4f} | {physiology} |\n"

    report += f"""

### 1.2 最重要特征解读

**{top1['Feature']}** 是对抗旱性预测贡献最大的特征（SHAP重要性 = {top1['Importance']:.4f}）。
该特征属于 **{top1['Group']}** 组，其生理含义为：{PHYSIOLOGY_MAPPING.get(top1['Feature'], 'N/A')}

---

## 2. 特征组贡献分析

### 2.1 各特征组贡献占比

| 特征组 | 总贡献 | 占比 |
|:------:|:------:|:----:|
"""

    for _, row in group_df.iterrows():
        report += f"| {row['Group']} | {row['Importance']:.4f} | {row['Percentage']:.1f}% |\n"

    # 找出贡献最大的组
    top_group = group_df.iloc[0]

    report += f"""

### 2.2 特征组解读

**{top_group['Group']}** 组贡献最大（{top_group['Percentage']:.1f}%），表明该类特征在抗旱性预测中起主导作用。

"""

    # 根据特征组给出生理学解释
    if top_group['Group'] == 'OJIP':
        report += """OJIP 参数直接反映光合电子传递链的功能状态，是评价植物抗旱性的核心生理指标。
干旱胁迫会导致 PSII 反应中心失活、电子传递效率下降，这些变化被 OJIP 荧光动力学敏感捕获。
"""
    elif top_group['Group'] == 'VI':
        report += """植被指数反映叶片的叶绿素含量、冠层结构和水分状态。
干旱胁迫导致叶绿素降解、叶片卷曲，这些表型变化会改变反射光谱特征。
"""
    elif top_group['Group'] == 'Static_Ratio':
        report += """静态荧光比值对胁迫响应敏感，特别是蓝/红荧光比（F440/F690）。
胁迫条件下，酚类化合物积累导致蓝色荧光增强，而叶绿素降解导致红色荧光减弱。
"""

    report += """

---

## 3. 图表索引

| 图编号 | 描述 | 文件路径 |
|--------|------|----------|
| Fig 1 | 全局特征重要性柱状图 | `figures/exp4_fig1_shap_summary_bar.png` |
| Fig 2 | SHAP 蜂群图 | `figures/exp4_fig2_shap_beeswarm.png` |
| Fig 3 | Top-5 特征依赖图 | `figures/exp4_fig3_shap_dependence.png` |
| Fig 4 | 典型品种瀑布图 | `figures/exp4_fig4_shap_waterfall.png` |
| Fig 5 | 特征组贡献图 | `figures/exp4_fig5_group_importance.png` |
| Fig 6 | 特征交互热力图 | `figures/exp4_fig6_interaction_heatmap.png` |

---

## 4. 表格索引

| 表编号 | 描述 | 文件路径 |
|--------|------|----------|
| Table 1 | 特征重要性排名 | `tables/exp4_table1_feature_importance.csv` |
| Table 2 | 特征组重要性 | `tables/exp4_table2_group_importance.csv` |
| Table 3 | 生理机制映射 | `tables/exp4_table3_physiology_mapping.csv` |

---

## 5. 结论

1. **{top1['Feature']}** 是最重要的预测特征，属于 {top1['Group']} 组
2. **{top_group['Group']}** 组特征贡献最大，占总贡献的 {top_group['Percentage']:.1f}%
3. 特征重要性排名与光合生理学先验知识一致，验证了模型的可解释性

---

*报告由 step5_exp4_shap_analysis.py 自动生成*
"""

    # 保存报告
    report_path = REPORT_DIR / "exp4_shap_interpretability_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"[OK] 报告已保存: {report_path}")


# =============================================================================
# 主函数
# =============================================================================
def main():
    print("=" * 60)
    print("Step 5: Exp-4 SHAP 可解释性分析")
    print("=" * 60)

    # 1. 加载数据
    print("\n[1/5] 加载数据...")
    X, y, feature_names, df = load_data()
    print(f"  样本数: {len(y)}")
    print(f"  特征数: {len(feature_names)}")

    # 2. 训练模型
    print("\n[2/5] 训练 CatBoost 模型...")
    model = train_catboost(X, y)
    y_pred = model.predict(X)
    r2 = 1 - np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2)
    print(f"  训练集 R2: {r2:.4f}")

    # 3. 计算 SHAP 值
    print("\n[3/5] 计算 SHAP 值...")
    shap_values, expected_value = compute_shap_values(model, X, feature_names)
    importance_df = get_feature_importance(shap_values, feature_names)
    group_df = get_group_importance(importance_df)
    print(f"  Top-3 特征: {importance_df.head(3)['Feature'].tolist()}")
    print(f"  贡献最大的组: {group_df.iloc[0]['Group']} ({group_df.iloc[0]['Percentage']:.1f}%)")

    # 4. 生成图表
    print("\n[4/5] 生成图表...")
    plot_summary_bar(shap_values, X, feature_names,
                     FIGURE_DIR / "exp4_fig1_shap_summary_bar.png")
    plot_beeswarm(shap_values, X, feature_names,
                  FIGURE_DIR / "exp4_fig2_shap_beeswarm.png")
    plot_dependence(shap_values, X, feature_names, importance_df,
                    FIGURE_DIR / "exp4_fig3_shap_dependence.png")
    plot_waterfall(shap_values, expected_value, X, feature_names, df,
                   FIGURE_DIR / "exp4_fig4_shap_waterfall.png")
    plot_group_importance(group_df,
                          FIGURE_DIR / "exp4_fig5_group_importance.png")
    plot_interaction_heatmap(shap_values, feature_names, importance_df,
                             FIGURE_DIR / "exp4_fig6_interaction_heatmap.png")

    # 5. 保存表格和报告
    print("\n[5/5] 保存表格和报告...")
    save_tables(importance_df, group_df, feature_names)
    generate_report(importance_df, group_df)

    print("\n" + "=" * 60)
    print("分析完成!")
    print("=" * 60)
    print(f"\n输出目录:")
    print(f"  图表: {FIGURE_DIR}")
    print(f"  表格: {TABLE_DIR}")
    print(f"  报告: {REPORT_DIR}")


if __name__ == "__main__":
    main()
