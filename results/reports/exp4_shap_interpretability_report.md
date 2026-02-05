# Exp-4: SHAP 可解释性分析报告

**报告生成时间:** 2026-02-03 17:42:58
**分析模型:** CatBoost (TreeExplainer)
**样本数:** 117
**特征数:** 37

---

## 1. 全局特征重要性

### 1.1 Top-10 最重要特征

| 排名 | 特征 | 特征组 | SHAP重要性 | 生理含义 |
|:----:|:-----|:------:|:----------:|:---------|
| 1 | OJIP_Vi | OJIP | 0.0454 | I相相对荧光，PQ池氧化还原状态 |
| 2 | OJIP_FvFm | OJIP | 0.0253 | PSII最大光化学效率，直接反映PSII损伤程度 |
| 3 | BF(F440) | Static | 0.0163 | 蓝色荧光，与酚类化合物和细胞壁相关 |
| 4 | SR_F440_F520 | Static_Ratio | 0.0132 | 蓝/绿荧光比，多酚类物质积累 |
| 5 | OJIP_ETo_RC | OJIP | 0.0122 | 每RC电子传递，电子传递链活性 |
| 6 | SR_F440_F690 | Static_Ratio | 0.0058 | 蓝/红荧光比，胁迫敏感指标 |
| 7 | VI_MTCI | VI | 0.0046 | 叶绿素红边指数，叶绿素含量敏感指标 |
| 8 | VI_SIPI | VI | 0.0038 | 结构不敏感色素指数，类胡萝卜素/叶绿素比 |
| 9 | FrF(f740) | Static | 0.0037 | 远红荧光，叶绿素荧光次峰 |
| 10 | R780 | Multi | 0.0030 | 近红外平台起始 |


### 1.2 最重要特征解读

**OJIP_Vi** 是对抗旱性预测贡献最大的特征（SHAP重要性 = 0.0454）。
该特征属于 **OJIP** 组，其生理含义为：I相相对荧光，PQ池氧化还原状态

---

## 2. 特征组贡献分析

### 2.1 各特征组贡献占比

| 特征组 | 总贡献 | 占比 |
|:------:|:------:|:----:|
| OJIP | 0.0901 | 51.3% |
| Static_Ratio | 0.0264 | 15.0% |
| Static | 0.0247 | 14.1% |
| VI | 0.0194 | 11.1% |
| Multi | 0.0149 | 8.5% |


### 2.2 特征组解读

**OJIP** 组贡献最大（51.3%），表明该类特征在抗旱性预测中起主导作用。

OJIP 参数直接反映光合电子传递链的功能状态，是评价植物抗旱性的核心生理指标。
干旱胁迫会导致 PSII 反应中心失活、电子传递效率下降，这些变化被 OJIP 荧光动力学敏感捕获。


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
