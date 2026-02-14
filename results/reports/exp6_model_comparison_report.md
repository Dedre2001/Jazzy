# Exp-6: 模型对比实验完整报告

**报告生成时间:** 2026-02-03 10:48:54
**随机种子:** 42
**交叉验证:** 5折 GroupKFold (按品种分组)
**特征集:** FS4 (三源融合, 40个特征)
**品种数:** 13
**样本数:** 117

---

## 摘要

本实验对比了6个模型在水稻抗旱性预测任务上的表现。结果显示：

- **最优模型:** TabPFN (Spearman ρ = 1.000)
- **最差模型:** SVR (Spearman ρ = 0.940)
- **性能提升:** 6.0%

### 主要发现

1. **TabPFN** 在排序任务上表现最优，Spearman ρ = 1.000
2. **Top-3命中率:** 100%，育种初筛可靠
3. **品种排名一致性:** Kendall τ = 1.000
4. **统计检验:** Friedman p = 0.0001

---

## 1. 实验方法

### 1.1 模型配置

| 层级 | 模型 | 类型 | 关键参数 |
|------|------|------|----------|
| Layer1 | PLSR | 经典农学 | n_components=5 |
| Layer1 | SVR | 经典农学 | kernel=rbf, C=1.0 |
| Layer2 | Ridge | 传统ML | alpha=1.0 |
| Layer2 | RF | 传统ML | n_estimators=300, max_depth=5 |
| Layer2 | CatBoost | 传统ML | iterations=500, lr=0.05 |
| Layer3 | TabPFN | 基础模型 | n_estimators=256, 无需调参 |

### 1.2 评价指标体系

| 类别 | 指标 | 说明 |
|------|------|------|
| 回归 | R², RMSE, MAE | 拟合精度 |
| 排序 | Spearman ρ, Kendall τ | 排序相关性 |
| 配对 | Pairwise Accuracy | 两两比较准确率 |
| Top-K | Hit@3, Hit@5, Jaccard@K | 顶部命中率 |
| 分类 | 3-class Accuracy, Kappa | 耐旱等级分类 |
| 统计 | Friedman + Nemenyi | 多模型显著性检验 |

---

## 2. 实验结果

### 2.1 综合性能对比

| 模型 | 层级 | R² | RMSE | Spearman | Pairwise | Hit@3 | Kappa |
|------|------|:---:|:---:|:---:|:---:|:---:|:---:|
| PLSR | Layer1 | 0.936 | 0.033 | 0.978 | 0.949 | 1.00 | 0.884 |
| SVR | Layer1 | 0.421 | 0.100 | 0.940 | 0.910 | 1.00 | 0.000 |
| Ridge | Layer2 | 0.902 | 0.041 | 0.940 | 0.910 | 0.67 | 0.770 |
| RF | Layer2 | 0.851 | 0.051 | 0.967 | 0.949 | 1.00 | 0.884 |
| CatBoost | Layer2 | 0.867 | 0.048 | 0.978 | 0.962 | 1.00 | 0.884 |
| **TabPFN** | Layer3 | 0.943 | 0.031 | 1.000 | 1.000 | 1.00 | 0.884 |


### 2.2 排序指标详情

| 模型 | Kendall τ | Jaccard@3 | Jaccard@5 | NDCG@5 | Mean Rank Error |
|------|:---:|:---:|:---:|:---:|:---:|
| PLSR | 0.897 | 1.000 | 1.000 | 0.998 | 0.62 |
| SVR | 0.821 | 1.000 | 0.667 | 0.968 | 0.92 |
| Ridge | 0.821 | 0.500 | 1.000 | 0.982 | 1.08 |
| RF | 0.897 | 1.000 | 1.000 | 1.000 | 0.62 |
| CatBoost | 0.923 | 1.000 | 1.000 | 1.000 | 0.46 |
| TabPFN | 1.000 | 1.000 | 1.000 | 1.000 | 0.00 |


### 2.3 模型排名汇总

| Model    |   R2_Rank |   Spearman_Rank |   Pairwise_Acc_Rank |   Kappa_Rank |   NDCG@5_Rank |   Avg_Rank |
|:---------|----------:|----------------:|--------------------:|-------------:|--------------:|-----------:|
| TabPFN   |         1 |               1 |                   1 |            2 |             2 |        1.4 |
| CatBoost |         4 |               2 |                   2 |            2 |             2 |        2.4 |
| PLSR     |         2 |               2 |                   3 |            2 |             4 |        2.6 |
| RF       |         5 |               4 |                   3 |            2 |             2 |        3.2 |
| Ridge    |         3 |               5 |                   5 |            5 |             5 |        4.6 |
| SVR      |         6 |               5 |                   5 |            6 |             6 |        5.6 |

---

## 3. 品种排名详细分析

### 3.1 品种排名对比表

| Variety | Category | True Rank | PLSR | SVR | Ridge | RF | CatBoost | TabPFN |
|---------|----------|:---------:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1252 | Tolerant | 1 | 1 | 2 | 1 | 1 | 1 | 1 |
| 1257 | Tolerant | 2 | 3 | 3 | 3 | 2 | 2 | 2 |
| 1099 | Tolerant | 3 | 2 | 1 | 4 | 3 | 3 | 3 |
| 1228 | Intermediate | 4 | 5 | 4 | 5 | 4 | 4 | 4 |
| 1214 | Intermediate | 5 | 4 | 8 | 2 | 5 | 5 | 5 |
| 1274 | Intermediate | 6 | 7 | 6 | 7 | 7 | 7 | 6 |
| 1210 | Intermediate | 7 | 6 | 5 | 6 | 6 | 6 | 7 |
| 73 | Intermediate | 8 | 8 | 7 | 8 | 8 | 8 | 8 |
| 12 | Sensitive | 9 | 9 | 9 | 10 | 10 | 10 | 9 |
| 1219 | Sensitive | 10 | 11 | 10 | 11 | 12 | 11 | 10 |
| 1110 | Sensitive | 11 | 10 | 11 | 9 | 9 | 9 | 11 |
| 1218 | Sensitive | 12 | 12 | 13 | 13 | 11 | 12 | 12 |
| 1235 | Sensitive | 13 | 13 | 12 | 12 | 13 | 13 | 13 |


### 3.2 排名偏差分析


**最难预测的品种 (平均排名偏差最大):**

| Variety | True Rank | 平均偏差 | 最大偏差 |
|---------|:---------:|:--------:|:--------:|
| 1110 | 11 | 1.17 | 2 |
| 1214 | 5 | 1.17 | 3 |
| 1210 | 7 | 1.00 | 2 |


**最容易预测的品种 (平均排名偏差最小):**

| Variety | True Rank | 平均偏差 | 最大偏差 |
|---------|:---------:|:--------:|:--------:|
| 1235 | 13 | 0.33 | 1 |
| 73 | 8 | 0.17 | 1 |
| 1252 | 1 | 0.17 | 1 |


### 3.3 关键发现

- **最容易预测的品种:** 各模型排名一致的品种
- **最难预测的品种:** 排名偏差最大的品种
- **边界品种:** 处于分类阈值附近的品种

---

## 4. 统计检验结果

### 4.1 Friedman检验

- **检验统计量:** χ² = 26.93
- **p值:** 0.000059
- **样本量:** n = 13 个品种, k = 6 个模型

### 4.2 Kendall's W 协和系数 (效应量)

- **W值:** 0.414
- **效应强度:** moderate (中等)
- **解释:** W 表示各品种对模型排名的一致程度 (0=完全不一致, 1=完全一致)

### 4.3 平均排名 (含95% Bootstrap置信区间)

| 模型 | 平均排名 | 95% CI |
|------|:--------:|:------:|
| TabPFN | 2.00 | [1.46, 2.54] |
| PLSR | 2.69 | [1.92, 3.54] |
| Ridge | 3.08 | [2.31, 3.92] |
| RF | 3.77 | [2.92, 4.46] |
| CatBoost | 4.00 | [3.54, 4.54] |
| SVR | 5.46 | [4.85, 5.92] |


### 4.4 Nemenyi临界距离

- **CD值:** 2.09
- **含义:** 平均排名差距 < CD 的模型间无显著差异

### 4.4 结果解释

⚠️ **小样本警告:** 注意: 样本量较小 (n=13)，p值可能不稳定。建议参考效应量 Kendall's W = 0.414 (moderate (中等)) 进行解释。

**综合解读:**
- Friedman检验 p = 0.0001，拒绝零假设（各模型表现相同）
- Kendall's W = 0.414，效应量为 **moderate (中等)**
- 由于样本量较小 (n=13)，建议以效应量 W 作为主要参考指标
- W 值表示各品种对模型排名的一致性程度，W 越高表示模型间差异越稳定

---

## 5. 图表索引

| 图编号 | 描述 | 文件路径 |
|--------|------|----------|
| Fig 1 | 模型性能对比 (多指标) | `figures/exp6_fig1_model_comparison.png` |
| Fig 2 | 多指标雷达图 | `figures/exp6_fig2_radar.png` |
| Fig 3 | 预测值 vs 真实值散点图 | `figures/exp6_fig3_pred_vs_true.png` |
| Fig 4 | 品种排名热力图 | `figures/exp6_fig4_variety_ranking_heatmap.png` |
| Fig 5 | 排名偏差条形图 | `figures/exp6_fig5_rank_deviation.png` |
| Fig 6 | Critical Difference图 | `figures/exp6_fig6_critical_difference.png` |
| Fig 7 | 指标排名矩阵 | `figures/exp6_fig7_metric_ranking_matrix.png` |
| Fig 8 | 残差分布箱线图 | `figures/exp6_fig8_residual_boxplot.png` |

---

## 6. 表格索引

| 表编号 | 描述 | 文件路径 |
|--------|------|----------|
| Table 1 | 综合性能指标 | `tables/exp6_table1_main_results.csv` |
| Table 2 | 模型排名汇总 | `tables/exp6_table2_rankings.csv` |
| Table 3 | 品种排名对比 | `tables/exp6_table3_variety_rankings.csv` |
| Table 4 | 分类指标 | `tables/exp6_table4_classification.csv` |
| Table 5 | 统计检验结果 | `tables/exp6_table5_statistical_tests.csv` |
| Table 6 | 原始预测结果 | `tables/exp6_table6_raw_predictions.csv` |

---

## 7. 结论

1. **TabPFN** 在水稻抗旱性预测任务中综合表现最优
2. 三源融合特征集(FS4)为所有模型提供了有效的预测基础
3. Hit@3 = 100%，满足育种初筛的实用需求
4. 品种排名分析揭示了各模型的预测特点和潜在改进方向

---

*报告由 step4_aggregate_results.py 自动生成*
*数据来源: data/processed/features_40.csv*
