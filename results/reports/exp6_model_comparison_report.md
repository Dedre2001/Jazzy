# Exp-6: 模型对比实验报告

**报告生成时间:** 2026-01-28 09:30:54
**随机种子:** 42
**交叉验证:** 5折 KFold
**特征集:** FS4 (三源融合, 40个特征)

---

## 摘要

本实验对比了6个模型在水稻抗旱性预测任务上的表现。结果显示，**CatBoost** 取得了最佳性能（Spearman rho = 0.989），较表现最差的 SVR（rho = 0.896）提升了 9.3%。

### 主要发现

1. **最优模型:** CatBoost，Spearman rho = 0.989
2. **Top-3命中率:** 1.00
3. **配对排序准确率:** 0.974

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
| Layer3 | TabPFN | 新型NN | n_estimators=32, 无需调参 |

### 1.2 验证策略

```
验证方式: 5折交叉验证 (样本层随机划分)
聚合规则: 样本预测 -> 品种层均值
评估层级: 品种层 (n=13)
```

---

## 2. 实验结果

### 2.1 性能对比

| 模型 | 层级 | R2 | RMSE | Spearman rho | 配对准确率 | Hit@3 | Hit@5 |
|------|------|-----|------|--------------|-----------|-------|-------|
| PLSR | Layer1 | 0.857 | 0.050 | 0.923 | 0.885 | 0.67 | 1.00 |
| SVR | Layer1 | 0.512 | 0.092 | 0.896 | 0.885 | 0.67 | 0.80 |
| Ridge | Layer2 | 0.878 | 0.046 | 0.934 | 0.923 | 1.00 | 1.00 |
| RF | Layer2 | 0.726 | 0.069 | 0.973 | 0.949 | 1.00 | 0.80 |
| **CatBoost** | Layer2 | 0.809 | 0.057 | 0.989 | 0.974 | 1.00 | 1.00 |

### 2.2 层级对比分析

| 层级 | 描述 | 模型 | 平均Spearman rho |
|------|------|------|------------------|
| Layer1 | 经典农学模型 | PLSR, SVR | 0.909 |
| Layer2 | 传统机器学习 | Ridge, RF, CatBoost | 0.965 |
| Layer3 | 新型神经网络 | TabPFN | 0.000 |

---

## 3. 讨论

### 3.1 模型排名

| Model    |   R2_Rank |   Spearman_Rank |   Pairwise_Acc_Rank |   Hit@3_Rank |   Avg_Rank |
|:---------|----------:|----------------:|--------------------:|-------------:|-----------:|
| CatBoost |         3 |               1 |                   1 |            2 |       1.75 |
| Ridge    |         1 |               3 |                   3 |            2 |       2.25 |
| RF       |         4 |               2 |                   2 |            2 |       2.5  |
| PLSR     |         2 |               4 |                   4 |            4 |       3.5  |
| SVR      |         5 |               5 |                   4 |            4 |       4.5  |

### 3.2 结论

1. **CatBoost** 在水稻抗旱性预测任务中表现最优
2. 三源融合特征集(FS4)为所有模型提供了有效的预测基础
3. Hit@3达到100%，支持育种初筛应用

---

## 4. 输出文件（可追溯性）

### 4.1 数据表格

| 编号 | 描述 | 文件路径 |
|------|------|----------|
| 表1 | 模型性能对比主表 | `F:/all_exp/results/tables/exp6_table1_main_results.csv` |
| 表2 | 各指标模型排名 | `F:/all_exp/results/tables/exp6_table2_rankings.csv` |

### 4.2 图表

| 编号 | 描述 | 文件路径 |
|------|------|----------|
| 图1 | 模型性能对比 | `F:/all_exp/results/figures/exp6_fig1_model_comparison.png` |
| 图2 | 模型层级性能对比 | `F:/all_exp/results/figures/exp6_fig2_layer_comparison.png` |
| 图3 | Top-3模型雷达图 | `F:/all_exp/results/figures/exp6_fig3_radar.png` |

---

*报告由 step4_aggregate_results.py 自动生成*
*数据来源: F:/all_exp/data/processed/features_40.csv*
