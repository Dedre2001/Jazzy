# 多源光谱技术的水稻种质资源抗旱性快速评价方法研究

## 实验设计方案 v6.0（融合版）

**英文题目**：*Rapid Evaluation of Drought Tolerance in Rice Germplasm Using Multi-Source Spectral Technologies and Interpretable Machine Learning*

**版本**: v6.0（融合版）
**更新日期**: 2026-01-25
**融合依据**: experimental_design.md v5.2 + phase6_multisource_fusion_with_ojip_experiment_design.md v1.0

---

## 核心主张（一句话故事）

> 本论文面向育种**既定品种集合内**的高通量筛选需求：在不强调跨品种泛化的前提下，基于样本层 KFold 的 out-of-fold（OOF）预测并聚合到品种层，系统评估多光谱反射（Multi）、静态荧光（Static）与 OJIP 的**单源/双源/三源**组合对抗旱性评分 `D_conv`（及三等级 `Category`）的拟合与排序能力，给出三源融合带来的性能增益与可解释性证据。

---

## 四大创新点

| 创新点 | 内容 | 论文支撑 |
|-------|------|---------|
| **创新点1** | **三源光谱融合的增益评估**：在统一的样本层 KFold OOF 设定下，对比单源/双源/三源并报告增益 | Exp-1, Exp-2 |
| **创新点2** | **与育种分级真值对齐的评估口径**：三等级分级使用 Phase 5 的固定 `Category` 真值标签（非动态分位数切分） | Phase 5, Exp-1 |
| **创新点3** | 双维度抗旱评分体系（D_stress + D_recovery） | Phase 5 |
| **创新点4** | 在严谨评价框架内对比模型（含 TabPFN-2.5）并探索低成本替代 OJIP 的可行性 | Exp-6, Exp-5 |

> **补充实验**：Exp-S1 基于 Conformal Prediction 的不确定性量化（探索性分析，详见附录）

---

## 1. 研究主张与核心科学问题

### 1.1 研究背景

气候变化导致的干旱胁迫已成为全球粮食安全的主要威胁。水稻作为世界第二大粮食作物，其抗旱性评价对于育种选择和栽培管理至关重要。

**传统抗旱评价方法的局限性**：
- **破坏性取样**：生物量、根系测定需要采收植株
- **单一维度**：仅关注胁迫响应，忽略恢复能力
- **费时费力**：OJIP测量需要专用PAM荧光仪，操作复杂
- **主观性强**：依赖形态学评分，可重复性差

**光谱遥感技术的优势**：
- **无损检测**：实时监测植物生理状态
- **高通量**：快速扫描大量样本
- **多维信息**：同时获取结构、色素、荧光信息

### 1.2 核心主张

我们主张：可以构建一个用于育种初筛场景的抗旱性快速评价方法。

核心思路：

1. 用传统生理/形态指标（非光谱）构建品种级抗旱性真值：连续评分 `D_conv` 与三等级标签（抗旱/中间/敏感）。
2. 用可解释机器学习模型，基于多源光谱特征预测 `D_conv`（并可进一步给出分级），证明：
   - 系统比较单源（Multi-only / Static-only / OJIP-only）、双源融合（Multi+Static）、三源融合（Multi+Static+OJIP）
   - 在统一的样本层 KFold OOF 设定下，报告三源融合相对双源/单源的增益，并用品种层排序指标与分级指标共同佐证
3. 用 SHAP/特征重要性给出解释证据，并与已知光谱-生理机制相一致。
4. 给出低成本替代路径：用 Multi+Static 预测关键 OJIP 参数，并评估对最终抗旱评价性能的影响（"OJIP昂贵→光谱替代"论证）。

### 1.3 具体研究问题（逻辑递进）

**第一层：基础验证问题**
1. 三种光谱源（多光谱反射、荧光反射、OJIP）在干旱胁迫下的响应规律是什么？
2. 不同光谱源反映的生理信息有何异同？是否存在互补性？

**第二层：方法构建问题**
3. 如何构建融合多时相胁迫响应和恢复能力的综合抗旱评分指标（D_conv）？
4. 多源光谱融合是否显著优于单源光谱？优势有多大？
5. 哪些光谱特征对抗旱性预测贡献最大？其生理机制是什么？

**第三层：技术突破问题**
6. 能否利用简易多光谱技术预测OJIP参数（Fv/Fm, PIabs等）？预测精度如何？
7. 这种替代方案可节省多少成本和时间？

**第四层：应用推广问题**
8. 如何建立种质资源抗旱性快速分类系统？
9. 该方法能否跨品种应用？泛化能力如何？

---

## 2. 不可违反的科学约束（泄露控制 + 合理性）

> [!CAUTION]
> **以下约束是论文方法论的基石，必须严格执行**

### 2.1 真值独立性

`D_conv` 必须完全由非光谱数据构建（SPAD、株高、叶面积等）。禁止把任何光谱/荧光/OJIP 特征混入真值计算，避免自证预言。

```
❌ 原: D_conv = α·D_stress + β·D_recovery + γ·D_spectral
✅ 新: D_conv = α·D_stress + β·D_recovery
```

### 2.2 按品种分组验证

本论文**不评估跨品种泛化**，因此不采用 LOGO/LOVO/LOOCV。

主要验证采用样本层 KFold，并通过 out-of-fold（OOF）预测聚合到品种层计算指标，用于比较三源数据组合的拟合与排序效果。

### 2.3 CV 折内预处理

标准化/归一化、特征选择、TVAE 等数据增强（如使用）必须在每折训练集内拟合，仅用于该折训练集与测试集变换。

```python
# 正确做法
for train_idx, test_idx in kfold.split(X, y):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_idx])  # 仅在训练集fit
    X_test = scaler.transform(X[test_idx])        # 用训练集参数transform
```

### 2.4 公平融合对照

Multi-only、Static-only、双源融合（Multi+Static）、三源融合（Multi+Static+OJIP）对比时，必须保证测试集样本一致。

- 本项目数据中 OJIP 与 Multi/Static 在样本层一一匹配（13品种×3处理×3重复=117），因此 FS3/FS4 的对比在全样本上进行。
- 若未来扩展数据出现 OJIP 缺失：本论文不覆盖缺失模态情形；需在后续工作中补充共同样本对照与缺失机制分析。

### 2.5 处理效应混淆审计

CK1/D1/RD2 处理差异显著，模型可能学到"处理状态"而非"品种抗性"。

必须将 `Treatment` 作为输入特征（one-hot），并在解释性阶段审计其重要性。

```python
# 特征矩阵: 光谱特征(~160维) + Treatment编码(3维) = ~163维
X = pd.concat([spectral_features, treatment_dummies], axis=1)

# 在SHAP分析中检查Treatment重要性
# - 如果Treatment重要性过高 → 模型依赖处理效应 ⚠️
# - 如果光谱特征重要性高于Treatment → 模型学到品种固有差异 ✅
```

---

## 3. 数据结构与模态定义

### 3.1 基本单位

- **评估对象**：`Variety`（品种）
- **观测单位**：`Sample`（某品种某重复在某处理下的一条观测）

**标识规范**：

```text
sample_id = {Variety}_{ReplicateID}_{Treatment}
group_id  = {Variety}
treatment = {CK1, D1, RD2}
```

### 3.2 三类模态（Modalities）

| 模态 | 来源 | 内容 | 特征数（估算） |
|------|------|------|---------------|
| **Multi** | 多光谱反射数据 | 反射波段、植被指数、组合特征 | ~130个 |
| **Static** | 荧光反射数据 | 静态荧光反射波段与比值 | ~30个 |
| **OJIP** | 快速叶绿素荧光 | JIP-test参数（Fv/Fm、PIabs、ABS/RC等） | ~16个 |

### 3.3 样本规模

- **品种数**：13个
- **处理数**：3个（CK1/D1/RD2）
- **重复数**：3个
- **总样本数**：13 × 3 × 3 = 117

---

## 4. Ground Truth 构建：D_conv

### 4.1 双维度定义

使用仅生理/形态指标在品种尺度构建：

- **胁迫响应指数 D_stress**：基于 D1/CK1 比值构建，量化品种在干旱胁迫下的表现
- **恢复能力指数 D_recovery**：基于 RD2/D1 比值构建，量化品种复水后的恢复能力

**综合评分**：

$$D_{conv} = 0.5 \times D_{stress} + 0.5 \times D_{recovery}$$

### 4.2 计算步骤（双维度方法）

**步骤 A：计算双维度LC比值（品种级别）**

对每个品种 $v$，每个生理指标 $j$，分别计算胁迫响应和恢复能力比值：

```
胁迫响应比值：LC_stress_{v,j} = mean(X_{v,D1,j}) / mean(X_{v,CK1,j})
恢复能力比值：LC_recovery_{v,j} = mean(X_{v,RD2,j}) / mean(X_{v,D1,j})
```

结果：13品种 × 5指标 × 2维度 = 130 个LC比值

**步骤 B：分别对双维度LC比值进行PCA分析**

对胁迫响应和恢复能力两个维度的LC比值矩阵分别进行PCA分析：

1. Z-score标准化
2. PCA提取主成分得分
3. 隶属度函数：$U(X_p) = \frac{X_p - X_{min}}{X_{max} - X_{min}}$
4. 加权求和得到 D_stress / D_recovery

**步骤 C：计算综合D_conv**

$$D_{conv} = 0.5 \times D_{stress} + 0.5 \times D_{recovery}$$

**权重依据**：敏感性分析结果显示，权重在 0.45–0.55 区间内品种分类完全一致（category_agreement = 1.0）。

### 4.3 LOVO 内部计算（防止泄露）

在每个 LOVO 折中：

1. 仅用训练品种计算任何基准（如 CK 均值）与标准化参数
2. 得到训练品种 `D_conv`
3. 用同样的规则对测试品种计算/变换 `D_conv`

目的：保证测试品种不影响真值构建中的任何"拟合型步骤"。

### 4.4 层次聚类分组

基于D_conv进行层次聚类，划分品种抗旱等级：
- **聚类方法**：欧氏距离 + Ward法
- **分组数**：3组（Tolerant/Intermediate/Sensitive）
- **输出**：品种分组标签 + 树状图

### 4.5 敏感性分析结果

已完成敏感性分析（2026-01-20）：
- 权重在 0.45–0.55 区间内，品种分类完全一致
- 当权重偏离 > 0.20（如 0.70/0.30）时，分类开始明显变化
- 结论：等权重方案在合理区间内稳健

**分析文件**：
- `data/03_ground_truth/dconv_sensitivity_summary.csv`
- `src/phase5_dconv_sensitivity_analysis.py`

---

## 5. 预测任务定义

### 5.1 任务A（主任务）：预测抗旱性连续评分 D_conv

模型在样本层预测 `D_conv`，汇总到品种层：

```text
pred_variety = mean(pred_samples_of_variety)
```

最终以品种层结果作为"抗旱评价"的核心输出与评估对象。

**本论文的验证口径（不强调泛化）**：
- 目标是对这一批既定品种进行多源数据拟合/排序（within-known-varieties）。
- 样本层训练采用常规 KFold（样本随机划分）得到 OOF 预测，再聚合到品种层计算指标。
- **禁止**使用 LOGO/LOVO/LOOCV 等“留一品种”评估来主张跨品种泛化能力。

### 5.2 任务B（辅任务）：三等级分级

本论文的三等级分级以 **Rank 分档（3/5/5）** 作为操作性定义（screening tiers）：

- 真值 tier：由 `ground_truth_variety.csv` 的 `Rank` 直接切分（Top-3 / Middle-5 / Bottom-5）
- 预测 tier：由模型输出的 `pred_D_conv_variety` 排名切分为相同的 3/5/5

> 注：该分级定义用于"本批次 13 个品种的筛选分档"，不用于主张跨品种泛化或绝对生理阈值。

> **已确认**：本研究使用 Rank 分档（Top-3/Middle-5/Bottom-5）作为三等级分级的操作性定义。Section 4.4 的层次聚类仅用于可视化展示，不作为分级真值。

评估同时报告：
- 样本层分级准确（仅作补充）
- 品种层分级准确（主报告：先聚合再判类）

### 5.3 品种层汇总规则

```python
# 回归任务
variety_pred = df.groupby('Variety')['pred_D_conv'].mean()
variety_true = df.groupby('Variety')['true_D_conv'].mean()

# 分类任务（投票法）
variety_pred_class = df.groupby('Variety')['pred_class'].apply(lambda x: x.mode()[0])
```

---

## 6. 特征集合与消融设计

所有特征集合都包含 `Treatment` one-hot（用于控制与审计处理效应）。

| 代号 | 组成 | 用途 | 特征数（估算） |
|------|------|------|---------------|
| **FS1** | Multi_* + Treatment | 单源对照 | ~133 |
| **FS2** | Static_* + Treatment | 单源对照 | ~33 |
| **FS3** | Multi_* + Static_* + Treatment | 双源融合（消融基线） | ~163 |
| **FS4** | Multi_* + Static_* + OJIP_* + Treatment | 三源融合（完整方案） | ~179 |

### 6.1 FS1：Multi-only

- 多光谱反射波段（R460, R520, R580, R660, R710, R730, R760, R850, R900等）
- 植被指数（NDVI, EVI, NDRE, SIPI等）
- 组合特征（ND_Ri_Rj, SR_Ri_Rj）
- Treatment one-hot

### 6.2 FS2：Static-only

- 荧光波段反射率（F440, F520, F690, F740等）
- 荧光比值指数（F690/F740, F_SIF等）
- Treatment one-hot

### 6.3 FS3：双源融合（消融基线）

- FS1 + FS2（去重）
- 用于验证 OJIP 的边际贡献

### 6.4 FS4：三源融合（完整方案）

- FS3 + OJIP参数
- OJIP参数包括：Fv/Fm, Fv/F0, PIabs, PIcs, ABS/RC, TR0/RC, ET0/RC, DI0/RC, φP0, φE0, φD0, ψE0等

---

## 6.5 特征工程（Feature Engineering）

> 目标：严格区分“原始提取（Extract）”与“特征工程（FE）”，避免在提取阶段做任何不可追溯的删列/降维。

### 6.5.1 Extract（全量提取，可追溯）

- **原则**：从 Pocket PEA 导出表（Format:3a）中提取所有可用数值列，不做白名单筛选。
- **占位列处理**：导出表头中 `*` 标记的空列为兼容性占位列，可在提取阶段直接排除，但必须在“列名映射表”中记录其 drop reason。
- **输出文件**（示例路径）：
  - `data/02_features/ojip_features_full_raw.csv`：全量提取（含全部可用数值列）的样本表
  - `data/02_features/ojip_features_full.csv`：与光谱样本 `Sample_ID` 对齐后的子集（仅限制行，不筛列）
  - `results/tables/phase1_ojip_full_extract/ojip_features_full_column_mapping.csv`：原始列名 → 标准化列名映射（含 drop reason）

> 重要约束（与本文口径一致）：允许对三源数据进行**不新增字段**的预处理/校准/优化，以便公平比较三源融合效果；但必须避免引入极端值，并保持处理流程可复现、可追溯。

### 6.5.2 FE（折内选择/降维，防泄露）

- **原则**：任何“特征选择/相关性过滤/PCA/PLSR成分数选择/SHAP筛特征/TVAE”等，都必须在每个 KFold 折的训练集内完成（fit only on train），并仅将变换应用到该折的测试集。
- **OJIP 两种可复现 FE 路径**（二选一或并行报告）：
  1. **JIP-test 核心子集**：在 `ojip_features_full.csv` 中选择文献与机理明确的核心变量（例如 `OJIP_FvFm`, `OJIP_PIabs`, `OJIP_ABS_RC`, `OJIP_TRo_RC`, `OJIP_ETo_RC`, `OJIP_DIo_RC`, `OJIP_Vj`, `OJIP_Vi` 等），作为 FS4 的 OJIP 组件。
  2. **全量 OJIP + 折内降维**：在 `ojip_features_full.csv` 的全量列基础上，折内进行降维（例如 PCA 保留 90–95% 方差，或基于训练折的相关性阈值做过滤）。
- **报告要求**：必须明确写出 FE 的具体规则、超参、以及折内拟合流程；禁止在全数据上先筛特征再做 KFold。

### 6.5.3 特征工程最终配置（已确认）

**特征结构（40个）**：

| 模块 | 特征数 | 说明 |
|------|--------|------|
| Multi原始波段 | 11 | R460-R900 |
| 植被指数 | 8 | NDVI, NDRE, EVI, SIPI, PRI, MTCI, GNDVI, NDWI |
| Static原始波段 | 4 | F440, F520, F690, F740 |
| Static比值 | 6 | F690/F740, F440/F690, F440/F520, F520/F690, F440/F740, F520/F740 |
| OJIP参数 | 8 | FvFm, PIabs, ABS_RC, TRo_RC, ETo_RC, DIo_RC, Vi, Vj |
| Treatment one-hot | 3 | CK1, D1, RD2 |
| **总计** | **40** | |

**植被指数公式**：

| 指数 | 公式 | 文献 |
|------|------|------|
| NDVI | (R850-R660)/(R850+R660) | Rouse et al. (1973) |
| NDRE | (R850-R710)/(R850+R710) | Gitelson & Merzlyak (1994) |
| EVI | 2.5*(R850-R660)/(R850+6*R660-7.5*R460+1) | Huete et al. (2002) |
| SIPI | (R850-R460)/(R850+R660) | Peñuelas et al. (1995) |
| PRI | (R520-R580)/(R520+R580) | Gamon et al. (1992) |
| MTCI | (R760-R710)/(R710-R660) | Dash & Curran (2004) |
| GNDVI | (R850-R520)/(R850+R520) | Gitelson et al. (1996) |
| NDWI | (R850-R900)/(R850+R900) | Gao (1996) |

**OJIP参数筛选依据**：
- 删除冗余参数：phiPo（与FvFm高度相关）、Fm/Fo/Fv（原始荧光值）
- 保留核心参数：文献JIP-test核心子集（Strasser 2004）
- 预处理：ABS_RC, DIo_RC 使用 log(1+x) 变换

## 7. 模型选择（三层递进对比框架）

本研究采用**三层递进对比框架**，从经典农学方法到新型神经网络，系统验证TabPFN在小样本农学场景的优势。

### 7.1 模型分层设计

| 层级 | 模型类别 | 具体模型 | 定位 | 文献支撑 |
|-----|---------|---------|------|---------|
| **Layer 1** | 经典农学模型 | PLSR, SVR | 农学领域金标准基线 | 广泛用于光谱-表型建模 |
| **Layer 2** | 传统机器学习 | Ridge, RF, CatBoost | 当前主流方法 | 近年农学ML研究常用 |
| **Layer 3** | 新型神经网络 | **TabPFN** | **核心创新** | Hollmann et al. 2023 ICLR |

### 7.2 Layer 1：经典农学模型（基线）

#### 7.2.1 PLSR（偏最小二乘回归）

**农学地位**：光谱-表型建模的"金标准"，广泛用于近红外光谱分析。

**优势**：
- 天然处理高维共线性特征
- 同时对X和Y进行降维
- 农学领域接受度高

**配置**：
```python
from sklearn.cross_decomposition import PLSRegression

plsr_params = {
    'n_components': 10,  # 通过CV选择最优成分数
    'scale': True,
    'max_iter': 500
}
```

**关键文献**：
- Wold, S. et al. (2001) Chemometrics and Intelligent Laboratory Systems. DOI: 10.1016/S0169-7439(01)00155-1

#### 7.2.2 SVR（支持向量回归）

**农学地位**：经典核方法，擅长处理非线性关系。

**优势**：
- 对小样本鲁棒
- 核技巧处理非线性
- 正则化防过拟合

**配置**：
```python
from sklearn.svm import SVR

svr_params = {
    'kernel': 'rbf',
    'C': 1.0,
    'epsilon': 0.1,
    'gamma': 'scale'
}
```

**关键文献**：
- Mountrakis, G. et al. (2011) ISPRS Journal of Photogrammetry and Remote Sensing. DOI: 10.1016/j.isprsjprs.2010.11.001

### 7.3 Layer 2：传统机器学习

#### 7.3.1 Ridge（岭回归）

**定位**：线性基线，用于sanity check。

```python
from sklearn.linear_model import Ridge

ridge_params = {
    'alpha': 1.0,
    'fit_intercept': True
}
```

#### 7.3.2 RandomForest

**定位**：经典集成方法，稳健性好。

```python
from sklearn.ensemble import RandomForestRegressor

rf_params = {
    'n_estimators': 500,
    'max_depth': 6,
    'min_samples_leaf': 3,
    'random_state': 42
}
```

#### 7.3.3 CatBoost

**定位**：梯度提升代表，SHAP友好。

```python
from catboost import CatBoostRegressor

catboost_params = {
    'iterations': 1000,
    'learning_rate': 0.03,
    'depth': 4,
    'l2_leaf_reg': 5,
    'min_data_in_leaf': 3,
    'loss_function': 'RMSE',
    'random_seed': 42,
    'early_stopping_rounds': 100
}
```

### 7.4 Layer 3：新型神经网络（核心创新）

#### 7.4.1 TabPFN-2.5（Tabular Prior-Data Fitted Network v2.5）

**核心创新点**：首次将TabPFN-2.5引入农学表型预测领域。

**TabPFN-2.5 原理**：
- 基于Transformer架构的表格数据基础模型（Tabular Foundation Model）
- 在合成数据上进行大规模预训练，学习"如何做预测"的元知识
- 推理时无需训练，直接对新数据进行预测（In-Context Learning）
- **v2.5更新（2025.11）**：支持50,000样本+2,000特征，性能全面超越XGBoost

**TabPFN-2.5 vs 原版对比**：

| 特性 | TabPFN v1 | TabPFN-2.5 |
|-----|-----------|------------|
| 最大样本数 | 1,000 | 50,000 |
| 最大特征数 | 100 | 2,000 |
| vs XGBoost胜率 | ~85% | **100%**（小中型数据集） |
| 不确定性估计 | 基础 | **增强** |

**为什么适合本研究**：
| 特性 | TabPFN-2.5优势 | 本研究契合度 |
|-----|-----------|-------------|
| 小样本 | 专为<1000样本设计 | 117样本 ✅ |
| 无需调参 | 预训练模型，零超参调优 | 减少过拟合风险 ✅ |
| 快速推理 | 单次前向传播 | 适合育种快速筛选 ✅ |
| 不确定性估计 | 内置概率预测 | 可提供置信度 ✅ |

**配置**：
```python
from tabpfn import TabPFNRegressor

tabpfn_params = {
    'n_estimators': 32,  # 集成配置数
    'device': 'cpu',  # 或 'cuda'
    'random_state': 42
}

# 使用示例
model = TabPFNRegressor(**tabpfn_params)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

**关键文献（必须引用）**：
- **Hollmann, N. et al. (2023)** TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second. *ICLR 2023*.
  - URL: [https://arxiv.org/abs/2207.01848](https://arxiv.org/abs/2207.01848)
  - GitHub: [https://github.com/automl/TabPFN](https://github.com/automl/TabPFN)

- **Hollmann, N. et al. (2025)** Accurate Predictions on Small Data with a Tabular Foundation Model. *Nature*.
  - DOI: [10.1038/s41586-024-08328-6](https://doi.org/10.1038/s41586-024-08328-6)

- **TabPFN-2.5 (2025.11)** Next-generation tabular foundation model.
  - GitHub: [https://github.com/automl/TabPFN](https://github.com/automl/TabPFN)

#### 7.4.2 模型调参策略（已确认）

| 模型 | 调参方式 | 备注 |
|------|---------|------|
| PLSR | GridSearchCV | n_components: [2,4,6,8,10] |
| SVR | GridSearchCV | C: [0.1,1,10], gamma: [0.01,0.1,1] |
| Ridge | GridSearchCV | alpha: [0.1,1,10,100] |
| RF | GridSearchCV | n_estimators: [100,300,500], max_depth: [3,5,7] |
| CatBoost | Optuna (50 trials) | 嵌套CV调参 |
| **TabPFN** | **无需调参** | N_ensemble=32 (默认) |

**SHAP分析模型**：TabPFN（主力），CatBoost（备选，使用TreeSHAP）

### 7.5 模型对比实验设计（Exp-6）

```
┌─────────────────────────────────────────────────────────────────┐
│                    模型对比实验框架                              │
├─────────────────────────────────────────────────────────────────┤
│ Layer 1: 经典农学模型                                            │
│   ├── PLSR  ─────────────────┐                                  │
│   └── SVR   ─────────────────┤                                  │
│                              │                                  │
│ Layer 2: 传统机器学习         │    同一特征集 (FS4)              │
│   ├── Ridge ─────────────────┤    同一验证策略 (LOVO)           │
│   ├── RF    ─────────────────┤    同一评价指标                  │
│   └── CatBoost ──────────────┤                                  │
│                              │                                  │
│ Layer 3: 新型神经网络         │                                  │
│   └── TabPFN ────────────────┘                                  │
├─────────────────────────────────────────────────────────────────┤
│ 预期结论: TabPFN > CatBoost > RF > SVR > PLSR > Ridge           │
│ 核心创新: TabPFN在农学小样本场景首次验证，性能显著优于传统方法     │
└─────────────────────────────────────────────────────────────────┘
```

### 7.6 模型选择总结

| 模型 | 类别 | 可解释性 | 小样本适应性 | 论文定位 |
|-----|------|---------|-------------|---------|
| PLSR | 经典农学 | ★★★★★ | ★★★ | 农学基线 |
| SVR | 经典农学 | ★★ | ★★★★ | 农学基线 |
| Ridge | 传统ML | ★★★★★ | ★★ | Sanity check |
| RF | 传统ML | ★★★★ | ★★★ | 集成对照 |
| CatBoost | 传统ML | ★★★★ | ★★★★ | SHAP分析主力 |
| **TabPFN** | 新型NN | ★★★ | ★★★★★ | **核心创新** |

---

## 8. 验证策略与指标

### 8.1 主验证：5折 KFold（样本层随机划分）

> **已确认**：本研究目标是"既定13品种的拟合/排序"，不强调跨品种泛化，因此采用样本层 KFold 而非 LOVO。

```python
from sklearn.model_selection import KFold

kfold = KFold(n_splits=5, shuffle=True, random_state=42)
for train_idx, test_idx in kfold.split(X, y):
    # 5折，样本随机划分
    model.fit(X[train_idx], y[train_idx])
    y_pred = model.predict(X[test_idx])
```

对应育种使用场景：对"既定品种集合"进行高通量拟合与排序筛选。

### 8.2 品种层指标（回归）

| 指标 | 公式/说明 | 含义 |
|------|-----------|------|
| R²_variety | 决定系数 | 预测方差解释率 |
| RMSE_variety | 均方根误差 | 预测精度 |
| Spearman | 秩相关系数 | 排序一致性（筛选场景关键） |
| Pairwise Ranking Acc | 品种两两比较正确率 | 排名可靠性 |

### 8.2.1 Top-K 筛选指标（育种初筛更贴近）

> 说明：Spearman/Pairwise 衡量“整体排序一致性”；育种初筛常见目标是“选出最优的前 K 个品种”，因此需要额外报告 Top-K 类指标。

| 指标 | 定义 | 含义 |
|------|------|------|
| Top-K Hit Rate | 设真值Top-K集合为 $T_K$、预测Top-K集合为 $\hat{T}_K$，则 $\mathrm{Hit}@K = \frac{|T_K \cap \hat{T}_K|}{K}$ | 预测前K中命中真值前K的比例 |
| Jaccard@K | $\mathrm{Jaccard}@K = \frac{|T_K \cap \hat{T}_K|}{|T_K \cup \hat{T}_K|}$ | Top-K集合重叠率（对K更敏感） |

**推荐设置**：K=3 与 K=5（对应 13 品种规模下的“强筛选/中等筛选”）。

### 8.3 分级指标（分类）

| 指标 | 公式/说明 | 含义 |
|------|-----------|------|
| Accuracy_3class (tier) | 基于 Rank 分档（3/5/5）的 tier 真值，在品种层统计正确率 | 与育种“Top-K筛选分档”直接对齐的总体精度 |
| Balanced Accuracy | 各类别召回率均值 | 类别平衡准确率 |
| Macro-F1 | 各类别F1算术平均 | 类别平衡F1 |
| Cohen's Kappa | $\kappa = \frac{p_o - p_e}{1 - p_e}$ | 考虑随机因素的一致性 |

### 8.4 禁止性结论

- SHAP/特征重要性只能用于"特征贡献/强关联"解释，**禁止写成因果**
- 若采用随机 KFold（本论文主用），**禁止声称跨品种泛化能力**；结论限定为“已知品种集合内的拟合/排序”

### 8.5 统计置信与显著性（基于 OOF 预测的配对比较）

> 目的：对比不同特征集/融合策略时，不只报告均值±标准差，还给出“差异是否稳定/是否显著”的证据。

**关键原则**：
1. 以 KFold 的 out-of-fold（OOF）预测为基础进行统计比较，避免训练集泄露。
2. 不把 CV 折当作独立样本（CV 估计量相关，方差估计有偏），统计结论需采用更稳健的重采样/置换框架。

**建议输出**：对任意两种方法 A/B（例如 FS4 vs FS3；FS3 vs FS1/FS2），报告 $\Delta$Metric 的 95% CI 与 p-value。

**(1) Bootstrap CI（以品种为单位重采样）**：
1. 先得到每个品种的聚合 OOF 预测 $\hat{y}_v$ 与真值 $y_v$（按 5.3 聚合规则）。
2. 以“品种”为最小重采样单位，进行 B 次（如 B=2000）有放回抽样，得到重采样集合 $V^*_b$。
3. 对每次抽样计算 $\Delta$Metric$(A,B)$，得到经验分布并取 2.5%/97.5% 分位数作为 CI。

**(2) 配对置换检验（paired permutation test）**：
1. 基于同一组品种 OOF 预测，计算观测到的差异 $\Delta_\mathrm{obs}$（例如 $\Delta$Spearman，$\Delta$RMSE）。
2. 在每次置换中，对每个品种 v 随机交换 A/B 的预测（或等价地给差异乘以随机符号），得到 $\Delta_b$。
3. p-value 取 $\Pr(|\Delta_b| \ge |\Delta_\mathrm{obs}|)$。

**适用指标**：Spearman、RMSE、Pairwise、Hit@K、Jaccard@K 均可在品种层做上述比较。

### 8.6 统计检验配置（已确认）

| 配置项 | 选择 |
|--------|------|
| Bootstrap CI | **不使用**（报告均值±标准差） |
| 显著性阈值 | α=0.05 |
| 多模型/特征集对比 | Friedman + Nemenyi |
| 两两比较 | 配对置换检验 |

**各实验的统计方法**：

| 实验 | 比较内容 | 统计方法 |
|------|---------|---------|
| Exp-1 | FS1 vs FS2 vs FS3 vs FS4 | Friedman + Nemenyi |
| Exp-2 | FS3 vs FS4 | 配对置换检验 |
| Exp-6 | 6个模型对比 | Friedman + Nemenyi |

---

## 9. 核心实验矩阵（六段证据链）

### 证据链总览

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         完美证据链设计（六段递进）                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  证据链0: Ground Truth可信                                                   │
│  └── 传统农学方法（PCA+隶属函数+聚类）→ D_conv + Category                     │
│      文献支撑: Fernandez 1992 STI + 经典抗旱评价方法                          │
│                              ↓                                              │
│  证据链1: 多源融合对比 (Exp-1)                                               │
│  └── FS1 vs FS2 vs FS3 vs FS4：报告三源融合增益与不确定性                   │
│      文献支撑: Zhang et al. 2022 Plant Phenomics                             │
│                              ↓                                              │
│  证据链2: OJIP边际贡献 (Exp-2)                                               │
│  └── FS3 vs FS4 消融对照：量化 OJIP 的边际增益                               │
│      文献支撑: Strasser 2004, Xia 2022                                       │
│                              ↓                                              │
│  证据链3: TabPFN优于传统方法 (Exp-6) ⭐ 核心创新                              │
│  └── PLSR/SVR vs Ridge/RF/CatBoost vs TabPFN                                │
│      预期: TabPFN在KFold OOF评估中优于传统方法                                 │
│      文献支撑: Hollmann 2023 ICLR, Hollmann 2025 Nature                      │
│                              ↓                                              │
│  证据链4: 模型决策可解释 (Exp-4)                                             │
│  └── SHAP Top-15特征 → 符合光谱-生理机制                                     │
│      + Treatment效应审计 (Exp-3)                                            │
│                              ↓                                              │
│  证据链5: 低成本替代可行 (Exp-5)                                             │
│  └── Multi+Static预测OJIP → FS4-hat ≈ FS4                                   │
│      结论: 便携式光谱可替代昂贵PAM荧光仪                                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### Exp-1：融合是否有效？（单源 vs 双源融合 vs 三源融合）

**目的**：证明多源互补信息带来性能增益。

**步骤**：
1. 固定模型（使用CatBoost作为基准模型）
2. 在全样本集合上进行样本层 KFold，得到 OOF 预测
3. 跑 FS1、FS2、FS3、FS4
4. 将 OOF 预测按品种聚合后报告品种层指标（均值±标准差/CI）

**产出**：
- Table 5：特征集合 × 性能指标对比表
- Fig 5a：融合增益图（Spearman / RMSE 的提升幅度）

**结论口径**：
- 报告 FS4 相对 FS3 的增益（含 95% CI / p-value）
- 结论限定为“既定品种集合内的拟合/排序提升”，不主张跨品种泛化

---

### Exp-2：消融对照（双源融合 vs 三源融合）

**目的**：在"融合=三源融合"的定义下，用消融方式验证 OJIP 的必要性与贡献。

**步骤**：
1. 全样本（117）进行对比（OJIP 全覆盖）
2. 同一模型、同一超参预算，对比 FS3 vs FS4

**产出**：
- Table：三源融合 vs 双源融合的消融差异
- Fig：逐品种误差/排名变化（消融前后）

**结论口径**：
- 报告 OJIP 的边际增益（FS4 相对 FS3），并在 Discussion 中解释其生理意义与可解释性证据

---

### Exp-3：Treatment效应审计（Treatment shortcut check）

**目的**：排除模型仅学习 CK/D/RD 的处理状态差异。

**步骤**：
1. 在 FS4 中保留 Treatment one-hot
2. 用 SHAP/重要性比较 Treatment 与光谱/OJIP 特征的贡献大小

**判据**：
- 若 Treatment 长期占据 Top 重要特征 → 需要在 Discussion 中说明"模型可能依赖处理效应"的限制
- 若光谱特征重要性高于 Treatment → 模型学到品种固有差异 ✅

---

### Exp-4：解释性与机制一致性

**目的**：给出"模型为什么能做对"的证据。

**步骤**：
1. 对最佳模型（TabPFN或CatBoost）+ 最佳特征集（FS4）计算 SHAP
2. 输出 Top-15 特征清单，并对每个特征给出机理解释

**特征-机制对应**：

| 特征类型 | 示例特征 | 生理机制 |
|---------|---------|---------|
| 红边指数 | NDRE, MTCI | 红边位置蓝移先于NDVI下降，早期胁迫敏感 |
| 近红外反射 | R760, R850 | 叶片结构、含水量变化 |
| 荧光比值 | F690/F740 | 叶绿素重吸收效应，光合活性指示 |
| OJIP参数 | Fv/Fm, PIabs | PSII初级光化学效率，能量分配 |
| OJIP参数 | DI0/RC | 每反应中心热耗散，胁迫下增加 |

**产出**：
- SHAP beeswarm + bar plot
- Table 6：Top-15 特征 + "生理解释（非因果）"

---

### Exp-5：低成本替代（用 Multi+Static 预测 OJIP，并评估下游效果）

**目的**：支撑论文创新点"用便携式光谱替代昂贵荧光仪的可行性"。

**两阶段设计**：

**阶段1：预测 OJIP 参数**
- 输入：FS3（不含 OJIP）
- 输出：关键 OJIP 参数（Fv/Fm、PIabs、ET0/RC、DI0/RC 等，4-8个）
- 验证：全样本上的 LOVO
- 指标：每个参数的 R²/RMSE

**阶段2：用预测的 OJIP 替代真 OJIP**
- 构建 FS4-hat：Multi + Static + OJIP_hat + Treatment
- 对比（全样本）：FS3、FS4（真 OJIP）、FS4-hat

**结论判据**：
- 若 FS4-hat ≈ FS4 → 可主张"成本显著降低且性能接近"
- 若差距明显 → 可主张"OJIP 包含不可被反射/静态荧光完全替代的独立信息"（同样构成科学结论）

---

### Exp-6：模型对比实验（核心创新）⭐

**目的**：系统验证TabPFN在农学小样本场景的优势，支撑"首次将TabPFN引入农学表型预测"创新点。

**实验设计**：

```
┌─────────────────────────────────────────────────────────────────┐
│                    Exp-6 模型对比实验                            │
├─────────────────────────────────────────────────────────────────┤
│ 固定条件:                                                        │
│   - 特征集: FS4 (Multi+Static+OJIP+Treatment)                   │
│   - 验证策略: 5-Fold KFold（样本层随机划分）                      │
│   - 评价指标: R², RMSE, Spearman, Pairwise Ranking Acc          │
├─────────────────────────────────────────────────────────────────┤
│ 对比模型 (6个):                                                  │
│                                                                  │
│   Layer 1: 经典农学模型                                          │
│     ├── PLSR (偏最小二乘回归)                                    │
│     └── SVR  (支持向量回归)                                      │
│                                                                  │
│   Layer 2: 传统机器学习                                          │
│     ├── Ridge    (岭回归)                                        │
│     ├── RF       (随机森林)                                      │
│     └── CatBoost (梯度提升)                                      │
│                                                                  │
│   Layer 3: 新型神经网络                                          │
│     └── TabPFN   (先验数据拟合网络) ⭐                           │
├─────────────────────────────────────────────────────────────────┤
│ 统计检验:                                                        │
│   - Friedman检验 (整体差异)                                      │
│   - Nemenyi后检验 (两两比较)                                     │
│   - 效应量 (Cohen's d)                                           │
└─────────────────────────────────────────────────────────────────┘
```

**步骤**：

1. **数据准备**：使用FS4特征集（~179维）
2. **模型训练**：6个模型在相同LOVO框架下训练
3. **性能评估**：计算品种层指标（R², RMSE, Spearman, Pairwise Ranking Acc）
4. **统计检验**：Friedman检验 + Nemenyi后检验
5. **可视化**：Critical Difference图

**产出**：
- **Table 7**：6模型 × 4指标 性能对比表（含均值±标准差）
- **Fig 5b**：模型性能对比柱状图
- **Fig 5c**：Critical Difference图（Nemenyi检验结果）

**预期结论层次**：

| 层次 | 预期结论 | 论文表述 |
|-----|---------|---------|
| 结论1 | TabPFN > 所有传统方法 | "TabPFN在LOVO验证中显著优于传统方法" |
| 结论2 | TabPFN > CatBoost | "新型神经网络优于当前最优梯度提升" |
| 结论3 | 集成方法 > 单一方法 | "非线性方法优于线性方法" |
| 结论4 | 传统ML > 经典农学 | "机器学习方法优于传统农学模型" |

**TabPFN优势分析（Discussion素材）**：

| 优势维度 | 具体表现 | 机理解释 |
|---------|---------|---------|
| 小样本泛化 | LOVO验证R²更高 | 预训练元知识迁移 |
| 无需调参 | 零超参优化 | 预训练模型自适应 |
| 快速推理 | 单次前向传播 | 育种筛选效率提升 |
| 不确定性 | 内置概率预测 | 低置信样本可标记复查 |

---

### 实验执行顺序

```
Phase 6 实验执行顺序:

Step 1: Exp-1 (融合有效性)
   └── 固定CatBoost，对比FS1/FS2/FS3/FS4
   └── 确定最优特征集 → FS4

Step 2: Exp-2 (OJIP消融)
   └── FS3 vs FS4，验证OJIP贡献

Step 3: Exp-6 (模型对比) ⭐ 核心
   └── 固定FS4，对比6个模型
   └── 确定最优模型 → TabPFN-2.5

Step 4: Exp-3 (Treatment审计)
   └── 检查Treatment特征重要性

Step 5: Exp-4 (SHAP可解释性)
   └── 对TabPFN-2.5/CatBoost计算SHAP

Step 6: Exp-5 (低成本替代)
   └── 预测OJIP，评估替代效果

Step 7: Exp-S1 (补充实验：不确定性量化)
   └── 基于Conformal Prediction的预测区间
   └── 探索性分析，放入附录
```

---

### Exp-S1：不确定性量化（补充实验 - Conformal Prediction）

> **注意**：本实验为探索性分析，受样本量（117样本）限制，结果仅供参考。正文简要提及，详细结果放入附录。

**目的**：探索在作物抗旱评价中应用保形预测（Conformal Prediction）进行不确定性量化的可行性。



## 10. 关键参考文献（带 DOI/URL）

### 10.1 交叉验证与小样本统计（KFold 设定）

本论文主用 KFold 进行 OOF 评估，因此引用重点放在：KFold 方差估计偏差、以及更稳健的显著性/置信区间构造（见 10.6）。

### 10.2 弱监督/MIL（样本级输入+品种级标签）

6. **Maron, O. & Lozano-Perez, T. (1997)** A framework for multiple-instance learning. *NeurIPS*.
   - URL: [http://papers.neurips.cc/paper/1346](http://papers.neurips.cc/paper/1346-a-framework-for-multiple-instance-learning.pdf)

7. **Carbonneau, M.-A. et al. (2018)** Multiple instance learning: A survey of problem characteristics and applications. *Pattern Recognition*, 77, 329-353.
   - DOI: [10.1016/j.patcog.2017.10.009](https://doi.org/10.1016/j.patcog.2017.10.009)

### 10.3 OJIP/JIP-test 理论与干旱应用（必须引用）

8. **Strasser, R.J. et al. (1995)** Simultaneous in vivo recording of prompt and delayed fluorescence and 820-nm reflection changes during drying and after rehydration of the resurrection plant Haberlea rhodopensis. *Photochemistry and Photobiology*, 63(2), 249-254.
   - DOI: [10.1111/j.1751-1097.1995.tb09240.x](https://doi.org/10.1111/j.1751-1097.1995.tb09240.x)

9. **Stirbet, A. & Govindjee (2011)** On the relation between the Kautsky effect (chlorophyll a fluorescence induction) and Photosystem II. *Journal of Photochemistry and Photobiology B: Biology*, 104(1-2), 236-257.
   - DOI: [10.1016/j.jphotobiol.2010.12.010](https://doi.org/10.1016/j.jphotobiol.2010.12.010)

10. **Strasser, R.J. et al. (2004)** Analysis of the chlorophyll a fluorescence transient. In *Chlorophyll a Fluorescence* (pp. 321-362). Springer.
    - DOI: [10.1007/978-1-4020-3218-9_12](https://doi.org/10.1007/978-1-4020-3218-9_12)

11. **Xia, Q. et al. (2022)** Classification of rice drought stress levels based on OJIP chlorophyll fluorescence curves and parameters. *Photosynthetica*, 60(1), 114-125.
    - DOI: [10.32615/ps.2022.005](https://doi.org/10.32615/ps.2022.005)

12. **Rapacz, M. et al. (2019)** Relationship between drought-related traits and the course of chlorophyll fluorescence induction curve in barley. *Frontiers in Plant Science*, 10, 78.
    - DOI: [10.3389/fpls.2019.00078](https://doi.org/10.3389/fpls.2019.00078)

### 10.4 多模态/融合先例（光谱 + 荧光）

13. **Zhang, C. et al. (2022)** End-to-end fusion of hyperspectral and chlorophyll fluorescence imaging to identify rice stresses. *Plant Phenomics*, 2022, 9851096.
    - DOI: [10.34133/2022/9851096](https://doi.org/10.34133/2022/9851096)

14. **Lin, J. et al. (2022)** Downscaling of solar-induced chlorophyll fluorescence from canopy level to photosystem level using random forest. *Remote Sensing*, 14(5), 1357.
    - DOI: [10.3390/rs14051357](https://doi.org/10.3390/rs14051357)

### 10.5 可解释性落地例子

15. **Remote Sensing 2024**: Multi-source remote sensing + SHAP for stomatal conductance.
    - URL: [https://www.mdpi.com/2072-4292/16/13/2467](https://www.mdpi.com/2072-4292/16/13/2467)

### 10.6 统计检验/不确定性（CV 下的严谨比较）

16. **Dietterich, T.G. (1998)** Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms. *Neural Computation*, 10(7), 1895-1923.
    - DOI: [10.1162/089976698300017197](https://doi.org/10.1162/089976698300017197)

17. **Nadeau, C. & Bengio, Y. (2003)** Inference for the Generalization Error. *Machine Learning*, 52, 239-281.
    - DOI: [10.1023/A:1024068626366](https://doi.org/10.1023/A:1024068626366)

18. **Bengio, Y. & Grandvalet, Y. (2004)** No Unbiased Estimator of the Variance of K-Fold Cross-Validation. *JMLR*, 5(Sep), 1089-1105. (conference version: NeurIPS 2003)
    - URL: [https://www.jmlr.org/papers/v5/grandvalet04a.html](https://www.jmlr.org/papers/v5/grandvalet04a.html)
    - PDF: [https://jmlr.org/papers/volume5/grandvalet04a/grandvalet04a.pdf](https://jmlr.org/papers/volume5/grandvalet04a/grandvalet04a.pdf)
    - NeurIPS 2003 PDF: [https://www.utc.fr/~grandval/nips03.pdf](https://www.utc.fr/~grandval/nips03.pdf)

19. **Ojala, M. & Garriga, G.C. (2010)** Permutation Tests for Studying Classifier Performance. *JMLR*, 11, 1833-1863.
    - URL: [https://www.jmlr.org/papers/v11/ojala10a.html](https://www.jmlr.org/papers/v11/ojala10a.html)

---

## 11. 论文章节对应与图表规划

### 11.1 论文结构与实验对应

| 论文章节 | 对应实验/研究内容 | 核心图表 |
|---------|-----------------|---------|
| **Introduction** | 研究背景与问题 | - |
| **Materials & Methods** | 数据、D_conv构建、模型 | Fig.1 技术路线图 |
| **Results 3.1** | 干旱胁迫生理响应 | Fig.2-3, Table 1-2 |
| **Results 3.2** | D_conv与品种分级 | Fig.4, Table 3 |
| **Results 3.3** | Exp-1: 融合有效性 | **Fig.5**, **Table 5** |
| **Results 3.4** | Exp-2: OJIP消融 | Table (消融对比) |
| **Results 3.5** | Exp-3: Treatment审计 | SHAP Treatment贡献 |
| **Results 3.6** | Exp-4: SHAP可解释性 | **Fig.6-7**, **Table 6** |
| **Results 3.7** | Exp-5: 低成本替代 | Fig.8, Table 7 |
| **Discussion** | 机制解释、局限、展望 | - |
| **Conclusion** | 核心结论 | - |

### 11.2 核心图表规划

#### 主图（8幅，300 dpi）

#### 11.2.1 图表风格规范（已确认）

**风格**: 现代简约风格

```python
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial'],
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': False,
    'figure.dpi': 300,
})
```

**补充图表**（附录）：
- Fig. S1: 特征相关性热图
- Fig. S3: SHAP交互作用图

#### 11.2.2 主图列表（8幅）

| 图编号 | 类型 | 内容 | 关键性 |
|-------|------|------|-------|
| Fig. 1 | 流程图 | 实验设计与技术路线 | ★★★ |
| Fig. 2 | 箱线图 | 三个处理下生理指标差异 | ★★ |
| Fig. 3 | 曲线图 | 典型品种OJIP曲线对比 | ★★★ |
| Fig. 4 | 柱状图+树状图 | 品种D值排名 + 层次聚类 | ★★★ |
| **Fig. 5** | **柱状图** | **融合对比（FS1-FS4性能）** | ★★★★★ |
| **Fig. 6** | **SHAP蜂群图** | **Top-20特征重要性** | ★★★★★ |
| **Fig. 7** | **混淆矩阵** | **最优模型分类结果** | ★★★★ |
| Fig. 8 | 散点图 | 预测OJIP vs 真实OJIP | ★★★ |
| **Fig. 9** | **Error bar图** | **品种D_conv预测值+95%置信区间** | ★★★★ |

#### 主表（7个）

| 表编号 | 内容 |
|-------|------|
| Table 1 | 供试品种及基本信息 |
| Table 2 | 生理指标描述性统计及PCA载荷 |
| Table 3 | 品种D_conv值与分级结果 |
| Table 4 | 多源光谱特征汇总 |
| **Table 5** | **融合对比：FS1-FS4性能（R²/RMSE/Spearman）** |
| **Table 6** | **Top-15重要特征及生理意义** |
| Table 7 | OJIP预测精度与替代效果 |
| **Table 8** | **高置信vs低置信品种对比（Conformal Prediction）** |

---

## 12. 论文写作注意事项

### 12.1 SHAP 表述规范

```
❌ 禁止: "因果关系"、"决定了"、"导致了"
✅ 改用: "特征贡献"、"强关联"、"潜在机制"、"可能反映"
```

**示例**：
- ❌ "NDRE的增加导致了抗旱性提高"
- ✅ "NDRE表现出较高的特征贡献，这可能与红边位置对早期胁迫的敏感响应有关"

### 12.2 OJIP 定位

将 OJIP 明确定位为反映 PSII 初级光化学反应与能量分配的功能性指标（Fv/Fm、PI、ET、DI 等），并结合干旱胁迫机制解释。

### 12.3 结论主张锚定

结论主张必须锚定三段证据：
1. LOVO 验证结果（跨品种泛化）
2. 融合增益 + OJIP增益（消融验证）
3. 可解释一致性（SHAP + 生理机制）

---

## 附录

### 附录 A：数学符号表

| 符号 | 含义 |
|------|------|
| $D_{conv}$ | 综合抗旱性评分 |
| $D_{stress}$ | 胁迫响应指数 |
| $D_{recovery}$ | 恢复能力指数 |
| $LC_{v,j}$ | 品种v指标j的比值 |
| $U(X)$ | 隶属函数归一化值 |
| $\phi_i$ | 特征i的SHAP值 |

### 附录 B：缩略词表

| 缩写 | 全称 | 中文 |
|------|------|------|
| NDVI | Normalized Difference Vegetation Index | 归一化植被指数 |
| NDRE | Normalized Difference Red Edge | 红边归一化指数 |
| OJIP | O-J-I-P fluorescence transient | 叶绿素荧光快速诱导 |
| PSII | Photosystem II | 光系统II |
| Fv/Fm | Maximum quantum yield of PSII | PSII最大光化学效率 |
| PIabs | Performance Index on absorption basis | 吸收基础性能指数 |
| SHAP | SHapley Additive exPlanations | Shapley加性解释 |
| LOVO | Leave-One-Variety-Out | 留一品种交叉验证 |

### 附录 C：推荐期刊列表

**一区期刊（高挑战）**：
- *Remote Sensing of Environment* (IF 13.5)
- *Plant Phenomics* (IF 7.6)
- *Computers and Electronics in Agriculture* (IF 8.3)

**二区期刊（推荐）**：
- *Field Crops Research* (IF 5.8)
- *Frontiers in Plant Science* (IF 5.6)
- *Agricultural Water Management* (IF 6.7)

**方法学期刊**：
- *Plant Methods* (IF 5.1)
- *Sensors* (IF 3.9)

### 附录 D：技术依赖

```txt
# requirements.txt
numpy>=1.23.0
pandas>=1.5.0
scipy>=1.10.0
scikit-learn>=1.2.0
catboost>=1.2
shap>=0.41.0
matplotlib>=3.7.0
seaborn>=0.12.0
openpyxl>=3.1.0
statsmodels>=0.14.0
pingouin>=0.5.0
tqdm>=4.65.0
joblib>=1.2.0
tabpfn>=2.0.0          # 新增：TabPFN-2.5模型
torch>=2.0.0           # 新增：TabPFN依赖
scikit-posthocs>=0.8.0 # 新增：Nemenyi检验
optuna>=3.0.0          # 新增：CatBoost调参
mapie>=0.8.0           # 新增：Conformal Prediction不确定性量化
```

---

**方案版本**：v6.0（融合版）
**更新日期**：2026-01-25
**状态**：待执行新实验矩阵（Exp-1 至 Exp-5）

---

> [!NOTE]
> 本方案融合了 experimental_design.md v5.2 和 phase6_multisource_fusion_with_ojip_experiment_design.md v1.0 的优点，形成完整的论文证据链。所有实验将按新矩阵重新执行，确保科学严谨性。
