# 第1章 绪论

## 1.1 研究背景与意义

水稻（*Oryza sativa* L.）作为全球最重要的粮食作物之一，为世界一半以上的人口提供主食。据联合国粮农组织（FAO）和美国农业部（USDA）的最新统计，2024/2025年度全球水稻产量预计将达到创纪录的5.43亿吨（FAO, 2025; USDA, 2025）。然而，这一总量的增长掩盖了区域性的严峻挑战。受厄尔尼诺现象及气候变化的持续影响，澳大利亚、美国密西西比三角洲等主要产区因高温和干旱出现了显著的产量波动（USDA, 2025）。经典的植物生理学研究指出，水份是限制植物生长和生产力的首要环境因子（Hsiao, 1973; Levitt, 1980）。研究证实，全球气温每升高1°C，水稻产量预计将下降约8%（Zhao et al., 2017），而干旱已成为威胁水稻生产的首要非生物胁迫因素。在我国，华南、西南等主要稻区频繁遭受季节性干旱，严重制约了水稻产量的稳定增长和粮食安全保障。

培育和推广抗旱品种是应对干旱胁迫最经济有效的策略。水稻抗旱性是一个复杂的数量性状，受多基因控制并与环境因素密切互作（Blum, 2011）。虽然近年来分子育种技术取得了长足进步，发掘了大量抗旱相关数量性状位点（QTL）和功能基因（Zhang et al., 2024），但基因型的优异并不总是能转化为表型的抗性。传统的抗旱性评价方法主要包括田间产量比较法和生理指标测定法。Fukai & Cooper (1995) 综述了利用生理性状开发抗旱品种的方法，指出田间产量比较法虽然结果直接，但需要完成作物的完整生育期，周期漫长且受环境波动干扰。Araus et al. (2002) 分析了C3禾谷类作物的育种策略，证实叶片相对含水量（RWC）、干旱敏感指数（DSI）等生理指标虽然准确，却多依赖破坏性取样，难以满足现代育种对大规模种质资源"高通量、无损"筛选的需求。本研究采用的多源光谱技术可在数秒内完成单株叶片的无损测量，结合自动化数据处理流程，单日可完成数百个样本的表型采集，在通量上较传统方法提升了一个数量级。

光谱技术作为一种非侵入性的检测手段，为作物表型的高通量获取提供了新途径。Carter (1991) 阐明了水分含量对叶片光谱反射率的原初和次生影响，为光谱监测水分胁迫奠定了物理基础。Govindjee (1995) 综述了叶绿素荧光技术的发展历程，将其定义为直接探测光合机构电子传递功能的“内在探针”。然而，单一光谱模态往往存在局限性：反射光谱对早期光合功能衰退不够敏感，而荧光信号易受环境光干扰且难以反映形态结构变化。因此，Murchie et al. (2024) 提出了整合多种光谱技术的新型评价框架，强调构建多源光谱融合（Multi-source Spectral Fusion）体系是突破当前抗旱鉴定瓶颈的关键。

基于上述背景，本研究提出基于多源光谱融合的水稻种质资源抗旱性快速评价方法。本研究旨在系统揭示多光谱反射率、静态荧光和OJIP荧光动力学三种模态对干旱胁迫的响应特征及其互补性，阐明"结构-功能"多维光谱信息的协同机制。在方法创新方面，本研究尝试将表格数据基础模型（TabPFN）引入作物表型预测领域，构建"TabPFN预测+TreeSHAP解释"的分析框架，探索解决小样本农学数据建模难、黑箱模型解释难等问题的技术路径，为大规模种质资源筛选提供高效、可解释的技术支撑。

---

## 1.2 国内外研究进展

### 1.2.1 水稻抗旱性评价方法研究进展

水稻抗旱性评价是一个多维度、多层次的系统工程，涉及形态学、生理生化及产量等多个层面的指标。随着育种目标从单一的"高产"转向"高产稳产"，评价方法也在不断演进。

从形态学角度，根系构型是植物适应干旱的核心性状。Uga et al. (2013) 发现了控制水稻根系构型的主效基因 *DEEPER ROOTING 1*，证实深根系比例和根冠比是水稻抗旱性的形态基础，发达的根系能显著提升植株对深层土壤水分的获取能力。在地上部性状方面，IRRI (2013) 发布了标准评价手册，采用1-9级视觉评分法对抗旱过程中的叶片卷曲度进行定性评价。Wasaya et al. (2018) 综述了干旱表型分析中的根系研究技术，指出传统的"挖掘-清洗"方法不仅具有破坏性，且难以保证根系的完整性，导致测量误差较大。

在生理生化层面，研究者开发了一系列反映植物水分状态和抗逆能力的指标。Barrs & Weatherley (1962) 建立了测定叶片相对含水量（RWC）的标准协议，该指标至今仍被视为衡量植物水分亏缺程度的金标准。Szabados & Savouré (2010) 阐明了脯氨酸（Proline）作为多功能氨基酸的生理作用，证实干旱胁迫会激活 P5CS 酶途径并抑制 P5CDH 酶途径，导致脯氨酸大量积累以维持细胞膨压并清除活性氧（ROS）。Dhindsa et al. (1981) 揭示了叶片衰老与膜通透性及脂质过氧化之间的关联，提出丙二醛（MDA）含量的升高可作为细胞膜系统受损的可靠指标。尽管这些指标具有明确的生物学意义，但测定过程繁琐，单一技术人员每天仅能处理数十个样本，且无法实现同一株植物的连续监测。

基于产量的综合评价指数是抗旱性评价的终极验证标准。Fischer & Maurer (1978) 提出了干旱敏感性指数（DSI），通过量化干旱下产量的下降幅度来评估品种的稳定性。Fernandez (1992) 进一步建立了抗旱性指数（STI），利用胁迫与正常环境下产量的乘积作为评价指标，倾向于筛选出"高产且抗逆"的基因型。但基于产量的评价周期通常长达120天以上，且极易受光温、病虫害等环境噪声的干扰，导致年份间结果的重现性欠佳。

### 1.2.2 多光谱反射技术在作物表型中的应用

多光谱技术基于“光与物质相互作用”的物理原理，利用植物对特定波长光线的反射特征来表征生理状态。Carter (1993) 系统分析了植物受胁迫后在可见光及近红外区域的光谱响应，证实红光（660 nm）吸收峰的减弱和反射率的升高是叶绿素受损的早期信号。Gitelson & Merzlyak (1994) 发现了秋季叶片衰老过程中光谱反射率的规律性变化，定义了红边蓝移（Red edge blue shift）现象，并指出其与叶绿素浓度的下降高度相关。

植被指数（VIs）的演进极大提升了光谱特征的提取效率。Rouse et al. (1974) 利用 Landsat 卫星数据提出了归一化植被指数（NDVI），该指数通过近红外与红光的差异化比值来反映植被盖度，成为近半个世纪以来遥感领域的基石。Huete (1988) 针对稀疏植被背景提出了土壤调整植被指数（SAVI），通过引入调节因子有效消除了土壤背景反射的干扰。Gamon et al. (1992) 研制了光化学反射指数（PRI），利用 531 nm 处叶黄素循环色素的瞬时变化，实现了对光合利用效率（LUE）的动态监测。Peñuelas et al. (1995) 则开发了水分指数（WI），利用 970 nm 处的微弱水吸收峰，实现了对植物水分状态的无损定量。

进入21世纪，传感器技术向着高通量、高空间分辨率方向快速发展。Feng et al. (2024) 采用多传感器数据融合技术对番茄干旱胁迫进行识别，利用集成在无人机（UAV）上的多光谱相机获取冠层特征，通过结合RGB、多光谱和热红外特征，在干旱状态分类中达到了90%以上的准确率。与Feng et al. (2024) 聚焦于干旱状态的二分类识别不同，本研究的目标是品种抗旱性的连续值预测与定量排名，因此在模型选择上侧重回归算法，并引入OJIP荧光动力学这一更能反映光合功能动态的模态。Zhang et al. (2024) 开展了基于全基因组关联分析（GWAS）的水稻抗旱性研究，利用高通量表型平台获取的大规模光谱数据作为表型输入，成功定位了多个控制水稻抗旱性的关键基因。

### 1.2.3 荧光光谱技术在植物胁迫检测中的应用

荧光光谱技术被誉为植物光合作用的“无损探针”，能够捕捉光合电子传递链的极微秒级变化。Schreiber et al. (1986) 研制了首台脉冲振幅调制（PAM）荧光仪，通过采用测量光、光化光和饱和脉冲的调制策略，首次实现在环境光干扰下对光化学效率（$F_v/F_m$）和非光化学淬灭（NPQ）的准确测定。

OJIP 快速荧光诱导动力学技术的成熟为精细诊断光合损伤提供了可能。Strasser et al. (2004) 系统构建了 JIP-test 分析框架，将 OJIP 曲线分解为 O-J-I-P 四个多相上升阶段，定义了包括光合性能指数（$PI_{abs}$）在内的30余个生物物理参数。Srivastava et al. (1995) 阐明了 OJIP 瞬态在植物及蓝细菌中的普适性，证实了 J 点（2 ms）的隆起反映了 $Q_A^-$ 的过度积累。Stirbet et al. (2024) 进一步分析了 OJIP 曲线在非生物胁迫下的精细结构，证实重度干旱导致的放氧复合体（OEC）损伤会在 300 $\mu s$ 处诱发出特异性的 K-step 信号。

多光谱荧光成像（MFI）技术则开辟了次生代谢监测的新路径。Lichtenthaler & Schweiger (1998) 证实了细胞壁结合的阿魏酸（Ferulic Acid）是植物发出蓝色荧光（F440）的主要物质来源。Buschmann et al. (2000) 采用多色荧光成像系统研究了甜菜叶片的养分状态，发现蓝/红荧光比值（BFR）能够比叶绿素荧光更早地预警植物的生理紊乱。Murchie et al. (2024) 的最新综述指出，将叶绿素荧光（CF）与多光谱荧光（MFI）结合，能够实现对植物“能量流”与“物质流”的同步监测，是构建未来智慧农业诊断系统的核心技术。

### 1.2.4 多源光谱融合与人工智能在农业中的研究进展

随着传感器维度的增加，如何有效处理高维异构数据成为研究焦点。Feng et al. (2024) 比较了不同层级的数据融合策略，证实在特征层级（Feature-level）进行多光谱与热红外特征拼接，能够显著提升模型对作物水分胁迫的判别能力。然而，在样本量受限（$N < 1000$）的典型农学实验中，传统的深度学习架构往往面临严重的过拟合风险。

表格数据基础模型（Tabular Foundation Models）的突破为这一难题提供了新解。Hollmann et al. (2023) 提出了TabPFN（Prior-Data Fitted Networks）表格数据基础模型，该模型基于12层Transformer编码器架构，在超过1亿个合成分类任务上进行离线预训练，通过学习贝叶斯推理的先验分布实现对新任务的零样本迁移。在OpenML-CC18基准测试的30个数据集上，TabPFN在样本量小于1000的任务中达到了0.898的平均AUC，较XGBoost（0.873）和CatBoost（0.881）分别提升2.5和1.7个百分点，且推理时间仅需约1秒。Badaro et al. (2023) 综述了表格数据 Transformer 的应用，指出此类模型在处理非均匀、非线性特征交互方面具有天然优势。Sabo et al. (2025) 率先将 TabPFN 应用于次国家级规模的作物产量预测，结果证实该模型在无需超参数调优的情况下，其预测精度显著超越了经典的随机森林和梯度提升模型。Sabo et al. (2025) 的研究聚焦于大尺度产量预测（样本量>1000），而本研究则探索TabPFN在小样本农学实验（N=117）中的适用性，验证其预训练范式能否在种质资源评价这一典型小样本场景下保持性能优势。

可解释人工智能（XAI）则是消除农业黑箱模型信任危机的关键。Lundberg et al. (2020) 将基于博弈论的 SHAP 算法应用于树模型解释，实现了特征贡献的局部与全局量化符。该方法目前已被广泛应用于解析环境因子（如降水、施肥）对产量的非线性影响，但在多源光谱协同机理的挖掘方面仍有待深入。

### 1.2.5 现有研究的不足与发展趋势

尽管光谱技术与 AI 算法各自取得了长足进展，但聚焦于水稻抗旱性快速评价的系统性研究仍存在明显空白。

现有研究多关注单一的光谱或荧光模态，缺乏对 OJIP 光合动力学与多光谱反射、静态荧光的系统性整合研究。例如，Feng et al. (2024) 虽然融合了RGB、多光谱和热红外三种模态，但未纳入荧光动力学信息；Murchie et al. (2024) 的综述虽然强调了荧光技术的重要性，但其引述的多数研究仍局限于单一荧光模态，缺乏与反射光谱的系统性整合。反射光谱擅长捕捉叶片色素和结构变化，荧光光谱擅长反映次生代谢状态，OJIP动力学则直接探测光合电子传递功能。三种模态各有侧重，但"结构-功能"信息的互补机制尚不明确，多源融合的协同增益缺乏定量验证。与此同时，在样本量有限的科研场景下（通常N<200），传统机器学习模型容易过拟合，深度学习模型则因参数量过大而难以收敛。TabPFN 等基础模型在农学领域的适用性尚未得到验证，预训练范式能否突破小样本表型建模的瓶颈有待探索。此外，多数研究止步于"黑箱预测"，未能利用 SHAP 交互值等前沿技术深入挖掘多源数据产生协同增益的生物学本质。即便模型预测精度较高，若无法解释"为什么多源融合优于单源"，其科学价值和育种指导意义将受到限制。

针对上述不足，本研究拟从三个方面开展工作。在数据层面，构建"反射+静态荧光+OJIP动力学"三源融合特征体系，通过消融实验定量验证融合增益。在模型层面，引入TabPFN基础模型，利用其在大规模合成数据上预训练获得的先验知识，探索零样本迁移在农学小样本数据上的可行性。在机制层面，采用CatBoost白盒代理结合TreeSHAP交互分析框架，从特征交互角度揭示跨模态协同的生理学机制。

---

## 1.3 研究内容与技术路线

### 1.3.1 研究目标

本研究的总体目标是：构建基于多源光谱融合的水稻种质资源抗旱性快速评价方法，验证 TabPFN 模型在小样本表型预测中的优势，并揭示多源信息的协同生理机制。

围绕上述总体目标，本研究设定了四个层层递进的具体目标。在特征解析层面，本研究拟系统阐明水稻在干旱胁迫下反射光谱、荧光光谱及OJIP动力学的响应规律，构建涵盖"结构-功能"多维信息的特征体系。在模型寻优层面，本研究将在小样本（N<200）条件下，系统比较TabPFN与经典机器学习模型（CatBoost、RF、PLSR等）的预测性能，验证基础模型在农学数据建模中的适用性。在机制揭示层面，本研究将利用SHAP交互值分析方法，量化不同模态间的协同效应，从生理学角度解释模型决策依据，揭示多源融合产生增益的科学本质。在应用验证层面，本研究将对不同水稻种质资源进行抗旱性排名，并与传统生理指标进行一致性验证，为育种实践提供参考。

### 1.3.2 研究内容

**（1）水稻干旱胁迫多源光谱响应特征研究**
选取具有代表性的籼稻品种，设置不同梯度的干旱胁迫试验。同步采集多光谱反射率（涵盖可见-近红外波段）、稳态荧光发射光谱（400-800 nm）以及 OJIP 快速叶绿素荧光动力学曲线。构建包含 $D_{\text{conv}}$（基于生理指标的综合抗旱指数）在内的真值标签体系。本部分内容将在第3章详细阐述。

**（2）基于多源光谱融合的抗旱性预测模型研究**
构建"反射+静态荧光+动力学"的多源特征池。建立包含线性模型（PLSR, Ridge）、传统集成模型（Random Forest, CatBoost）和基础模型（TabPFN）的对比框架。重点考察 TabPFN 在免调参、小样本场景下的预测精度（$R^2$, RMSE）与排序能力（Spearman $\rho$）。本部分内容将在第4章详细阐述。

**（3）多源融合策略有效性验证与消融分析**
设计全组合消融实验（Ablation Study），对比单模态、双模态及三模态融合的性能差异。计算"融合增益系数"，量化各模态的边际贡献，验证多源融合的必要性。本部分内容将在第4章详细阐述。

**（4）基于特征交互分析的多源协同机制研究**
构建可解释性分析流程。由于 TabPFN 目前缺乏原生的 SHAP 接口，本研究采用与其预测行为高度一致（Spearman $\rho > 0.9$）的 CatBoost 模型作为"白盒代理"，计算 TreeSHAP 特征重要性及 SHAP Interaction Values。重点分析跨模态特征对（如"荧光参数 $\times$ 光谱指数"）的交互作用，揭示光合功能与叶片结构信息的互补机制。本部分内容将在第5章详细阐述。

### 1.3.3 技术路线

本研究的技术路线遵循“数据采集—模型构建—机制解析—应用验证”的逻辑主线。首先，通过控制试验获取水稻生理指标及多源光谱数据；其次，构建多模态融合特征体系，并分别训练线性、集成学习及 TabPFN 基础模型；随后，通过消融实验和模型对比筛选最优预测方案；最后，利用 SHAP 可解释性分析揭示多源特征的协同机制，最终建立可靠的水稻抗旱性评价方法。

*[在此处插入 Figure 1-1: 技术路线图]*
*说明：图 1-1 展示了从数据采集、特征提取、模型训练到机制解析的全流程。*

---

## 参考文献

Araus, J. L., et al. (2002). Plant breeding and drought in C3 cereals: what should we breed for? *Annals of Botany*.

Badaro, G., et al. (2023). Transformers for tabular data representation: A survey of models and applications. *Transactions of the Association for Computational Linguistics*.

Barrs, H. D., & Weatherley, P. E. (1962). A re-examination of the relative turgidity technique for estimating water deficits in leaves. *Australian Journal of Biological Sciences*.

Blum, A. (2011). *Plant Breeding for Water-Limited Environments*. Springer.

Buschmann, C., et al. (2000). Multicolor fluorescence imaging of sugar beet leaves with different nitrogen status. *Plant Biology*.

Carter, G. A. (1991). Primary and secondary effects of water content on the spectral reflectance of leaves. *American Journal of Botany*.

Carter, G. A. (1993). Responses of leaf spectral reflectance to plant stress. *American Journal of Botany*.

Dash, J., & Curran, P. J. (2004). The MERIS terrestrial chlorophyll index. *International Journal of Remote Sensing*.

Dhindsa, R. S., et al. (1981). Leaf senescence: correlated with increased levels of membrane permeability and lipid peroxidation. *Journal of Experimental Botany*.

FAO. (2025). *World Food Situation*.

Feng, L., Chen, S., Zhang, C., Zhang, Y., & He, Y. (2024). Multi-sensor fusion for crop drought stress detection using machine learning. *Computers and Electronics in Agriculture*, 218, 108693.

Fernandez, G. C. (1992). Effective selection criteria for assessing plant stress tolerance.

Fischer, R. A., & Maurer, R. (1978). Drought resistance in spring wheat cultivars. I. Grain yield responses. *Australian Journal of Agricultural Research*.

Fukai, S., & Cooper, M. (1995). Development of drought-resistant cultivars using physiological traits in rice. *Field Crops Research*.

Gamon, J. A., et al. (1992). A narrow-waveband spectral index that tracks diurnal changes in photosynthetic efficiency. *Remote Sensing of Environment*.

Gitelson, A. A., & Merzlyak, M. N. (1994). Spectral reflectance changes associated with autumn senescence. *Journal of Plant Physiology*.

Govindjee. (1995). Sixty-three years since Kautsky: Chlorophyll a fluorescence. *Australian Journal of Plant Physiology*.

Hollmann, N., et al. (2023). TabPFN: A transformer that solves small tabular classification problems in a second. *ICLR*.

Hsiao, T. C. (1973). Plant responses to water stress. *Annual Review of Plant Physiology*.

Huete, A. R. (1988). A soil-adjusted vegetation index (SAVI). *Remote Sensing of Environment*.

IRRI. (2013). *Standard Evaluation System for Rice*.

Levitt, J. (1980). *Responses of Plants to Environmental Stresses*. Academic Press.

Lichtenthaler, H. K., & Schweiger, J. (1998). Cell wall bound ferulic acid, the major substance of the blue-green fluorescence emission. *Journal of Plant Physiology*.

Lundberg, S. M., et al. (2020). From local explanations to global understanding with explainable AI for trees. *Nature Machine Intelligence*.

Murchie, E. H., et al. (2024). Chlorophyll fluorescence analysis: a guide to good practice. *Journal of Experimental Botany*.

Peñuelas, J., et al. (1995). The reflectance at the 950–970 nm region as an indicator of plant water status. *International Journal of Remote Sensing*.

Peñuelas, J., et al. (2011). Reflectance assessment of seasonal and annual changes in biomass. *Remote Sensing of Environment*.

Rouse, J. W., et al. (1974). Monitoring vegetation systems in the Great Plains with ERTS.

Sabo, F., et al. (2025). From Rows to Yields: How Foundation Models for Tabular Data Simplify Crop Yield Prediction. *arXiv preprint arXiv:2506.19046*.

Schreiber, U., et al. (1986). Continuous recording of photochemical and non-photochemical chlorophyll fluorescence quenching. *Photosynthesis Research*.

Sitko, K., et al. (2024). Advanced analysis of OJIP chlorophyll fluorescence induction kinetics. *Photosynthetica*.

Srivastava, A., et al. (1995). Polyphasic chlorophyll a fluorescence transient in plants and cyanobacteria. *Photochemistry and Photobiology*.

Stirbet, A., et al. (2024). Chlorophyll a fluorescence induction. *Photosynthetica*.

Strasser, R. J., et al. (2004). Analysis of the chlorophyll a fluorescence transient.

Szabados, L., & Savouré, A. (2010). Proline: a multifunctional amino acid. *Trends in Plant Science*.

Uga, Y., et al. (2013). Control of root system architecture by DEEPER ROOTING 1. *Nature Genetics*.

Wasaya, A., et al. (2018). Root phenotyping for drought tolerance: a review. *Agronomy*.

Zhang, Y., et al. (2024). Genome-wide association study for drought tolerance in rice. *The Crop Journal*.

Zhao, C., et al. (2017). Temperature increase reduces global yields of major crops. *PNAS*.