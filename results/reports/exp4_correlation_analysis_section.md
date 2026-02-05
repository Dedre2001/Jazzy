# 2.Z 多源光谱特征相关性分析

## 2.Z.1 分析目的

为验证多源光谱融合策略的合理性，本研究对三种测量模态（多光谱反射率、静态荧光、OJIP荧光动力学）的特征进行Pearson相关性分析。若模态间相关性较低，则表明不同测量手段捕获了互补的生理信息，融合具有理论依据；若模态内相关性适中，则表明特征间既有共性又有差异，信息结构合理。

## 2.Z.2 模态定义与特征构成

本研究采用的37个特征源自三种独立的光谱测量模态（表Z-1）。多光谱反射率模态包含11个原始波段（460-900 nm）及8个衍生植被指数；静态荧光模态包含4个原始荧光波段（F440、F520、F690、F740）及6个荧光比值；OJIP荧光动力学模态包含8个JIP-test参数。

**表Z-1 三种测量模态的特征构成**

| 模态 | 测量原理 | 原始特征 | 衍生特征 | 总计 |
|:-----|:---------|:--------:|:--------:|:----:|
| Multi | 叶片反射光谱 (460-900 nm) | 11 | 8 (VI) | 19 |
| Static | 稳态叶绿素荧光 | 4 | 6 (比值) | 10 |
| OJIP | 荧光诱导动力学 (0.01 ms - 1 s) | 8 | — | 8 |
| **合计** | | | | **37** |

*注：Multi模态的衍生特征为植被指数（VI），包括NDVI、NDRE、EVI等；Static模态的衍生特征为荧光比值，如F690/F740、F440/F690等。*

## 2.Z.3 模态内相关性分析

各模态内部特征的Pearson相关系数统计结果如表Z-2所示。Multi模态内部平均相关系数为0.322，其中近红外波段（760-900 nm）间呈现较高相关性（r = 0.90-0.96），这与叶片光学理论一致：近红外区域反射率主要由叶肉细胞结构决定，不同波段响应于相同的物理机制（Jacquemoud & Baret, 1990）。可见光波段（460-660 nm）间相关性较低（r = 0.30-0.50），反映了不同色素（叶绿素、类胡萝卜素）的差异化吸收特性。

**表Z-2 模态内特征相关性统计**

| 模态 | 平均相关系数 | 范围 | 特征说明 |
|:-----|:------------:|:----:|:---------|
| Multi | 0.322 | [-0.84, 0.97] | 近红外高相关，可见光低相关 |
| Static | 0.063 | [-0.74, 0.83] | 荧光比值间存在负相关 |
| OJIP | 0.099 | [-0.95, 0.87] | PIabs与Vi/Vj强负相关 |

Static模态内部平均相关系数仅为0.063，表明荧光比值间存在显著差异甚至负相关。例如，SR_F440_F690（蓝/红荧光比）与SR_F690_F740（红/远红荧光比）呈负相关（r = -0.65），这是因为两者分别反映了胁迫诱导的酚类积累和叶绿素荧光重吸收效应，具有不同的生理指示意义（Buschmann, 2007）。

OJIP模态内部平均相关系数为0.099，其中PIabs与Vj呈强负相关（r = -0.95）。这一结果具有明确的生理学基础：PIabs（综合性能指数）反映光合电子传递链的整体效率，而Vj（J相相对荧光）反映初级电子受体QA的还原累积程度。干旱胁迫导致电子传递受阻时，QA累积还原态（Vj升高），同时整体光合性能下降（PIabs降低），两者呈现负向耦合（Oukarroum et al., 2007; Stirbet & Govindjee, 2011）。

## 2.Z.4 模态间相关性分析

三种测量模态间的Pearson相关系数统计结果如表Z-3所示。Multi与Static模态间的平均绝对相关系数为0.27，Multi与OJIP模态间为0.29，Static与OJIP模态间为0.25。三组模态间相关性均处于较低水平（|r| < 0.30），表明不同测量手段捕获了互补的生理信息。

**表Z-3 模态间特征相关性统计**

| 模态对 | 平均相关系数 | 平均|r| | 范围 |
|:-------|:------------:|:-------:|:----:|
| Multi - Static | +0.08 | 0.27 | [-0.56, 0.68] |
| Multi - OJIP | +0.07 | 0.29 | [-0.71, 0.82] |
| Static - OJIP | +0.11 | 0.25 | [-0.57, 0.65] |

Multi与Static/OJIP模态间低相关性的物理基础在于测量原理的本质差异：反射光谱测量叶片表面和内部结构对入射光的散射与吸收特性，而荧光测量光合系统吸收光能后的再发射过程。前者主要反映叶片的光学特性和色素含量，后者直接反映光合机构的功能状态（Lichtenthaler & Miehé, 1997）。

Static与OJIP模态虽然都涉及叶绿素荧光测量，但两者的低相关性（|r| = 0.25）源于测量时间尺度的差异。Static荧光测量稳态条件下的荧光发射强度，主要反映色素组成和荧光量子产率；OJIP荧光测量暗适应叶片受光激发后0.01 ms至1 s内的荧光瞬态变化，直接反映光合电子传递链各组分的氧化还原动态（Strasser et al., 2004）。这种时间尺度的差异使得两类荧光测量提供了互补的光合功能信息。

## 2.Z.5 相关性模式的生理学意义

特征相关性模式不仅验证了多源融合的合理性，还反映了植物的生理状态。文献报道表明，干旱胁迫会改变光合参数间的相关性结构：对照条件下参数间相关性较弱，各环节独立运作；而胁迫条件下相关性增强，反映光合电子传递链各环节的耦合效应加剧（Živčák et al., 2008）。

本研究数据中OJIP参数间存在的强负相关（PIabs-Vj: r = -0.95）正是这种胁迫耦合效应的体现。当干旱导致气孔关闭、CO₂供应不足时，Calvin循环对ATP和NADPH的消耗减少，电子传递链下游受体处于还原态，导致电子在QA处累积（Vj升高）。这种"电子拥堵"状态使得整体光合性能下降（PIabs降低），形成两参数间的强负相关。抗旱性强的品种能够通过替代电子流、抗氧化系统等机制维持电子传递的畅通，表现为较低的Vj和较高的PIabs。

## 2.Z.6 小结

本节通过Pearson相关性分析，系统评估了三种测量模态的特征关联结构。主要发现如下：

**（1）模态间相关性较低（|r| = 0.25-0.29）**，表明多光谱反射率、静态荧光和OJIP荧光动力学分别捕获了叶片光学特性、稳态荧光发射和光合电子传递动态等不同层面的生理信息，为多源融合策略提供了理论依据。

**（2）模态内相关性结构符合物理和生理学预期**：近红外波段间高相关（r > 0.9）反映了相同的叶片结构散射机制；OJIP参数中PIabs与Vj强负相关（r = -0.95）反映了干旱胁迫下电子传递链的耦合效应。

**（3）相关性模式具有生理指示意义**：OJIP参数间的负相关结构可作为胁迫响应的指纹特征，不同抗旱性品种在该相关结构中的位置可能存在系统性差异。

综上所述，相关性分析从统计学角度验证了多源光谱融合的合理性，并揭示了特征关联结构与植物生理状态的内在联系。

---

## 参考文献

Buschmann, C. (2007). Variability and application of the chlorophyll fluorescence emission ratio red/far-red of leaves. *Photosynthesis Research*, 92(2), 261-271.

Jacquemoud, S., & Baret, F. (1990). PROSPECT: A model of leaf optical properties spectra. *Remote Sensing of Environment*, 34(2), 75-91.

Lichtenthaler, H. K., & Miehé, J. A. (1997). Fluorescence imaging as a diagnostic tool for plant stress. *Trends in Plant Science*, 2(8), 316-320.

Oukarroum, A., El Madidi, S., Schansker, G., & Strasser, R. J. (2007). Probing the responses of barley cultivars (*Hordeum vulgare* L.) by chlorophyll a fluorescence OLKJIP under drought stress and re-watering. *Environmental and Experimental Botany*, 60(3), 438-446.

Stirbet, A., & Govindjee. (2011). On the relation between the Kautsky effect (chlorophyll a fluorescence induction) and Photosystem II: Basics and applications of the OJIP fluorescence transient. *Journal of Photochemistry and Photobiology B: Biology*, 104(1-2), 236-257.

Strasser, R. J., Tsimilli-Michael, M., & Srivastava, A. (2004). Analysis of the chlorophyll a fluorescence transient. In *Chlorophyll a Fluorescence* (pp. 321-362). Springer, Dordrecht.

Živčák, M., Brestič, M., Olšovská, K., & Slamka, P. (2008). Performance index as a sensitive indicator of water stress in *Triticum aestivum* L. *Plant, Soil and Environment*, 54(4), 133-139.
