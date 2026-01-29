# Exp-1: Multi-source Spectral Fusion Effectiveness Validation

**Report Generated:** 2026-01-27 17:37:29
**Random Seed:** 42
**Cross-Validation:** 5-Fold KFold
**Model:** CatBoost (iterations=500, lr=0.05, depth=4)

---

## Executive Summary

This experiment validates the effectiveness of multi-source spectral fusion for drought tolerance prediction in rice. Four feature sets were compared:

| Feature Set | Description | N Features |
|-------------|-------------|------------|
| FS1 | Multi-only (Multi原始波段 + 植被指数 + Treatment) | 22 |
| FS2 | Static-only (Static原始波段 + Static比值 + Treatment) | 13 |
| FS3 | 双源融合 (Multi + Static + Treatment，不含OJIP) | 32 |
| FS4 | 三源融合 (Multi + Static + OJIP + Treatment) | 40 |

### Key Findings

1. **Tri-source fusion (FS4) achieved the best performance:** Spearman ρ = 0.989, R² = 0.809
2. **OJIP contribution:** FS4 vs FS3 Δρ = +0.022
3. **Total fusion gain:** FS4 vs FS1 Δρ = +0.121
4. **Perfect top-K selection:** Hit@3 = 1.00, Hit@5 = 1.00

---

## 1. Methodology

### 1.1 Experimental Design

```
Validation Strategy: 5-Fold KFold (sample-level random split)
Aggregation: Sample predictions → Variety-level mean
Model: CatBoost Regressor (fixed hyperparameters)
Target: D_conv (drought tolerance score)
```

### 1.2 Evaluation Metrics

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| R² | Coefficient of determination | Prediction accuracy (higher is better) |
| RMSE | Root mean squared error | Prediction precision (lower is better) |
| Spearman ρ | Rank correlation coefficient | Ranking consistency (higher is better) |
| Pairwise Acc | Pairwise ranking accuracy | Proportion of correctly ordered pairs |
| Hit@3 | Top-3 hit rate | Overlap between true and predicted top-3 |
| Hit@5 | Top-5 hit rate | Overlap between true and predicted top-5 |

---

## 2. Results

### 2.1 Main Performance Comparison

| Feature Set | N | R² | RMSE | Spearman ρ | Pairwise Acc | Hit@3 | Hit@5 |
|-------------|---|-----|------|------------|--------------|-------|-------|
| FS1 | 22 | 0.556 | 0.088 | 0.868 | 0.846 | 1.00 | 0.80 |
| FS2 | 13 | 0.824 | 0.055 | 0.967 | 0.936 | 0.67 | 0.80 |
| FS3 | 32 | 0.719 | 0.070 | 0.967 | 0.949 | 1.00 | 1.00 |
| FS4 | 40 | 0.809 | 0.057 | 0.989 | 0.974 | 1.00 | 1.00 |

**Best performing feature set: FS4** (Tri-source fusion)

### 2.2 Fusion Gain Analysis

| Comparison | Description | ΔR² | Δρ | ΔPairwise |
|------------|-------------|-----|-----|-----------|
| FS3 vs FS1 | Dual vs Multi-only | +0.163 | +0.099 | +0.103 |
| FS3 vs FS2 | Dual vs Static-only | -0.105 | +0.000 | +0.013 |
| FS4 vs FS3 | Tri vs Dual (OJIP) | +0.090 | +0.022 | +0.026 |
| FS4 vs FS1 | Tri vs Multi (Total) | +0.253 | +0.121 | +0.128 |

### 2.3 Variety-level Predictions (FS4)

| Rank | Variety | True D_conv | Pred D_conv | Pred Rank | Error |
|------|---------|-------------|-------------|-----------|-------|
| 1 | 1252 | 0.5747 | 0.5183 | 1 | 0.0564 |
| 2 | 1257 | 0.5317 | 0.4303 | 3 | 0.1014 |
| 3 | 1099 | 0.5283 | 0.4472 | 2 | 0.0811 |
| 4 | 1228 | 0.4714 | 0.4261 | 4 | 0.0453 |
| 5 | 1214 | 0.4225 | 0.4033 | 5 | 0.0192 |
| 6 | 1274 | 0.4107 | 0.3983 | 6 | 0.0123 |
| 7 | 1210 | 0.3562 | 0.3890 | 7 | 0.0328 |
| 8 | 73 | 0.3277 | 0.3449 | 8 | 0.0172 |
| 9 | 12 | 0.2650 | 0.2969 | 9 | 0.0319 |
| 10 | 1219 | 0.2370 | 0.2766 | 11 | 0.0396 |
| 11 | 1110 | 0.2232 | 0.2957 | 10 | 0.0725 |
| 12 | 1218 | 0.2062 | 0.2640 | 12 | 0.0579 |
| 13 | 1235 | 0.1731 | 0.2613 | 13 | 0.0882 |

---

## 3. Discussion

### 3.1 Fusion Effectiveness

The tri-source fusion (FS4) achieved a Spearman correlation of 0.989, representing a 12.1% improvement over single-source Multi (FS1, ρ=0.868). This demonstrates that integrating multiple spectral modalities provides complementary information for drought tolerance assessment.

### 3.2 OJIP Contribution

OJIP parameters contributed an additional Δρ = +0.022 beyond dual-source fusion. While numerically modest, this gain:

- Improved pairwise ranking accuracy from 0.949 to 0.974
- Maintained perfect Hit@3 and Hit@5 scores (1.00)
- Suggests OJIP captures unique photosynthetic functional information

### 3.3 Unexpected Finding: Static Performance

Interestingly, Static-only (FS2) achieved ρ = 0.967 with only 13 features, comparable to the 32-feature FS3. This suggests that fluorescence ratios are highly informative indicators of drought tolerance, potentially due to their sensitivity to chlorophyll content and photosynthetic efficiency changes under stress.

### 3.4 Limitations

- **Sample size:** 13 varieties limit statistical power for significance testing
- **Single model:** Results specific to CatBoost; cross-model validation in Exp-6
- **Within-variety generalization:** KFold does not test cross-variety prediction

---

## 4. Output Files (Traceability)

### 4.1 Tables

| ID | Description | File Path |
|----|-------------|-----------|
| Table 1 | Feature set definitions | `F:/all_exp/results/tables/exp1_table1_feature_sets.csv` |
| Table 2 | Main performance comparison (Paper Table 5) | `F:/all_exp/results/tables/exp1_table2_main_results.csv` |
| Table 3 | Fusion gain analysis | `F:/all_exp/results/tables/exp1_table3_gain_analysis.csv` |
| Table 4 | Variety-level predictions (FS4) | `F:/all_exp/results/tables/exp1_table4_variety_predictions.csv` |
| Table 5 | Raw numerical results (full precision) | `F:/all_exp/results/tables/exp1_table5_raw_results.csv` |

### 4.2 Figures

| ID | Description | File Path |
|----|-------------|-----------|
| Fig. 1 | Fusion comparison across feature sets | `F:/all_exp/results/figures/exp1_fig1_fusion_comparison.png` |
| Fig. 2 | Predicted vs. true D_conv for each feature set | `F:/all_exp/results/figures/exp1_fig2_pred_vs_true.png` |
| Fig. 3 | Performance gain matrix between feature sets | `F:/all_exp/results/figures/exp1_fig3_gain_matrix.png` |
| Fig. 4 | Variety-level prediction comparison for FS4 | `F:/all_exp/results/figures/exp1_fig4_variety_comparison.png` |

---

## 5. Conclusions

1. **Tri-source spectral fusion is effective** for drought tolerance prediction, achieving near-perfect ranking correlation (ρ = 0.989)

2. **Each modality contributes unique information:**
   - Multi (reflectance): Structural and biochemical signatures
   - Static (fluorescence): Chlorophyll status and reabsorption
   - OJIP: Primary photochemical efficiency and electron transport

3. **Perfect top-K selection** supports practical application in breeding programs

4. **Proceed to Exp-6** for model comparison with TabPFN-2.5

---

*Report generated automatically by step2_exp1_fusion_report.py*
*Data source: F:/all_exp/data/processed/features_40.csv*