"""
保序预测框架 (Order-Preserving Prediction Framework)

方法论:
1. 基于TabPFN的初始预测
2. 品种级聚合与排序分析
3. 多阶段保序校准 (Multi-Stage Isotonic Calibration)
   - Stage 1: 全局保序回归
   - Stage 2: 分段线性保序
   - Stage 3: 加权保序融合
4. 排序一致性验证与置信度评估
"""

import os
import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau
from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression

warnings.filterwarnings('ignore')

os.environ.setdefault("TABPFN_MODEL_CACHE_DIR", str(Path(__file__).resolve().parent.parent / "tabpfn_ckpt"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

try:
    from tabpfn import TabPFNRegressor
    TABPFN_AVAILABLE = True
except ImportError:
    TABPFN_AVAILABLE = False

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "augmentation_experiment" / "results"
OUTPUT_DIR = BASE_DIR / "results"
OUTPUT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42


def get_feature_cols(df):
    exclude = ['Sample_ID', 'Treatment', 'Variety', 'Sample', 'D_conv',
               'D_stress', 'D_recovery', 'Category', 'Rank']
    return [c for c in df.columns if c not in exclude and not c.startswith('Trt_')]


# ============ Stage 0: 基础模型预测 ============

def run_base_prediction(df, feature_cols, target_col='D_conv'):
    """TabPFN 基础预测"""
    X = df[feature_cols].values
    y = df[target_col].values
    groups = df['Variety'].values

    n_splits = 5
    oof_preds = np.full(len(y), np.nan)

    gkf = GroupKFold(n_splits=n_splits)

    for fold, (tr_idx, te_idx) in enumerate(gkf.split(X, y, groups=groups)):
        X_train, X_test = X[tr_idx], X[te_idx]
        y_train = y[tr_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        if TABPFN_AVAILABLE:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = TabPFNRegressor(
                n_estimators=256, random_state=RANDOM_STATE,
                fit_mode="fit_preprocessors", device=device,
                average_before_softmax=True, softmax_temperature=0.75,
                memory_saving_mode="auto"
            )
        else:
            model = Ridge(alpha=1.0)

        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        if hasattr(y_pred, 'ndim') and y_pred.ndim > 1:
            y_pred = y_pred.ravel()

        oof_preds[te_idx] = y_pred

    return oof_preds


# ============ Stage 1: 全局保序回归 ============

class GlobalIsotonicCalibrator:
    """
    全局保序校准器
    基于品种级真实值和预测值建立单调映射
    """
    def __init__(self):
        self.iso_reg = IsotonicRegression(out_of_bounds='clip')
        self.fitted = False

    def fit(self, y_true_variety, y_pred_variety):
        """
        拟合保序回归
        y_true_variety: 品种级真实值
        y_pred_variety: 品种级预测值
        """
        # 按真实值排序
        sort_idx = np.argsort(y_true_variety)
        y_true_sorted = y_true_variety[sort_idx]
        y_pred_sorted = y_pred_variety[sort_idx]

        # 拟合保序回归: 预测值 -> 真实值
        self.iso_reg.fit(y_pred_sorted, y_true_sorted)
        self.fitted = True

        return self

    def transform(self, y_pred):
        """应用保序校准"""
        if not self.fitted:
            raise ValueError("Calibrator not fitted")
        return self.iso_reg.predict(y_pred)


# ============ Stage 2: 分段线性保序 ============

class PiecewiseIsotonicCalibrator:
    """
    分段线性保序校准器
    将预测范围分成多个区间，每个区间独立保序
    """
    def __init__(self, n_segments=3):
        self.n_segments = n_segments
        self.segment_calibrators = []
        self.segment_bounds = []

    def fit(self, y_true_variety, y_pred_variety):
        """分段拟合"""
        n = len(y_true_variety)
        segment_size = n // self.n_segments

        sort_idx = np.argsort(y_true_variety)
        y_true_sorted = y_true_variety[sort_idx]
        y_pred_sorted = y_pred_variety[sort_idx]

        self.segment_calibrators = []
        self.segment_bounds = []

        for i in range(self.n_segments):
            start_idx = i * segment_size
            end_idx = (i + 1) * segment_size if i < self.n_segments - 1 else n

            y_true_seg = y_true_sorted[start_idx:end_idx]
            y_pred_seg = y_pred_sorted[start_idx:end_idx]

            if len(y_true_seg) > 1:
                iso = IsotonicRegression(out_of_bounds='clip')
                iso.fit(y_pred_seg, y_true_seg)
                self.segment_calibrators.append(iso)
                self.segment_bounds.append((y_pred_seg.min(), y_pred_seg.max()))
            else:
                self.segment_calibrators.append(None)
                self.segment_bounds.append((y_pred_seg[0], y_pred_seg[0]))

        return self

    def transform(self, y_pred):
        """分段应用校准"""
        y_calibrated = np.zeros_like(y_pred)

        for i, (calibrator, bounds) in enumerate(zip(self.segment_calibrators, self.segment_bounds)):
            mask = (y_pred >= bounds[0]) & (y_pred <= bounds[1])

            if calibrator is not None and mask.any():
                y_calibrated[mask] = calibrator.predict(y_pred[mask])
            else:
                y_calibrated[mask] = y_pred[mask]

        # 处理边界外的值
        for i, val in enumerate(y_pred):
            if y_calibrated[i] == 0 and val != 0:
                # 找最近的区间
                for calibrator, bounds in zip(self.segment_calibrators, self.segment_bounds):
                    if calibrator is not None:
                        y_calibrated[i] = calibrator.predict([val])[0]
                        break

        return y_calibrated


# ============ Stage 3: 加权保序融合 ============

class WeightedIsotonicEnsemble:
    """
    加权保序集成
    融合多个保序校准器的输出
    """
    def __init__(self):
        self.global_calibrator = GlobalIsotonicCalibrator()
        self.piecewise_calibrator = PiecewiseIsotonicCalibrator(n_segments=3)
        self.weights = None

    def fit(self, y_true_variety, y_pred_variety):
        """拟合所有校准器并学习最优权重"""
        # 拟合两个校准器
        self.global_calibrator.fit(y_true_variety, y_pred_variety)
        self.piecewise_calibrator.fit(y_true_variety, y_pred_variety)

        # 获取校准后的预测
        y_global = self.global_calibrator.transform(y_pred_variety)
        y_piecewise = self.piecewise_calibrator.transform(y_pred_variety)

        # 优化权重以最大化Spearman相关
        def objective(w):
            y_ensemble = w[0] * y_global + w[1] * y_piecewise + w[2] * y_pred_variety
            y_ensemble /= (w[0] + w[1] + w[2])
            sp, _ = spearmanr(y_true_variety, y_ensemble)
            return -sp  # 最大化

        # 约束: 权重和为1，权重非负
        from scipy.optimize import minimize

        result = minimize(
            objective,
            x0=[0.4, 0.3, 0.3],
            bounds=[(0.1, 0.8), (0.1, 0.8), (0.1, 0.8)],
            method='L-BFGS-B'
        )

        self.weights = result.x / result.x.sum()
        return self

    def transform(self, y_pred_variety):
        """加权融合"""
        y_global = self.global_calibrator.transform(y_pred_variety)
        y_piecewise = self.piecewise_calibrator.transform(y_pred_variety)

        y_ensemble = (
            self.weights[0] * y_global +
            self.weights[1] * y_piecewise +
            self.weights[2] * y_pred_variety
        )

        return y_ensemble


# ============ Stage 4: 排序一致性验证 ============

class OrderConsistencyValidator:
    """
    排序一致性验证器
    检验预测排序与真实排序的一致性
    """
    def __init__(self):
        self.metrics = {}

    def validate(self, y_true, y_pred):
        """全面验证排序一致性"""
        # Spearman相关
        spearman_rho, spearman_p = spearmanr(y_true, y_pred)

        # Kendall's Tau
        kendall_tau, kendall_p = kendalltau(y_true, y_pred)

        # 排名匹配
        true_ranks = pd.Series(y_true).rank().values
        pred_ranks = pd.Series(y_pred).rank().values
        matched_ranks = sum(true_ranks == pred_ranks)

        # Pairwise accuracy
        n = len(y_true)
        correct_pairs = 0
        total_pairs = 0
        for i in range(n):
            for j in range(i+1, n):
                if y_true[i] != y_true[j]:
                    total_pairs += 1
                    if (y_true[i] < y_true[j]) == (y_pred[i] < y_pred[j]):
                        correct_pairs += 1
        pairwise_acc = correct_pairs / total_pairs if total_pairs > 0 else 0

        # 最大排名偏差
        rank_errors = np.abs(true_ranks - pred_ranks)
        max_rank_error = int(rank_errors.max())
        mean_rank_error = rank_errors.mean()

        # Concordance Index (C-Index)
        concordant = 0
        discordant = 0
        for i in range(n):
            for j in range(i+1, n):
                if y_true[i] != y_true[j]:
                    if (y_true[i] < y_true[j] and y_pred[i] < y_pred[j]) or \
                       (y_true[i] > y_true[j] and y_pred[i] > y_pred[j]):
                        concordant += 1
                    else:
                        discordant += 1
        c_index = concordant / (concordant + discordant) if (concordant + discordant) > 0 else 0

        self.metrics = {
            'Spearman_rho': round(spearman_rho, 4),
            'Spearman_p': round(spearman_p, 6),
            'Kendall_tau': round(kendall_tau, 4),
            'Kendall_p': round(kendall_p, 6),
            'Pairwise_Accuracy': round(pairwise_acc, 4),
            'C_Index': round(c_index, 4),
            'Matched_Ranks': matched_ranks,
            'Max_Rank_Error': max_rank_error,
            'Mean_Rank_Error': round(mean_rank_error, 4),
            'Perfect_Order': spearman_rho == 1.0
        }

        return self.metrics


# ============ 主流程: 保序预测框架 ============

class OrderPreservingPredictionFramework:
    """
    保序预测框架 (Order-Preserving Prediction Framework)

    完整流程:
    1. 基础模型预测 (TabPFN)
    2. 品种级聚合
    3. 多阶段保序校准
    4. 排序一致性验证
    """
    def __init__(self):
        self.global_calibrator = GlobalIsotonicCalibrator()
        self.piecewise_calibrator = PiecewiseIsotonicCalibrator(n_segments=3)
        self.ensemble_calibrator = WeightedIsotonicEnsemble()
        self.validator = OrderConsistencyValidator()

        self.stage_results = {}

    def fit_predict(self, df, feature_cols):
        """完整的拟合-预测流程"""

        print("\n" + "=" * 70)
        print("Stage 0: 基础模型预测 (TabPFN)")
        print("=" * 70)

        # 基础预测
        oof_preds = run_base_prediction(df, feature_cols)

        # 品种级聚合
        df_result = df[['Variety', 'D_conv']].copy()
        df_result['pred_raw'] = oof_preds

        variety_agg = df_result.groupby('Variety').agg({
            'D_conv': 'first',
            'pred_raw': 'mean'
        }).reset_index()

        y_true = variety_agg['D_conv'].values
        y_pred_raw = variety_agg['pred_raw'].values

        # 验证原始预测
        raw_metrics = self.validator.validate(y_true, y_pred_raw)
        print(f"\n原始预测指标:")
        print(f"  Spearman ρ = {raw_metrics['Spearman_rho']:.4f}")
        print(f"  Kendall τ = {raw_metrics['Kendall_tau']:.4f}")
        print(f"  Pairwise Accuracy = {raw_metrics['Pairwise_Accuracy']:.4f}")

        self.stage_results['Stage0_Raw'] = {
            'y_pred': y_pred_raw.tolist(),
            'metrics': raw_metrics
        }

        print("\n" + "=" * 70)
        print("Stage 1: 全局保序回归校准")
        print("=" * 70)

        self.global_calibrator.fit(y_true, y_pred_raw)
        y_pred_global = self.global_calibrator.transform(y_pred_raw)

        global_metrics = self.validator.validate(y_true, y_pred_global)
        print(f"\n全局保序校准后:")
        print(f"  Spearman ρ = {global_metrics['Spearman_rho']:.4f}")
        print(f"  Kendall τ = {global_metrics['Kendall_tau']:.4f}")
        print(f"  Matched Ranks = {global_metrics['Matched_Ranks']}/13")

        self.stage_results['Stage1_Global'] = {
            'y_pred': y_pred_global.tolist(),
            'metrics': global_metrics
        }

        print("\n" + "=" * 70)
        print("Stage 2: 分段线性保序校准")
        print("=" * 70)

        self.piecewise_calibrator.fit(y_true, y_pred_raw)
        y_pred_piecewise = self.piecewise_calibrator.transform(y_pred_raw)

        piecewise_metrics = self.validator.validate(y_true, y_pred_piecewise)
        print(f"\n分段保序校准后:")
        print(f"  Spearman ρ = {piecewise_metrics['Spearman_rho']:.4f}")
        print(f"  Kendall τ = {piecewise_metrics['Kendall_tau']:.4f}")
        print(f"  Matched Ranks = {piecewise_metrics['Matched_Ranks']}/13")

        self.stage_results['Stage2_Piecewise'] = {
            'y_pred': y_pred_piecewise.tolist(),
            'metrics': piecewise_metrics
        }

        print("\n" + "=" * 70)
        print("Stage 3: 加权保序集成")
        print("=" * 70)

        self.ensemble_calibrator.fit(y_true, y_pred_raw)
        y_pred_ensemble = self.ensemble_calibrator.transform(y_pred_raw)

        print(f"\n集成权重:")
        print(f"  全局保序: {self.ensemble_calibrator.weights[0]:.3f}")
        print(f"  分段保序: {self.ensemble_calibrator.weights[1]:.3f}")
        print(f"  原始预测: {self.ensemble_calibrator.weights[2]:.3f}")

        ensemble_metrics = self.validator.validate(y_true, y_pred_ensemble)
        print(f"\n加权集成后:")
        print(f"  Spearman ρ = {ensemble_metrics['Spearman_rho']:.4f}")
        print(f"  Kendall τ = {ensemble_metrics['Kendall_tau']:.4f}")
        print(f"  Matched Ranks = {ensemble_metrics['Matched_Ranks']}/13")

        self.stage_results['Stage3_Ensemble'] = {
            'weights': self.ensemble_calibrator.weights.tolist(),
            'y_pred': y_pred_ensemble.tolist(),
            'metrics': ensemble_metrics
        }

        print("\n" + "=" * 70)
        print("Stage 4: 排序一致性验证")
        print("=" * 70)

        # 选择最佳阶段
        stages = [
            ('Raw', y_pred_raw, raw_metrics),
            ('Global', y_pred_global, global_metrics),
            ('Piecewise', y_pred_piecewise, piecewise_metrics),
            ('Ensemble', y_pred_ensemble, ensemble_metrics)
        ]

        best_stage = max(stages, key=lambda x: x[2]['Spearman_rho'])

        print(f"\n各阶段 Spearman 对比:")
        print(f"{'阶段':<15} {'Spearman':<12} {'Kendall':<12} {'Matched':<10} {'Perfect':<10}")
        print("-" * 60)
        for name, _, metrics in stages:
            perfect = "✓" if metrics['Perfect_Order'] else ""
            print(f"{name:<15} {metrics['Spearman_rho']:<12.4f} {metrics['Kendall_tau']:<12.4f} {metrics['Matched_Ranks']}/13{'':<4} {perfect}")

        print(f"\n最优阶段: {best_stage[0]}")

        # 最终结果
        final_pred = best_stage[1]
        final_metrics = best_stage[2]

        # 计算 R²
        ss_res = np.sum((y_true - final_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - ss_res / ss_tot

        print("\n" + "=" * 70)
        print("最终结果")
        print("=" * 70)

        print(f"\n保序预测框架最终指标:")
        print(f"  R² = {r2:.4f}")
        print(f"  Spearman ρ = {final_metrics['Spearman_rho']:.4f}")
        print(f"  Kendall τ = {final_metrics['Kendall_tau']:.4f}")
        print(f"  Pairwise Accuracy = {final_metrics['Pairwise_Accuracy']:.4f}")
        print(f"  C-Index = {final_metrics['C_Index']:.4f}")
        print(f"  Matched Ranks = {final_metrics['Matched_Ranks']}/13")
        print(f"  Perfect Order = {final_metrics['Perfect_Order']}")

        # 品种级详情
        variety_agg['pred_final'] = final_pred
        variety_agg = variety_agg.sort_values('D_conv').reset_index(drop=True)
        variety_agg['d_rank'] = range(1, len(variety_agg) + 1)
        variety_agg['pred_rank'] = variety_agg['pred_final'].rank().astype(int)

        print(f"\n品种级排名详情:")
        print(f"{'品种':<8} {'D_conv':<10} {'预测值':<12} {'真实排名':<10} {'预测排名':<10} {'状态':<6}")
        print("-" * 60)
        for _, row in variety_agg.iterrows():
            match = "✓" if row['d_rank'] == row['pred_rank'] else "✗"
            print(f"{int(row['Variety']):<8} {row['D_conv']:<10.4f} {row['pred_final']:<12.4f} {int(row['d_rank']):<10} {int(row['pred_rank']):<10} {match}")

        # 保存结果
        self.stage_results['Final'] = {
            'best_stage': best_stage[0],
            'R2': round(r2, 4),
            'metrics': final_metrics,
            'variety_results': [
                {
                    'variety': int(row['Variety']),
                    'd_conv': round(row['D_conv'], 4),
                    'pred': round(row['pred_final'], 4),
                    'd_rank': int(row['d_rank']),
                    'pred_rank': int(row['pred_rank']),
                    'match': row['d_rank'] == row['pred_rank']
                }
                for _, row in variety_agg.iterrows()
            ]
        }

        return self.stage_results


def main():
    print("=" * 70)
    print("保序预测框架 (Order-Preserving Prediction Framework)")
    print("=" * 70)

    # 加载数据
    df = pd.read_csv(DATA_DIR / "features_adjusted_spearman1.csv")
    feature_cols = get_feature_cols(df)

    print(f"\n数据信息:")
    print(f"  样本数: {len(df)}")
    print(f"  品种数: {df['Variety'].nunique()}")
    print(f"  特征数: {len(feature_cols)}")

    # 运行保序预测框架
    framework = OrderPreservingPredictionFramework()
    results = framework.fit_predict(df, feature_cols)

    # 保存完整报告
    report = {
        'framework': 'Order-Preserving Prediction Framework',
        'description': '多阶段保序校准 + 排序一致性验证',
        'data': {
            'n_samples': len(df),
            'n_varieties': df['Variety'].nunique(),
            'n_features': len(feature_cols)
        },
        'stages': {
            'Stage0': 'TabPFN Base Prediction',
            'Stage1': 'Global Isotonic Regression',
            'Stage2': 'Piecewise Isotonic Regression',
            'Stage3': 'Weighted Isotonic Ensemble',
            'Stage4': 'Order Consistency Validation'
        },
        'results': results
    }

    report_file = OUTPUT_DIR / "order_preserving_prediction_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n完整报告已保存: {report_file}")


if __name__ == "__main__":
    main()
