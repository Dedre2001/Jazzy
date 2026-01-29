"""
增强版诊断脚本
A/B测试框架：对比不同改进方案对GroupKFold性能的影响

方案对比:
- Baseline: 原始FS4特征
- A: +光谱导数特征
- B: +样本加权
- C: 导数替代原始光谱
- D: A+B组合
"""

import os
import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# 设置环境变量
os.environ.setdefault("TABPFN_MODEL_CACHE_DIR", str(Path(__file__).resolve().parent.parent / "tabpfn_ckpt"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

try:
    from tabpfn import TabPFNRegressor
    TABPFN_AVAILABLE = True
except ImportError:
    TABPFN_AVAILABLE = False
    print("[WARN] TabPFN 未安装，将使用 Ridge 替代")

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from utils_enhanced import (
    load_data, get_variety_metrics, compute_sample_weights,
    run_group_kfold_cv, run_random_kfold_cv, sample_level_normalize,
    RESULTS_DIR
)

BASE_DIR = Path(__file__).resolve().parent


def get_model():
    """获取模型实例"""
    if TABPFN_AVAILABLE:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return TabPFNRegressor, {
            "n_estimators": 256,
            "random_state": 42,
            "fit_mode": "fit_preprocessors",
            "device": device,
            "average_before_softmax": True,
            "softmax_temperature": 0.75,
            "memory_saving_mode": "auto"
        }
    else:
        return Ridge, {"alpha": 1.0, "random_state": 42}


def run_experiment(name, df, feature_cols, sample_weight=None, use_sample_norm=True):
    """运行单个实验"""
    print(f"\n  [{name}]")

    model_class, model_params = get_model()

    # GroupKFold
    group_metrics, group_agg, _ = run_group_kfold_cv(
        df, feature_cols, model_class, model_params,
        sample_weight=sample_weight,
        use_sample_norm=use_sample_norm
    )

    # Random KFold (用于对比)
    random_metrics, random_agg, _ = run_random_kfold_cv(
        df, feature_cols, model_class, model_params,
        use_sample_norm=use_sample_norm
    )

    print(f"    GroupKFold R²: {group_metrics['R2']:.4f}")
    print(f"    Random KFold R²: {random_metrics['R2']:.4f}")
    print(f"    Gap: {random_metrics['R2'] - group_metrics['R2']:.4f}")

    # 难预测品种分析
    group_agg['error'] = np.abs(group_agg['D_conv'] - group_agg['pred'])
    worst = group_agg.nlargest(3, 'error')

    return {
        'name': name,
        'n_features': len(feature_cols),
        'group_metrics': group_metrics,
        'random_metrics': random_metrics,
        'gap': random_metrics['R2'] - group_metrics['R2'],
        'worst_varieties': worst[['Variety', 'D_conv', 'pred', 'error']].to_dict('records'),
        'variety_results': group_agg
    }


def analyze_difficult_varieties(results):
    """分析难预测品种"""
    print("\n" + "=" * 60)
    print("难预测品种分析")
    print("=" * 60)

    # 收集所有方案的品种误差
    variety_errors = {}

    for result in results:
        name = result['name']
        for row in result['worst_varieties']:
            variety = row['Variety']
            if variety not in variety_errors:
                variety_errors[variety] = {}
            variety_errors[variety][name] = row['error']

    # 找出在多个方案中都表现差的品种
    print("\n各方案难预测品种误差对比:")
    print("-" * 60)

    header = "品种".ljust(10) + "".join([r['name'][:12].ljust(14) for r in results])
    print(header)
    print("-" * 60)

    for variety in variety_errors:
        row = str(variety).ljust(10)
        for result in results:
            err = variety_errors[variety].get(result['name'], '-')
            if isinstance(err, float):
                row += f"{err:.4f}".ljust(14)
            else:
                row += "-".ljust(14)
        print(row)


def main():
    print("=" * 60)
    print("GroupKFold 性能提升实验")
    print("=" * 60)

    # 1. 首先运行特征工程生成增强特征
    print("\n[Step 1] 生成增强特征...")
    import feature_engineering_enhanced
    feature_engineering_enhanced.main()

    # 2. 加载数据
    print("\n[Step 2] 加载数据...")
    df, feature_sets = load_data(use_enhanced=True)
    print(f"  样本数: {len(df)}")
    print(f"  品种数: {df['Variety'].nunique()}")

    # 3. 定义实验方案
    print("\n[Step 3] 运行A/B测试...")
    print("=" * 60)

    results = []

    # Baseline: 原始FS4
    result_baseline = run_experiment(
        "Baseline",
        df,
        feature_sets['FS4']['features'],
        sample_weight=None,
        use_sample_norm=True
    )
    results.append(result_baseline)

    # 方案A: +光谱导数
    result_a = run_experiment(
        "A: +导数",
        df,
        feature_sets['FS4_enhanced']['features'],
        sample_weight=None,
        use_sample_norm=True
    )
    results.append(result_a)

    # 方案B: +样本加权
    weights = compute_sample_weights(df['D_conv'].values, q_low=0.1, q_high=0.9, boost=2.0)
    result_b = run_experiment(
        "B: +加权",
        df,
        feature_sets['FS4']['features'],
        sample_weight=weights,
        use_sample_norm=True
    )
    results.append(result_b)

    # 方案C: 导数替代原始光谱
    result_c = run_experiment(
        "C: 导数替代",
        df,
        feature_sets['FS4_derivatives']['features'],
        sample_weight=None,
        use_sample_norm=True
    )
    results.append(result_c)

    # 方案D: A+B组合
    result_d = run_experiment(
        "D: 导数+加权",
        df,
        feature_sets['FS4_enhanced']['features'],
        sample_weight=weights,
        use_sample_norm=True
    )
    results.append(result_d)

    # 4. 汇总结果
    print("\n" + "=" * 60)
    print("实验结果汇总")
    print("=" * 60)

    print("\n方案对比:")
    print("-" * 60)
    print(f"{'方案':<15} {'特征数':>8} {'GroupKFold R²':>15} {'Random R²':>12} {'Gap':>8}")
    print("-" * 60)

    for r in results:
        print(f"{r['name']:<15} {r['n_features']:>8} {r['group_metrics']['R2']:>15.4f} "
              f"{r['random_metrics']['R2']:>12.4f} {r['gap']:>8.4f}")

    # 找出最佳方案
    best = max(results, key=lambda x: x['group_metrics']['R2'])
    baseline_r2 = results[0]['group_metrics']['R2']
    improvement = best['group_metrics']['R2'] - baseline_r2

    print("-" * 60)
    print(f"\n最佳方案: {best['name']}")
    print(f"  GroupKFold R²: {best['group_metrics']['R2']:.4f}")
    print(f"  相比Baseline提升: {improvement:+.4f}")

    # 5. 难预测品种分析
    analyze_difficult_varieties(results)

    # 6. 保存结果
    report = {
        "experiments": [
            {
                "name": r['name'],
                "n_features": r['n_features'],
                "group_r2": r['group_metrics']['R2'],
                "random_r2": r['random_metrics']['R2'],
                "gap": r['gap'],
                "group_metrics": r['group_metrics'],
                "worst_varieties": r['worst_varieties']
            }
            for r in results
        ],
        "best_experiment": best['name'],
        "baseline_r2": baseline_r2,
        "best_r2": best['group_metrics']['R2'],
        "improvement": improvement
    }

    report_path = RESULTS_DIR / "ab_test_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n报告已保存: {report_path}")

    # 7. 可视化
    if MATPLOTLIB_AVAILABLE:
        visualize_results(results)

    return results


def visualize_results(results):
    """生成可视化图表"""
    print("\n生成可视化...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. R²对比柱状图
    ax1 = axes[0]
    names = [r['name'] for r in results]
    group_r2 = [r['group_metrics']['R2'] for r in results]
    random_r2 = [r['random_metrics']['R2'] for r in results]

    x = np.arange(len(names))
    width = 0.35

    bars1 = ax1.bar(x - width/2, group_r2, width, label='GroupKFold', color='#2ca02c')
    bars2 = ax1.bar(x + width/2, random_r2, width, label='Random KFold', color='#ff7f0e', alpha=0.7)

    ax1.set_ylabel('R² Score')
    ax1.set_title('各方案性能对比')
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=15)
    ax1.legend()
    ax1.set_ylim(0, 1.0)

    # 标注数值
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    # 2. Gap分析
    ax2 = axes[1]
    gaps = [r['gap'] for r in results]
    colors = ['#d62728' if g > 0.15 else '#2ca02c' for g in gaps]

    bars = ax2.bar(names, gaps, color=colors, alpha=0.7)
    ax2.axhline(y=0.15, color='red', linestyle='--', label='泄露阈值 (0.15)')
    ax2.set_ylabel('Random - Group Gap')
    ax2.set_title('数据泄露程度分析')
    ax2.set_xticklabels(names, rotation=15)
    ax2.legend()

    plt.tight_layout()
    save_path = RESULTS_DIR / "ab_test_results.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"可视化已保存: {save_path}")


if __name__ == "__main__":
    results = main()
