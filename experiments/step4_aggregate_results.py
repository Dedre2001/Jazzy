"""
Step 4: 汇总所有模型结果，生成报告
"""
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt

# 路径配置
BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
EXP6_DIR = RESULTS_DIR / "exp6"
TABLE_DIR = RESULTS_DIR / "tables"
FIG_DIR = RESULTS_DIR / "figures"
REPORT_DIR = RESULTS_DIR / "reports"

for d in [TABLE_DIR, FIG_DIR, REPORT_DIR]:
    os.makedirs(d, exist_ok=True)

# 图表风格
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['SimHei', 'Microsoft YaHei', 'Arial'],
    'font.size': 11,
    'axes.unicode_minus': False,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 300,
})

MODELS = ['PLSR', 'SVR', 'Ridge', 'RF', 'CatBoost', 'TabPFN']

def load_all_results():
    """加载所有模型结果"""
    results = []
    for model in MODELS:
        json_path = f"{EXP6_DIR}/model_{model.lower()}_results.json"
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                row = {'Model': data['model'], 'Layer': data['layer']}
                row.update(data['metrics'])
                results.append(row)
                print(f"[OK] 加载 {model}")
        else:
            print(f"[WARN] 未找到 {model} 结果: {json_path}")
    return pd.DataFrame(results)

def generate_tables(df):
    """生成表格"""
    # 表1: 主结果表
    table1 = df[['Model', 'Layer', 'R2', 'RMSE', 'Spearman', 'Pairwise_Acc', 'Hit@3', 'Hit@5']].copy()
    table1.to_csv(f"{TABLE_DIR}/exp6_table1_main_results.csv", index=False, encoding='utf-8-sig')
    print(f"[OK] 表1已保存: {TABLE_DIR}/exp6_table1_main_results.csv")

    # 表2: 排名表
    rank_cols = ['R2', 'Spearman', 'Pairwise_Acc', 'Hit@3']
    table2 = df[['Model']].copy()
    for col in rank_cols:
        table2[f'{col}_Rank'] = df[col].rank(ascending=False).astype(int)
    table2['Avg_Rank'] = table2[[f'{c}_Rank' for c in rank_cols]].mean(axis=1)
    table2 = table2.sort_values('Avg_Rank')
    table2.to_csv(f"{TABLE_DIR}/exp6_table2_rankings.csv", index=False, encoding='utf-8-sig')
    print(f"[OK] 表2已保存: {TABLE_DIR}/exp6_table2_rankings.csv")

    return table1, table2

def generate_figures(df):
    """生成可视化图表"""
    colors = {'Layer1': '#66c2a5', 'Layer2': '#fc8d62', 'Layer3': '#8da0cb'}

    # 图1: 模型性能对比 (Spearman)
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(df['Model'], df['Spearman'],
                  color=[colors[l] for l in df['Layer']],
                  edgecolor='black', linewidth=1.2)
    ax.set_ylabel('Spearman rho', fontsize=12)
    ax.set_xlabel('Model', fontsize=12)
    ax.set_title('Model Comparison: Spearman Correlation', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.1)
    for bar, val in zip(bars, df['Spearman']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[l], label=l) for l in ['Layer1', 'Layer2', 'Layer3']]
    ax.legend(handles=legend_elements, loc='upper left')
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/exp6_fig1_model_comparison.png", dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[OK] 图1已保存: {FIG_DIR}/exp6_fig1_model_comparison.png")

    # 图2: 层级对比
    layer_stats = df.groupby('Layer')['Spearman'].mean().reset_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(layer_stats['Layer'], layer_stats['Spearman'],
                  color=[colors[l] for l in layer_stats['Layer']],
                  edgecolor='black', linewidth=1.2)
    ax.set_ylabel('Average Spearman rho', fontsize=12)
    ax.set_xlabel('Model Layer', fontsize=12)
    ax.set_title('Layer-wise Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.1)
    for bar, val in zip(bars, layer_stats['Spearman']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/exp6_fig2_layer_comparison.png", dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[OK] 图2已保存: {FIG_DIR}/exp6_fig2_layer_comparison.png")

    # 图3: 雷达图 (Top 3 模型)
    top3 = df.nlargest(3, 'Spearman')
    categories = ['R2', 'Spearman', 'Pairwise_Acc', 'Hit@3', 'Hit@5']
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    for idx, row in top3.iterrows():
        values = [row[c] for c in categories]
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=row['Model'])
        ax.fill(angles, values, alpha=0.25)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.set_title('Top-3 Models Radar Chart', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/exp6_fig3_radar.png", dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[OK] 图3已保存: {FIG_DIR}/exp6_fig3_radar.png")

def generate_report(df, table2):
    """生成中文报告"""
    best_model = df.loc[df['Spearman'].idxmax()]
    worst_model = df.loc[df['Spearman'].idxmin()]

    report = f"""# Exp-6: 模型对比实验报告

**报告生成时间:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**随机种子:** 42
**交叉验证:** 5折 KFold
**特征集:** FS4 (三源融合, 40个特征)

---

## 摘要

本实验对比了6个模型在水稻抗旱性预测任务上的表现。结果显示，**{best_model['Model']}** 取得了最佳性能（Spearman rho = {best_model['Spearman']:.3f}），较表现最差的 {worst_model['Model']}（rho = {worst_model['Spearman']:.3f}）提升了 {(best_model['Spearman'] - worst_model['Spearman'])*100:.1f}%。

### 主要发现

1. **最优模型:** {best_model['Model']}，Spearman rho = {best_model['Spearman']:.3f}
2. **Top-3命中率:** {best_model['Hit@3']:.2f}
3. **配对排序准确率:** {best_model['Pairwise_Acc']:.3f}

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
"""

    for _, row in df.iterrows():
        highlight = "**" if row['Model'] == best_model['Model'] else ""
        report += f"| {highlight}{row['Model']}{highlight} | {row['Layer']} | {row['R2']:.3f} | {row['RMSE']:.3f} | {row['Spearman']:.3f} | {row['Pairwise_Acc']:.3f} | {row['Hit@3']:.2f} | {row['Hit@5']:.2f} |\n"

    # 层级对比
    layer_stats = df.groupby('Layer')['Spearman'].mean()
    report += f"""
### 2.2 层级对比分析

| 层级 | 描述 | 模型 | 平均Spearman rho |
|------|------|------|------------------|
| Layer1 | 经典农学模型 | PLSR, SVR | {layer_stats.get('Layer1', 0):.3f} |
| Layer2 | 传统机器学习 | Ridge, RF, CatBoost | {layer_stats.get('Layer2', 0):.3f} |
| Layer3 | 新型神经网络 | TabPFN | {layer_stats.get('Layer3', 0):.3f} |

---

## 3. 讨论

### 3.1 模型排名

{table2.to_markdown(index=False)}

### 3.2 结论

1. **{best_model['Model']}** 在水稻抗旱性预测任务中表现最优
2. 三源融合特征集(FS4)为所有模型提供了有效的预测基础
3. Hit@3达到{best_model['Hit@3']:.0%}，支持育种初筛应用

---

## 4. 输出文件（可追溯性）

### 4.1 数据表格

| 编号 | 描述 | 文件路径 |
|------|------|----------|
| 表1 | 模型性能对比主表 | `{TABLE_DIR}/exp6_table1_main_results.csv` |
| 表2 | 各指标模型排名 | `{TABLE_DIR}/exp6_table2_rankings.csv` |

### 4.2 图表

| 编号 | 描述 | 文件路径 |
|------|------|----------|
| 图1 | 模型性能对比 | `{FIG_DIR}/exp6_fig1_model_comparison.png` |
| 图2 | 模型层级性能对比 | `{FIG_DIR}/exp6_fig2_layer_comparison.png` |
| 图3 | Top-3模型雷达图 | `{FIG_DIR}/exp6_fig3_radar.png` |

---

*报告由 step4_aggregate_results.py 自动生成*
*数据来源: data/processed/features_40.csv*
"""

    report_path = f"{REPORT_DIR}/exp6_model_comparison_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"[OK] 报告已保存: {report_path}")

def main():
    print("="*60)
    print("Step 4: 汇总模型对比结果")
    print("="*60)

    # 加载结果
    df = load_all_results()

    if len(df) == 0:
        print("[ERROR] 没有找到任何模型结果，请先运行各模型脚本")
        return

    print(f"\n已加载 {len(df)} 个模型结果")

    # 生成表格
    table1, table2 = generate_tables(df)

    # 生成图表
    generate_figures(df)

    # 生成报告
    generate_report(df, table2)

    print("\n" + "="*60)
    print("汇总完成!")
    print("="*60)

if __name__ == "__main__":
    main()
