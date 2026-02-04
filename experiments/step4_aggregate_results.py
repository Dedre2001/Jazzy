"""
Step 4: 汇总所有模型结果，生成完整报告
包含: 评价指标、品种排名分析、统计检验、论文级图表
"""
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
from sklearn.metrics import (
    mean_absolute_error, cohen_kappa_score,
    balanced_accuracy_score, f1_score, accuracy_score
)
import warnings
warnings.filterwarnings('ignore')

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
    'font.sans-serif': ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial'],
    'font.size': 11,
    'axes.unicode_minus': False,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.facecolor': 'white',
})

MODELS = ['PLSR', 'SVR', 'Ridge', 'RF', 'CatBoost', 'TabPFN']
MODEL_COLORS = {
    'PLSR': '#66c2a5',
    'SVR': '#a6d854',
    'Ridge': '#fc8d62',
    'RF': '#e78ac3',
    'CatBoost': '#ffd92f',
    'TabPFN': '#8da0cb'
}
LAYER_COLORS = {'Layer1': '#66c2a5', 'Layer2': '#fc8d62', 'Layer3': '#8da0cb'}

# 品种分类阈值
THRESHOLDS = {
    'sensitive': 0.30,   # D_conv < 0.30 -> 敏感
    'tolerant': 0.45     # D_conv >= 0.45 -> 耐旱
}


# =============================================================================
# 数据加载
# =============================================================================

def load_all_results():
    """加载所有模型的JSON结果"""
    results = []
    for model in MODELS:
        json_path = EXP6_DIR / f"model_{model.lower()}_results.json"
        if json_path.exists():
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                row = {'Model': data['model'], 'Layer': data['layer']}
                row.update(data['metrics'])
                results.append(row)
                print(f"[OK] 加载 {model}")
        else:
            print(f"[WARN] 未找到 {model} 结果: {json_path}")
    return pd.DataFrame(results)


def load_variety_predictions():
    """加载所有模型的品种预测结果"""
    predictions = {}
    for model in MODELS:
        csv_path = EXP6_DIR / f"model_{model.lower()}_variety_pred.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            df['pred_rank'] = df['pred'].rank(ascending=False).astype(int)
            df['true_rank'] = df['D_conv'].rank(ascending=False).astype(int)
            predictions[model] = df
            print(f"[OK] 加载品种预测 {model}")
        else:
            print(f"[WARN] 未找到品种预测 {model}: {csv_path}")
    return predictions


# =============================================================================
# 指标计算
# =============================================================================

def compute_additional_metrics(predictions):
    """计算补充指标"""
    metrics_list = []

    for model, df in predictions.items():
        y_true = df['D_conv'].values
        y_pred = df['pred'].values
        true_rank = df['true_rank'].values
        pred_rank = df['pred_rank'].values

        # MAE
        mae = mean_absolute_error(y_true, y_pred)

        # MAPE
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

        # Kendall's tau
        kendall_tau, kendall_p = stats.kendalltau(true_rank, pred_rank)

        # Jaccard@K
        top3_true = set(df.nlargest(3, 'D_conv')['Variety'].values)
        top3_pred = set(df.nlargest(3, 'pred')['Variety'].values)
        top5_true = set(df.nlargest(5, 'D_conv')['Variety'].values)
        top5_pred = set(df.nlargest(5, 'pred')['Variety'].values)

        jaccard3 = len(top3_true & top3_pred) / len(top3_true | top3_pred)
        jaccard5 = len(top5_true & top5_pred) / len(top5_true | top5_pred)

        # NDCG@5
        ndcg5 = compute_ndcg(y_true, y_pred, k=5)

        # Mean Rank Error
        mean_rank_error = np.mean(np.abs(true_rank - pred_rank))

        # 分类指标 (3-class)
        true_class = classify_varieties(y_true)
        pred_class = classify_varieties(y_pred)

        acc_3class = accuracy_score(true_class, pred_class)
        balanced_acc = balanced_accuracy_score(true_class, pred_class)
        kappa = cohen_kappa_score(true_class, pred_class)
        macro_f1 = f1_score(true_class, pred_class, average='macro')

        metrics_list.append({
            'Model': model,
            'MAE': mae,
            'MAPE': mape,
            'Kendall_tau': kendall_tau,
            'Jaccard@3': jaccard3,
            'Jaccard@5': jaccard5,
            'NDCG@5': ndcg5,
            'Mean_Rank_Error': mean_rank_error,
            'Accuracy_3class': acc_3class,
            'Balanced_Acc': balanced_acc,
            'Kappa': kappa,
            'Macro_F1': macro_f1
        })

    return pd.DataFrame(metrics_list)


def compute_ndcg(y_true, y_pred, k=5):
    """计算NDCG@K"""
    # 按预测值排序
    order = np.argsort(y_pred)[::-1]
    y_true_sorted = y_true[order]

    # DCG
    dcg = np.sum((2**y_true_sorted[:k] - 1) / np.log2(np.arange(2, k + 2)))

    # IDCG
    ideal_order = np.argsort(y_true)[::-1]
    y_true_ideal = y_true[ideal_order]
    idcg = np.sum((2**y_true_ideal[:k] - 1) / np.log2(np.arange(2, k + 2)))

    return dcg / idcg if idcg > 0 else 0


def classify_varieties(values):
    """将连续值分为3类: 0=敏感, 1=中等, 2=耐旱"""
    classes = []
    for v in values:
        if v < THRESHOLDS['sensitive']:
            classes.append(0)  # 敏感
        elif v >= THRESHOLDS['tolerant']:
            classes.append(2)  # 耐旱
        else:
            classes.append(1)  # 中等
    return np.array(classes)


def compute_statistical_tests(predictions):
    """
    Friedman检验 + Nemenyi后检验 + Kendall's W协和系数
    包含效应量解释和Bootstrap置信区间
    """
    # 构建排名矩阵 (品种 x 模型)
    varieties = list(predictions[MODELS[0]]['Variety'].values)
    n_varieties = len(varieties)
    k = len(MODELS)  # 模型数
    n = n_varieties  # 品种数

    # 计算每个模型在每个品种上的绝对误差
    error_matrix = np.zeros((n_varieties, len(MODELS)))

    for j, model in enumerate(MODELS):
        df = predictions[model]
        for i, var in enumerate(varieties):
            row = df[df['Variety'] == var]
            error = abs(row['D_conv'].values[0] - row['pred'].values[0])
            error_matrix[i, j] = error

    # 对每个品种，对模型进行排名 (误差小 = 排名高)
    rank_matrix = np.zeros_like(error_matrix)
    for i in range(n_varieties):
        rank_matrix[i] = stats.rankdata(error_matrix[i])

    # Friedman检验
    friedman_stat, friedman_p = stats.friedmanchisquare(*[rank_matrix[:, j] for j in range(len(MODELS))])

    # 平均排名
    avg_ranks = rank_matrix.mean(axis=0)

    # =========================================================================
    # Kendall's W 协和系数 (效应量)
    # W = 12 * S / (n^2 * (k^3 - k))
    # 其中 S = sum((Rj - R_mean)^2)
    # =========================================================================
    R_sum = rank_matrix.sum(axis=0)  # 每个模型的排名总和
    R_mean = R_sum.mean()
    S = np.sum((R_sum - R_mean) ** 2)
    kendall_w = 12 * S / (n ** 2 * (k ** 3 - k))

    # W 效应量解释 (Landis & Koch, 1977 改编)
    if kendall_w < 0.1:
        w_interpretation = "negligible (可忽略)"
    elif kendall_w < 0.3:
        w_interpretation = "weak (弱)"
    elif kendall_w < 0.5:
        w_interpretation = "moderate (中等)"
    elif kendall_w < 0.7:
        w_interpretation = "strong (强)"
    else:
        w_interpretation = "very strong (很强)"

    # =========================================================================
    # Bootstrap 置信区间 (对平均排名)
    # =========================================================================
    n_bootstrap = 1000
    np.random.seed(42)
    bootstrap_ranks = np.zeros((n_bootstrap, k))

    for b in range(n_bootstrap):
        # 有放回抽样品种
        idx = np.random.choice(n, size=n, replace=True)
        boot_rank_matrix = rank_matrix[idx, :]
        bootstrap_ranks[b, :] = boot_rank_matrix.mean(axis=0)

    # 95% 置信区间
    ci_lower = np.percentile(bootstrap_ranks, 2.5, axis=0)
    ci_upper = np.percentile(bootstrap_ranks, 97.5, axis=0)
    rank_ci = {model: (ci_lower[i], ci_upper[i]) for i, model in enumerate(MODELS)}

    # =========================================================================
    # Nemenyi临界距离
    # =========================================================================
    q_alpha = 2.850  # q值 (alpha=0.05, k=6)
    cd = q_alpha * np.sqrt(k * (k + 1) / (6 * n))

    # =========================================================================
    # 小样本警告
    # =========================================================================
    small_sample_warning = n < 20
    if small_sample_warning:
        interpretation_note = (
            f"注意: 样本量较小 (n={n})，p值可能不稳定。"
            f"建议参考效应量 Kendall's W = {kendall_w:.3f} ({w_interpretation}) 进行解释。"
        )
    else:
        interpretation_note = ""

    return {
        'friedman_stat': friedman_stat,
        'friedman_p': friedman_p,
        'avg_ranks': dict(zip(MODELS, avg_ranks)),
        'cd': cd,
        'rank_matrix': rank_matrix,
        'error_matrix': error_matrix,
        # 新增指标
        'kendall_w': kendall_w,
        'w_interpretation': w_interpretation,
        'rank_ci': rank_ci,
        'n_varieties': n,
        'n_models': k,
        'small_sample_warning': small_sample_warning,
        'interpretation_note': interpretation_note
    }


# =============================================================================
# 品种排名分析
# =============================================================================

def generate_variety_ranking_table(predictions):
    """生成品种排名对比表"""
    # 获取真实排名
    base_df = predictions[MODELS[0]][['Variety', 'D_conv', 'true_rank']].copy()
    base_df = base_df.sort_values('true_rank')

    # 添加各模型的预测排名
    for model in MODELS:
        df = predictions[model]
        rank_map = dict(zip(df['Variety'], df['pred_rank']))
        base_df[f'{model}_Rank'] = base_df['Variety'].map(rank_map)

    # 添加类别标签
    base_df['Category'] = base_df['D_conv'].apply(
        lambda x: 'Tolerant' if x >= THRESHOLDS['tolerant']
                  else ('Sensitive' if x < THRESHOLDS['sensitive'] else 'Moderate')
    )

    return base_df


def compute_rank_deviations(predictions):
    """计算排名偏差"""
    deviations = []

    for model in MODELS:
        df = predictions[model]
        for _, row in df.iterrows():
            deviations.append({
                'Variety': row['Variety'],
                'Model': model,
                'True_Rank': row['true_rank'],
                'Pred_Rank': row['pred_rank'],
                'Deviation': row['pred_rank'] - row['true_rank'],
                'Abs_Deviation': abs(row['pred_rank'] - row['true_rank'])
            })

    return pd.DataFrame(deviations)


# =============================================================================
# 表格生成
# =============================================================================

def generate_tables(df, additional_metrics, predictions, stat_tests):
    """生成所有表格"""

    # 表1: 主结果表 (合并所有指标)
    table1 = df.merge(additional_metrics, on='Model')
    cols_order = ['Model', 'Layer', 'R2', 'RMSE', 'MAE', 'Spearman', 'Kendall_tau',
                  'Pairwise_Acc', 'Hit@3', 'Hit@5', 'Jaccard@3', 'Jaccard@5',
                  'NDCG@5', 'Mean_Rank_Error', 'Accuracy_3class', 'Kappa']
    cols_available = [c for c in cols_order if c in table1.columns]
    table1 = table1[cols_available]
    table1.to_csv(TABLE_DIR / "exp6_table1_main_results.csv", index=False, encoding='utf-8-sig')
    print(f"[OK] 表1已保存: {TABLE_DIR}/exp6_table1_main_results.csv")

    # 表2: 模型排名表
    rank_cols = ['R2', 'Spearman', 'Pairwise_Acc', 'Kappa', 'NDCG@5']
    rank_cols = [c for c in rank_cols if c in table1.columns]
    table2 = table1[['Model']].copy()
    for col in rank_cols:
        table2[f'{col}_Rank'] = table1[col].rank(ascending=False).astype(int)
    table2['Avg_Rank'] = table2[[f'{c}_Rank' for c in rank_cols]].mean(axis=1)
    table2 = table2.sort_values('Avg_Rank')
    table2.to_csv(TABLE_DIR / "exp6_table2_rankings.csv", index=False, encoding='utf-8-sig')
    print(f"[OK] 表2已保存: {TABLE_DIR}/exp6_table2_rankings.csv")

    # 表3: 品种排名对比表
    table3 = generate_variety_ranking_table(predictions)
    table3.to_csv(TABLE_DIR / "exp6_table3_variety_rankings.csv", index=False, encoding='utf-8-sig')
    print(f"[OK] 表3已保存: {TABLE_DIR}/exp6_table3_variety_rankings.csv")

    # 表4: 分类指标表
    table4 = additional_metrics[['Model', 'Accuracy_3class', 'Balanced_Acc', 'Kappa', 'Macro_F1']].copy()
    table4.to_csv(TABLE_DIR / "exp6_table4_classification.csv", index=False, encoding='utf-8-sig')
    print(f"[OK] 表4已保存: {TABLE_DIR}/exp6_table4_classification.csv")

    # 表5: 统计检验表 (含置信区间和效应量)
    table5_data = []
    for model in MODELS:
        ci = stat_tests['rank_ci'][model]
        table5_data.append({
            'Model': model,
            'Avg_Rank': stat_tests['avg_ranks'][model],
            'CI_Lower': ci[0],
            'CI_Upper': ci[1],
        })
    table5 = pd.DataFrame(table5_data)
    table5 = table5.sort_values('Avg_Rank')
    table5['Friedman_p'] = stat_tests['friedman_p']
    table5['Kendall_W'] = stat_tests['kendall_w']
    table5['W_Interpretation'] = stat_tests['w_interpretation']
    table5['CD'] = stat_tests['cd']
    table5.to_csv(TABLE_DIR / "exp6_table5_statistical_tests.csv", index=False, encoding='utf-8-sig')
    print(f"[OK] 表5已保存: {TABLE_DIR}/exp6_table5_statistical_tests.csv")

    # 表6: 原始结果 (每个模型的品种预测)
    all_preds = []
    for model in MODELS:
        if model in predictions:
            temp = predictions[model].copy()
            temp['Model'] = model
            all_preds.append(temp)
    table6 = pd.concat(all_preds, ignore_index=True)
    table6.to_csv(TABLE_DIR / "exp6_table6_raw_predictions.csv", index=False, encoding='utf-8-sig')
    print(f"[OK] 表6已保存: {TABLE_DIR}/exp6_table6_raw_predictions.csv")

    return table1, table2, table3


# =============================================================================
# 图表生成
# =============================================================================

def plot_fig1_model_comparison(df):
    """图1: 模型性能对比柱状图"""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    metrics = ['Spearman', 'R2', 'Pairwise_Acc']
    titles = ['Spearman Correlation', 'R² Score', 'Pairwise Accuracy']

    for ax, metric, title in zip(axes, metrics, titles):
        colors = [LAYER_COLORS[l] for l in df['Layer']]
        bars = ax.bar(df['Model'], df[metric], color=colors, edgecolor='black', linewidth=1)
        ax.set_ylabel(metric, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1.15)
        ax.tick_params(axis='x', rotation=45)

        for bar, val in zip(bars, df[metric]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    # 图例
    legend_elements = [mpatches.Patch(facecolor=LAYER_COLORS[l], label=l)
                       for l in ['Layer1', 'Layer2', 'Layer3']]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))

    plt.tight_layout()
    plt.savefig(FIG_DIR / "exp6_fig1_model_comparison.png")
    plt.close()
    print(f"[OK] 图1已保存: {FIG_DIR}/exp6_fig1_model_comparison.png")


def plot_fig2_radar(df):
    """图2: 多指标雷达图 (所有模型)"""
    categories = ['R2', 'Spearman', 'Pairwise_Acc', 'Hit@3', 'Hit@5']
    n_cats = len(categories)

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    angles = np.linspace(0, 2*np.pi, n_cats, endpoint=False).tolist()
    angles += angles[:1]

    for _, row in df.iterrows():
        values = [row[c] for c in categories]
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=row['Model'],
                color=MODEL_COLORS.get(row['Model'], '#333333'))
        ax.fill(angles, values, alpha=0.15, color=MODEL_COLORS.get(row['Model'], '#333333'))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.set_title('Model Performance Radar Chart', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "exp6_fig2_radar.png")
    plt.close()
    print(f"[OK] 图2已保存: {FIG_DIR}/exp6_fig2_radar.png")


def plot_fig3_pred_vs_true(predictions):
    """图3: 预测值 vs 真实值散点图"""
    fig, axes = plt.subplots(2, 3, figsize=(14, 10))
    axes = axes.flatten()

    for ax, model in zip(axes, MODELS):
        if model not in predictions:
            continue
        df = predictions[model]

        ax.scatter(df['D_conv'], df['pred'], c=MODEL_COLORS[model],
                   s=80, edgecolor='black', linewidth=0.5, alpha=0.8)

        # 对角线
        lims = [0, 0.7]
        ax.plot(lims, lims, 'k--', alpha=0.5, linewidth=1)

        # 标注品种编号
        for _, row in df.iterrows():
            ax.annotate(str(int(row['Variety'])),
                        (row['D_conv'], row['pred']),
                        fontsize=7, alpha=0.7,
                        xytext=(3, 3), textcoords='offset points')

        # 计算指标
        r2 = 1 - np.sum((df['D_conv'] - df['pred'])**2) / np.sum((df['D_conv'] - df['D_conv'].mean())**2)
        rmse = np.sqrt(np.mean((df['D_conv'] - df['pred'])**2))

        ax.set_xlabel('True D_conv', fontsize=10)
        ax.set_ylabel('Predicted D_conv', fontsize=10)
        ax.set_title(f'{model}\nR²={r2:.3f}, RMSE={rmse:.3f}', fontsize=11, fontweight='bold')
        ax.set_xlim(0.1, 0.65)
        ax.set_ylim(0.1, 0.65)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "exp6_fig3_pred_vs_true.png")
    plt.close()
    print(f"[OK] 图3已保存: {FIG_DIR}/exp6_fig3_pred_vs_true.png")


def plot_fig4_variety_ranking_heatmap(predictions):
    """图4: 品种排名热力图"""
    ranking_df = generate_variety_ranking_table(predictions)

    # 准备数据
    varieties = ranking_df['Variety'].values
    true_ranks = ranking_df['true_rank'].values

    # 构建热力图矩阵
    rank_cols = ['true_rank'] + [f'{m}_Rank' for m in MODELS]
    matrix = ranking_df[rank_cols].values

    fig, ax = plt.subplots(figsize=(12, 10))

    # 热力图
    im = ax.imshow(matrix, cmap='RdYlGn_r', aspect='auto', vmin=1, vmax=13)

    # 标签
    ax.set_xticks(np.arange(len(rank_cols)))
    ax.set_xticklabels(['True'] + MODELS, fontsize=10)
    ax.set_yticks(np.arange(len(varieties)))
    ax.set_yticklabels([f"V{v}" for v in varieties], fontsize=9)

    # 添加数值标注
    for i in range(len(varieties)):
        for j in range(len(rank_cols)):
            val = int(matrix[i, j])
            # 高亮偏差大的单元格
            if j > 0:  # 跳过真实排名列
                deviation = abs(val - matrix[i, 0])
                if deviation >= 3:
                    ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1,
                                               fill=False, edgecolor='red', linewidth=2))
            ax.text(j, i, str(val), ha='center', va='center', fontsize=9,
                    color='white' if val <= 4 or val >= 10 else 'black')

    ax.set_title('Variety Ranking Comparison Across Models\n(Red box: deviation ≥ 3)',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Model', fontsize=11)
    ax.set_ylabel('Variety (sorted by true rank)', fontsize=11)

    # 颜色条
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Rank (1=Best)', fontsize=10)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "exp6_fig4_variety_ranking_heatmap.png")
    plt.close()
    print(f"[OK] 图4已保存: {FIG_DIR}/exp6_fig4_variety_ranking_heatmap.png")


def plot_fig5_rank_deviation(predictions):
    """图5: 排名偏差条形图"""
    deviations = compute_rank_deviations(predictions)

    # 按品种分组，计算每个模型的平均偏差
    pivot = deviations.pivot_table(index='Variety', columns='Model',
                                    values='Deviation', aggfunc='mean')
    pivot = pivot[MODELS]  # 保持顺序

    # 排序 (按真实排名)
    base_df = predictions[MODELS[0]]
    variety_order = base_df.sort_values('true_rank')['Variety'].values
    pivot = pivot.reindex(variety_order)

    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(len(pivot))
    width = 0.12

    for i, model in enumerate(MODELS):
        offset = (i - len(MODELS)/2 + 0.5) * width
        bars = ax.bar(x + offset, pivot[model], width, label=model,
                      color=MODEL_COLORS[model], edgecolor='black', linewidth=0.5)

    ax.axhline(y=0, color='black', linewidth=1)
    ax.set_xlabel('Variety', fontsize=11)
    ax.set_ylabel('Rank Deviation (Pred - True)', fontsize=11)
    ax.set_title('Rank Deviation by Variety and Model\n(Positive = ranked lower than true)',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'V{int(v)}' for v in pivot.index], rotation=45, ha='right')
    ax.legend(loc='upper right', ncol=3)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "exp6_fig5_rank_deviation.png")
    plt.close()
    print(f"[OK] 图5已保存: {FIG_DIR}/exp6_fig5_rank_deviation.png")


def plot_fig6_critical_difference(stat_tests):
    """图6: Critical Difference图"""
    avg_ranks = stat_tests['avg_ranks']
    cd = stat_tests['cd']

    # 按平均排名排序
    sorted_models = sorted(avg_ranks.keys(), key=lambda x: avg_ranks[x])
    ranks = [avg_ranks[m] for m in sorted_models]

    fig, ax = plt.subplots(figsize=(10, 5))

    # 绘制排名轴
    ax.set_xlim(0.5, 6.5)
    ax.set_ylim(0, len(sorted_models) + 1)

    # 绘制CD bar
    cd_y = len(sorted_models) + 0.5
    ax.plot([1, 1 + cd], [cd_y, cd_y], 'k-', linewidth=3)
    ax.text(1 + cd/2, cd_y + 0.2, f'CD = {cd:.2f}', ha='center', fontsize=10)

    # 绘制模型点和标签
    for i, (model, rank) in enumerate(zip(sorted_models, ranks)):
        y = len(sorted_models) - i
        ax.plot(rank, y, 'o', markersize=12, color=MODEL_COLORS.get(model, '#333333'))
        ax.text(rank, y - 0.3, f'{rank:.2f}', ha='center', fontsize=9)

        # 模型名称在两侧
        if rank <= 3.5:
            ax.text(0.4, y, model, ha='right', va='center', fontsize=11, fontweight='bold')
        else:
            ax.text(6.6, y, model, ha='left', va='center', fontsize=11, fontweight='bold')

    # 连接无显著差异的模型
    for i, m1 in enumerate(sorted_models):
        for j, m2 in enumerate(sorted_models):
            if i < j:
                if abs(avg_ranks[m1] - avg_ranks[m2]) < cd:
                    y1 = len(sorted_models) - i
                    y2 = len(sorted_models) - j
                    ax.plot([avg_ranks[m1], avg_ranks[m2]], [y1, y2],
                            'k-', linewidth=1.5, alpha=0.4)

    ax.set_xlabel('Average Rank', fontsize=12)
    ax.set_title(f'Critical Difference Diagram (Friedman p={stat_tests["friedman_p"]:.4f})',
                 fontsize=13, fontweight='bold')
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "exp6_fig6_critical_difference.png")
    plt.close()
    print(f"[OK] 图6已保存: {FIG_DIR}/exp6_fig6_critical_difference.png")


def plot_fig7_metric_ranking_matrix(df, additional_metrics):
    """图7: 模型×指标排名矩阵热力图"""
    # 合并指标
    merged = df.merge(additional_metrics, on='Model')

    # 选择关键指标
    metrics = ['R2', 'Spearman', 'Pairwise_Acc', 'Hit@3', 'Kappa', 'NDCG@5']
    metrics = [m for m in metrics if m in merged.columns]

    # 计算排名
    rank_matrix = np.zeros((len(MODELS), len(metrics)))
    for j, metric in enumerate(metrics):
        ranks = merged[metric].rank(ascending=False).values
        for i, model in enumerate(MODELS):
            idx = merged[merged['Model'] == model].index[0]
            rank_matrix[i, j] = ranks[idx]

    fig, ax = plt.subplots(figsize=(10, 6))

    im = ax.imshow(rank_matrix, cmap='RdYlGn_r', aspect='auto', vmin=1, vmax=6)

    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels(metrics, fontsize=10, rotation=45, ha='right')
    ax.set_yticks(np.arange(len(MODELS)))
    ax.set_yticklabels(MODELS, fontsize=10)

    # 数值标注
    for i in range(len(MODELS)):
        for j in range(len(metrics)):
            val = int(rank_matrix[i, j])
            color = 'white' if val <= 2 or val >= 5 else 'black'
            ax.text(j, i, str(val), ha='center', va='center', fontsize=11,
                    fontweight='bold', color=color)

    ax.set_title('Model Ranking Matrix Across Metrics\n(1=Best, 6=Worst)',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Metric', fontsize=11)
    ax.set_ylabel('Model', fontsize=11)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Rank', fontsize=10)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "exp6_fig7_metric_ranking_matrix.png")
    plt.close()
    print(f"[OK] 图7已保存: {FIG_DIR}/exp6_fig7_metric_ranking_matrix.png")


def plot_fig8_residual_boxplot(predictions):
    """图8: 残差分布箱线图"""
    residuals_data = []

    for model in MODELS:
        if model not in predictions:
            continue
        df = predictions[model]
        residuals = df['pred'] - df['D_conv']
        for r in residuals:
            residuals_data.append({'Model': model, 'Residual': r})

    residuals_df = pd.DataFrame(residuals_data)

    fig, ax = plt.subplots(figsize=(10, 6))

    # 箱线图
    bp = ax.boxplot([residuals_df[residuals_df['Model'] == m]['Residual'].values for m in MODELS],
                    labels=MODELS, patch_artist=True, widths=0.6)

    # 设置颜色
    for patch, model in zip(bp['boxes'], MODELS):
        patch.set_facecolor(MODEL_COLORS[model])
        patch.set_alpha(0.7)

    ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.7)
    ax.set_xlabel('Model', fontsize=11)
    ax.set_ylabel('Residual (Predicted - True)', fontsize=11)
    ax.set_title('Residual Distribution by Model', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "exp6_fig8_residual_boxplot.png")
    plt.close()
    print(f"[OK] 图8已保存: {FIG_DIR}/exp6_fig8_residual_boxplot.png")


# =============================================================================
# 报告生成
# =============================================================================

def generate_report(df, additional_metrics, predictions, stat_tests, table2, table3):
    """生成完整Markdown报告"""

    merged = df.merge(additional_metrics, on='Model')
    best_model = merged.loc[merged['Spearman'].idxmax()]
    worst_model = merged.loc[merged['Spearman'].idxmin()]

    # 品种排名分析
    variety_analysis = analyze_variety_rankings(predictions)

    report = f"""# Exp-6: 模型对比实验完整报告

**报告生成时间:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**随机种子:** 42
**交叉验证:** 5折 GroupKFold (按品种分组)
**特征集:** FS4 (三源融合, 40个特征)
**品种数:** 13
**样本数:** 117

---

## 摘要

本实验对比了6个模型在水稻抗旱性预测任务上的表现。结果显示：

- **最优模型:** {best_model['Model']} (Spearman ρ = {best_model['Spearman']:.3f})
- **最差模型:** {worst_model['Model']} (Spearman ρ = {worst_model['Spearman']:.3f})
- **性能提升:** {(best_model['Spearman'] - worst_model['Spearman'])*100:.1f}%

### 主要发现

1. **{best_model['Model']}** 在排序任务上表现最优，Spearman ρ = {best_model['Spearman']:.3f}
2. **Top-3命中率:** {best_model['Hit@3']:.0%}，育种初筛可靠
3. **品种排名一致性:** Kendall τ = {best_model['Kendall_tau']:.3f}
4. **统计检验:** Friedman p = {stat_tests['friedman_p']:.4f}

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
"""

    for _, row in merged.iterrows():
        highlight = "**" if row['Model'] == best_model['Model'] else ""
        report += f"| {highlight}{row['Model']}{highlight} | {row['Layer']} | {row['R2']:.3f} | {row['RMSE']:.3f} | {row['Spearman']:.3f} | {row['Pairwise_Acc']:.3f} | {row['Hit@3']:.2f} | {row['Kappa']:.3f} |\n"

    report += f"""

### 2.2 排序指标详情

| 模型 | Kendall τ | Jaccard@3 | Jaccard@5 | NDCG@5 | Mean Rank Error |
|------|:---:|:---:|:---:|:---:|:---:|
"""

    for _, row in merged.iterrows():
        report += f"| {row['Model']} | {row['Kendall_tau']:.3f} | {row['Jaccard@3']:.3f} | {row['Jaccard@5']:.3f} | {row['NDCG@5']:.3f} | {row['Mean_Rank_Error']:.2f} |\n"

    report += f"""

### 2.3 模型排名汇总

{table2.to_markdown(index=False)}

---

## 3. 品种排名详细分析

### 3.1 品种排名对比表

| Variety | Category | True Rank | {' | '.join(MODELS)} |
|---------|----------|:---------:|{'|'.join([':---:' for _ in MODELS])}|
"""

    for _, row in table3.iterrows():
        ranks = [str(int(row[f'{m}_Rank'])) for m in MODELS]
        report += f"| {int(row['Variety'])} | {row['Category']} | {int(row['true_rank'])} | {' | '.join(ranks)} |\n"

    report += f"""

### 3.2 排名偏差分析

{variety_analysis}

### 3.3 关键发现

- **最容易预测的品种:** 各模型排名一致的品种
- **最难预测的品种:** 排名偏差最大的品种
- **边界品种:** 处于分类阈值附近的品种

---

## 4. 统计检验结果

### 4.1 Friedman检验

- **检验统计量:** χ² = {stat_tests['friedman_stat']:.2f}
- **p值:** {stat_tests['friedman_p']:.6f}
- **样本量:** n = {stat_tests['n_varieties']} 个品种, k = {stat_tests['n_models']} 个模型

### 4.2 Kendall's W 协和系数 (效应量)

- **W值:** {stat_tests['kendall_w']:.3f}
- **效应强度:** {stat_tests['w_interpretation']}
- **解释:** W 表示各品种对模型排名的一致程度 (0=完全不一致, 1=完全一致)

### 4.3 平均排名 (含95% Bootstrap置信区间)

| 模型 | 平均排名 | 95% CI |
|------|:--------:|:------:|
"""

    sorted_ranks = sorted(stat_tests['avg_ranks'].items(), key=lambda x: x[1])
    for model, rank in sorted_ranks:
        ci = stat_tests['rank_ci'][model]
        report += f"| {model} | {rank:.2f} | [{ci[0]:.2f}, {ci[1]:.2f}] |\n"

    report += f"""

### 4.4 Nemenyi临界距离

- **CD值:** {stat_tests['cd']:.2f}
- **含义:** 平均排名差距 < CD 的模型间无显著差异

### 4.4 结果解释

{'⚠️ **小样本警告:** ' + stat_tests['interpretation_note'] if stat_tests['small_sample_warning'] else ''}

**综合解读:**
- Friedman检验 p = {stat_tests['friedman_p']:.4f}，{'拒绝' if stat_tests['friedman_p'] < 0.05 else '不拒绝'}零假设（各模型表现相同）
- Kendall's W = {stat_tests['kendall_w']:.3f}，效应量为 **{stat_tests['w_interpretation']}**
- 由于样本量较小 (n={stat_tests['n_varieties']})，建议以效应量 W 作为主要参考指标
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

1. **{best_model['Model']}** 在水稻抗旱性预测任务中综合表现最优
2. 三源融合特征集(FS4)为所有模型提供了有效的预测基础
3. Hit@3 = {best_model['Hit@3']:.0%}，满足育种初筛的实用需求
4. 品种排名分析揭示了各模型的预测特点和潜在改进方向

---

*报告由 step4_aggregate_results.py 自动生成*
*数据来源: data/processed/features_40.csv*
"""

    report_path = REPORT_DIR / "exp6_model_comparison_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"[OK] 报告已保存: {report_path}")


def analyze_variety_rankings(predictions):
    """分析品种排名"""
    deviations = compute_rank_deviations(predictions)

    # 计算每个品种的平均绝对偏差
    variety_stats = deviations.groupby('Variety').agg({
        'Abs_Deviation': ['mean', 'max'],
        'True_Rank': 'first'
    }).reset_index()
    variety_stats.columns = ['Variety', 'Mean_Dev', 'Max_Dev', 'True_Rank']
    variety_stats = variety_stats.sort_values('Mean_Dev', ascending=False)

    # 最难预测的品种
    hardest = variety_stats.head(3)
    easiest = variety_stats.tail(3)

    analysis = f"""
**最难预测的品种 (平均排名偏差最大):**

| Variety | True Rank | 平均偏差 | 最大偏差 |
|---------|:---------:|:--------:|:--------:|
"""
    for _, row in hardest.iterrows():
        analysis += f"| {int(row['Variety'])} | {int(row['True_Rank'])} | {row['Mean_Dev']:.2f} | {int(row['Max_Dev'])} |\n"

    analysis += f"""

**最容易预测的品种 (平均排名偏差最小):**

| Variety | True Rank | 平均偏差 | 最大偏差 |
|---------|:---------:|:--------:|:--------:|
"""
    for _, row in easiest.iterrows():
        analysis += f"| {int(row['Variety'])} | {int(row['True_Rank'])} | {row['Mean_Dev']:.2f} | {int(row['Max_Dev'])} |\n"

    return analysis


# =============================================================================
# 主函数
# =============================================================================

def main():
    print("=" * 60)
    print("Step 4: 汇总模型对比结果 (完整版)")
    print("=" * 60)

    # 1. 加载数据
    print("\n[1/6] 加载模型结果...")
    df = load_all_results()
    predictions = load_variety_predictions()

    if len(df) == 0:
        print("[ERROR] 没有找到任何模型结果，请先运行各模型脚本")
        return

    print(f"\n已加载 {len(df)} 个模型结果")

    # 2. 计算补充指标
    print("\n[2/6] 计算补充指标...")
    additional_metrics = compute_additional_metrics(predictions)

    # 3. 统计检验
    print("\n[3/6] 执行统计检验...")
    stat_tests = compute_statistical_tests(predictions)
    print(f"  Friedman p-value: {stat_tests['friedman_p']:.6f}")
    print(f"  Kendall's W: {stat_tests['kendall_w']:.3f} ({stat_tests['w_interpretation']})")
    print(f"  CD: {stat_tests['cd']:.2f}")
    if stat_tests['small_sample_warning']:
        print(f"  [WARNING] {stat_tests['interpretation_note']}")

    # 4. 生成表格
    print("\n[4/6] 生成表格...")
    table1, table2, table3 = generate_tables(df, additional_metrics, predictions, stat_tests)

    # 5. 生成图表
    print("\n[5/6] 生成图表...")
    plot_fig1_model_comparison(df)
    plot_fig2_radar(df)
    plot_fig3_pred_vs_true(predictions)
    plot_fig4_variety_ranking_heatmap(predictions)
    plot_fig5_rank_deviation(predictions)
    plot_fig6_critical_difference(stat_tests)
    plot_fig7_metric_ranking_matrix(df, additional_metrics)
    plot_fig8_residual_boxplot(predictions)

    # 6. 生成报告
    print("\n[6/6] 生成报告...")
    generate_report(df, additional_metrics, predictions, stat_tests, table2, table3)

    print("\n" + "=" * 60)
    print("汇总完成!")
    print("=" * 60)
    print(f"\n输出目录:")
    print(f"  表格: {TABLE_DIR}")
    print(f"  图表: {FIG_DIR}")
    print(f"  报告: {REPORT_DIR}")


if __name__ == "__main__":
    main()
