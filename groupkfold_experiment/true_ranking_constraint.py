"""
真正的保序预测: 训练时约束

方法:
1. Pairwise Ranking Loss - 训练时优化排序
2. XGBoost with rank objective - 排序学习
3. LightGBM with lambdarank - 排序优化
4. 组合损失: MSE + Ranking Loss

这些方法在训练时就优化排序，不是后处理！
"""

import os
import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau
from itertools import combinations

warnings.filterwarnings('ignore')

os.environ.setdefault("TABPFN_MODEL_CACHE_DIR", str(Path(__file__).resolve().parent.parent / "tabpfn_ckpt"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("XGBoost 未安装")

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False
    print("LightGBM 未安装")

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
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


def get_feature_cols(df):
    exclude = ['Sample_ID', 'Treatment', 'Variety', 'Sample', 'D_conv',
               'D_stress', 'D_recovery', 'Category', 'Rank']
    return [c for c in df.columns if c not in exclude and not c.startswith('Trt_')]


# ============ 方法1: Pairwise Ranking Loss 神经网络 ============

class RankingMLP(nn.Module):
    """带排序损失的MLP"""
    def __init__(self, input_dim, hidden_dims=[64, 32]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze(-1)


def pairwise_ranking_loss(y_pred, y_true, margin=0.1):
    """
    Pairwise Ranking Loss
    对于所有 (i,j) 对，如果 y_true[i] < y_true[j]，则要求 y_pred[i] < y_pred[j]
    """
    n = len(y_true)
    loss = torch.tensor(0.0, device=y_pred.device)
    count = 0

    for i in range(n):
        for j in range(i+1, n):
            if y_true[i] < y_true[j]:
                # 要求 y_pred[i] < y_pred[j]
                # 如果 y_pred[i] >= y_pred[j]，惩罚
                diff = y_pred[i] - y_pred[j] + margin
                loss += torch.relu(diff)
                count += 1
            elif y_true[i] > y_true[j]:
                # 要求 y_pred[i] > y_pred[j]
                diff = y_pred[j] - y_pred[i] + margin
                loss += torch.relu(diff)
                count += 1

    return loss / max(count, 1)


def combined_loss(y_pred, y_true, alpha=0.5):
    """
    组合损失: MSE + Ranking Loss
    alpha: 排序损失权重
    """
    mse_loss = nn.MSELoss()(y_pred, y_true)
    rank_loss = pairwise_ranking_loss(y_pred, y_true)
    return (1 - alpha) * mse_loss + alpha * rank_loss


def train_ranking_mlp(X_train, y_train, X_test, input_dim, alpha=0.5, epochs=200):
    """训练带排序损失的MLP"""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = RankingMLP(input_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        y_pred = model(X_train_t)
        loss = combined_loss(y_pred, y_train_t, alpha=alpha)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        y_pred_test = model(X_test_t).cpu().numpy()

    return y_pred_test


# ============ 方法2: XGBoost Ranking ============

def train_xgb_ranking(X_train, y_train, X_test, groups_train):
    """XGBoost 排序模型"""
    if not XGB_AVAILABLE:
        return None

    # 创建排序数据格式
    # group: 每个查询（品种）有多少样本
    unique_groups = np.unique(groups_train)
    group_sizes = [np.sum(groups_train == g) for g in unique_groups]

    # 排序目标需要 relevance labels (我们用 D_conv 的排名)
    # 归一化 D_conv 到 0-10 作为相关性分数
    y_rank = (y_train - y_train.min()) / (y_train.max() - y_train.min()) * 10

    dtrain = xgb.DMatrix(X_train, label=y_rank)
    dtrain.set_group(group_sizes)

    params = {
        'objective': 'rank:pairwise',
        'eval_metric': 'ndcg',
        'eta': 0.1,
        'max_depth': 4,
        'min_child_weight': 3,
        'seed': RANDOM_STATE
    }

    model = xgb.train(params, dtrain, num_boost_round=100)

    dtest = xgb.DMatrix(X_test)
    y_pred = model.predict(dtest)

    return y_pred


# ============ 方法3: LightGBM LambdaRank ============

def train_lgb_ranking(X_train, y_train, X_test, groups_train):
    """LightGBM 排序模型"""
    if not LGB_AVAILABLE:
        return None

    unique_groups = np.unique(groups_train)
    group_sizes = [np.sum(groups_train == g) for g in unique_groups]

    y_rank = (y_train - y_train.min()) / (y_train.max() - y_train.min()) * 10

    train_data = lgb.Dataset(X_train, label=y_rank, group=group_sizes)

    params = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'learning_rate': 0.1,
        'max_depth': 4,
        'num_leaves': 15,
        'min_data_in_leaf': 3,
        'seed': RANDOM_STATE,
        'verbose': -1
    }

    model = lgb.train(params, train_data, num_boost_round=100)
    y_pred = model.predict(X_test)

    return y_pred


# ============ 方法4: Differentiable Spearman Loss ============

def differentiable_spearman_loss(y_pred, y_true, eps=1e-6):
    """
    可微分的 Spearman 相关损失
    使用软排名近似
    """
    n = len(y_pred)

    # 软排名 (用 sigmoid 近似)
    def soft_rank(x, temperature=1.0):
        # 计算每个元素比多少其他元素大
        diff = x.unsqueeze(1) - x.unsqueeze(0)  # n x n
        ranks = torch.sigmoid(diff / temperature).sum(dim=1)
        return ranks

    pred_ranks = soft_rank(y_pred)
    true_ranks = soft_rank(y_true)

    # Spearman 相关系数
    pred_centered = pred_ranks - pred_ranks.mean()
    true_centered = true_ranks - true_ranks.mean()

    numerator = (pred_centered * true_centered).sum()
    denominator = torch.sqrt((pred_centered ** 2).sum() * (true_centered ** 2).sum() + eps)

    spearman = numerator / denominator

    # 最大化 Spearman = 最小化 (1 - Spearman)
    return 1 - spearman


def train_spearman_mlp(X_train, y_train, X_test, input_dim, epochs=300):
    """训练直接优化Spearman的MLP"""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = RankingMLP(input_dim, hidden_dims=[64, 32]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        y_pred = model(X_train_t)

        # 组合 MSE 和 Spearman 损失
        mse = nn.MSELoss()(y_pred, y_train_t)
        spearman_loss = differentiable_spearman_loss(y_pred, y_train_t)
        loss = 0.3 * mse + 0.7 * spearman_loss

        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        y_pred_test = model(X_test_t).cpu().numpy()

    return y_pred_test


# ============ 评估函数 ============

def run_method_cv(df, feature_cols, method_name, method_func):
    """5折 GroupKFold 评估"""
    X = df[feature_cols].values
    y = df['D_conv'].values
    groups = df['Variety'].values

    n_splits = 5
    oof_preds = np.full(len(y), np.nan)

    gkf = GroupKFold(n_splits=n_splits)

    for fold, (tr_idx, te_idx) in enumerate(gkf.split(X, y, groups=groups)):
        X_train, X_test = X[tr_idx], X[te_idx]
        y_train = y[tr_idx]
        groups_train = groups[tr_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        y_pred = method_func(X_train_scaled, y_train, X_test_scaled, groups_train, len(feature_cols))

        if y_pred is not None:
            oof_preds[te_idx] = y_pred

    return oof_preds


def get_variety_metrics(df, oof_preds):
    """计算品种级指标"""
    if np.isnan(oof_preds).all():
        return None

    df_result = df[['Variety', 'D_conv']].copy()
    df_result['pred'] = oof_preds

    variety_agg = df_result.groupby('Variety').agg({
        'D_conv': 'first',
        'pred': 'mean'
    }).reset_index()

    y_true = variety_agg['D_conv'].values
    y_pred = variety_agg['pred'].values

    if np.isnan(y_pred).any():
        return None

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    sp, _ = spearmanr(y_true, y_pred)
    kt, _ = kendalltau(y_true, y_pred)

    variety_agg = variety_agg.sort_values('D_conv').reset_index(drop=True)
    variety_agg['d_rank'] = range(1, len(variety_agg) + 1)
    variety_agg['pred_rank'] = variety_agg['pred'].rank().astype(int)
    matched = sum(variety_agg['d_rank'] == variety_agg['pred_rank'])

    return {
        'R2': round(r2, 4),
        'Spearman': round(sp, 4),
        'Kendall': round(kt, 4),
        'matched_ranks': matched
    }


def main():
    print("=" * 70)
    print("真正的保序预测: 训练时排序约束")
    print("=" * 70)

    df = pd.read_csv(DATA_DIR / "features_adjusted_spearman1.csv")
    feature_cols = get_feature_cols(df)

    print(f"\n数据: {len(df)} 样本, {df['Variety'].nunique()} 品种")
    print(f"特征数: {len(feature_cols)}")

    results = []

    # ============ 方法1: Pairwise Ranking Loss MLP ============
    print("\n" + "=" * 70)
    print("方法1: Pairwise Ranking Loss MLP")
    print("=" * 70)

    def method_ranking_mlp(X_train, y_train, X_test, groups, input_dim):
        return train_ranking_mlp(X_train, y_train, X_test, input_dim, alpha=0.7, epochs=300)

    oof_preds = run_method_cv(df, feature_cols, "RankingMLP", method_ranking_mlp)
    metrics = get_variety_metrics(df, oof_preds)
    if metrics:
        print(f"R² = {metrics['R2']:.4f}, Spearman = {metrics['Spearman']:.4f}, Kendall = {metrics['Kendall']:.4f}, 匹配 = {metrics['matched_ranks']}/13")
        results.append({'method': 'Pairwise Ranking MLP', **metrics})

    # ============ 方法2: Differentiable Spearman MLP ============
    print("\n" + "=" * 70)
    print("方法2: Differentiable Spearman Loss MLP")
    print("=" * 70)

    def method_spearman_mlp(X_train, y_train, X_test, groups, input_dim):
        return train_spearman_mlp(X_train, y_train, X_test, input_dim, epochs=300)

    oof_preds = run_method_cv(df, feature_cols, "SpearmanMLP", method_spearman_mlp)
    metrics = get_variety_metrics(df, oof_preds)
    if metrics:
        print(f"R² = {metrics['R2']:.4f}, Spearman = {metrics['Spearman']:.4f}, Kendall = {metrics['Kendall']:.4f}, 匹配 = {metrics['matched_ranks']}/13")
        results.append({'method': 'Differentiable Spearman MLP', **metrics})

    # ============ 方法3: XGBoost Ranking ============
    if XGB_AVAILABLE:
        print("\n" + "=" * 70)
        print("方法3: XGBoost rank:pairwise")
        print("=" * 70)

        def method_xgb(X_train, y_train, X_test, groups, input_dim):
            return train_xgb_ranking(X_train, y_train, X_test, groups)

        oof_preds = run_method_cv(df, feature_cols, "XGBRank", method_xgb)
        metrics = get_variety_metrics(df, oof_preds)
        if metrics:
            print(f"R² = {metrics['R2']:.4f}, Spearman = {metrics['Spearman']:.4f}, Kendall = {metrics['Kendall']:.4f}, 匹配 = {metrics['matched_ranks']}/13")
            results.append({'method': 'XGBoost Pairwise Ranking', **metrics})

    # ============ 方法4: LightGBM LambdaRank ============
    if LGB_AVAILABLE:
        print("\n" + "=" * 70)
        print("方法4: LightGBM LambdaRank")
        print("=" * 70)

        def method_lgb(X_train, y_train, X_test, groups, input_dim):
            return train_lgb_ranking(X_train, y_train, X_test, groups)

        oof_preds = run_method_cv(df, feature_cols, "LGBRank", method_lgb)
        metrics = get_variety_metrics(df, oof_preds)
        if metrics:
            print(f"R² = {metrics['R2']:.4f}, Spearman = {metrics['Spearman']:.4f}, Kendall = {metrics['Kendall']:.4f}, 匹配 = {metrics['matched_ranks']}/13")
            results.append({'method': 'LightGBM LambdaRank', **metrics})

    # ============ 对比: TabPFN (无排序约束) ============
    print("\n" + "=" * 70)
    print("对比: TabPFN (无排序约束)")
    print("=" * 70)

    def method_tabpfn(X_train, y_train, X_test, groups, input_dim):
        if not TABPFN_AVAILABLE:
            return Ridge(alpha=1.0).fit(X_train, y_train).predict(X_test)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = TabPFNRegressor(n_estimators=256, random_state=RANDOM_STATE, device=device)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        return pred.ravel() if hasattr(pred, 'ravel') else pred

    oof_preds = run_method_cv(df, feature_cols, "TabPFN", method_tabpfn)
    metrics = get_variety_metrics(df, oof_preds)
    if metrics:
        print(f"R² = {metrics['R2']:.4f}, Spearman = {metrics['Spearman']:.4f}, Kendall = {metrics['Kendall']:.4f}, 匹配 = {metrics['matched_ranks']}/13")
        results.append({'method': 'TabPFN (baseline)', **metrics})

    # ============ 结果汇总 ============
    print("\n" + "=" * 70)
    print("结果汇总: 训练时排序约束 vs 无约束")
    print("=" * 70)

    print(f"\n{'方法':<35} {'R²':<10} {'Spearman':<12} {'Kendall':<10} {'匹配':<8}")
    print("-" * 75)
    for r in sorted(results, key=lambda x: -x['Spearman']):
        star = "★" if r['Spearman'] == 1.0 else ""
        print(f"{r['method']:<35} {r['R2']:<10.4f} {r['Spearman']:<12.4f} {r['Kendall']:<10.4f} {r['matched_ranks']}/13 {star}")

    # 保存报告
    report = {
        'description': '训练时排序约束 vs 后处理校准',
        'methods': {
            'Pairwise Ranking MLP': '训练时优化两两排序损失',
            'Differentiable Spearman MLP': '训练时直接优化Spearman',
            'XGBoost Pairwise Ranking': '排序学习目标函数',
            'LightGBM LambdaRank': 'Lambda梯度排序优化',
            'TabPFN': '基线，无排序约束'
        },
        'results': results
    }

    report_file = OUTPUT_DIR / "true_ranking_constraint_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n报告已保存: {report_file}")


if __name__ == "__main__":
    main()
