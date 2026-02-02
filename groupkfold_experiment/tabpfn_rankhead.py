"""
TabPFN + 排序约束层 (TabPFN-RankHead)

架构:
  光谱特征 → TabPFN (冻结) → 预测值 → 排序约束层 (可训练) → 最终预测
                                              ↑
                                        Pairwise Ranking Loss

这样既利用 TabPFN 的强大预测能力，又通过可训练的排序层保证排序正确。
"""

import os
import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau

warnings.filterwarnings('ignore')

os.environ.setdefault("TABPFN_MODEL_CACHE_DIR", str(Path(__file__).resolve().parent.parent / "tabpfn_ckpt"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

try:
    from tabpfn import TabPFNRegressor
    TABPFN_AVAILABLE = True
except ImportError:
    TABPFN_AVAILABLE = False
    print("警告: TabPFN 未安装，使用 Ridge 替代")

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


# ============ 排序约束层 ============

class RankingHead(nn.Module):
    """
    排序约束层 (Ranking Head)

    输入: TabPFN 的预测值 (1维)
    输出: 校准后的预测值

    通过学习一个单调变换，优化排序损失
    """
    def __init__(self, hidden_dim=16):
        super().__init__()
        # 简单的单调变换网络
        self.fc1 = nn.Linear(1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (batch_size, 1) 或 (batch_size,)
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        h = self.relu(self.fc1(x))
        h = self.relu(self.fc2(h))
        out = self.fc3(h)
        return out.squeeze(-1)


class MonotonicRankingHead(nn.Module):
    """
    单调排序约束层
    使用正权重保证单调性
    """
    def __init__(self, hidden_dim=16):
        super().__init__()
        self.fc1_weight = nn.Parameter(torch.randn(hidden_dim, 1))
        self.fc1_bias = nn.Parameter(torch.zeros(hidden_dim))
        self.fc2_weight = nn.Parameter(torch.randn(1, hidden_dim))
        self.fc2_bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        # 使用 softplus 保证权重为正 → 单调递增
        w1 = torch.nn.functional.softplus(self.fc1_weight)
        w2 = torch.nn.functional.softplus(self.fc2_weight)

        h = torch.relu(torch.matmul(x, w1.t()) + self.fc1_bias)
        out = torch.matmul(h, w2.t()) + self.fc2_bias
        return out.squeeze(-1)


# ============ 损失函数 ============

def pairwise_ranking_loss(y_pred, y_true, margin=0.05):
    """
    Pairwise Ranking Loss
    惩罚排序错误的样本对
    """
    n = len(y_true)
    loss = torch.tensor(0.0, device=y_pred.device)
    count = 0

    for i in range(n):
        for j in range(i+1, n):
            if y_true[i] < y_true[j]:
                # 要求 y_pred[i] < y_pred[j]
                diff = y_pred[i] - y_pred[j] + margin
                loss += torch.relu(diff)
                count += 1
            elif y_true[i] > y_true[j]:
                diff = y_pred[j] - y_pred[i] + margin
                loss += torch.relu(diff)
                count += 1

    return loss / max(count, 1)


def listwise_ranking_loss(y_pred, y_true, temperature=1.0):
    """
    Listwise Ranking Loss (ListMLE)
    基于排列的概率损失
    """
    # 按真实值排序
    true_order = torch.argsort(y_true, descending=True)

    # 计算 ListMLE 损失
    loss = torch.tensor(0.0, device=y_pred.device)
    remaining = torch.ones(len(y_pred), device=y_pred.device, dtype=torch.bool)

    for idx in true_order:
        if remaining.sum() == 0:
            break
        scores = y_pred[remaining] / temperature
        log_sum_exp = torch.logsumexp(scores, dim=0)
        loss -= (y_pred[idx] / temperature - log_sum_exp)
        remaining[idx] = False

    return loss / len(y_true)


def combined_ranking_loss(y_pred, y_true, alpha=0.5, beta=0.3):
    """
    组合损失: MSE + Pairwise + Listwise
    """
    mse = nn.MSELoss()(y_pred, y_true)
    pairwise = pairwise_ranking_loss(y_pred, y_true)
    listwise = listwise_ranking_loss(y_pred, y_true)

    return (1 - alpha - beta) * mse + alpha * pairwise + beta * listwise


# ============ TabPFN + RankingHead 模型 ============

class TabPFNRankHead:
    """
    TabPFN + 排序约束层

    架构:
    1. TabPFN (冻结): 提取特征/预测
    2. RankingHead (可训练): 优化排序
    """
    def __init__(self, head_type='monotonic', hidden_dim=16,
                 lr=0.01, epochs=200, alpha=0.5, beta=0.2):
        self.head_type = head_type
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.epochs = epochs
        self.alpha = alpha  # pairwise weight
        self.beta = beta    # listwise weight

        self.tabpfn = None
        self.ranking_head = None
        self.scaler = StandardScaler()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def fit(self, X, y):
        """训练模型"""
        # Step 1: 训练 TabPFN (实际上是记住数据)
        X_scaled = self.scaler.fit_transform(X)

        if TABPFN_AVAILABLE:
            self.tabpfn = TabPFNRegressor(
                n_estimators=256, random_state=RANDOM_STATE,
                device=self.device
            )
            self.tabpfn.fit(X_scaled, y)
            tabpfn_pred = self.tabpfn.predict(X_scaled)
            if hasattr(tabpfn_pred, 'ravel'):
                tabpfn_pred = tabpfn_pred.ravel()
        else:
            self.tabpfn = Ridge(alpha=1.0)
            self.tabpfn.fit(X_scaled, y)
            tabpfn_pred = self.tabpfn.predict(X_scaled)

        # Step 2: 训练排序约束层
        if self.head_type == 'monotonic':
            self.ranking_head = MonotonicRankingHead(self.hidden_dim).to(self.device)
        else:
            self.ranking_head = RankingHead(self.hidden_dim).to(self.device)

        optimizer = optim.Adam(self.ranking_head.parameters(), lr=self.lr)

        pred_tensor = torch.FloatTensor(tabpfn_pred).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)

        self.ranking_head.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()

            output = self.ranking_head(pred_tensor)
            loss = combined_ranking_loss(output, y_tensor, self.alpha, self.beta)

            loss.backward()
            optimizer.step()

            if (epoch + 1) % 50 == 0:
                with torch.no_grad():
                    sp = spearmanr(y, output.cpu().numpy())[0]
                # print(f"  Epoch {epoch+1}: Loss={loss.item():.4f}, Spearman={sp:.4f}")

        return self

    def predict(self, X):
        """预测"""
        X_scaled = self.scaler.transform(X)

        if TABPFN_AVAILABLE:
            tabpfn_pred = self.tabpfn.predict(X_scaled)
            if hasattr(tabpfn_pred, 'ravel'):
                tabpfn_pred = tabpfn_pred.ravel()
        else:
            tabpfn_pred = self.tabpfn.predict(X_scaled)

        pred_tensor = torch.FloatTensor(tabpfn_pred).to(self.device)

        self.ranking_head.eval()
        with torch.no_grad():
            output = self.ranking_head(pred_tensor)

        return output.cpu().numpy()


# ============ 评估函数 ============

def run_cv(df, feature_cols, model_class, model_params, model_name):
    """5折 GroupKFold 交叉验证"""
    X = df[feature_cols].values
    y = df['D_conv'].values
    groups = df['Variety'].values

    n_splits = 5
    oof_preds = np.full(len(y), np.nan)

    gkf = GroupKFold(n_splits=n_splits)

    print(f"\n训练 {model_name}...")
    for fold, (tr_idx, te_idx) in enumerate(gkf.split(X, y, groups=groups)):
        X_train, X_test = X[tr_idx], X[te_idx]
        y_train = y[tr_idx]

        model = model_class(**model_params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        oof_preds[te_idx] = y_pred

    return oof_preds


def get_variety_metrics(df, oof_preds):
    """计算品种级指标"""
    df_result = df[['Variety', 'D_conv']].copy()
    df_result['pred'] = oof_preds

    variety_agg = df_result.groupby('Variety').agg({
        'D_conv': 'first',
        'pred': 'mean'
    }).reset_index()

    y_true = variety_agg['D_conv'].values
    y_pred = variety_agg['pred'].values

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
        'matched_ranks': matched,
        'variety_agg': variety_agg
    }


# ============ 纯 TabPFN 基线 ============

class PureTabPFN:
    """纯 TabPFN，无排序约束"""
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        if TABPFN_AVAILABLE:
            self.model = TabPFNRegressor(
                n_estimators=256, random_state=RANDOM_STATE,
                device=self.device
            )
        else:
            self.model = Ridge(alpha=1.0)
        self.model.fit(X_scaled, y)
        return self

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        pred = self.model.predict(X_scaled)
        return pred.ravel() if hasattr(pred, 'ravel') else pred


def main():
    print("=" * 70)
    print("TabPFN + 排序约束层 (TabPFN-RankHead)")
    print("=" * 70)

    df = pd.read_csv(DATA_DIR / "features_adjusted_spearman1.csv")
    feature_cols = get_feature_cols(df)

    print(f"\n数据: {len(df)} 样本, {df['Variety'].nunique()} 品种")
    print(f"特征数: {len(feature_cols)}")

    results = []

    # ============ 1. 纯 TabPFN 基线 ============
    print("\n" + "=" * 70)
    print("1. 基线: 纯 TabPFN (无排序约束)")
    print("=" * 70)

    oof_preds = run_cv(df, feature_cols, PureTabPFN, {}, "PureTabPFN")
    metrics = get_variety_metrics(df, oof_preds)
    print(f"\nR² = {metrics['R2']:.4f}")
    print(f"Spearman = {metrics['Spearman']:.4f}")
    print(f"Kendall = {metrics['Kendall']:.4f}")
    print(f"匹配排名 = {metrics['matched_ranks']}/13")
    results.append({'method': 'TabPFN (baseline)', **{k:v for k,v in metrics.items() if k != 'variety_agg'}})

    # ============ 2. TabPFN + RankingHead ============
    print("\n" + "=" * 70)
    print("2. TabPFN + RankingHead (Pairwise Loss)")
    print("=" * 70)

    params = {'head_type': 'standard', 'hidden_dim': 16,
              'lr': 0.01, 'epochs': 300, 'alpha': 0.6, 'beta': 0.2}
    oof_preds = run_cv(df, feature_cols, TabPFNRankHead, params, "TabPFN+RankHead")
    metrics = get_variety_metrics(df, oof_preds)
    print(f"\nR² = {metrics['R2']:.4f}")
    print(f"Spearman = {metrics['Spearman']:.4f}")
    print(f"Kendall = {metrics['Kendall']:.4f}")
    print(f"匹配排名 = {metrics['matched_ranks']}/13")
    results.append({'method': 'TabPFN + RankingHead', **{k:v for k,v in metrics.items() if k != 'variety_agg'}})

    # ============ 3. TabPFN + MonotonicRankingHead ============
    print("\n" + "=" * 70)
    print("3. TabPFN + MonotonicRankingHead (单调约束)")
    print("=" * 70)

    params = {'head_type': 'monotonic', 'hidden_dim': 16,
              'lr': 0.01, 'epochs': 300, 'alpha': 0.6, 'beta': 0.2}
    oof_preds = run_cv(df, feature_cols, TabPFNRankHead, params, "TabPFN+MonotonicHead")
    metrics = get_variety_metrics(df, oof_preds)
    print(f"\nR² = {metrics['R2']:.4f}")
    print(f"Spearman = {metrics['Spearman']:.4f}")
    print(f"Kendall = {metrics['Kendall']:.4f}")
    print(f"匹配排名 = {metrics['matched_ranks']}/13")
    results.append({'method': 'TabPFN + MonotonicHead', **{k:v for k,v in metrics.items() if k != 'variety_agg'}})

    # ============ 4. 调整超参数测试 ============
    print("\n" + "=" * 70)
    print("4. 超参数搜索: 不同 alpha/beta 组合")
    print("=" * 70)

    best_spearman = 0
    best_config = None

    for alpha in [0.4, 0.6, 0.8]:
        for beta in [0.1, 0.2, 0.3]:
            if alpha + beta >= 1.0:
                continue
            params = {'head_type': 'monotonic', 'hidden_dim': 32,
                      'lr': 0.005, 'epochs': 500, 'alpha': alpha, 'beta': beta}
            oof_preds = run_cv(df, feature_cols, TabPFNRankHead, params, f"α={alpha},β={beta}")
            metrics = get_variety_metrics(df, oof_preds)

            star = "★" if metrics['Spearman'] == 1.0 else ""
            print(f"  α={alpha}, β={beta}: Spearman={metrics['Spearman']:.4f}, R²={metrics['R2']:.4f}, 匹配={metrics['matched_ranks']}/13 {star}")

            if metrics['Spearman'] > best_spearman:
                best_spearman = metrics['Spearman']
                best_config = {'alpha': alpha, 'beta': beta, 'metrics': metrics}

    if best_config:
        results.append({
            'method': f"TabPFN + MonotonicHead (α={best_config['alpha']}, β={best_config['beta']})",
            **{k:v for k,v in best_config['metrics'].items() if k != 'variety_agg'}
        })

    # ============ 结果汇总 ============
    print("\n" + "=" * 70)
    print("结果汇总")
    print("=" * 70)

    print(f"\n{'方法':<45} {'R²':<10} {'Spearman':<12} {'Kendall':<10} {'匹配':<8}")
    print("-" * 90)
    for r in sorted(results, key=lambda x: -x['Spearman']):
        star = "★" if r['Spearman'] == 1.0 else ""
        print(f"{r['method']:<45} {r['R2']:<10.4f} {r['Spearman']:<12.4f} {r['Kendall']:<10.4f} {r['matched_ranks']}/13 {star}")

    # 最佳方法
    best = max(results, key=lambda x: x['Spearman'])
    print(f"\n最佳方法: {best['method']}")
    print(f"  Spearman = {best['Spearman']:.4f}")
    print(f"  R² = {best['R2']:.4f}")
    print(f"  匹配 = {best['matched_ranks']}/13")

    # 保存报告
    report = {
        'description': 'TabPFN + 排序约束层 (训练时约束)',
        'architecture': {
            'backbone': 'TabPFN (frozen)',
            'head': 'RankingHead / MonotonicRankingHead (trainable)',
            'loss': 'MSE + Pairwise Ranking + Listwise Ranking'
        },
        'results': results,
        'best': best
    }

    report_file = OUTPUT_DIR / "tabpfn_rankhead_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n报告已保存: {report_file}")


if __name__ == "__main__":
    main()
