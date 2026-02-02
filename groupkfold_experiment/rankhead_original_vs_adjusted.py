"""
TabPFN + 排序约束层: 原始数据 vs 修正数据 对比

关键问题: 训练时排序约束能否在原始数据上也达到好的效果？
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
DATA_DIR_ORIGINAL = BASE_DIR.parent / "data" / "processed"
DATA_DIR_ADJUSTED = BASE_DIR.parent / "augmentation_experiment" / "results"
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

class MonotonicRankingHead(nn.Module):
    """单调排序约束层"""
    def __init__(self, hidden_dim=32):
        super().__init__()
        self.fc1_weight = nn.Parameter(torch.randn(hidden_dim, 1) * 0.1)
        self.fc1_bias = nn.Parameter(torch.zeros(hidden_dim))
        self.fc2_weight = nn.Parameter(torch.randn(1, hidden_dim) * 0.1)
        self.fc2_bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        w1 = torch.nn.functional.softplus(self.fc1_weight)
        w2 = torch.nn.functional.softplus(self.fc2_weight)
        h = torch.relu(torch.matmul(x, w1.t()) + self.fc1_bias)
        out = torch.matmul(h, w2.t()) + self.fc2_bias
        return out.squeeze(-1)


class DeepRankingHead(nn.Module):
    """更深的排序头"""
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        return self.net(x).squeeze(-1)


# ============ 损失函数 ============

def pairwise_ranking_loss(y_pred, y_true, margin=0.02):
    """Pairwise Ranking Loss"""
    n = len(y_true)
    loss = torch.tensor(0.0, device=y_pred.device)
    count = 0

    for i in range(n):
        for j in range(i+1, n):
            if y_true[i] < y_true[j]:
                diff = y_pred[i] - y_pred[j] + margin
                loss += torch.relu(diff)
                count += 1
            elif y_true[i] > y_true[j]:
                diff = y_pred[j] - y_pred[i] + margin
                loss += torch.relu(diff)
                count += 1

    return loss / max(count, 1)


def spearman_loss(y_pred, y_true, temperature=0.1):
    """可微分 Spearman 损失"""
    def soft_rank(x, temp):
        n = len(x)
        diff = x.unsqueeze(0) - x.unsqueeze(1)
        return torch.sigmoid(diff / temp).sum(dim=1)

    pred_ranks = soft_rank(y_pred, temperature)
    true_ranks = soft_rank(y_true, temperature)

    pred_centered = pred_ranks - pred_ranks.mean()
    true_centered = true_ranks - true_ranks.mean()

    num = (pred_centered * true_centered).sum()
    den = torch.sqrt((pred_centered**2).sum() * (true_centered**2).sum() + 1e-8)

    return 1 - num / den


def combined_loss(y_pred, y_true, alpha=0.3, beta=0.5):
    """组合损失: MSE + Pairwise + Spearman"""
    mse = nn.MSELoss()(y_pred, y_true)
    pairwise = pairwise_ranking_loss(y_pred, y_true)
    spearman = spearman_loss(y_pred, y_true)

    return (1 - alpha - beta) * mse + alpha * pairwise + beta * spearman


# ============ TabPFN + RankHead ============

class TabPFNRankHead:
    """TabPFN + 排序约束层"""
    def __init__(self, head_type='deep', hidden_dim=64,
                 lr=0.005, epochs=500, alpha=0.3, beta=0.5):
        self.head_type = head_type
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.epochs = epochs
        self.alpha = alpha
        self.beta = beta
        self.tabpfn = None
        self.ranking_head = None
        self.scaler = StandardScaler()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def fit(self, X, y):
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

        if self.head_type == 'monotonic':
            self.ranking_head = MonotonicRankingHead(self.hidden_dim).to(self.device)
        else:
            self.ranking_head = DeepRankingHead(self.hidden_dim).to(self.device)

        optimizer = optim.Adam(self.ranking_head.parameters(), lr=self.lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, self.epochs)

        pred_tensor = torch.FloatTensor(tabpfn_pred).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)

        self.ranking_head.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            output = self.ranking_head(pred_tensor)
            loss = combined_loss(output, y_tensor, self.alpha, self.beta)
            loss.backward()
            optimizer.step()
            scheduler.step()

        return self

    def predict(self, X):
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


class PureTabPFN:
    """纯 TabPFN"""
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        if TABPFN_AVAILABLE:
            self.model = TabPFNRegressor(n_estimators=256, random_state=RANDOM_STATE, device=self.device)
        else:
            self.model = Ridge(alpha=1.0)
        self.model.fit(X_scaled, y)
        return self

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        pred = self.model.predict(X_scaled)
        return pred.ravel() if hasattr(pred, 'ravel') else pred


# ============ 评估函数 ============

def run_cv(df, feature_cols, model_class, model_params):
    X = df[feature_cols].values
    y = df['D_conv'].values
    groups = df['Variety'].values

    n_splits = 5
    oof_preds = np.full(len(y), np.nan)

    gkf = GroupKFold(n_splits=n_splits)

    for fold, (tr_idx, te_idx) in enumerate(gkf.split(X, y, groups=groups)):
        X_train, X_test = X[tr_idx], X[te_idx]
        y_train = y[tr_idx]

        model = model_class(**model_params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        oof_preds[te_idx] = y_pred

    return oof_preds


def get_variety_metrics(df, oof_preds):
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
        'matched_ranks': matched
    }


def test_on_dataset(df, feature_cols, dataset_name):
    """在一个数据集上测试多种方法"""
    print(f"\n{'='*70}")
    print(f"数据集: {dataset_name}")
    print(f"{'='*70}")

    results = []

    # 1. 纯 TabPFN
    print("\n[1] 纯 TabPFN (无排序约束)")
    oof = run_cv(df, feature_cols, PureTabPFN, {})
    m = get_variety_metrics(df, oof)
    print(f"    R²={m['R2']:.4f}, Spearman={m['Spearman']:.4f}, 匹配={m['matched_ranks']}/13")
    results.append({'method': 'TabPFN (baseline)', **m})

    # 2. TabPFN + MonotonicHead
    print("\n[2] TabPFN + MonotonicRankHead")
    params = {'head_type': 'monotonic', 'hidden_dim': 32, 'lr': 0.01, 'epochs': 500, 'alpha': 0.3, 'beta': 0.5}
    oof = run_cv(df, feature_cols, TabPFNRankHead, params)
    m = get_variety_metrics(df, oof)
    print(f"    R²={m['R2']:.4f}, Spearman={m['Spearman']:.4f}, 匹配={m['matched_ranks']}/13")
    results.append({'method': 'TabPFN + MonotonicHead', **m})

    # 3. TabPFN + DeepRankHead
    print("\n[3] TabPFN + DeepRankHead")
    params = {'head_type': 'deep', 'hidden_dim': 64, 'lr': 0.005, 'epochs': 500, 'alpha': 0.3, 'beta': 0.5}
    oof = run_cv(df, feature_cols, TabPFNRankHead, params)
    m = get_variety_metrics(df, oof)
    print(f"    R²={m['R2']:.4f}, Spearman={m['Spearman']:.4f}, 匹配={m['matched_ranks']}/13")
    results.append({'method': 'TabPFN + DeepRankHead', **m})

    # 4. 更激进的排序权重
    print("\n[4] TabPFN + DeepRankHead (高排序权重 α=0.4, β=0.5)")
    params = {'head_type': 'deep', 'hidden_dim': 64, 'lr': 0.005, 'epochs': 800, 'alpha': 0.4, 'beta': 0.5}
    oof = run_cv(df, feature_cols, TabPFNRankHead, params)
    m = get_variety_metrics(df, oof)
    print(f"    R²={m['R2']:.4f}, Spearman={m['Spearman']:.4f}, 匹配={m['matched_ranks']}/13")
    results.append({'method': 'TabPFN + DeepRankHead (high rank)', **m})

    # 5. 只优化 Spearman
    print("\n[5] TabPFN + DeepRankHead (纯 Spearman Loss)")
    params = {'head_type': 'deep', 'hidden_dim': 64, 'lr': 0.003, 'epochs': 800, 'alpha': 0.0, 'beta': 0.9}
    oof = run_cv(df, feature_cols, TabPFNRankHead, params)
    m = get_variety_metrics(df, oof)
    print(f"    R²={m['R2']:.4f}, Spearman={m['Spearman']:.4f}, 匹配={m['matched_ranks']}/13")
    results.append({'method': 'TabPFN + DeepRankHead (pure spearman)', **m})

    return results


def main():
    print("=" * 70)
    print("TabPFN + 排序约束层: 原始数据 vs 修正数据")
    print("=" * 70)

    # 加载两份数据
    df_original = pd.read_csv(DATA_DIR_ORIGINAL / "features_40.csv")
    df_adjusted = pd.read_csv(DATA_DIR_ADJUSTED / "features_adjusted_spearman1.csv")

    feature_cols = get_feature_cols(df_original)

    print(f"\n原始数据: {DATA_DIR_ORIGINAL / 'features_40.csv'}")
    print(f"修正数据: {DATA_DIR_ADJUSTED / 'features_adjusted_spearman1.csv'}")
    print(f"样本数: {len(df_original)}, 品种数: {df_original['Variety'].nunique()}")

    # 在原始数据上测试
    results_original = test_on_dataset(df_original, feature_cols, "原始数据 (features_40.csv)")

    # 在修正数据上测试
    results_adjusted = test_on_dataset(df_adjusted, feature_cols, "修正数据 (features_adjusted_spearman1.csv)")

    # ============ 对比汇总 ============
    print("\n" + "=" * 70)
    print("对比汇总")
    print("=" * 70)

    print("\n原始数据:")
    print(f"{'方法':<40} {'R²':<10} {'Spearman':<12} {'匹配':<8}")
    print("-" * 70)
    for r in sorted(results_original, key=lambda x: -x['Spearman']):
        star = "★" if r['Spearman'] >= 0.99 else ""
        print(f"{r['method']:<40} {r['R2']:<10.4f} {r['Spearman']:<12.4f} {r['matched_ranks']}/13 {star}")

    print("\n修正数据:")
    print(f"{'方法':<40} {'R²':<10} {'Spearman':<12} {'匹配':<8}")
    print("-" * 70)
    for r in sorted(results_adjusted, key=lambda x: -x['Spearman']):
        star = "★" if r['Spearman'] >= 0.99 else ""
        print(f"{r['method']:<40} {r['R2']:<10.4f} {r['Spearman']:<12.4f} {r['matched_ranks']}/13 {star}")

    # 关键对比
    print("\n" + "=" * 70)
    print("关键发现")
    print("=" * 70)

    best_original = max(results_original, key=lambda x: x['Spearman'])
    best_adjusted = max(results_adjusted, key=lambda x: x['Spearman'])

    print(f"\n原始数据最佳: {best_original['method']}")
    print(f"  Spearman = {best_original['Spearman']:.4f}")

    print(f"\n修正数据最佳: {best_adjusted['method']}")
    print(f"  Spearman = {best_adjusted['Spearman']:.4f}")

    print(f"\n排序约束提升 (原始数据):")
    baseline_orig = [r for r in results_original if 'baseline' in r['method']][0]
    print(f"  基线 Spearman = {baseline_orig['Spearman']:.4f}")
    print(f"  最佳 Spearman = {best_original['Spearman']:.4f}")
    print(f"  提升 = {best_original['Spearman'] - baseline_orig['Spearman']:.4f}")

    # 保存报告
    report = {
        'description': 'TabPFN + 排序约束层: 原始数据 vs 修正数据对比',
        'original_data': {
            'file': 'features_40.csv',
            'results': results_original,
            'best': best_original
        },
        'adjusted_data': {
            'file': 'features_adjusted_spearman1.csv',
            'results': results_adjusted,
            'best': best_adjusted
        }
    }

    report_file = OUTPUT_DIR / "rankhead_original_vs_adjusted_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n报告已保存: {report_file}")


if __name__ == "__main__":
    main()
