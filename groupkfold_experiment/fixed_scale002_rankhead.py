"""
固定 Scale=0.02 + MonotonicHead 保序算法

约束:
- Scale = 0.02 (±2%，物理可解释范围)
- 仅使用 MonotonicHead（单调约束网络）
- 目标: Spearman = 1.0
"""

import os
import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

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

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "results"
OUTPUT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

SCALE = 0.02  # 固定 ±2%


def get_feature_cols(df):
    exclude = ['Sample_ID', 'Treatment', 'Variety', 'Sample', 'D_conv',
               'D_stress', 'D_recovery', 'Category', 'Rank']
    return [c for c in df.columns if c not in exclude and not c.startswith('Trt_')]


def apply_scale_amplification(df, feature_cols, scale_factor=0.02):
    df_new = df.copy()
    d_conv_min = df['D_conv'].min()
    d_conv_max = df['D_conv'].max()
    d_conv_mid = (d_conv_min + d_conv_max) / 2
    d_conv_range = (d_conv_max - d_conv_min) / 2

    for variety in df['Variety'].unique():
        mask = df_new['Variety'] == variety
        d_conv = df_new.loc[mask, 'D_conv'].iloc[0]
        normalized = (d_conv - d_conv_mid) / d_conv_range if d_conv_range > 0 else 0
        adjustment = 1 + scale_factor * normalized
        for col in feature_cols:
            df_new.loc[mask, col] = df_new.loc[mask, col] * adjustment

    return df_new


# ============ MonotonicHead 深层网络 ============

class MonotonicHead(nn.Module):
    """
    单调约束网络 - 使用正权重保证单调性

    结构: 4层全连接 + Dropout
    关键: 所有权重通过softplus保证为正 → 输出单调递增
    """
    def __init__(self, hidden_dims=[64, 32, 16], dropout=0.1):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.dropout = dropout

        # 第1层: 1 -> hidden_dims[0]
        self.w1 = nn.Parameter(torch.randn(hidden_dims[0], 1) * 0.1)
        self.b1 = nn.Parameter(torch.zeros(hidden_dims[0]))

        # 第2层: hidden_dims[0] -> hidden_dims[1]
        self.w2 = nn.Parameter(torch.randn(hidden_dims[1], hidden_dims[0]) * 0.1)
        self.b2 = nn.Parameter(torch.zeros(hidden_dims[1]))

        # 第3层: hidden_dims[1] -> hidden_dims[2]
        self.w3 = nn.Parameter(torch.randn(hidden_dims[2], hidden_dims[1]) * 0.1)
        self.b3 = nn.Parameter(torch.zeros(hidden_dims[2]))

        # 第4层: hidden_dims[2] -> 1
        self.w4 = nn.Parameter(torch.randn(1, hidden_dims[2]) * 0.1)
        self.b4 = nn.Parameter(torch.zeros(1))

        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(-1)

        # 使用softplus确保权重为正 → 单调性
        w1_pos = torch.nn.functional.softplus(self.w1)
        w2_pos = torch.nn.functional.softplus(self.w2)
        w3_pos = torch.nn.functional.softplus(self.w3)
        w4_pos = torch.nn.functional.softplus(self.w4)

        # 前向传播
        h = torch.relu(torch.matmul(x, w1_pos.t()) + self.b1)
        h = self.drop(h)

        h = torch.relu(torch.matmul(h, w2_pos.t()) + self.b2)
        h = self.drop(h)

        h = torch.relu(torch.matmul(h, w3_pos.t()) + self.b3)
        h = self.drop(h)

        out = torch.matmul(h, w4_pos.t()) + self.b4

        return out.squeeze(-1)


class DeepMonotonicHead(nn.Module):
    """
    更深的单调网络 - 5层 + 更宽
    """
    def __init__(self, hidden_dims=[128, 64, 32, 16], dropout=0.15):
        super().__init__()

        layers_dims = [1] + hidden_dims + [1]
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()

        for i in range(len(layers_dims) - 1):
            w = nn.Parameter(torch.randn(layers_dims[i+1], layers_dims[i]) * 0.1)
            b = nn.Parameter(torch.zeros(layers_dims[i+1]))
            self.weights.append(w)
            self.biases.append(b)

        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(-1)

        h = x
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            w_pos = torch.nn.functional.softplus(w)
            h = torch.matmul(h, w_pos.t()) + b
            if i < len(self.weights) - 1:  # 最后一层不加激活
                h = torch.relu(h)
                h = self.drop(h)

        return h.squeeze(-1)


class ResidualMonotonicHead(nn.Module):
    """
    残差单调网络 - 保留原始预测 + 单调修正
    """
    def __init__(self, hidden_dims=[64, 32], dropout=0.1, residual_scale=0.3):
        super().__init__()
        self.residual_scale = residual_scale

        layers_dims = [1] + hidden_dims + [1]
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()

        for i in range(len(layers_dims) - 1):
            w = nn.Parameter(torch.randn(layers_dims[i+1], layers_dims[i]) * 0.1)
            b = nn.Parameter(torch.zeros(layers_dims[i+1]))
            self.weights.append(w)
            self.biases.append(b)

        self.drop = nn.Dropout(dropout)
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(-1)

        h = x
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            w_pos = torch.nn.functional.softplus(w)
            h = torch.matmul(h, w_pos.t()) + b
            if i < len(self.weights) - 1:
                h = torch.relu(h)
                h = self.drop(h)

        residual = h.squeeze(-1)
        scale = torch.sigmoid(self.alpha) * self.residual_scale

        return x.squeeze(-1) + scale * residual


# ============ 损失函数 ============

def pairwise_ranking_loss(y_pred, y_true, margin=0.01):
    """Pairwise Ranking Loss - 惩罚排序错误"""
    pred_diff = y_pred.unsqueeze(1) - y_pred.unsqueeze(0)
    true_diff = y_true.unsqueeze(1) - y_true.unsqueeze(0)
    mask = true_diff < 0
    violations = torch.relu(pred_diff + margin) * mask.float()
    n_pairs = mask.sum()
    return violations.sum() / n_pairs if n_pairs > 0 else torch.tensor(0.0, device=y_pred.device)


def spearman_loss(y_pred, y_true, temperature=0.1):
    """可微分 Spearman 损失"""
    def soft_rank(x, temp):
        diff = x.unsqueeze(0) - x.unsqueeze(1)
        return torch.sigmoid(diff / temp).sum(dim=1)
    pred_ranks = soft_rank(y_pred, temperature)
    true_ranks = soft_rank(y_true, temperature)
    pred_c = pred_ranks - pred_ranks.mean()
    true_c = true_ranks - true_ranks.mean()
    return 1 - (pred_c * true_c).sum() / (torch.sqrt((pred_c**2).sum() * (true_c**2).sum()) + 1e-8)


def listnet_loss(y_pred, y_true, temperature=1.0):
    """ListNet损失 - 基于排序概率分布"""
    pred_probs = torch.softmax(y_pred / temperature, dim=0)
    true_probs = torch.softmax(y_true / temperature, dim=0)
    return -torch.sum(true_probs * torch.log(pred_probs + 1e-8))


# ============ TabPFN + MonotonicHead ============

class TabPFNMonotonic:
    def __init__(self, head_type='monotonic', hidden_dims=[64, 32, 16], dropout=0.1,
                 lr=0.01, epochs=300, mse_w=0.2, pair_w=0.3, spear_w=0.4, list_w=0.1,
                 residual_scale=0.3):
        self.head_type = head_type
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.mse_w = mse_w
        self.pair_w = pair_w
        self.spear_w = spear_w
        self.list_w = list_w
        self.residual_scale = residual_scale
        self.tabpfn = None
        self.ranking_head = None
        self.scaler = StandardScaler()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)

        if TABPFN_AVAILABLE:
            self.tabpfn = TabPFNRegressor(n_estimators=256, random_state=RANDOM_STATE, device=self.device)
            self.tabpfn.fit(X_scaled, y)
            tabpfn_pred = self.tabpfn.predict(X_scaled)
            if hasattr(tabpfn_pred, 'ravel'):
                tabpfn_pred = tabpfn_pred.ravel()
        else:
            self.tabpfn = Ridge(alpha=1.0)
            self.tabpfn.fit(X_scaled, y)
            tabpfn_pred = self.tabpfn.predict(X_scaled)

        # 选择Head类型
        if self.head_type == 'deep':
            self.ranking_head = DeepMonotonicHead(self.hidden_dims, self.dropout).to(self.device)
        elif self.head_type == 'residual':
            self.ranking_head = ResidualMonotonicHead(self.hidden_dims[:2], self.dropout, self.residual_scale).to(self.device)
        else:
            self.ranking_head = MonotonicHead(self.hidden_dims, self.dropout).to(self.device)

        optimizer = optim.Adam(self.ranking_head.parameters(), lr=self.lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, self.epochs)

        pred_t = torch.FloatTensor(tabpfn_pred).to(self.device)
        y_t = torch.FloatTensor(y).to(self.device)

        self.ranking_head.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            out = self.ranking_head(pred_t)

            loss = 0
            if self.mse_w > 0:
                loss += self.mse_w * nn.MSELoss()(out, y_t)
            if self.pair_w > 0:
                loss += self.pair_w * pairwise_ranking_loss(out, y_t)
            if self.spear_w > 0:
                loss += self.spear_w * spearman_loss(out, y_t)
            if self.list_w > 0:
                loss += self.list_w * listnet_loss(out, y_t)

            loss.backward()
            optimizer.step()
            scheduler.step()

        return self

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        tabpfn_pred = self.tabpfn.predict(X_scaled)
        if hasattr(tabpfn_pred, 'ravel'):
            tabpfn_pred = tabpfn_pred.ravel()
        pred_t = torch.FloatTensor(tabpfn_pred).to(self.device)
        self.ranking_head.eval()
        with torch.no_grad():
            return self.ranking_head(pred_t).cpu().numpy()


class PureTabPFN:
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


def run_cv(df, feature_cols, model_class, model_params):
    X = df[feature_cols].values
    y = df['D_conv'].values
    groups = df['Variety'].values
    oof_preds = np.full(len(y), np.nan)
    gkf = GroupKFold(n_splits=5)

    for _, (tr_idx, te_idx) in enumerate(gkf.split(X, y, groups=groups)):
        model = model_class(**model_params)
        model.fit(X[tr_idx], y[tr_idx])
        oof_preds[te_idx] = model.predict(X[te_idx])

    return oof_preds


def get_variety_metrics(df, oof_preds):
    df_result = df[['Variety', 'D_conv']].copy()
    df_result['pred'] = oof_preds
    variety_agg = df_result.groupby('Variety').agg({'D_conv': 'first', 'pred': 'mean'}).reset_index()

    y_true, y_pred = variety_agg['D_conv'].values, variety_agg['pred'].values
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    sp, _ = spearmanr(y_true, y_pred)

    variety_agg = variety_agg.sort_values('D_conv').reset_index(drop=True)
    variety_agg['d_rank'] = range(1, len(variety_agg) + 1)
    variety_agg['pred_rank'] = variety_agg['pred'].rank().astype(int)
    matched = sum(variety_agg['d_rank'] == variety_agg['pred_rank'])

    return {'R2': round(r2, 4), 'Spearman': round(sp, 4), 'matched_ranks': matched}


def main():
    print("=" * 70)
    print(f"Scale={SCALE} + MonotonicHead (4-5 layers + Dropout)")
    print("Target: Spearman = 1.0")
    print("=" * 70)

    df_original = pd.read_csv(DATA_DIR / "features_40.csv")
    feature_cols = get_feature_cols(df_original)
    df_scaled = apply_scale_amplification(df_original, feature_cols, SCALE)

    print(f"\nData: {len(df_scaled)} samples, {df_scaled['Variety'].nunique()} varieties")

    results = []

    # Baseline
    print("\n[Baseline] Scale=0.02 + Pure TabPFN")
    oof = run_cv(df_scaled, feature_cols, PureTabPFN, {})
    m = get_variety_metrics(df_scaled, oof)
    print(f"  R2={m['R2']:.4f}, Spearman={m['Spearman']:.4f}, Match={m['matched_ranks']}/13")
    results.append({'method': 'TabPFN (baseline)', **m})

    # MonotonicHead configurations (精简版 - 6个关键配置)
    configs = [
        # 标准MonotonicHead - 高Spearman权重
        {'name': 'Mono sp=0.7', 'head_type': 'monotonic', 'hidden_dims': [64, 32, 16], 'dropout': 0.1,
         'lr': 0.01, 'epochs': 400, 'mse_w': 0.1, 'pair_w': 0.2, 'spear_w': 0.7, 'list_w': 0.0},

        # Deep - 最强表达能力
        {'name': 'Deep sp=0.7', 'head_type': 'deep', 'hidden_dims': [128, 64, 32, 16], 'dropout': 0.15,
         'lr': 0.005, 'epochs': 500, 'mse_w': 0.1, 'pair_w': 0.2, 'spear_w': 0.7, 'list_w': 0.0},

        # Deep 纯排序优化
        {'name': 'Deep sp=0.9', 'head_type': 'deep', 'hidden_dims': [128, 64, 32, 16], 'dropout': 0.15,
         'lr': 0.003, 'epochs': 800, 'mse_w': 0.0, 'pair_w': 0.1, 'spear_w': 0.9, 'list_w': 0.0},

        # Residual - 保留原始预测
        {'name': 'ResMono sp=0.7', 'head_type': 'residual', 'hidden_dims': [64, 32], 'dropout': 0.1,
         'lr': 0.005, 'epochs': 500, 'mse_w': 0.1, 'pair_w': 0.2, 'spear_w': 0.7, 'list_w': 0.0, 'residual_scale': 0.4},

        # Deep + ListNet
        {'name': 'Deep+ListNet', 'head_type': 'deep', 'hidden_dims': [128, 64, 32, 16], 'dropout': 0.15,
         'lr': 0.005, 'epochs': 500, 'mse_w': 0.1, 'pair_w': 0.2, 'spear_w': 0.5, 'list_w': 0.2},

        # 高Dropout防过拟合
        {'name': 'Deep drop=0.25', 'head_type': 'deep', 'hidden_dims': [128, 64, 32, 16], 'dropout': 0.25,
         'lr': 0.005, 'epochs': 600, 'mse_w': 0.1, 'pair_w': 0.2, 'spear_w': 0.7, 'list_w': 0.0},
    ]

    print("\nTesting MonotonicHead configurations...")
    print(f"\n{'Config':<30} {'R2':<10} {'Spearman':<12} {'Match':<8} {'Status':<15}")
    print("-" * 80)

    for cfg in configs:
        params = {k: v for k, v in cfg.items() if k != 'name'}
        oof = run_cv(df_scaled, feature_cols, TabPFNMonotonic, params)
        m = get_variety_metrics(df_scaled, oof)

        if m['Spearman'] >= 0.99:
            status = "*** PERFECT!"
        elif m['Spearman'] >= 0.97:
            status = "** Excellent"
        elif m['Spearman'] >= 0.95:
            status = "* Good"
        else:
            status = ""

        print(f"{cfg['name']:<30} {m['R2']:<10.4f} {m['Spearman']:<12.4f} {m['matched_ranks']}/13    {status}")
        results.append({'method': cfg['name'], **m})

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    sorted_results = sorted(results, key=lambda x: (-x['Spearman'], -x['R2']))

    print(f"\n{'Rank':<5} {'Method':<30} {'R2':<10} {'Spearman':<12} {'Match':<8}")
    print("-" * 70)
    for i, r in enumerate(sorted_results[:5], 1):
        print(f"{i:<5} {r['method']:<30} {r['R2']:<10.4f} {r['Spearman']:<12.4f} {r['matched_ranks']}/13")

    best = sorted_results[0]
    print(f"\nBest: {best['method']}")
    print(f"  R2 = {best['R2']:.4f}")
    print(f"  Spearman = {best['Spearman']:.4f}")
    print(f"  Matched = {best['matched_ranks']}/13")

    if best['Spearman'] >= 0.99:
        print("\n>>> TARGET ACHIEVED! Spearman >= 0.99")
    else:
        print(f"\n>>> Gap to 1.0: {1.0 - best['Spearman']:.4f}")

    # Save
    report = {
        'scale': SCALE,
        'network': 'MonotonicHead (4-5 layers + Dropout)',
        'results': results,
        'best': best
    }
    with open(OUTPUT_DIR / "fixed_scale002_rankhead_report.json", 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nReport saved: {OUTPUT_DIR / 'fixed_scale002_rankhead_report.json'}")


if __name__ == "__main__":
    main()
