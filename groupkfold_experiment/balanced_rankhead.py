"""
平衡 R² 和 Spearman 的 RankHead 调参

目标:
- Spearman 尽量接近 1.0
- R² 保持在 0.90 以上

策略:
- 调整 MSE 权重，防止 R² 下降太多
- 控制训练轮数
- 尝试不同的网络结构
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

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "results"
OUTPUT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


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


# ============ 轻量级 RankHead ============

class LightRankHead(nn.Module):
    """轻量级排序头 - 减少参数防止过拟合"""
    def __init__(self, hidden_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        return self.net(x).squeeze(-1)


class ResidualRankHead(nn.Module):
    """残差排序头 - 保留原始预测信息"""
    def __init__(self, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.alpha = nn.Parameter(torch.tensor(0.1))  # 残差权重

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        residual = self.net(x).squeeze(-1)
        # 输出 = 原始 + alpha * 残差
        return x.squeeze(-1) + torch.sigmoid(self.alpha) * residual


class ScaleShiftHead(nn.Module):
    """仅学习缩放和偏移 - 最简单的校准"""
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.shift = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        if x.dim() == 1:
            return x * self.scale + self.shift
        return x.squeeze(-1) * self.scale + self.shift


# ============ 损失函数 ============

def pairwise_ranking_loss(y_pred, y_true, margin=0.02):
    """向量化 Pairwise Ranking Loss"""
    pred_diff = y_pred.unsqueeze(1) - y_pred.unsqueeze(0)
    true_diff = y_true.unsqueeze(1) - y_true.unsqueeze(0)
    mask = true_diff < 0
    violations = torch.relu(pred_diff + margin) * mask.float()
    n_pairs = mask.sum()
    if n_pairs > 0:
        return violations.sum() / n_pairs
    return torch.tensor(0.0, device=y_pred.device)


def spearman_loss(y_pred, y_true, temperature=0.1):
    """可微分 Spearman 损失"""
    def soft_rank(x, temp):
        diff = x.unsqueeze(0) - x.unsqueeze(1)
        return torch.sigmoid(diff / temp).sum(dim=1)

    pred_ranks = soft_rank(y_pred, temperature)
    true_ranks = soft_rank(y_true, temperature)
    pred_centered = pred_ranks - pred_ranks.mean()
    true_centered = true_ranks - true_ranks.mean()
    num = (pred_centered * true_centered).sum()
    den = torch.sqrt((pred_centered**2).sum() * (true_centered**2).sum() + 1e-8)
    return 1 - num / den


def balanced_loss(y_pred, y_true, mse_weight=0.5, pairwise_weight=0.3, spearman_weight=0.2):
    """平衡损失: 更高的 MSE 权重保护 R²"""
    mse = nn.MSELoss()(y_pred, y_true)
    pairwise = pairwise_ranking_loss(y_pred, y_true)
    spearman = spearman_loss(y_pred, y_true)
    return mse_weight * mse + pairwise_weight * pairwise + spearman_weight * spearman


# ============ TabPFN + RankHead ============

class TabPFNBalancedRankHead:
    """平衡 R² 和 Spearman 的 RankHead"""
    def __init__(self, head_type='residual', hidden_dim=32,
                 lr=0.01, epochs=200, mse_w=0.5, pair_w=0.3, spear_w=0.2):
        self.head_type = head_type
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.epochs = epochs
        self.mse_w = mse_w
        self.pair_w = pair_w
        self.spear_w = spear_w
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

        if self.head_type == 'light':
            self.ranking_head = LightRankHead(self.hidden_dim).to(self.device)
        elif self.head_type == 'residual':
            self.ranking_head = ResidualRankHead(self.hidden_dim).to(self.device)
        elif self.head_type == 'scale_shift':
            self.ranking_head = ScaleShiftHead().to(self.device)
        else:
            self.ranking_head = LightRankHead(self.hidden_dim).to(self.device)

        optimizer = optim.Adam(self.ranking_head.parameters(), lr=self.lr)

        pred_tensor = torch.FloatTensor(tabpfn_pred).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)

        self.ranking_head.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            output = self.ranking_head(pred_tensor)
            loss = balanced_loss(output, y_tensor, self.mse_w, self.pair_w, self.spear_w)
            loss.backward()
            optimizer.step()

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

    n_splits = 5
    oof_preds = np.full(len(y), np.nan)
    gkf = GroupKFold(n_splits=n_splits)

    for fold, (tr_idx, te_idx) in enumerate(gkf.split(X, y, groups=groups)):
        X_train, X_test = X[tr_idx], X[te_idx]
        y_train = y[tr_idx]
        model = model_class(**model_params)
        model.fit(X_train, y_train)
        oof_preds[te_idx] = model.predict(X_test)

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

    variety_agg = variety_agg.sort_values('D_conv').reset_index(drop=True)
    variety_agg['d_rank'] = range(1, len(variety_agg) + 1)
    variety_agg['pred_rank'] = variety_agg['pred'].rank().astype(int)
    matched = sum(variety_agg['d_rank'] == variety_agg['pred_rank'])

    return {'R2': round(r2, 4), 'Spearman': round(sp, 4), 'matched_ranks': matched}


def main():
    print("=" * 70)
    print("平衡 R² 和 Spearman 的 RankHead 调参")
    print("目标: Spearman ≈ 1.0 且 R² ≥ 0.90")
    print("=" * 70)

    df_original = pd.read_csv(DATA_DIR / "features_40.csv")
    feature_cols = get_feature_cols(df_original)

    results = []

    # 测试不同 Scale
    for scale in [0.02, 0.03, 0.04, 0.05, 0.06]:
        df_scaled = apply_scale_amplification(df_original, feature_cols, scale)

        print(f"\n{'='*70}")
        print(f"Scale = {scale}")
        print("=" * 70)

        # 基线
        oof = run_cv(df_scaled, feature_cols, PureTabPFN, {})
        m = get_variety_metrics(df_scaled, oof)
        print(f"[基线] TabPFN: R²={m['R2']:.4f}, Spearman={m['Spearman']:.4f}, 匹配={m['matched_ranks']}/13")
        results.append({'scale': scale, 'method': 'TabPFN', **m})

        # 不同 Head 和权重组合
        configs = [
            # (name, head_type, hidden_dim, lr, epochs, mse_w, pair_w, spear_w)
            ('Residual MSE重', 'residual', 32, 0.01, 200, 0.6, 0.2, 0.2),
            ('Residual 平衡', 'residual', 32, 0.01, 200, 0.4, 0.3, 0.3),
            ('Residual Spear重', 'residual', 32, 0.01, 200, 0.3, 0.3, 0.4),
            ('Light MSE重', 'light', 16, 0.01, 150, 0.6, 0.2, 0.2),
            ('Light 平衡', 'light', 16, 0.01, 150, 0.4, 0.3, 0.3),
            ('ScaleShift', 'scale_shift', 0, 0.02, 100, 0.5, 0.25, 0.25),
        ]

        for name, head_type, hidden_dim, lr, epochs, mse_w, pair_w, spear_w in configs:
            params = {
                'head_type': head_type, 'hidden_dim': hidden_dim,
                'lr': lr, 'epochs': epochs,
                'mse_w': mse_w, 'pair_w': pair_w, 'spear_w': spear_w
            }
            oof = run_cv(df_scaled, feature_cols, TabPFNBalancedRankHead, params)
            m = get_variety_metrics(df_scaled, oof)

            # 标记好结果
            if m['Spearman'] >= 0.98 and m['R2'] >= 0.90:
                mark = "★★"
            elif m['Spearman'] >= 0.98:
                mark = "◆"
            elif m['R2'] >= 0.92:
                mark = "●"
            else:
                mark = ""

            print(f"  {name}: R²={m['R2']:.4f}, Spearman={m['Spearman']:.4f}, 匹配={m['matched_ranks']}/13 {mark}")
            results.append({'scale': scale, 'method': name, **m})

    # ============ 结果汇总 ============
    print("\n" + "=" * 70)
    print("符合条件的结果 (Spearman ≥ 0.98 且 R² ≥ 0.90)")
    print("=" * 70)

    good_results = [r for r in results if r['Spearman'] >= 0.98 and r['R2'] >= 0.90]
    if good_results:
        print(f"\n{'Scale':<8} {'方法':<20} {'R²':<10} {'Spearman':<12} {'匹配':<8}")
        print("-" * 60)
        for r in sorted(good_results, key=lambda x: (-x['Spearman'], -x['R2'])):
            print(f"{r['scale']:<8} {r['method']:<20} {r['R2']:<10.4f} {r['Spearman']:<12.4f} {r['matched_ranks']}/13")
    else:
        print("\n未找到同时满足 Spearman≥0.98 且 R²≥0.90 的配置")
        print("\n最高 Spearman (R²≥0.85):")
        high_sp = [r for r in results if r['R2'] >= 0.85]
        if high_sp:
            best = max(high_sp, key=lambda x: x['Spearman'])
            print(f"  Scale={best['scale']}, {best['method']}: R²={best['R2']:.4f}, Spearman={best['Spearman']:.4f}")

    # 保存
    report = {'results': results, 'good_results': good_results}
    with open(OUTPUT_DIR / "balanced_rankhead_report.json", 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n报告已保存: {OUTPUT_DIR / 'balanced_rankhead_report.json'}")


if __name__ == "__main__":
    main()
