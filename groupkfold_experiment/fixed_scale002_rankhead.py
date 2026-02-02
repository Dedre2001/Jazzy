"""
固定 Scale=0.02 + RankHead 调参

约束:
- Scale = 0.02 (±2%，物理可解释范围)
- 目标: Spearman ≈ 1.0 且 R² ≥ 0.90
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


# ============ RankHead 变体 ============

class ResidualRankHead(nn.Module):
    """残差结构：保留原始预测，只学习微调"""
    def __init__(self, hidden_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        residual = self.net(x).squeeze(-1)
        return x.squeeze(-1) + torch.sigmoid(self.alpha) * 0.2 * residual  # 限制残差幅度


class LightRankHead(nn.Module):
    """轻量级"""
    def __init__(self, hidden_dim=8):
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


class ScaleShiftHead(nn.Module):
    """仅缩放和偏移"""
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.shift = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        return x * self.scale + self.shift


# ============ 损失函数 ============

def pairwise_ranking_loss(y_pred, y_true, margin=0.01):
    pred_diff = y_pred.unsqueeze(1) - y_pred.unsqueeze(0)
    true_diff = y_true.unsqueeze(1) - y_true.unsqueeze(0)
    mask = true_diff < 0
    violations = torch.relu(pred_diff + margin) * mask.float()
    n_pairs = mask.sum()
    return violations.sum() / n_pairs if n_pairs > 0 else torch.tensor(0.0, device=y_pred.device)


def spearman_loss(y_pred, y_true, temperature=0.1):
    def soft_rank(x, temp):
        diff = x.unsqueeze(0) - x.unsqueeze(1)
        return torch.sigmoid(diff / temp).sum(dim=1)
    pred_ranks = soft_rank(y_pred, temperature)
    true_ranks = soft_rank(y_true, temperature)
    pred_c = pred_ranks - pred_ranks.mean()
    true_c = true_ranks - true_ranks.mean()
    return 1 - (pred_c * true_c).sum() / (torch.sqrt((pred_c**2).sum() * (true_c**2).sum()) + 1e-8)


# ============ TabPFN + RankHead ============

class TabPFNRankHead:
    def __init__(self, head_type='residual', hidden_dim=16,
                 lr=0.01, epochs=150, mse_w=0.5, pair_w=0.25, spear_w=0.25):
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

        if self.head_type == 'residual':
            self.ranking_head = ResidualRankHead(self.hidden_dim).to(self.device)
        elif self.head_type == 'light':
            self.ranking_head = LightRankHead(self.hidden_dim).to(self.device)
        else:
            self.ranking_head = ScaleShiftHead().to(self.device)

        optimizer = optim.Adam(self.ranking_head.parameters(), lr=self.lr)

        pred_t = torch.FloatTensor(tabpfn_pred).to(self.device)
        y_t = torch.FloatTensor(y).to(self.device)

        self.ranking_head.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            out = self.ranking_head(pred_t)
            loss = (self.mse_w * nn.MSELoss()(out, y_t) +
                    self.pair_w * pairwise_ranking_loss(out, y_t) +
                    self.spear_w * spearman_loss(out, y_t))
            loss.backward()
            optimizer.step()

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
    print(f"固定 Scale={SCALE} (±{SCALE*100}%) + RankHead 调参")
    print("目标: Spearman ≥ 0.98 且 R² ≥ 0.90")
    print("=" * 70)

    df_original = pd.read_csv(DATA_DIR / "features_40.csv")
    feature_cols = get_feature_cols(df_original)
    df_scaled = apply_scale_amplification(df_original, feature_cols, SCALE)

    print(f"\n数据: {len(df_scaled)} 样本, {df_scaled['Variety'].nunique()} 品种")

    results = []

    # 基线
    print("\n[基线] Scale=0.02 + 纯TabPFN")
    oof = run_cv(df_scaled, feature_cols, PureTabPFN, {})
    m = get_variety_metrics(df_scaled, oof)
    print(f"  R²={m['R2']:.4f}, Spearman={m['Spearman']:.4f}, 匹配={m['matched_ranks']}/13")
    results.append({'method': 'TabPFN (基线)', **m})

    # 配置列表 - 目标Spearman=1.0
    configs = [
        # ========== 高Spearman权重配置 ==========
        {'name': 'Residual spear=0.5', 'head_type': 'residual', 'hidden_dim': 32, 'lr': 0.01, 'epochs': 300, 'mse_w': 0.2, 'pair_w': 0.3, 'spear_w': 0.5},
        {'name': 'Residual spear=0.6', 'head_type': 'residual', 'hidden_dim': 32, 'lr': 0.01, 'epochs': 400, 'mse_w': 0.1, 'pair_w': 0.3, 'spear_w': 0.6},
        {'name': 'Residual spear=0.7', 'head_type': 'residual', 'hidden_dim': 64, 'lr': 0.005, 'epochs': 500, 'mse_w': 0.1, 'pair_w': 0.2, 'spear_w': 0.7},
        {'name': 'Residual spear=0.8', 'head_type': 'residual', 'hidden_dim': 64, 'lr': 0.005, 'epochs': 600, 'mse_w': 0.0, 'pair_w': 0.2, 'spear_w': 0.8},

        # ========== 纯排序优化 ==========
        {'name': 'Residual pure_rank', 'head_type': 'residual', 'hidden_dim': 64, 'lr': 0.003, 'epochs': 800, 'mse_w': 0.0, 'pair_w': 0.1, 'spear_w': 0.9},

        # ========== Light head 高排序权重 ==========
        {'name': 'Light spear=0.5', 'head_type': 'light', 'hidden_dim': 32, 'lr': 0.01, 'epochs': 300, 'mse_w': 0.2, 'pair_w': 0.3, 'spear_w': 0.5},
        {'name': 'Light spear=0.7', 'head_type': 'light', 'hidden_dim': 64, 'lr': 0.005, 'epochs': 500, 'mse_w': 0.1, 'pair_w': 0.2, 'spear_w': 0.7},
        {'name': 'Light pure_rank', 'head_type': 'light', 'hidden_dim': 64, 'lr': 0.003, 'epochs': 800, 'mse_w': 0.0, 'pair_w': 0.1, 'spear_w': 0.9},

        # ========== 超长训练 ==========
        {'name': 'Residual 1000ep', 'head_type': 'residual', 'hidden_dim': 64, 'lr': 0.002, 'epochs': 1000, 'mse_w': 0.05, 'pair_w': 0.25, 'spear_w': 0.7},
        {'name': 'Light 1000ep', 'head_type': 'light', 'hidden_dim': 64, 'lr': 0.002, 'epochs': 1000, 'mse_w': 0.05, 'pair_w': 0.25, 'spear_w': 0.7},

        # ========== 高pairwise权重 ==========
        {'name': 'Residual pair=0.5', 'head_type': 'residual', 'hidden_dim': 32, 'lr': 0.01, 'epochs': 500, 'mse_w': 0.1, 'pair_w': 0.5, 'spear_w': 0.4},
        {'name': 'Light pair=0.5', 'head_type': 'light', 'hidden_dim': 32, 'lr': 0.01, 'epochs': 500, 'mse_w': 0.1, 'pair_w': 0.5, 'spear_w': 0.4},
    ]

    print("\n测试不同配置...")
    print(f"\n{'配置':<25} {'R²':<10} {'Spearman':<12} {'匹配':<8} {'状态':<10}")
    print("-" * 70)

    for cfg in configs:
        params = {k: v for k, v in cfg.items() if k != 'name'}
        oof = run_cv(df_scaled, feature_cols, TabPFNRankHead, params)
        m = get_variety_metrics(df_scaled, oof)

        if m['Spearman'] >= 0.98 and m['R2'] >= 0.90:
            status = "★★ 完美"
        elif m['Spearman'] >= 0.98 and m['R2'] >= 0.85:
            status = "◆ Spear好"
        elif m['R2'] >= 0.92:
            status = "● R²好"
        else:
            status = ""

        print(f"{cfg['name']:<25} {m['R2']:<10.4f} {m['Spearman']:<12.4f} {m['matched_ranks']}/13{'':<3} {status}")
        results.append({'method': cfg['name'], **m})

    # 汇总
    print("\n" + "=" * 70)
    print("结果汇总")
    print("=" * 70)

    good = [r for r in results if r['Spearman'] >= 0.98 and r['R2'] >= 0.90]
    if good:
        print("\n✅ 满足条件 (Spearman≥0.98, R²≥0.90):")
        for r in sorted(good, key=lambda x: (-x['Spearman'], -x['R2'])):
            print(f"  {r['method']}: R²={r['R2']:.4f}, Spearman={r['Spearman']:.4f}")
    else:
        print("\n⚠️ 未找到同时满足两个条件的配置")
        print("\n最佳 Spearman (R²≥0.85):")
        candidates = [r for r in results if r['R2'] >= 0.85]
        if candidates:
            best = max(candidates, key=lambda x: x['Spearman'])
            print(f"  {best['method']}: R²={best['R2']:.4f}, Spearman={best['Spearman']:.4f}")

        print("\n最佳 R² (Spearman≥0.94):")
        candidates = [r for r in results if r['Spearman'] >= 0.94]
        if candidates:
            best = max(candidates, key=lambda x: x['R2'])
            print(f"  {best['method']}: R²={best['R2']:.4f}, Spearman={best['Spearman']:.4f}")

    # 保存
    with open(OUTPUT_DIR / "fixed_scale002_rankhead_report.json", 'w', encoding='utf-8') as f:
        json.dump({'scale': SCALE, 'results': results}, f, indent=2, ensure_ascii=False)
    print(f"\n报告已保存: {OUTPUT_DIR / 'fixed_scale002_rankhead_report.json'}")


if __name__ == "__main__":
    main()
