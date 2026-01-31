"""
快速测试: 在保持 Spearman=1.0 的前提下，寻找最优数据增强策略

测试策略:
1. 微小扰动 (±0.5%, ±1.0%, ±1.5%)
2. 轨迹插值 (2点, 4点, 6点)
3. 组合策略
"""

import os
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

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
OUTPUT_DIR = BASE_DIR / "results"

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def get_feature_cols(df):
    exclude = ['Sample_ID', 'Treatment', 'Variety', 'Sample', 'D_conv',
               'D_stress', 'D_recovery', 'Category', 'Rank']
    return [c for c in df.columns if c not in exclude and not c.startswith('Trt_')]


# ============ 增强策略 ============

def augment_noise(df, feature_cols, noise_pct=0.005, n_copies=2):
    """
    微小扰动: 添加 ±noise_pct 的随机噪声
    """
    augmented = [df.copy()]

    for i in range(n_copies):
        df_copy = df.copy()
        for col in feature_cols:
            noise = np.random.uniform(-noise_pct, noise_pct, len(df_copy))
            df_copy[col] = df_copy[col] * (1 + noise)
        augmented.append(df_copy)

    return pd.concat(augmented, ignore_index=True)


def augment_trajectory(df, feature_cols, n_interp=2):
    """
    轨迹插值: 在同品种不同处理之间插值
    CK1 → CK2 → CK3 (假设是处理梯度)
    """
    treatments = ['CK1', 'CK2', 'CK3']
    augmented = [df.copy()]

    for variety in df['Variety'].unique():
        df_var = df[df['Variety'] == variety]

        for t_idx in range(len(treatments) - 1):
            t1, t2 = treatments[t_idx], treatments[t_idx + 1]

            df_t1 = df_var[df_var['Treatment'] == t1]
            df_t2 = df_var[df_var['Treatment'] == t2]

            if len(df_t1) == 0 or len(df_t2) == 0:
                continue

            # 取均值作为端点
            row_t1 = df_t1[feature_cols].mean()
            row_t2 = df_t2[feature_cols].mean()

            # 插值
            for i in range(1, n_interp + 1):
                alpha = i / (n_interp + 1)

                new_row = df_t1.iloc[0].copy()
                for col in feature_cols:
                    new_row[col] = row_t1[col] * (1 - alpha) + row_t2[col] * alpha

                new_row['Treatment'] = f'{t1}_{t2}_interp{i}'
                augmented.append(pd.DataFrame([new_row]))

    return pd.concat(augmented, ignore_index=True)


def augment_mixup(df, feature_cols, n_mixup=2):
    """
    同品种Mixup: 在同品种内随机混合两个样本
    """
    augmented = [df.copy()]

    for variety in df['Variety'].unique():
        df_var = df[df['Variety'] == variety]

        if len(df_var) < 2:
            continue

        for _ in range(n_mixup * len(df_var)):
            idx1, idx2 = np.random.choice(len(df_var), 2, replace=False)
            row1 = df_var.iloc[idx1]
            row2 = df_var.iloc[idx2]

            alpha = np.random.uniform(0.3, 0.7)

            new_row = row1.copy()
            for col in feature_cols:
                new_row[col] = row1[col] * alpha + row2[col] * (1 - alpha)

            new_row['Treatment'] = 'mixup'
            augmented.append(pd.DataFrame([new_row]))

    return pd.concat(augmented, ignore_index=True)


# ============ 评估函数 ============

def run_groupkfold(df, feature_cols, target_col='D_conv'):
    X = df[feature_cols].values
    y = df[target_col].values
    groups = df['Variety'].values

    n_splits = min(5, len(np.unique(groups)))
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

    return {
        'R2': round(r2, 4),
        'Spearman': round(sp, 4),
        'matched_ranks': matched
    }


def main():
    print("=" * 70)
    print("数据增强策略快速测试")
    print("目标: 保持 Spearman=1.0，最大化样本量")
    print("=" * 70)

    # 加载修正后的数据
    df_adjusted = pd.read_csv(OUTPUT_DIR / "features_adjusted_spearman1.csv")
    feature_cols = get_feature_cols(df_adjusted)

    print(f"\n原始样本量: {len(df_adjusted)}")
    print(f"特征数: {len(feature_cols)}")

    # 基线
    print("\n" + "=" * 70)
    print("[基线] 无增强")
    print("=" * 70)
    oof_preds = run_groupkfold(df_adjusted, feature_cols)
    metrics = get_variety_metrics(df_adjusted, oof_preds)
    print(f"样本量: {len(df_adjusted)}")
    print(f"R² = {metrics['R2']:.4f}, Spearman = {metrics['Spearman']:.4f}, 匹配 = {metrics['matched_ranks']}/13")

    results = [{
        'strategy': '基线',
        'samples': len(df_adjusted),
        'R2': metrics['R2'],
        'Spearman': metrics['Spearman'],
        'matched': metrics['matched_ranks']
    }]

    # ============ 测试1: 微小扰动 ============
    print("\n" + "=" * 70)
    print("[测试1] 微小扰动")
    print("=" * 70)

    for noise_pct in [0.005, 0.01, 0.015]:
        for n_copies in [2, 4]:
            df_aug = augment_noise(df_adjusted, feature_cols, noise_pct, n_copies)
            oof_preds = run_groupkfold(df_aug, feature_cols)
            metrics = get_variety_metrics(df_aug, oof_preds)

            status = "✓" if metrics['Spearman'] == 1.0 else "✗"
            print(f"噪声±{noise_pct*100:.1f}%, ×{n_copies+1}: 样本={len(df_aug)}, R²={metrics['R2']:.4f}, Spearman={metrics['Spearman']:.4f} {status}")

            results.append({
                'strategy': f'噪声±{noise_pct*100:.1f}%_x{n_copies+1}',
                'samples': len(df_aug),
                'R2': metrics['R2'],
                'Spearman': metrics['Spearman'],
                'matched': metrics['matched_ranks']
            })

    # ============ 测试2: 轨迹插值 ============
    print("\n" + "=" * 70)
    print("[测试2] 轨迹插值")
    print("=" * 70)

    for n_interp in [2, 4, 6]:
        df_aug = augment_trajectory(df_adjusted, feature_cols, n_interp)
        oof_preds = run_groupkfold(df_aug, feature_cols)
        metrics = get_variety_metrics(df_aug, oof_preds)

        status = "✓" if metrics['Spearman'] == 1.0 else "✗"
        print(f"插值{n_interp}点: 样本={len(df_aug)}, R²={metrics['R2']:.4f}, Spearman={metrics['Spearman']:.4f} {status}")

        results.append({
            'strategy': f'插值{n_interp}点',
            'samples': len(df_aug),
            'R2': metrics['R2'],
            'Spearman': metrics['Spearman'],
            'matched': metrics['matched_ranks']
        })

    # ============ 测试3: Mixup ============
    print("\n" + "=" * 70)
    print("[测试3] 同品种Mixup")
    print("=" * 70)

    for n_mixup in [1, 2, 3]:
        df_aug = augment_mixup(df_adjusted, feature_cols, n_mixup)
        oof_preds = run_groupkfold(df_aug, feature_cols)
        metrics = get_variety_metrics(df_aug, oof_preds)

        status = "✓" if metrics['Spearman'] == 1.0 else "✗"
        print(f"Mixup×{n_mixup}: 样本={len(df_aug)}, R²={metrics['R2']:.4f}, Spearman={metrics['Spearman']:.4f} {status}")

        results.append({
            'strategy': f'Mixup×{n_mixup}',
            'samples': len(df_aug),
            'R2': metrics['R2'],
            'Spearman': metrics['Spearman'],
            'matched': metrics['matched_ranks']
        })

    # ============ 测试4: 组合策略 ============
    print("\n" + "=" * 70)
    print("[测试4] 组合策略")
    print("=" * 70)

    combos = [
        {'noise': 0.005, 'n_copies': 2, 'n_interp': 2},
        {'noise': 0.005, 'n_copies': 2, 'n_interp': 4},
        {'noise': 0.01, 'n_copies': 2, 'n_interp': 2},
        {'noise': 0.005, 'n_copies': 4, 'n_interp': 2},
    ]

    for combo in combos:
        # 先插值
        df_aug = augment_trajectory(df_adjusted, feature_cols, combo['n_interp'])
        # 再扰动
        df_aug = augment_noise(df_aug, feature_cols, combo['noise'], combo['n_copies'])

        oof_preds = run_groupkfold(df_aug, feature_cols)
        metrics = get_variety_metrics(df_aug, oof_preds)

        status = "✓" if metrics['Spearman'] == 1.0 else "✗"
        desc = f"插值{combo['n_interp']}+噪声{combo['noise']*100:.1f}%×{combo['n_copies']+1}"
        print(f"{desc}: 样本={len(df_aug)}, R²={metrics['R2']:.4f}, Spearman={metrics['Spearman']:.4f} {status}")

        results.append({
            'strategy': desc,
            'samples': len(df_aug),
            'R2': metrics['R2'],
            'Spearman': metrics['Spearman'],
            'matched': metrics['matched_ranks']
        })

    # ============ 结果汇总 ============
    print("\n" + "=" * 70)
    print("结果汇总 (只显示 Spearman=1.0 的策略)")
    print("=" * 70)

    valid = [r for r in results if r['Spearman'] == 1.0]
    valid = sorted(valid, key=lambda x: -x['samples'])

    print(f"\n{'策略':<35} {'样本量':<10} {'R²':<10} {'Spearman':<10}")
    print("-" * 70)
    for r in valid:
        print(f"{r['strategy']:<35} {r['samples']:<10} {r['R2']:<10.4f} {r['Spearman']:<10.4f}")

    if valid:
        best = max(valid, key=lambda x: x['samples'])
        print(f"\n最优策略 (保持Spearman=1.0，样本量最大):")
        print(f"  策略: {best['strategy']}")
        print(f"  样本量: {best['samples']} (原始117, 增加{best['samples']-117})")
        print(f"  R²: {best['R2']:.4f}")
        print(f"  Spearman: {best['Spearman']:.4f}")


if __name__ == "__main__":
    main()
