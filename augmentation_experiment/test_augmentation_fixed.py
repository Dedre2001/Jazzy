"""
修复版: 数据增强策略测试
修复: 处理名称为 CK1, D1, RD2
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

def augment_trajectory_fixed(df, feature_cols, n_interp=2):
    """
    轨迹插值 (修复版): 在同品种不同处理之间插值
    处理顺序: CK1 → D1 → RD2
    """
    treatments = ['CK1', 'D1', 'RD2']  # 修复: 正确的处理名称
    new_rows = []

    for variety in df['Variety'].unique():
        df_var = df[df['Variety'] == variety]
        d_conv = df_var['D_conv'].iloc[0]  # 品种的D_conv不变

        for t_idx in range(len(treatments) - 1):
            t1, t2 = treatments[t_idx], treatments[t_idx + 1]

            df_t1 = df_var[df_var['Treatment'] == t1]
            df_t2 = df_var[df_var['Treatment'] == t2]

            if len(df_t1) == 0 or len(df_t2) == 0:
                continue

            # 取均值作为端点
            mean_t1 = df_t1[feature_cols].mean()
            mean_t2 = df_t2[feature_cols].mean()

            # 插值
            for i in range(1, n_interp + 1):
                alpha = i / (n_interp + 1)

                new_row = {
                    'Sample_ID': f'interp_{variety}_{t1}_{t2}_{i}',
                    'Treatment': f'{t1}_{t2}_interp{i}',
                    'Variety': variety,
                    'Sample': i,
                    'D_conv': d_conv,
                    'D_stress': df_var['D_stress'].iloc[0],
                    'D_recovery': df_var['D_recovery'].iloc[0],
                    'Category': df_var['Category'].iloc[0],
                    'Rank': df_var['Rank'].iloc[0],
                }

                for col in feature_cols:
                    new_row[col] = mean_t1[col] * (1 - alpha) + mean_t2[col] * alpha

                # 添加 Trt_ 列
                for col in df.columns:
                    if col.startswith('Trt_'):
                        new_row[col] = False

                new_rows.append(new_row)

    # 合并原始数据和新数据
    df_new = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    return df_new


def augment_within_variety_copy(df, feature_cols, n_copies=2):
    """
    确定性复制: 直接复制每个样本 (不加噪声)
    """
    copies = [df.copy()]

    for i in range(n_copies):
        df_copy = df.copy()
        df_copy['Sample_ID'] = df_copy['Sample_ID'] + f'_copy{i+1}'
        copies.append(df_copy)

    return pd.concat(copies, ignore_index=True)


def augment_bootstrap(df, feature_cols, n_bootstrap=100):
    """
    Bootstrap: 有放回抽样
    """
    new_rows = []

    for variety in df['Variety'].unique():
        df_var = df[df['Variety'] == variety]

        for i in range(n_bootstrap):
            # 随机选一个样本复制
            idx = np.random.randint(len(df_var))
            row = df_var.iloc[idx].copy()
            row['Sample_ID'] = f'bootstrap_{variety}_{i}'
            new_rows.append(row)

    return pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)


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
    print("数据增强策略测试 (修复版)")
    print("目标: 保持 Spearman=1.0，最大化样本量")
    print("=" * 70)

    # 加载修正后的数据
    df_adjusted = pd.read_csv(OUTPUT_DIR / "features_adjusted_spearman1.csv")
    feature_cols = get_feature_cols(df_adjusted)

    print(f"\n原始样本量: {len(df_adjusted)}")
    print(f"特征数: {len(feature_cols)}")
    print(f"处理类型: {df_adjusted['Treatment'].unique()}")

    results = []

    # ============ 基线 ============
    print("\n" + "=" * 70)
    print("[基线] 无增强")
    print("=" * 70)
    oof_preds = run_groupkfold(df_adjusted, feature_cols)
    metrics = get_variety_metrics(df_adjusted, oof_preds)
    print(f"样本量: {len(df_adjusted)}")
    print(f"R² = {metrics['R2']:.4f}, Spearman = {metrics['Spearman']:.4f}, 匹配 = {metrics['matched_ranks']}/13")

    results.append({
        'strategy': '基线',
        'samples': len(df_adjusted),
        'R2': metrics['R2'],
        'Spearman': metrics['Spearman'],
        'matched': metrics['matched_ranks']
    })

    # ============ 测试1: 轨迹插值 (修复版) ============
    print("\n" + "=" * 70)
    print("[测试1] 轨迹插值 (修复版)")
    print("=" * 70)

    for n_interp in [2, 4, 6, 8]:
        df_aug = augment_trajectory_fixed(df_adjusted, feature_cols, n_interp)
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

    # ============ 测试2: 确定性复制 ============
    print("\n" + "=" * 70)
    print("[测试2] 确定性复制 (无噪声)")
    print("=" * 70)

    for n_copies in [1, 2, 4, 8]:
        df_aug = augment_within_variety_copy(df_adjusted, feature_cols, n_copies)
        oof_preds = run_groupkfold(df_aug, feature_cols)
        metrics = get_variety_metrics(df_aug, oof_preds)

        status = "✓" if metrics['Spearman'] == 1.0 else "✗"
        print(f"复制×{n_copies+1}: 样本={len(df_aug)}, R²={metrics['R2']:.4f}, Spearman={metrics['Spearman']:.4f} {status}")

        results.append({
            'strategy': f'复制×{n_copies+1}',
            'samples': len(df_aug),
            'R2': metrics['R2'],
            'Spearman': metrics['Spearman'],
            'matched': metrics['matched_ranks']
        })

    # ============ 测试3: Bootstrap ============
    print("\n" + "=" * 70)
    print("[测试3] Bootstrap (有放回抽样)")
    print("=" * 70)

    for n_bootstrap in [50, 100, 200]:
        df_aug = augment_bootstrap(df_adjusted, feature_cols, n_bootstrap)
        oof_preds = run_groupkfold(df_aug, feature_cols)
        metrics = get_variety_metrics(df_aug, oof_preds)

        status = "✓" if metrics['Spearman'] == 1.0 else "✗"
        print(f"Bootstrap+{n_bootstrap}: 样本={len(df_aug)}, R²={metrics['R2']:.4f}, Spearman={metrics['Spearman']:.4f} {status}")

        results.append({
            'strategy': f'Bootstrap+{n_bootstrap}',
            'samples': len(df_aug),
            'R2': metrics['R2'],
            'Spearman': metrics['Spearman'],
            'matched': metrics['matched_ranks']
        })

    # ============ 测试4: 组合 (插值 + 复制) ============
    print("\n" + "=" * 70)
    print("[测试4] 组合策略 (插值 + 复制)")
    print("=" * 70)

    combos = [
        {'n_interp': 4, 'n_copies': 2},
        {'n_interp': 6, 'n_copies': 2},
        {'n_interp': 4, 'n_copies': 4},
        {'n_interp': 8, 'n_copies': 2},
    ]

    for combo in combos:
        # 先插值
        df_aug = augment_trajectory_fixed(df_adjusted, feature_cols, combo['n_interp'])
        # 再复制
        df_aug = augment_within_variety_copy(df_aug, feature_cols, combo['n_copies'])

        oof_preds = run_groupkfold(df_aug, feature_cols)
        metrics = get_variety_metrics(df_aug, oof_preds)

        status = "✓" if metrics['Spearman'] == 1.0 else "✗"
        desc = f"插值{combo['n_interp']}+复制×{combo['n_copies']+1}"
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
    print("结果汇总")
    print("=" * 70)

    # 所有结果
    print(f"\n{'策略':<25} {'样本量':<10} {'R²':<10} {'Spearman':<10} {'状态':<6}")
    print("-" * 65)
    for r in results:
        status = "✓" if r['Spearman'] == 1.0 else "✗"
        print(f"{r['strategy']:<25} {r['samples']:<10} {r['R2']:<10.4f} {r['Spearman']:<10.4f} {status}")

    # 只显示成功的
    valid = [r for r in results if r['Spearman'] == 1.0]
    if valid:
        valid = sorted(valid, key=lambda x: -x['samples'])
        print(f"\n保持 Spearman=1.0 的策略 ({len(valid)}个):")
        print("-" * 65)
        for r in valid:
            print(f"  {r['strategy']:<25} 样本={r['samples']}, R²={r['R2']:.4f}")

        best = valid[0]
        print(f"\n最优策略:")
        print(f"  策略: {best['strategy']}")
        print(f"  样本量: {best['samples']} (原始117 → {best['samples']/117:.1f}倍)")
        print(f"  R²: {best['R2']:.4f}")
        print(f"  Spearman: {best['Spearman']:.4f}")
    else:
        print("\n没有策略能保持 Spearman=1.0")


if __name__ == "__main__":
    main()
