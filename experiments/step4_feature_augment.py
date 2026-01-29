"""
特征层数据增强（不改原训练脚本）
- 读取 data/processed/features_40.csv
- 生成 3x（默认）增强数据，输出：
  - data/processed/features_40_aug.csv（原始+增强）
  - data/processed/features_40_aug_only.csv（仅增强部分）
  - data/processed/feature_augment_report.json（参数与统计）

环境变量可配置：
  AUG_MULTIPLIER      默认 3            # 总样本倍数（含原始）
  AUG_NOISE_SCALE     默认 0.02         # 噪声系数，乘以列标准差
  AUG_MIXUP_ALPHA     默认 0.4          # Beta 分布参数，0 表示关闭 MixUp
  AUG_SEED            默认 42
  INPUT_PATH          默认 data/processed/features_40.csv
  OUTPUT_PATH         默认 data/processed/features_40_aug.csv
  OUTPUT_AUG_ONLY     默认 data/processed/features_40_aug_only.csv
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


def load_base_data(path: Path):
    df = pd.read_csv(path)
    return df


def detect_bool_columns(df: pd.DataFrame):
    bool_cols = []
    num_cols = []
    for col in df.columns:
        if df[col].dtype == bool:
            bool_cols.append(col)
        elif pd.api.types.is_numeric_dtype(df[col]):
            uniques = df[col].dropna().unique()
            if len(uniques) <= 2 and set(uniques).issubset({0, 1}):
                bool_cols.append(col)
            else:
                num_cols.append(col)
    return bool_cols, num_cols


def make_noise_samples(df, num_cols, count, noise_scale, rng):
    """在数值列上加噪声，布尔/其他列保持不变"""
    base_idx = rng.integers(0, len(df), size=count)
    samples = df.iloc[base_idx].copy().reset_index(drop=True)
    if count == 0 or not num_cols:
        return samples
    stds = df[num_cols].std()
    noise = rng.normal(loc=0.0, scale=1.0, size=(count, len(num_cols)))
    samples[num_cols] = samples[num_cols] + noise * (stds.values * noise_scale)
    return samples


def make_mixup_samples(df, num_cols, bool_cols, count, alpha, rng):
    """同品种内 MixUp；布尔列用阈值0.5 还原"""
    if count == 0 or alpha <= 0:
        return pd.DataFrame(columns=df.columns)
    # 按 Variety 分组
    if "Variety" not in df.columns:
        raise ValueError("MixUp 需要 Variety 列进行分组")
    groups = df.groupby("Variety")
    out_rows = []
    varieties = list(groups.groups.keys())
    for _ in range(count):
        var = rng.choice(varieties)
        g = groups.get_group(var)
        if len(g) == 1:
            a = b = g.iloc[0]
        else:
            a, b = g.sample(2, replace=True, random_state=int(rng.integers(0, 1e9))).to_dict(
                "records"
            )
        lam = rng.beta(alpha, alpha)
        mixed = {}
        for col in df.columns:
            va, vb = a[col], b[col]
            if col in num_cols:
                mixed[col] = lam * va + (1 - lam) * vb
            elif col in bool_cols:
                val = lam * va + (1 - lam) * vb
                mixed[col] = 1 if val >= 0.5 else 0
            else:
                # 非数值列（例如 Sample_ID）直接取 a
                mixed[col] = va
        out_rows.append(mixed)
    return pd.DataFrame(out_rows)


def main():
    input_path = Path(os.environ.get("INPUT_PATH", "data/processed/features_40.csv"))
    output_path = Path(os.environ.get("OUTPUT_PATH", "data/processed/features_40_aug.csv"))
    output_aug_only = Path(
        os.environ.get("OUTPUT_AUG_ONLY", "data/processed/features_40_aug_only.csv")
    )
    multiplier = float(os.environ.get("AUG_MULTIPLIER", 3))
    noise_scale = float(os.environ.get("AUG_NOISE_SCALE", 0.02))
    mixup_alpha = float(os.environ.get("AUG_MIXUP_ALPHA", 0.4))
    seed = int(os.environ.get("AUG_SEED", 42))

    rng = np.random.default_rng(seed)
    df = load_base_data(input_path)

    bool_cols, num_cols = detect_bool_columns(df)

    base_n = len(df)
    target_total = int(round(base_n * multiplier))
    aug_n = max(target_total - base_n, 0)
    noise_n = aug_n // 2
    mixup_n = aug_n - noise_n

    noise_df = make_noise_samples(df, num_cols, noise_n, noise_scale, rng)
    mixup_df = make_mixup_samples(df, num_cols, bool_cols, mixup_n, mixup_alpha, rng)
    aug_df = pd.concat([noise_df, mixup_df], axis=0).reset_index(drop=True)

    combined = pd.concat([df, aug_df], axis=0).sample(frac=1.0, random_state=seed).reset_index(
        drop=True
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_aug_only.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)
    aug_df.to_csv(output_aug_only, index=False)

    # 简要报告
    report = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "output_aug_only": str(output_aug_only),
        "base_samples": base_n,
        "aug_samples": aug_n,
        "total_samples": len(combined),
        "multiplier": multiplier,
        "noise_scale": noise_scale,
        "mixup_alpha": mixup_alpha,
        "seed": seed,
        "num_cols": num_cols,
        "bool_cols": bool_cols,
    }
    with open("data/processed/feature_augment_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("[DONE] augmentation finished")
    print("base:", base_n, "aug:", aug_n, "total:", len(combined))
    print("saved:", output_path)


if __name__ == "__main__":
    main()
