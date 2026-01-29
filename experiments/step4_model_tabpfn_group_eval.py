"""
分组评估脚本（不改主训练文件）：
- 按品种分组做 GroupKFold（品种级 CV），估计“新样本同品种”泛化
- 可选留出测试集按品种分层（默认 80/20），报告测试指标

默认参数对齐主训练脚本，可用环境变量覆盖：
TABPFN_N_ESTIMATORS=256
TABPFN_SOFTMAX_T=0.5
TABPFN_AVG_BEFORE_SOFTMAX=1
TABPFN_INFER_PRECISION=float32
TABPFN_DEVICE=cuda
TABPFN_MEMORY_SAVING_MODE=auto
RANDOM_STATE=42
TEST_SIZE=0.2
TABPFN_MODEL_CACHE_DIR=项目根目录/tabpfn_ckpt
"""

import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler

from step4_utils import (
    load_data,
    get_variety_metrics,
    RANDOM_STATE as DEFAULT_RANDOM_STATE,
)


def parse_env():
    cfg = {
        # 显式写死的默认最优参数
        "n_estimators": 256,
        "softmax_temperature": 0.75,
        "average_before_softmax": True,
        "inference_precision": "float32",
        "device": "cuda",
        "memory_saving_mode": "auto",
        "random_state": int(os.environ.get("RANDOM_STATE", DEFAULT_RANDOM_STATE)),
        "test_size": float(os.environ.get("TEST_SIZE", 0.2)),
        "model_cache_dir": str(Path(__file__).resolve().parent.parent / "tabpfn_ckpt"),
    }
    os.environ["TABPFN_MODEL_CACHE_DIR"] = cfg["model_cache_dir"]
    os.environ["HF_HUB_OFFLINE"] = "1"
    return cfg


def resolve_precision_and_device(cfg):
    infer_precision = cfg["inference_precision"]
    if isinstance(infer_precision, str):
        lower = infer_precision.lower()
        if lower in {"float32", "fp32", "torch.float32"}:
            infer_precision = torch.float32
        elif lower in {"float64", "fp64", "torch.float64"}:
            infer_precision = torch.float64
        elif lower in {"auto", "autocast"}:
            infer_precision = lower
        else:
            print(f"[WARN] 未识别 TABPFN_INFER_PRECISION={infer_precision}, 回退 auto")
            infer_precision = "auto"
    device = cfg["device"]
    if isinstance(device, str) and device.lower() == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if isinstance(infer_precision, str) and infer_precision == "autocast" and device == "cpu":
        infer_precision = "auto"
    return infer_precision, device


def run_group_cv(cfg, X, y, varieties, infer_precision, device):
    from tabpfn import TabPFNRegressor

    gkf = GroupKFold(n_splits=len(np.unique(varieties)))  # Leave-One-Variety-Out
    oof = np.zeros_like(y)
    for train_idx, test_idx in gkf.split(X, y, groups=varieties):
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X[train_idx])
        Xte = scaler.transform(X[test_idx])
        model = TabPFNRegressor(
            n_estimators=cfg["n_estimators"],
            random_state=cfg["random_state"],
            fit_mode="fit_preprocessors",
            n_preprocessing_jobs=1,
            device=device,
            average_before_softmax=cfg["average_before_softmax"],
            softmax_temperature=cfg["softmax_temperature"],
            inference_precision=infer_precision,
            memory_saving_mode=cfg["memory_saving_mode"],
        )
        model.fit(Xtr, y[train_idx])
        preds = model.predict(Xte)
        if hasattr(preds, "ndim") and preds.ndim > 1:
            preds = preds.ravel()
        oof[test_idx] = preds
    df_cv = pd.DataFrame({"Variety": varieties, "D_conv": y, "pred": oof})
    var_cv = (
        df_cv.groupby("Variety").agg({"D_conv": "first", "pred": "mean"}).reset_index()
    )
    metrics_cv = get_variety_metrics(var_cv["D_conv"].values, var_cv["pred"].values)
    return metrics_cv, var_cv


def run_train_test(cfg, X, y, varieties, infer_precision, device):
    from tabpfn import TabPFNRegressor

    Xtr, Xte, ytr, yte, var_tr, var_te = train_test_split(
        X,
        y,
        varieties,
        test_size=cfg["test_size"],
        random_state=cfg["random_state"],
        stratify=varieties,
    )
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)
    model = TabPFNRegressor(
        n_estimators=cfg["n_estimators"],
        random_state=cfg["random_state"],
        fit_mode="fit_preprocessors",
        n_preprocessing_jobs=1,
        device=device,
        average_before_softmax=cfg["average_before_softmax"],
        softmax_temperature=cfg["softmax_temperature"],
        inference_precision=infer_precision,
        memory_saving_mode=cfg["memory_saving_mode"],
    )
    model.fit(Xtr_s, ytr)
    preds = model.predict(Xte_s)
    if hasattr(preds, "ndim") and preds.ndim > 1:
        preds = preds.ravel()
    df_test = pd.DataFrame({"Variety": var_te, "D_conv": yte, "pred": preds})
    var_test = (
        df_test.groupby("Variety").agg({"D_conv": "first", "pred": "mean"}).reset_index()
    )
    metrics_test = get_variety_metrics(var_test["D_conv"].values, var_test["pred"].values)
    return metrics_test, var_test


def main():
    cfg = parse_env()
    infer_precision, device = resolve_precision_and_device(cfg)

    df, feature_sets = load_data()
    feature_cols = feature_sets["FS4"]["features"]
    X = df[feature_cols].values
    y = df["D_conv"].values
    varieties = df["Variety"].values

    metrics_cv, var_cv = run_group_cv(cfg, X, y, varieties, infer_precision, device)
    metrics_test, var_test = run_train_test(cfg, X, y, varieties, infer_precision, device)

    out_dir = "results/exp6"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "model_tabpfn_group_eval.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": cfg,
                "group_cv_metrics": metrics_cv,
                "test_metrics": metrics_test,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    var_cv.to_csv(os.path.join(out_dir, "model_tabpfn_group_cv_variety_pred.csv"), index=False)
    var_test.to_csv(os.path.join(out_dir, "model_tabpfn_group_test_variety_pred.csv"), index=False)

    print("[DONE] group eval saved to", out_path)
    print("GroupCV (leave-one-variety-out):", metrics_cv)
    print("Test (stratified holdout):", metrics_test)


if __name__ == "__main__":
    main()
