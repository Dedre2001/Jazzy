"""
评估脚本（独立于原训练脚本）：
- 按 Variety 分层划分训练/测试（默认 80/20，可改 TEST_SIZE）
- 训练集上做 5 折 CV
- 在测试集上评估

默认参数与当前训练脚本保持一致，可用环境变量覆盖：
TABPFN_N_ESTIMATORS (默认 256)
TABPFN_SOFTMAX_T (默认 0.5)
TABPFN_AVG_BEFORE_SOFTMAX (默认 "1" -> True)
TABPFN_INFER_PRECISION (默认 "float32")
TABPFN_DEVICE (默认 "cuda")
TABPFN_MEMORY_SAVING_MODE (默认 "auto")
RANDOM_STATE (默认 42)
TEST_SIZE (默认 0.2)
  TABPFN_MODEL_CACHE_DIR (默认 项目根目录/tabpfn_ckpt)
"""

import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler

from step4_utils import (
    load_data,
    get_variety_metrics,
    RANDOM_STATE as DEFAULT_RANDOM_STATE,
)


def parse_env():
    cfg = {
        # 显式写死的默认最优参数（可直接修改数值，如需试验）
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
    # 强制使用本地权重并离线
    os.environ["TABPFN_MODEL_CACHE_DIR"] = cfg["model_cache_dir"]
    os.environ["HF_HUB_OFFLINE"] = "1"
    return cfg


def aggregate_variety(df, target_col="D_conv", pred_col="pred"):
    return (
        df[["Variety", target_col, pred_col]]
        .groupby("Variety")
        .agg({target_col: "first", pred_col: "mean"})
        .reset_index()
    )


def train_and_eval(cfg):
    from tabpfn import TabPFNRegressor

    df, feature_sets = load_data()
    feature_cols = feature_sets["FS4"]["features"]

    X = df[feature_cols].values
    y = df["D_conv"].values
    varieties = df["Variety"].values

    # 留出测试集（按品种分层）
    X_train, X_test, y_train, y_test, var_train, var_test = train_test_split(
        X,
        y,
        varieties,
        test_size=cfg["test_size"],
        random_state=cfg["random_state"],
        stratify=varieties,
    )

    # precision / device 解析，保持与主训练脚本一致
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
            print(f"[WARN] 未识别的 TABPFN_INFER_PRECISION={infer_precision}, 回退 auto")
            infer_precision = "auto"

    resolved_device = cfg["device"]
    if isinstance(resolved_device, str) and resolved_device.lower() == "auto":
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
    if isinstance(infer_precision, str) and infer_precision == "autocast" and resolved_device == "cpu":
        infer_precision = "auto"

    # 训练集 5 折 CV
    kfold = KFold(n_splits=5, shuffle=True, random_state=cfg["random_state"])
    oof = np.zeros_like(y_train)
    for tr_idx, te_idx in kfold.split(X_train):
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X_train[tr_idx])
        Xte = scaler.transform(X_train[te_idx])
        model = TabPFNRegressor(
            n_estimators=cfg["n_estimators"],
            random_state=cfg["random_state"],
            fit_mode="fit_preprocessors",
            n_preprocessing_jobs=1,
            device=resolved_device,
            average_before_softmax=cfg["average_before_softmax"],
            softmax_temperature=cfg["softmax_temperature"],
            inference_precision=infer_precision,
            memory_saving_mode=cfg["memory_saving_mode"],
        )
        model.fit(Xtr, y_train[tr_idx])
        preds = model.predict(Xte)
        if hasattr(preds, "ndim") and preds.ndim > 1:
            preds = preds.ravel()
        oof[te_idx] = preds
    cv_df = pd.DataFrame({"Variety": var_train, "D_conv": y_train, "pred": oof})
    cv_var = aggregate_variety(cv_df)
    cv_metrics = get_variety_metrics(cv_var["D_conv"].values, cv_var["pred"].values)

    # 测试集训练+评估
    scaler = StandardScaler()
    Xtr_full = scaler.fit_transform(X_train)
    Xte_full = scaler.transform(X_test)
    model = TabPFNRegressor(
        n_estimators=cfg["n_estimators"],
        random_state=cfg["random_state"],
        fit_mode="fit_preprocessors",
        n_preprocessing_jobs=1,
        device=resolved_device,
        average_before_softmax=cfg["average_before_softmax"],
        softmax_temperature=cfg["softmax_temperature"],
        inference_precision=infer_precision,
        memory_saving_mode=cfg["memory_saving_mode"],
    )
    model.fit(Xtr_full, y_train)
    preds = model.predict(Xte_full)
    if hasattr(preds, "ndim") and preds.ndim > 1:
        preds = preds.ravel()
    test_df = pd.DataFrame({"Variety": var_test, "D_conv": y_test, "pred": preds})
    test_var = aggregate_variety(test_df)
    test_metrics = get_variety_metrics(test_var["D_conv"].values, test_var["pred"].values)

    return cv_metrics, test_metrics, cv_var, test_var, cfg


def main():
    cfg = parse_env()
    cv_metrics, test_metrics, cv_var, test_var, cfg = train_and_eval(cfg)

    out_dir = "results/exp6"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "model_tabpfn_eval.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": cfg,
                "cv_metrics": cv_metrics,
                "test_metrics": test_metrics,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    cv_var.to_csv(os.path.join(out_dir, "model_tabpfn_cv_variety_pred.csv"), index=False)
    test_var.to_csv(os.path.join(out_dir, "model_tabpfn_test_variety_pred.csv"), index=False)

    print("[DONE] eval saved to", out_path)
    print("CV:", cv_metrics)
    print("TEST:", test_metrics)


if __name__ == "__main__":
    main()
