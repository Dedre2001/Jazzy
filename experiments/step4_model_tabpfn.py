"""
Step 4: TabPFN-2.5 模型 (Layer 3 - 新型NN)
"""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

# 配置 Hugging Face Token
os.environ['HUGGING_FACE_HUB_TOKEN'] = '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'
os.environ['HF_TOKEN'] = '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'

import numpy as np
import pandas as pd
import json
import time
import torch
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from step4_utils import load_data, get_variety_metrics, save_model_results, RANDOM_STATE, N_SPLITS, EXP6_DIR
from tabpfn import TabPFNRegressor

MODEL_NAME = "TabPFN"
LAYER = "Layer3"
# 默认参数（GPU验证最优；代码内显式写死，可按需改数值）
N_ESTIMATORS = 256                # 集成数
AVG_BEFORE_SOFTMAX = True         # 先平均再 softmax
SOFTMAX_T = 0.75                  # softmax 温度
INFER_PRECISION = "float32"       # 推理精度
TABPFN_DEVICE = "cuda"            # 设备
MEMORY_SAVING_MODE = "auto"       # 内存节省
# 本地权重目录（默认读取环境变量，若未设置则使用下方硬编码路径）
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_CACHE_DIR = os.environ.get("TABPFN_MODEL_CACHE_DIR", str(BASE_DIR / "tabpfn_ckpt"))
MODEL_FILE = Path(MODEL_CACHE_DIR) / "tabpfn-v2.5-regressor-v2.5_default.ckpt"
# 强制使用本地权重文件，避免联网下载
os.environ["TABPFN_MODEL_CACHE_DIR"] = str(MODEL_CACHE_DIR)
os.environ["HF_HUB_OFFLINE"] = "1"

def main():
    print(f"{'='*50}")
    print(f"运行 {MODEL_NAME} 模型")
    print(f"{'='*50}")

    # 尝试导入TabPFN
    try:
        from tabpfn import TabPFNRegressor
        print("[INFO] TabPFN 加载成功")
    except ImportError as e:
        print(f"[ERROR] TabPFN 未安装: {e}")
        print("[INFO] 请运行: pip install tabpfn")
        return None

    # 加载数据
    df, feature_sets = load_data()
    feature_cols = feature_sets['FS4']['features']
    print(f"特征数: {len(feature_cols)}")
    print(f"样本数: {len(df)}")

    X = df[feature_cols].values
    y = df['D_conv'].values

    # 存储OOF预测
    oof_predictions = np.zeros(len(y))

    kfold = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    # 指定本地权重目录（确保远程无法联网时仍可加载）
    os.environ.setdefault("TABPFN_MODEL_CACHE_DIR", MODEL_CACHE_DIR)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    # env → dtype 解析
    infer_precision = INFER_PRECISION
    if isinstance(infer_precision, str):
        lower = infer_precision.lower()
        if lower in {"float32", "fp32", "torch.float32"}:
            infer_precision = torch.float32
        elif lower in {"float64", "fp64", "torch.float64"}:
            infer_precision = torch.float64
        elif lower in {"auto", "autocast"}:
            infer_precision = lower  # 仍用 str 给 TabPFN
        else:
            print(f"[WARN] 未识别的 TABPFN_INFER_PRECISION={INFER_PRECISION}, 回退 auto")
            infer_precision = "auto"

    # 仅初始化一次模型，避免每折重复加载权重耗时；多次 fit 会覆盖旧状态
    resolved_device = TABPFN_DEVICE
    if isinstance(resolved_device, str) and resolved_device.lower() == "auto":
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
    if isinstance(infer_precision, str) and infer_precision == "autocast" and resolved_device == "cpu":
        infer_precision = "auto"

    base_model = TabPFNRegressor(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        fit_mode="fit_preprocessors",
        n_preprocessing_jobs=1,
        device=resolved_device,
        average_before_softmax=AVG_BEFORE_SOFTMAX,
        softmax_temperature=SOFTMAX_T if SOFTMAX_T > 0 else 0.9,
        inference_precision=infer_precision,
        memory_saving_mode=MEMORY_SAVING_MODE,
        model_path=str(MODEL_FILE),
    )

    for fold, (train_idx, test_idx) in enumerate(kfold.split(X)):
        fold_start = time.time()
        print(f"  Fold {fold+1}/{N_SPLITS}...")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # 折内标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 复用已加载权重，减少初始化开销
        model = base_model
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        if hasattr(y_pred, 'ndim') and y_pred.ndim > 1:
            y_pred = y_pred.ravel()

        oof_predictions[test_idx] = y_pred
        print(f"    fold耗时: {time.time() - fold_start:.1f}s")

    # 聚合到品种层
    df_result = df[['Variety', 'D_conv']].copy()
    df_result['pred'] = oof_predictions

    variety_agg = df_result.groupby('Variety').agg({
        'D_conv': 'first',
        'pred': 'mean'
    }).reset_index()

    # 计算品种层指标
    metrics = get_variety_metrics(
        variety_agg['D_conv'].values,
        variety_agg['pred'].values
    )

    # 打印结果
    print(f"\n品种层指标:")
    print(f"  R2: {metrics['R2']:.4f}")
    print(f"  RMSE: {metrics['RMSE']:.4f}")
    print(f"  Spearman rho: {metrics['Spearman']:.4f}")
    print(f"  Pairwise Acc: {metrics['Pairwise_Acc']:.4f}")
    print(f"  Hit@3: {metrics['Hit@3']:.4f}")
    print(f"  Hit@5: {metrics['Hit@5']:.4f}")

    # 保存结果
    save_model_results(MODEL_NAME, LAYER, metrics, variety_agg)

    return metrics

if __name__ == "__main__":
    main()
