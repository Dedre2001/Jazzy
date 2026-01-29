"""
完整过拟合诊断脚本
- 对比随机 KFold vs GroupKFold（品种划分）
- 检测数据泄露、过拟合、欠拟合
- 生成可视化图表和诊断报告
"""

import os
import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold, GroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 尝试导入 matplotlib，如果没有则跳过绘图
try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("[WARN] matplotlib 未安装，将跳过可视化")

# 假设你的 utils 文件路径
try:
    from step4_utils import load_data, get_variety_metrics, RANDOM_STATE
except ImportError:
    # 如果导入失败，使用默认配置
    RANDOM_STATE = 42


    def load_data():
        """请替换为你的实际数据加载逻辑"""
        raise NotImplementedError("请修改代码中的 load_data() 函数或使用你的 step4_utils")


    def get_variety_metrics(y_true, y_pred):
        """计算品种级指标"""
        return {
            "r2": r2_score(y_true, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
            "mae": mean_absolute_error(y_true, y_pred),
            "pearson": np.corrcoef(y_true, y_pred)[0, 1] if len(y_true) > 1 else 0
        }


def parse_env():
    """解析环境变量配置"""
    cfg = {
        "n_estimators": int(os.environ.get("TABPFN_N_ESTIMATORS", 256)),
        "softmax_temperature": float(os.environ.get("TABPFN_SOFTMAX_T", 0.75)),
        "average_before_softmax": os.environ.get("TABPFN_AVG_BEFORE_SOFTMAX", "1") == "1",
        "inference_precision": os.environ.get("TABPFN_INFER_PRECISION", "float32"),
        "device": os.environ.get("TABPFN_DEVICE", "cuda"),
        "memory_saving_mode": os.environ.get("TABPFN_MEMORY_SAVING_MODE", "auto"),
        "random_state": int(os.environ.get("RANDOM_STATE", RANDOM_STATE)),
        "test_size": float(os.environ.get("TEST_SIZE", 0.2)),
        "model_cache_dir": os.environ.get(
            "TABPFN_MODEL_CACHE_DIR",
            str(Path(__file__).resolve().parent.parent / "tabpfn_ckpt"),
        ),
        # 样本级幅值归一化开关；默认开启
        "sample_level_norm": os.environ.get("SAMPLE_LEVEL_NORM", "1") == "1",
        # 可选剔除样本列表（逗号分隔 Sample_ID），默认空
        "exclude_sample_ids": [
            s for s in os.environ.get("EXCLUDE_SAMPLE_IDS", "").split(",") if s.strip()
        ],
    }
    os.environ.setdefault("TABPFN_MODEL_CACHE_DIR", cfg["model_cache_dir"])
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    return cfg


def resolve_precision_and_device(cfg):
    """处理精度和设备配置"""
    infer_precision = cfg["inference_precision"]
    if isinstance(infer_precision, str):
        lower = infer_precision.lower()
        if lower in {"float32", "fp32"}:
            infer_precision = torch.float32
        elif lower in {"float64", "fp64"}:
            infer_precision = torch.float64
        elif lower in {"auto", "autocast"}:
            infer_precision = "auto"

    device = cfg["device"]
    if isinstance(device, str) and device.lower() == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if isinstance(infer_precision, str) and infer_precision == "autocast" and device == "cpu":
        infer_precision = "auto"

    return infer_precision, device


def get_metrics(y_true, y_pred):
    """计算回归指标"""
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "pearson": float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 1 else 0.0
    }


def sample_level_normalize(df, band_cols, static_cols):
    """按样本对光谱/静态荧光做幅值归一化（z-score per sample）"""
    X = df.copy()
    # 光谱
    mean_band = X[band_cols].mean(axis=1)
    std_band = X[band_cols].std(axis=1) + 1e-9
    X[band_cols] = X[band_cols].sub(mean_band, axis=0).div(std_band, axis=0)
    # 静态荧光
    mean_static = X[static_cols].mean(axis=1)
    std_static = X[static_cols].std(axis=1) + 1e-9
    X[static_cols] = X[static_cols].sub(mean_static, axis=0).div(std_static, axis=0)
    return X


def train_eval_fold(X_train, y_train, X_test, y_test, cfg, infer_precision, device):
    """训练并评估一个 fold"""
    from tabpfn import TabPFNRegressor

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)

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

    model.fit(Xtr, y_train)
    preds = model.predict(Xte)
    if hasattr(preds, "ndim") and preds.ndim > 1:
        preds = preds.ravel()

    return preds, get_metrics(y_test, preds)


def run_random_kfold(X, y, varieties, cfg, infer_precision, device):
    """
    随机 KFold（可能泄露品种信息）
    这是原来的方式，用于对比
    """
    print("\n[1/4] 运行随机 KFold CV（原方法，可能泄露）...")

    kf = KFold(n_splits=5, shuffle=True, random_state=cfg["random_state"])
    oof_preds = np.zeros_like(y)
    fold_metrics = []

    for fold, (tr_idx, te_idx) in enumerate(kf.split(X)):
        print(f"  Fold {fold + 1}/5...")
        preds, metrics = train_eval_fold(
            X[tr_idx], y[tr_idx], X[te_idx], y[te_idx],
            cfg, infer_precision, device
        )
        oof_preds[te_idx] = preds
        fold_metrics.append(metrics)

    # 聚合到品种级（如果有多行同品种，取平均）
    df = pd.DataFrame({"Variety": varieties, "y": y, "pred": oof_preds})
    var_level = df.groupby("Variety").agg({"y": "first", "pred": "mean"}).reset_index()

    overall_metrics = get_metrics(var_level["y"].values, var_level["pred"].values)

    return {
        "method": "Random KFold",
        "overall": overall_metrics,
        "folds": fold_metrics,
        "predictions": oof_preds,
        "variety_level": var_level
    }


def run_group_kfold(X, y, varieties, cfg, infer_precision, device):
    """
    GroupKFold（严格按品种划分，无泄露）
    这是诊断的关键
    """
    print("\n[2/4] 运行 GroupKFold CV（品种级划分，严格）...")

    n_unique = len(np.unique(varieties))
    n_splits = min(5, n_unique)  # 避免折数过多导致不稳定

    gkf = GroupKFold(n_splits=n_splits)
    oof_preds = np.zeros_like(y)
    fold_metrics = []

    for fold, (tr_idx, te_idx) in enumerate(gkf.split(X, y, groups=varieties)):
        print(f"  Fold {fold + 1}/{n_splits}...")
        # 显示当前 fold 的划分情况
        tr_var = set(varieties[tr_idx])
        te_var = set(varieties[te_idx])
        print(f"    训练品种数: {len(tr_var)}, 验证品种数: {len(te_var)}")

        preds, metrics = train_eval_fold(
            X[tr_idx], y[tr_idx], X[te_idx], y[te_idx],
            cfg, infer_precision, device
        )
        oof_preds[te_idx] = preds
        fold_metrics.append(metrics)

    # 聚合到品种级
    df = pd.DataFrame({"Variety": varieties, "y": y, "pred": oof_preds})
    var_level = df.groupby("Variety").agg({"y": "first", "pred": "mean"}).reset_index()

    overall_metrics = get_metrics(var_level["y"].values, var_level["pred"].values)

    return {
        "method": "GroupKFold",
        "overall": overall_metrics,
        "folds": fold_metrics,
        "predictions": oof_preds,
        "variety_level": var_level
    }


def run_holdout_test(X, y, varieties, cfg, infer_precision, device):
    """
    留出法测试集（80/20 分层）
    """
    print("\n[3/4] 运行留出测试集（80/20 分层）...")

    X_tr, X_te, y_tr, y_te, var_tr, var_te = train_test_split(
        X, y, varieties,
        test_size=cfg["test_size"],
        random_state=cfg["random_state"],
        stratify=varieties
    )

    preds, metrics = train_eval_fold(X_tr, y_tr, X_te, y_te, cfg, infer_precision, device)

    # 聚合到品种级
    df = pd.DataFrame({"Variety": var_te, "y": y_te, "pred": preds})
    var_level = df.groupby("Variety").agg({"y": "first", "pred": "mean"}).reset_index()

    return {
        "method": "Holdout Test",
        "overall": metrics,
        "predictions": preds,
        "variety_level": var_level,
        "test_varieties": list(set(var_te))
    }


def run_internal_overfit_check(X, y, cfg, infer_precision, device):
    """
    训练集内部过拟合检查（训练集上再跑 KFold，看 train vs val gap）
    """
    print("\n[4/4] 运行训练集内部过拟合检查...")

    # 先划分训练集
    X_tr, _, y_tr, _, _, _ = train_test_split(
        X, y, np.arange(len(y)),  # 占位
        test_size=cfg["test_size"],
        random_state=cfg["random_state"]
    )

    # 在训练集内部做 KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=cfg["random_state"])
    train_scores = []
    val_scores = []

    for tr_idx, val_idx in kf.split(X_tr):
        scaler = StandardScaler()
        X_subtrain = scaler.fit_transform(X_tr[tr_idx])
        X_val = scaler.transform(X_tr[val_idx])

        from tabpfn import TabPFNRegressor
        model = TabPFNRegressor(
            n_estimators=cfg["n_estimators"],
            random_state=cfg["random_state"],
            fit_mode="fit_preprocessors",
            device=device,
            average_before_softmax=cfg["average_before_softmax"],
            softmax_temperature=cfg["softmax_temperature"],
            inference_precision=infer_precision,
            memory_saving_mode=cfg["memory_saving_mode"],
        )
        model.fit(X_subtrain, y_tr[tr_idx])

        # 训练集分数（检查记忆能力）
        train_pred = model.predict(X_subtrain)
        train_r2 = r2_score(y_tr[tr_idx], train_pred)

        # 验证集分数（检查泛化能力）
        val_pred = model.predict(X_val)
        val_r2 = r2_score(y_tr[val_idx], val_pred)

        train_scores.append(train_r2)
        val_scores.append(val_r2)

    gap = np.mean(train_scores) - np.mean(val_scores)

    return {
        "method": "Internal Overfit Check",
        "train_r2_mean": float(np.mean(train_scores)),
        "train_r2_std": float(np.std(train_scores)),
        "val_r2_mean": float(np.mean(val_scores)),
        "val_r2_std": float(np.std(val_scores)),
        "gap": float(gap),
        "is_overfitting": gap > 0.15  # R2 差距 > 0.15 认为过拟合
    }


def diagnose(random_cv, group_cv, holdout, internal):
    """
    生成诊断结论
    """
    print("\n" + "=" * 60)
    print("诊断结论")
    print("=" * 60)

    r2_random = random_cv["overall"]["r2"]
    r2_group = group_cv["overall"]["r2"]
    r2_test = holdout["overall"]["r2"]

    print(f"\n1. 性能对比:")
    print(f"   随机 KFold (可能泄露): R² = {r2_random:.3f}")
    print(f"   GroupKFold (严格划分): R² = {r2_group:.3f}")
    print(f"   留出测试集:           R² = {r2_test:.3f}")

    gap_leak = r2_random - r2_group
    gap_general = r2_group - r2_test

    print(f"\n2. 差距分析:")
    print(f"   Random vs Group 差距: {gap_leak:.3f}")
    print(f"   Group vs Test 差距:   {gap_general:.3f}")

    # 判断逻辑
    if gap_leak > 0.15:
        print(f"\n3. [主要问题] CV 数据泄露!")
        print("   -> 随机 KFold 把同品种分到不同折，模型记忆了品种特异性")
        print("   -> 真实泛化能力应以 GroupKFold 为准")
        primary_issue = "data_leakage"
    elif internal["is_overfitting"]:
        print(f"\n3. [主要问题] 模型过拟合!")
        print(f"   -> 训练集内部 R² = {internal['train_r2_mean']:.3f}")
        print(f"   -> 验证集内部 R² = {internal['val_r2_mean']:.3f}")
        print(f"   -> Gap = {internal['gap']:.3f}")
        primary_issue = "overfitting"
    elif gap_general > 0.10:
        print(f"\n3. [主要问题] 分布偏移 (Domain Shift)!")
        print("   -> Group CV 与 Test 差距大，测试集品种与训练集有系统性差异")
        print("   -> 建议检查：测试集品种是否与训练集来自不同年份/产地?")
        primary_issue = "domain_shift"
    else:
        print(f"\n3. [结论] 模型正常，无严重过拟合")
        print("   -> 各评估方式结果一致，性能稳定")
        primary_issue = "normal"

    # 针对品种的分析
    group_df = group_cv["variety_level"]
    worst_varieties = group_df.copy()
    worst_varieties["error"] = np.abs(worst_varieties["y"] - worst_varieties["pred"])
    worst_varieties = worst_varieties.nlargest(3, "error")

    print(f"\n4. 难预测品种 (Group CV):")
    for _, row in worst_varieties.iterrows():
        print(f"   {row['Variety']}: 真实={row['y']:.2f}, 预测={row['pred']:.2f}, 误差={row['error']:.2f}")

    return {
        "primary_issue": primary_issue,
        "random_r2": r2_random,
        "group_r2": r2_group,
        "test_r2": r2_test,
        "gap_leakage": gap_leak,
        "gap_generalization": gap_general,
        "worst_varieties": worst_varieties.to_dict()
    }


def visualize(random_cv, group_cv, holdout, save_dir):
    """生成可视化图表"""
    if not MATPLOTLIB_AVAILABLE:
        print("\n[WARN] 跳过可视化（matplotlib 未安装）")
        return

    print("\n生成可视化...")
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. 方法对比柱状图
    ax1 = axes[0, 0]
    methods = ["Random\nKFold", "Group\nKFold", "Holdout\nTest"]
    r2_values = [
        random_cv["overall"]["r2"],
        group_cv["overall"]["r2"],
        holdout["overall"]["r2"]
    ]
    colors = ["#ff7f0e", "#2ca02c", "#1f77b4"]
    bars = ax1.bar(methods, r2_values, color=colors, alpha=0.7, edgecolor="black")
    ax1.set_ylabel("R² Score")
    ax1.set_title("Model Performance Comparison")
    ax1.set_ylim(0, 1.0)
    ax1.axhline(y=max(r2_values), color="gray", linestyle="--", alpha=0.5)

    # 在柱子上标数值
    for bar, val in zip(bars, r2_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height,
                 f"{val:.3f}", ha="center", va="bottom", fontweight="bold")

    # 2. 残差分布对比
    ax2 = axes[0, 1]
    random_resid = random_cv["variety_level"]["y"] - random_cv["variety_level"]["pred"]
    group_resid = group_cv["variety_level"]["y"] - group_cv["variety_level"]["pred"]

    ax2.hist(random_resid, bins=15, alpha=0.5, label="Random KFold", color="#ff7f0e")
    ax2.hist(group_resid, bins=15, alpha=0.5, label="Group KFold", color="#2ca02c")
    ax2.axvline(x=0, color="red", linestyle="--")
    ax2.set_xlabel("Residual (True - Pred)")
    ax2.set_ylabel("Count")
    ax2.set_title("Residual Distribution")
    ax2.legend()

    # 3. 预测值 vs 真实值散点图 (Group CV)
    ax3 = axes[1, 0]
    x_vals = group_cv["variety_level"]["y"]
    y_vals = group_cv["variety_level"]["pred"]
    ax3.scatter(x_vals, y_vals, alpha=0.6, s=60, color="#2ca02c")

    # 对角线
    min_val = min(x_vals.min(), y_vals.min())
    max_val = max(x_vals.max(), y_vals.max())
    ax3.plot([min_val, max_val], [min_val, max_val], "r--", lw=2)

    ax3.set_xlabel("True D_conv")
    ax3.set_ylabel("Predicted D_conv")
    ax3.set_title(f"GroupKFold: True vs Pred (R²={group_cv['overall']['r2']:.3f})")

    # 4. 品种误差条形图
    ax4 = axes[1, 1]
    group_df = group_cv["variety_level"].copy()
    group_df["abs_error"] = np.abs(group_df["y"] - group_df["pred"])
    group_df = group_df.sort_values("abs_error", ascending=True).tail(10)  # 取误差最大的10个

    ax4.barh(range(len(group_df)), group_df["abs_error"].values, color="coral")
    ax4.set_yticks(range(len(group_df)))
    ax4.set_yticklabels(group_df["Variety"].values, fontsize=8)
    ax4.set_xlabel("Absolute Error")
    ax4.set_title("Top 10 Worst Predicted Varieties (Group CV)")

    plt.tight_layout()
    save_path = save_dir / "diagnosis_plots.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[保存] 可视化图表: {save_path}")


def main():
    print("=" * 60)
    print("TabPFN 过拟合诊断工具")
    print("=" * 60)

    cfg = parse_env()
    infer_precision, device = resolve_precision_and_device(cfg)

    print(f"\n配置信息:")
    print(f"  设备: {device}")
    print(f"  精度: {cfg['inference_precision']}")
    print(f"  n_estimators: {cfg['n_estimators']}")

    # 加载数据
    print("\n加载数据...")
    df, feature_sets = load_data()
    feature_cols = feature_sets["FS4"]["features"]

    X = df[feature_cols].values
    y = df["D_conv"].values
    varieties = df["Variety"].values

    print(f"  样本数: {len(y)}")
    print(f"  特征数: {X.shape[1]}")
    print(f"  品种数: {len(np.unique(varieties))}")
    print(f"  特征列表: {feature_cols[:5]}..." if len(feature_cols) > 5 else f"  特征列表: {feature_cols}")

    # 运行三种验证方式
    random_cv = run_random_kfold(X, y, varieties, cfg, infer_precision, device)
    group_cv = run_group_kfold(X, y, varieties, cfg, infer_precision, device)
    holdout = run_holdout_test(X, y, varieties, cfg, infer_precision, device)
    internal = run_internal_overfit_check(X, y, cfg, infer_precision, device)

    # 生成诊断
    diagnosis_result = diagnose(random_cv, group_cv, holdout, internal)

    # 可视化
    out_dir = Path("results/exp6/diagnosis")
    visualize(random_cv, group_cv, holdout, out_dir)

    # 保存完整报告
    report = {
        "config": cfg,
        "data_stats": {
            "n_samples": len(y),
            "n_features": X.shape[1],
            "n_varieties": int(len(np.unique(varieties))),
            "feature_names": list(feature_cols)
        },
        "random_kfold": {
            "overall": random_cv["overall"],
            "folds": random_cv["folds"]
        },
        "group_kfold": {
            "overall": group_cv["overall"],
            "folds": group_cv["folds"]
        },
        "holdout_test": holdout["overall"],
        "internal_check": internal,
        "diagnosis": diagnosis_result
    }

    report_path = out_dir / "diagnosis_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n[保存] 完整报告: {report_path}")
    print("\n" + "=" * 60)
    print("诊断完成!")
    print("=" * 60)

    # 下一步建议
    print("\n下一步建议:")
    if diagnosis_result["primary_issue"] == "data_leakage":
        print("  1. 使用 GroupKFold 的结果作为真实性能指标")
        print("  2. 如果 GroupKFold R² < 0.7，考虑特征工程或增加数据")
        print("  3. 检查是否包含品种ID等泄露特征")
    elif diagnosis_result["primary_issue"] == "overfitting":
        print("  1. 减少特征数量（FS4可能包含冗余特征）")
        print("  2. 删除高基数特征（ID、时间戳等）")
        print("  3. 降低 n_estimators 或减少 TabPFN 复杂度")
    elif diagnosis_result["primary_issue"] == "domain_shift":
        print("  1. 检查测试集品种是否与训练集来自不同批次/年份")
        print("  2. 考虑使用领域自适应（Domain Adaptation）技术")
        print("  3. 确保训练集包含测试集品种的近亲或同生态型")
    else:
        print("  模型表现稳定，可进行超参数优化或集成学习进一步提升")


if __name__ == "__main__":
    main()
