"""
Step 4: Ridge 模型 (Layer 2 - 传统ML)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from step4_utils import load_data, run_kfold_cv, save_model_results
from sklearn.linear_model import Ridge

MODEL_NAME = "Ridge"
LAYER = "Layer2"

def main():
    print(f"{'='*50}")
    print(f"运行 {MODEL_NAME} 模型")
    print(f"{'='*50}")

    # 加载数据
    df, feature_sets = load_data()
    feature_cols = feature_sets['FS4']['features']
    print(f"特征数: {len(feature_cols)}")
    print(f"样本数: {len(df)}")

    # 模型配置
    model_class = Ridge
    model_params = {
        'alpha': 1.0,
        'random_state': 42
    }

    # 运行交叉验证
    metrics, variety_agg = run_kfold_cv(df, feature_cols, model_class, model_params)

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
