"""
OJIP数据预处理脚本
功能:
1. 对ABS/RC和DIo/RC进行对数转换
2. 计算群体统计(CK1/D1/RD2的均值、标准差、变化率)
3. 计算三个代表品种(1252/1228/1235)的详细统计
4. 输出处理后的数据和统计摘要
"""

import pandas as pd
import numpy as np
from pathlib import Path

# 设置路径
DATA_DIR = Path("F:/all_exp/data")
INPUT_FILE = DATA_DIR / "ojip_8params_raw.csv"
OUTPUT_FILE = DATA_DIR / "processed" / "ojip_8params_processed.csv"
STATS_FILE = DATA_DIR / "processed" / "ojip_stats_summary.txt"

# 确保输出目录存在
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

def load_data():
    """加载原始数据"""
    df = pd.read_csv(INPUT_FILE)
    print(f"加载数据: {len(df)} 个样本")
    print(f"品种数: {df['Variety'].nunique()}")
    print(f"处理: {df['Treatment'].unique()}")
    return df

def apply_log_transform(df):
    """对ABS/RC和DIo/RC进行对数转换"""
    df['ABS_RC_log'] = np.log(df['ABS/RC'])
    df['DIo_RC_log'] = np.log(df['DIo/RC'])

    # 检查对数转换后的分布
    print("\n对数转换后的偏度:")
    print(f"ABS_RC_log偏度: {df['ABS_RC_log'].skew():.3f}")
    print(f"DIo_RC_log偏度: {df['DIo_RC_log'].skew():.3f}")

    return df

def compute_population_stats(df):
    """计算群体统计"""
    stats = []

    params = ['Fv/Fm', 'PIabs', 'Vj', 'Vi', 'TRo/RC', 'ETo/RC', 'ABS/RC', 'DIo/RC']

    for param in params:
        ck1 = df[df['Treatment'] == 'CK1'][param]
        d1 = df[df['Treatment'] == 'D1'][param]
        rd2 = df[df['Treatment'] == 'RD2'][param]

        ck1_mean = ck1.mean()
        d1_mean = d1.mean()
        rd2_mean = rd2.mean()

        d1_change = (d1_mean - ck1_mean) / ck1_mean * 100
        rd2_change = (rd2_mean - ck1_mean) / ck1_mean * 100

        stats.append({
            'Parameter': param,
            'CK1_mean': ck1_mean,
            'CK1_std': ck1.std(),
            'D1_mean': d1_mean,
            'D1_std': d1.std(),
            'D1_change_%': d1_change,
            'RD2_mean': rd2_mean,
            'RD2_std': rd2.std(),
            'RD2_change_%': rd2_change
        })

    return pd.DataFrame(stats)

def compute_variety_stats(df, varieties=[1252, 1228, 1235]):
    """计算代表品种的详细统计"""
    variety_stats = []

    params = ['Fv/Fm', 'PIabs', 'Vj', 'Vi', 'TRo/RC', 'ETo/RC', 'ABS/RC', 'DIo/RC']

    for variety in varieties:
        df_var = df[df['Variety'] == variety]

        for param in params:
            ck1 = df_var[df_var['Treatment'] == 'CK1'][param].mean()
            d1 = df_var[df_var['Treatment'] == 'D1'][param].mean()
            rd2 = df_var[df_var['Treatment'] == 'RD2'][param].mean()

            d1_change = (d1 - ck1) / ck1 * 100
            rd2_change = (rd2 - d1) / d1 * 100

            variety_stats.append({
                'Variety': variety,
                'Parameter': param,
                'CK1': ck1,
                'D1': d1,
                'RD2': rd2,
                'CK1_to_D1_%': d1_change,
                'D1_to_RD2_%': rd2_change
            })

    return pd.DataFrame(variety_stats)

def main():
    """主函数"""
    print("="*60)
    print("OJIP数据预处理")
    print("="*60)

    # 1. 加载数据
    df = load_data()

    # 2. 对数转换
    print("\n" + "="*60)
    print("执行对数转换")
    print("="*60)
    df = apply_log_transform(df)

    # 3. 计算群体统计
    print("\n" + "="*60)
    print("计算群体统计")
    print("="*60)
    pop_stats = compute_population_stats(df)
    print(pop_stats.to_string(index=False))

    # 4. 计算品种统计
    print("\n" + "="*60)
    print("计算代表品种统计")
    print("="*60)
    var_stats = compute_variety_stats(df)
    print(var_stats.to_string(index=False))

    # 5. 保存处理后的数据
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n处理后的数据已保存到: {OUTPUT_FILE}")

    # 6. 保存统计摘要
    with open(STATS_FILE, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("OJIP数据统计摘要\n")
        f.write("="*60 + "\n\n")

        f.write("群体统计:\n")
        f.write(pop_stats.to_string(index=False))
        f.write("\n\n")

        f.write("代表品种统计:\n")
        f.write(var_stats.to_string(index=False))
        f.write("\n")

    print(f"统计摘要已保存到: {STATS_FILE}")

    print("\n" + "="*60)
    print("数据处理完成!")
    print("="*60)

if __name__ == "__main__":
    main()
