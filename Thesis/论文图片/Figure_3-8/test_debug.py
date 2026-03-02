import pandas as pd
import numpy as np

# 加载数据
df = pd.read_csv("F:/all_exp/data/ojip_curves.csv")
print(f"总数据行数: {len(df)}")
print(f"列名: {df.columns[:5].tolist()}...")
print(f"\nVariety列的唯一值: {sorted(df['Variety'].unique())}")
print(f"Treatment列的唯一值: {df['Treatment'].unique()}")

# 测试过滤
varieties = [1252, 1228, 1235]
for variety in varieties:
    for treatment in ['CK1', 'D1', 'RD2']:
        df_subset = df[(df['Variety'] == variety) & (df['Treatment'] == treatment)]
        print(f"\n品种{variety}, 处理{treatment}: {len(df_subset)}行")
        if len(df_subset) > 0:
            # 获取时间点列
            time_cols = [col for col in df.columns if col not in ['Variety', 'Treatment']]
            print(f"  时间点列数: {len(time_cols)}")

            # 检查第一行数据
            first_row = df_subset.iloc[0]
            fluorescence = first_row[time_cols].values
            print(f"  荧光数据类型: {fluorescence.dtype}")
            print(f"  荧光数据前5个值: {fluorescence[:5]}")
            print(f"  荧光数据是否有NaN: {np.isnan(fluorescence).any()}")
            print(f"  荧光数据最大值: {np.nanmax(fluorescence)}")
