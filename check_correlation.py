import pandas as pd
import numpy as np

# 读取原始数据
df = pd.read_excel('/Users/Celia/Downloads/dataA/Data0.xlsx')

# 清理列名 - 更彻底的清理，去除所有空格
print("原始列名:")
for col in df.columns:
    print(f"'{col}'")

df.columns = [col.strip().replace(' ', '') for col in df.columns]
print("\n清理后的列名:", df.columns.tolist())

# 查看积极配合调查列的内容
print("\n积极配合调查列的唯一值:", df['积极配合调查'].unique())

# 对所有分类变量进行OneHot编码
df_encoded = pd.get_dummies(df)

print("\n编码后的数据形状:", df_encoded.shape)
print("编码后的列数量:", len(df_encoded.columns))

# 打印包含'积极配合调查'的列名
print("\n包含'积极配合调查'的列名:")
found = False
for col in df_encoded.columns:
    if '积极配合调查' in col:
        print(f"'{col}'")
        found = True
if not found:
    print("未找到包含'积极配合调查'的列")

# 直接查看数据中的罚款比例列
print("\n罚款比例列名:")
for col in df_encoded.columns:
    if '罚款比例' in col:
        print(f"'{col}'")

# 计算相关性
print("\n计算相关性...")

# 创建一个小的测试子集，只包含需要的列
test_cols = []
for col in df_encoded.columns:
    if '积极配合调查' in col or '罚款比例' in col:
        test_cols.append(col)

if test_cols:
    print("\n用于测试的列:", test_cols)
    test_df = df_encoded[test_cols]
    print("测试数据的前5行:")
    print(test_df.head())
    
    # 计算相关性
    corr = test_df.corr()
    print("\n相关性矩阵:")
    print(corr)
else:
    print("未找到相关列")

# 手动计算积极配合调查变量的相关性
def calculate_manual_correlation(df):
    # 创建虚拟变量
    df['积极配合调查_是'] = (df['积极配合调查'] == '是').astype(int)
    df['积极配合调查_否'] = (df['积极配合调查'] == '否').astype(int)
    df['积极配合调查_未知'] = (df['积极配合调查'] == '未知').astype(int)
    
    # 确保罚款比例是数值型
    fine_col = [col for col in df.columns if '罚款比例' in col][0]
    df[fine_col] = pd.to_numeric(df[fine_col], errors='coerce')
    
    print("\n手动编码后的相关列:", [col for col in df.columns if '积极配合调查' in col])
    
    # 计算相关性
    corr_是 = df[fine_col].corr(df['积极配合调查_是'])
    corr_否 = df[fine_col].corr(df['积极配合调查_否'])
    corr_未知 = df[fine_col].corr(df['积极配合调查_未知'])
    
    print(f"\n手动计算的相关性:\n积极配合调查_是: {corr_是:.4f}\n积极配合调查_否: {corr_否:.4f}\n积极配合调查_未知: {corr_未知:.4f}")
    
    # 查看各分类的平均罚款比例
    print("\n各分类的平均罚款比例:")
    print(df.groupby('积极配合调查')[fine_col].mean())

# 调用手动计算函数
calculate_manual_correlation(df.copy())