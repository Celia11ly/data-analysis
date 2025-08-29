import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np

# 读取数据
df = pd.read_excel('/Users/Celia/Downloads/dataA/Data0.xlsx')

# 获取所有分类变量（除了违法行为持续时间和罚款比例）
categorical_vars = ['是否没收违法所得', '垄断行为性质', '社会危害程度', '是否是组织者', '积极配合调查', '主动整改', '主动停止违法行为', '提供证据', '有无行业协会参与']

print("=== 所有分类变量的编码检查和分析 ===\n")

# 存储所有变量的编码信息和分析结果
all_results = {}

for var in categorical_vars:
    print(f"\n\n========== {var} ==========")
    
    # 检查变量的取值范围和分布
    unique_values = df[var].unique()
    value_counts = df[var].value_counts()
    
    print(f"原始取值: {unique_values}")
    print(f"取值分布:\n{value_counts}")
    
    # 使用LabelEncoder进行编码
    le = LabelEncoder()
    le.fit(df[var].astype(str))
    encoded_values = le.transform(df[var].astype(str))
    
    # 保存编码映射
    encoding_map = {cls: code for cls, code in zip(le.classes_, le.transform(le.classes_))}
    print(f"\n编码映射: {encoding_map}")
    
    # 添加编码后的值到数据框
    df[f'{var}_编码'] = encoded_values
    
    # 计算不同编码值的平均罚款比例
    mean_fine_by_code = df.groupby(f'{var}_编码')['罚款比例（%）'].mean()
    print(f"\n各编码值对应的平均罚款比例:\n{mean_fine_by_code}")
    
    # 计算原始类别与罚款比例的关系
    mean_fine_by_original = df.groupby(var)['罚款比例（%）'].mean()
    print(f"\n原始类别对应的平均罚款比例:\n{mean_fine_by_original}")
    
    # 保存结果用于后续分析
    all_results[var] = {
        'unique_values': unique_values,
        'value_counts': value_counts,
        'encoding_map': encoding_map,
        'mean_fine_by_code': mean_fine_by_code,
        'mean_fine_by_original': mean_fine_by_original
    }

# 保存包含编码后数据的数据框
df.to_excel('/Users/Celia/Downloads/dataA/Data_with_encodings.xlsx', index=False)
print("\n\n数据已保存到 Data_with_encodings.xlsx")
print("\n=== 检查完成 ===")