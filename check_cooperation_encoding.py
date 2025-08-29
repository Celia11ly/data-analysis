import pandas as pd
from sklearn.preprocessing import LabelEncoder

# 读取原始数据
df = pd.read_excel('/Users/Celia/Downloads/dataA/Data0.xlsx')

# 清理列名，去除可能的空格
df.columns = [col.strip() for col in df.columns]

# 检查社会危害程度列名
if '社会危害程度' not in df.columns and '社会危害程 度' in df.columns:
    df.rename(columns={'社会危害程 度': '社会危害程度'}, inplace=True)

# 定义目标变量
target_var = '罚款比例（%）'

# 重点检查"积极配合调查"变量
print("=== 积极配合调查变量分析 ===")

# 查看变量的原始值分布
print("\n1. 原始值分布:")
print(df['积极配合调查'].value_counts())

# 查看不同值对应的平均罚款比例
print("\n2. 不同值对应的平均罚款比例:")
mean_fine_by_cooperation = df.groupby('积极配合调查')[target_var].mean()
print(mean_fine_by_cooperation)

# 进行LabelEncoder编码
le = LabelEncoder()
cooperation_encoded = le.fit_transform(df['积极配合调查'].astype(str))

# 保存编码映射
encoding_map = {cls: code for cls, code in zip(le.classes_, le.transform(le.classes_))}
print("\n3. LabelEncoder编码映射:")
print(encoding_map)

# 添加编码后的值到数据框
df['积极配合调查_编码'] = cooperation_encoded

# 查看编码后不同值对应的平均罚款比例
print("\n4. 编码后不同值对应的平均罚款比例:")
mean_fine_by_encoded = df.groupby('积极配合调查_编码')[target_var].mean()
print(mean_fine_by_encoded)

# 计算编码值与罚款比例的相关性
print("\n5. 编码值与罚款比例的相关性:")
correlation = df['积极配合调查_编码'].corr(df[target_var])
print(f"相关性系数: {correlation:.4f}")

# 保存详细分析结果
detail_df = pd.DataFrame({
    '原始值': df['积极配合调查'],
    '编码值': df['积极配合调查_编码'],
    '罚款比例(%)': df[target_var]
})

detail_df.to_excel('/Users/Celia/Downloads/dataA/cooperation_encoding_detail.xlsx', index=False)

print("\n详细分析结果已保存到 cooperation_encoding_detail.xlsx")

# 生成结论报告
with open('/Users/Celia/Downloads/dataA/cooperation_encoding_analysis.txt', 'w', encoding='utf-8') as f:
    f.write("# 积极配合调查变量编码分析报告\n\n")
    
    f.write("## 一、变量基本情况\n")
    f.write(f"- 原始值分布:\n{df['积极配合调查'].value_counts().to_string()}\n\n")
    
    f.write("## 二、编码映射分析\n")
    f.write(f"- LabelEncoder编码映射:\n{encoding_map}\n\n")
    
    f.write("## 三、不同值对应的平均罚款比例\n")
    f.write(f"- 原始值:\n{mean_fine_by_cooperation.to_string()}\n\n")
    f.write(f"- 编码值:\n{mean_fine_by_encoded.to_string()}\n\n")
    
    f.write("## 四、相关性分析\n")
    f.write(f"- 编码值与罚款比例的相关性系数: {correlation:.4f}\n\n")
    
    f.write("## 五、结论\n")
    if encoding_map.get('是') == 1 and encoding_map.get('否') == 0:
        if mean_fine_by_cooperation['是'] < mean_fine_by_cooperation['否']:
            f.write("1. 编码顺序与实际语义顺序一致（'否'=0，'是'=1）\n")
            f.write("2. 但数据显示'是'的平均罚款比例反而低于'否'，这可能是由于样本选择偏差或其他混淆因素导致的\n")
        else:
            f.write("1. 编码顺序与实际语义顺序一致（'否'=0，'是'=1）\n")
            f.write("2. 数据显示'是'的平均罚款比例高于'否'，这可能是因为配合调查的案件本身性质更严重\n")
    elif encoding_map.get('是') == 0 and encoding_map.get('否') == 1:
        f.write("1. 编码顺序与实际语义顺序相反（'是'=0，'否'=1）\n")
        f.write("2. 这是导致回归分析结果看似矛盾的主要原因\n")
    
    f.write("\n建议在未来的研究中，明确指定有序分类变量的编码顺序，确保与实际语义顺序一致。")

print("\n分析报告已保存到 cooperation_encoding_analysis.txt")