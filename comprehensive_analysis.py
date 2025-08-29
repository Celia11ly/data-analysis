import pandas as pd
from sklearn.preprocessing import LabelEncoder
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
import numpy as np

# 读取原始数据
df = pd.read_excel('/Users/Celia/Downloads/dataA/Data0.xlsx')

# 清理列名，去除可能的空格
df.columns = [col.strip() for col in df.columns]

print("数据列名清理完成\n")
print("数据形状:", df.shape)
print("列名:", df.columns.tolist())

# 定义所有变量
continuous_vars = ['违法行为持续时间（月）']
categorical_vars = ['是否没收违法所得', '垄断行为性质', '社会危害程度', '是否是组织者', '积极配合调查', '主动整改', '主动停止违法行为', '提供证据', '有无行业协会参与']
target_var = '罚款比例（%）'

# 创建一个新的数据框用于分析
analysis_df = df.copy()

# 存储所有变量的编码信息
encoding_info = {}

# 对所有分类变量进行LabelEncoder编码并保存编码映射
for var in categorical_vars:
    le = LabelEncoder()
    # 将变量转换为字符串以处理可能的混合类型
    str_values = analysis_df[var].astype(str)
    encoded_values = le.fit_transform(str_values)
    analysis_df[f'{var}_编码'] = encoded_values
    
    # 保存编码映射
    encoding_map = {cls: code for cls, code in zip(le.classes_, le.transform(le.classes_))}
    encoding_info[var] = encoding_map
    
    print(f"\n{var}的编码映射:")
    for cls, code in encoding_map.items():
        print(f"  {cls}: {code}")

# 标准化连续变量
scaler = StandardScaler()
for var in continuous_vars:
    # 处理可能的缺失值
    analysis_df[var] = analysis_df[var].fillna(analysis_df[var].mean())
    analysis_df[f'{var}_标准化'] = scaler.fit_transform(analysis_df[[var]])

# 准备回归分析的特征变量
feature_vars = [f'{var}_编码' for var in categorical_vars] + [f'{var}_标准化' for var in continuous_vars]
X = analysis_df[feature_vars]
y = analysis_df[target_var]

# 添加常数项
X = sm.add_constant(X)

# 进行线性回归分析
model = sm.OLS(y, X).fit()

# 打印回归分析结果摘要
print("\n\n=== 回归分析结果摘要 ===")
print(model.summary())

# 保存完整的分析结果
alysis_results = {
    'encoding_info': encoding_info,
    'model_summary': model.summary().as_text(),
    'params': model.params.to_dict(),
    'pvalues': model.pvalues.to_dict(),
    'rsquared': model.rsquared,
    'rsquared_adj': model.rsquared_adj
}

# 创建详细的变量影响分析
print("\n\n=== 变量对罚款比例的影响分析 ===")
variable_analysis = {}

for var in categorical_vars:
    encoded_var = f'{var}_编码'
    if encoded_var in model.params:
        coeff = model.params[encoded_var]
        pvalue = model.pvalues[encoded_var]
        significant = pvalue < 0.05
        
        # 计算原始类别与罚款比例的关系
        mean_fine_by_category = df.groupby(var)[target_var].mean()
        
        variable_analysis[var] = {
            'coefficient': coeff,
            'pvalue': pvalue,
            'significant': significant,
            'encoding_map': encoding_info[var],
            'mean_fine_by_category': mean_fine_by_category
        }
        
        print(f"\n{var}:")
        print(f"  回归系数: {coeff:.4f}")
        print(f"  p值: {pvalue:.4f} {'(显著)' if significant else '(不显著)'}")
        print(f"  编码映射: {encoding_info[var]}")
        print(f"  各原始类别的平均罚款比例:")
        for category, mean_fine in mean_fine_by_category.items():
            print(f"    {category}: {mean_fine:.4f}%")

# 分析连续变量
for var in continuous_vars:
    standardized_var = f'{var}_标准化'
    if standardized_var in model.params:
        coeff = model.params[standardized_var]
        pvalue = model.pvalues[standardized_var]
        significant = pvalue < 0.05
        
        variable_analysis[var] = {
            'coefficient': coeff,
            'pvalue': pvalue,
            'significant': significant
        }
        
        print(f"\n{var}:")
        print(f"  标准化回归系数: {coeff:.4f}")
        print(f"  p值: {pvalue:.4f} {'(显著)' if significant else '(不显著)'}")
        print(f"  说明: 每变化一个标准差，罚款比例平均变化{coeff:.4f}%")

# 保存分析结果
excel_writer = pd.ExcelWriter('/Users/Celia/Downloads/dataA/comprehensive_analysis_results.xlsx', engine='openpyxl')

# 保存编码信息
encoding_df = pd.DataFrame([(var, cls, code) for var, mapping in encoding_info.items() for cls, code in mapping.items()], 
                          columns=['变量', '原始类别', '编码值'])
encoding_df.to_excel(excel_writer, sheet_name='编码信息', index=False)

# 保存各变量的平均罚款比例
mean_fine_dfs = []
for var in categorical_vars:
    mean_fine_by_category = df.groupby(var)[target_var].mean().reset_index()
    mean_fine_by_category.columns = ['类别', '平均罚款比例(%)']
    mean_fine_by_category['变量'] = var
    mean_fine_dfs.append(mean_fine_by_category)

mean_fine_df = pd.concat(mean_fine_dfs)
mean_fine_df.to_excel(excel_writer, sheet_name='各变量平均罚款比例', index=False)

# 保存回归分析结果
params_df = pd.DataFrame(model.params).reset_index()
params_df.columns = ['变量', '系数']
pvalues_df = pd.DataFrame(model.pvalues).reset_index()
pvalues_df.columns = ['变量', 'p值']
regression_results = pd.merge(params_df, pvalues_df, on='变量')
regression_results['显著性'] = regression_results['p值'].apply(lambda x: '显著' if x < 0.05 else '不显著')
regression_results.to_excel(excel_writer, sheet_name='回归分析结果', index=False)

# 保存模型评估指标
model_metrics = pd.DataFrame({
    '指标': ['R²', '调整后R²', 'F统计量', 'F统计量p值'],
    '值': [model.rsquared, model.rsquared_adj, model.fvalue, model.f_pvalue]
})
model_metrics.to_excel(excel_writer, sheet_name='模型评估', index=False)

excel_writer.close()

print("\n\n综合分析完成。详细结果已保存到 comprehensive_analysis_results.xlsx")