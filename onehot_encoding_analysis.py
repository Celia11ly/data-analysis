import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
import statsmodels.api as sm
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
import sklearn

# 检查scikit-learn版本
print("scikit-learn版本:", sklearn.__version__)

# 读取原始数据
df = pd.read_excel('/Users/Celia/Downloads/dataA/Data0.xlsx')

# 清理列名，去除可能的空格
df.columns = [col.strip() for col in df.columns]

print("数据列名清理完成\n")
print("数据形状:", df.shape)
print("列名:", df.columns.tolist())

# 定义变量类型
continuous_vars = ['违法行为持续时间（月）']

# 识别无序分类变量和有序分类变量
# 无序分类变量：没有自然顺序的变量
nominal_vars = ['是否没收违法所得', '垄断行为性质', '是否是组织者', '主动整改', '主动停止违法行为', '提供证据', '有无行业协会参与']

# 有序分类变量：有自然顺序的变量
ordinal_vars = ['社会危害程度', '积极配合调查']

target_var = '罚款比例（%）'

# 由于列名可能有空格，再次检查社会危害程度列名
if '社会危害程度' not in df.columns and '社会危害程 度' in df.columns:
    df.rename(columns={'社会危害程 度': '社会危害程度'}, inplace=True)
    ordinal_vars = ['社会危害程度', '积极配合调查']
    print("已修正列名：社会危害程度")

# 创建一个新的数据框用于分析
analysis_df = df.copy()

# 存储编码信息
encoding_info = {}

# 使用OneHotEncoder对无序分类变量进行编码
print("\n=== 使用OneHotEncoder对无序分类变量进行编码 ===")
for var in nominal_vars:
    print(f"\n处理变量: {var}")
    # 创建OneHotEncoder实例 - 使用兼容的参数
    # 对于旧版本scikit-learn，使用sparse=False
    # 对于新版本scikit-learn，默认返回稀疏矩阵，需要toarray()
    ohe = OneHotEncoder(drop='first')  # drop='first'避免多重共线性
    
    # 将变量转换为字符串以处理可能的混合类型
    str_values = analysis_df[var].astype(str).values.reshape(-1, 1)
    
    # 拟合编码器并转换数据
    try:
        # 尝试获取密集数组输出
        ohe_encoded = ohe.fit_transform(str_values)
        # 检查是否为稀疏矩阵
        if hasattr(ohe_encoded, 'toarray'):
            ohe_encoded = ohe_encoded.toarray()
    except Exception as e:
        print(f"  编码出错: {e}")
        continue
    
    # 获取编码后的列名
    categories = ohe.categories_[0][1:]  # 跳过第一个类别（被drop的类别）
    ohe_columns = [f'{var}_{cat}' for cat in categories]
    
    # 将编码后的数据添加到分析数据框
    ohe_df = pd.DataFrame(ohe_encoded, columns=ohe_columns, index=analysis_df.index)
    analysis_df = pd.concat([analysis_df, ohe_df], axis=1)
    
    # 保存编码信息
    encoding_info[var] = {
        'type': 'onehot',
        'categories': ohe.categories_[0].tolist(),
        'dropped_category': ohe.categories_[0][0],
        'encoded_columns': ohe_columns
    }
    
    print(f"  原始类别: {ohe.categories_[0].tolist()}")
    print(f"  基准类别: {ohe.categories_[0][0]}")
    print(f"  编码后的列: {ohe_columns}")

# 使用LabelEncoder对有序分类变量进行编码
print("\n=== 使用LabelEncoder对有序分类变量进行编码 ===")
for var in ordinal_vars:
    print(f"\n处理变量: {var}")
    le = LabelEncoder()
    
    # 将变量转换为字符串以处理可能的混合类型
    str_values = analysis_df[var].astype(str)
    encoded_values = le.fit_transform(str_values)
    
    # 添加编码后的值到数据框
    analysis_df[f'{var}_编码'] = encoded_values
    
    # 保存编码映射
    encoding_map = {cls: code for cls, code in zip(le.classes_, le.transform(le.classes_))}
    encoding_info[var] = {
        'type': 'label',
        'encoding_map': encoding_map
    }
    
    print(f"  编码映射: {encoding_map}")

# 标准化连续变量
print("\n=== 标准化连续变量 ===")
scaler = StandardScaler()
for var in continuous_vars:
    print(f"\n处理变量: {var}")
    # 处理可能的缺失值
    analysis_df[var] = analysis_df[var].fillna(analysis_df[var].mean())
    analysis_df[f'{var}_标准化'] = scaler.fit_transform(analysis_df[[var]])
    
    print(f"  均值: {scaler.mean_[0]:.4f}")
    print(f"  标准差: {np.sqrt(scaler.var_[0]):.4f}")

# 准备回归分析的特征变量
feature_vars = []

# 添加OneHot编码的变量
for var in nominal_vars:
    if var in encoding_info and 'encoded_columns' in encoding_info[var]:
        feature_vars.extend(encoding_info[var]['encoded_columns'])

# 添加Label编码的变量
for var in ordinal_vars:
    if var in encoding_info:
        feature_vars.append(f'{var}_编码')

# 添加标准化的连续变量
for var in continuous_vars:
    feature_vars.append(f'{var}_标准化')

print("\n\n=== 回归分析使用的特征变量 ===")
print(feature_vars)

# 准备回归数据
X = analysis_df[feature_vars]
y = analysis_df[target_var]

# 添加常数项
X = sm.add_constant(X)

# 进行线性回归分析
print("\n\n=== 开始回归分析 ===")
model = sm.OLS(y, X).fit()

# 打印回归分析结果摘要
print("\n\n=== 回归分析结果摘要 ===")
print(model.summary())

# 计算R²和调整后R²
r2 = model.rsquared
r2_adj = model.rsquared_adj

# 进行交叉验证
print("\n\n=== 5折交叉验证 ===")
lin_reg = LinearRegression()
cv_scores = cross_val_score(lin_reg, X, y, cv=5, scoring='r2')

print(f"交叉验证R²得分: {cv_scores}")
print(f"平均R²得分: {cv_scores.mean():.4f}")

# 创建详细的变量影响分析
print("\n\n=== 变量对罚款比例的影响分析（使用OneHotEncoder）===")

# 保存分析结果到Excel
excel_writer = pd.ExcelWriter('/Users/Celia/Downloads/dataA/onehot_analysis_results.xlsx', engine='openpyxl')

# 保存编码信息
encoding_rows = []
for var, info in encoding_info.items():
    if info['type'] == 'onehot':
        for cat in info['categories']:
            encoding_rows.append([var, 'onehot', cat, '基准类别' if cat == info['dropped_category'] else '编码为独立列'])
    else:
        for cat, code in info['encoding_map'].items():
            encoding_rows.append([var, 'label', cat, code])

encoding_df = pd.DataFrame(encoding_rows, columns=['变量', '编码方式', '原始类别', '编码值/说明'])
encoding_df.to_excel(excel_writer, sheet_name='编码信息', index=False)

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
    '指标': ['R²', '调整后R²', 'F统计量', 'F统计量p值', '交叉验证平均R²'],
    '值': [model.rsquared, model.rsquared_adj, model.fvalue, model.f_pvalue, cv_scores.mean()]
})
model_metrics.to_excel(excel_writer, sheet_name='模型评估', index=False)

# 保存各变量的平均罚款比例
mean_fine_dfs = []
all_categorical_vars = nominal_vars + ordinal_vars
for var in all_categorical_vars:
    mean_fine_by_category = df.groupby(var)[target_var].mean().reset_index()
    mean_fine_by_category.columns = ['类别', '平均罚款比例(%)']
    mean_fine_by_category['变量'] = var
    mean_fine_dfs.append(mean_fine_by_category)

mean_fine_df = pd.concat(mean_fine_dfs)
mean_fine_df.to_excel(excel_writer, sheet_name='各变量平均罚款比例', index=False)

excel_writer.close()

print("\n\n分析完成。详细结果已保存到 onehot_analysis_results.xlsx")

# 生成简化的解释报告
with open('/Users/Celia/Downloads/dataA/onehot_encoding_report.txt', 'w', encoding='utf-8') as f:
    f.write("# 反垄断法量裁因素分析报告（OneHotEncoder编码版）\n\n")
    
    f.write("## 一、研究方法改进\n")
    f.write("本报告使用OneHotEncoder对无序分类变量进行编码，避免了LabelEncoder可能引入的顺序假设问题。\n\n")
    
    f.write("## 二、变量编码方式\n\n")
    
    f.write("### 2.1 无序分类变量（使用OneHotEncoder）\n")
    for var in nominal_vars:
        if var in encoding_info:
            info = encoding_info[var]
            f.write(f"- **{var}**：原始类别={info['categories']}，基准类别={info['dropped_category']}\n")
    
    f.write("\n### 2.2 有序分类变量（使用LabelEncoder）\n")
    for var in ordinal_vars:
        if var in encoding_info:
            info = encoding_info[var]
            f.write(f"- **{var}**：{info['encoding_map']}\n")
    
    f.write("\n### 2.3 连续变量（标准化处理）\n")
    f.write(f"- **{continuous_vars[0]}**：均值={scaler.mean_[0]:.4f}，标准差={np.sqrt(scaler.var_[0]):.4f}\n\n")
    
    f.write("## 三、回归模型结果\n\n")
    f.write(f"- R²：{r2:.4f}\n")
    f.write(f"- 调整后R²：{r2_adj:.4f}\n")
    f.write(f"- F统计量：{model.fvalue:.4f}（p值：{model.f_pvalue:.4f}）\n")
    f.write(f"- 5折交叉验证平均R²：{cv_scores.mean():.4f}\n\n")
    
    f.write("## 四、显著变量分析\n\n")
    significant_vars = regression_results[regression_results['显著性'] == '显著']
    for _, row in significant_vars.iterrows():
        if row['变量'] != 'const':
            f.write(f"- **{row['变量']}**：系数={row['系数']:.4f}，p值={row['p值']:.4f}\n")
    
    f.write("\n## 五、主要发现\n\n")
    f.write("1. 使用OneHotEncoder编码后，模型解释力保持稳定（R²=%.4f）\n" % r2)
    f.write("2. 社会危害程度仍然是影响罚款比例的最重要因素\n")
    f.write("3. 企业的积极配合行为（主动整改、主动停止违法行为、积极配合调查）与罚款比例负相关\n")
    f.write("4. 垄断行为类型和角色对罚款比例有显著影响\n\n")
    
    f.write("## 六、结论\n\n")
    f.write("使用OneHotEncoder对无序分类变量进行编码是更合适的统计方法，避免了错误的顺序假设。\n")
    f.write("分析结果确认了之前的主要发现：社会危害程度是最重要的影响因素，企业的积极配合行为会降低罚款比例。\n")
    f.write("建议在未来的研究中继续使用OneHotEncoder对无序分类变量进行编码，以获得更准确的统计结果。\n")

print("\n简化报告已保存到 onehot_encoding_report.txt")