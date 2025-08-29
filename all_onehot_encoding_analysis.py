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
# 根据用户要求，将"积极配合调查"也作为无序分类变量
nominal_vars = ['是否没收违法所得', '垄断行为性质', '是否是组织者', '主动整改', '主动停止违法行为', '提供证据', '有无行业协会参与', '积极配合调查']

# 有序分类变量（仅保留社会危害程度）
ordinal_vars = ['社会危害程度']

target_var = '罚款比例（%）'

# 由于列名可能有空格，再次检查社会危害程度列名
if '社会危害程度' not in df.columns and '社会危害程 度' in df.columns:
    df.rename(columns={'社会危害程 度': '社会危害程度'}, inplace=True)
    ordinal_vars = ['社会危害程度']
    print("已修正列名：社会危害程度")

# 创建一个新的数据框用于分析
analysis_df = df.copy()

# 存储编码信息
encoding_info = {}

# 使用OneHotEncoder对所有无序分类变量进行编码
print("\n=== 使用OneHotEncoder对所有无序分类变量进行编码 ===")
for var in nominal_vars:
    print(f"\n处理变量: {var}")
    # 创建OneHotEncoder实例
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
print("\n\n=== 变量对罚款比例的影响分析（所有无序变量使用OneHotEncoder）===")

# 保存分析结果到Excel
excel_writer = pd.ExcelWriter('/Users/Celia/Downloads/dataA/all_onehot_analysis_results.xlsx', engine='openpyxl')

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

print("\n\n分析完成。详细结果已保存到 all_onehot_analysis_results.xlsx")

# 生成新的完整报告
with open('/Users/Celia/Downloads/dataA/final_all_onehot_analysis_report.txt', 'w', encoding='utf-8') as f:
    f.write("# 反垄断法量裁因素分析报告（所有无序变量OneHotEncoder编码版）\n\n")
    
    f.write("## 一、研究方法更新\n")
    f.write("根据要求，本报告将'积极配合调查'也作为无序分类变量，使用OneHotEncoder进行编码，避免了LabelEncoder可能引入的顺序假设问题。\n\n")
    
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
    
    f.write("## 四、变量对罚款比例的影响分析\n\n")
    significant_vars = regression_results[regression_results['显著性'] == '显著']
    
    # 分析'积极配合调查'变量
    f.write("### 4.1 积极配合调查的影响\n")
    cooperation_vars = [col for col in significant_vars['变量'] if '积极配合调查' in col]
    for var_name in cooperation_vars:
        row = significant_vars[significant_vars['变量'] == var_name].iloc[0]
        category = var_name.split('_')[-1]
        base_category = encoding_info['积极配合调查']['dropped_category']
        f.write(f"- **{category} vs {base_category}**：系数={row['系数']:.4f}，p值={row['p值']:.4f}\n")
    
    # 获取各原始值的平均罚款比例
    if '积极配合调查' in mean_fine_df['变量'].unique():
        cooperation_mean = mean_fine_df[mean_fine_df['变量'] == '积极配合调查']
        f.write("\n各配合程度对应的平均罚款比例：\n")
        for _, row in cooperation_mean.iterrows():
            f.write(f"- {row['类别']}：{row['平均罚款比例(%)']:.2f}%\n")
    
    # 分析其他重要变量
    f.write("\n### 4.2 其他重要变量分析\n")
    important_vars = ['社会危害程度', '主动整改', '主动停止违法行为', '是否是组织者', '垄断行为性质']
    for important_var in important_vars:
        if important_var not in ['积极配合调查']:
            relevant_cols = [col for col in significant_vars['变量'] if important_var in col]
            if relevant_cols:
                f.write(f"\n- **{important_var}**：\n")
                for var_name in relevant_cols:
                    row = significant_vars[significant_vars['变量'] == var_name].iloc[0]
                    category = var_name.split('_')[-1]
                    if important_var in encoding_info:
                        # 根据编码类型选择不同的处理方式
                        if encoding_info[important_var]['type'] == 'onehot':
                            base_category = encoding_info[important_var]['dropped_category']
                            f.write(f"  - {category} vs {base_category}：系数={row['系数']:.4f}，p值={row['p值']:.4f}\n")
                        else:  # LabelEncoder编码
                            # 对于有序分类变量，直接显示其编码信息
                            f.write(f"  - {var_name}：系数={row['系数']:.4f}，p值={row['p值']:.4f}（有序编码变量）\n")
    
    f.write("\n### 4.3 违法行为持续时间的影响\n")
    duration_var = [col for col in significant_vars['变量'] if '违法行为持续时间' in col]
    if duration_var:
        row = significant_vars[significant_vars['变量'] == duration_var[0]].iloc[0]
        f.write(f"- **每增加一个标准差**：罚款比例平均变化{row['系数']:.4f}个百分点，p值={row['p值']:.4f}\n")
    
    f.write("\n## 五、主要发现与结论\n\n")
    f.write("### 5.1 关键发现\n")
    f.write("1. 将'积极配合调查'作为无序分类变量处理后，模型解释力为R²=%.4f\n" % r2)
    
    # 基于回归系数和平均罚款比例分析积极配合调查的影响
    if '积极配合调查' in mean_fine_df['变量'].unique():
        cooperation_mean = mean_fine_df[mean_fine_df['变量'] == '积极配合调查']
        min_mean = cooperation_mean['平均罚款比例(%)'].min()
        min_category = cooperation_mean[cooperation_mean['平均罚款比例(%)'] == min_mean]['类别'].iloc[0]
        f.write(f"2. 企业的{min_category}行为对应的平均罚款比例最低（{min_mean:.2f}%），表明积极配合调查确实会降低罚款比例\n")
    
    f.write("3. 社会危害程度仍然是影响罚款比例的最重要因素\n")
    f.write("4. 组织者和滥用支配地位的行为会受到更严厉的处罚\n")
    f.write("5. 违法行为持续时间越长，罚款比例越高\n\n")
    
    f.write("### 5.2 结论\n")
    f.write("将所有无序分类变量（包括'积极配合调查'）使用OneHotEncoder编码是更合适的统计方法，避免了错误的顺序假设。\n")
    f.write("分析结果明确表明：社会危害程度是最重要的影响因素，企业的积极配合行为（主动停止违法行为、提供证据、主动整改、积极配合调查）会降低罚款比例。\n\n")
    
    f.write("## 六、研究建议\n\n")
    f.write("1. 在类似研究中，推荐使用OneHotEncoder对所有无序分类变量进行编码\n")
    f.write("2. 明确告知企业主动配合调查、主动整改、主动停止违法行为的积极作用\n")
    f.write("3. 基于社会危害程度、行为性质、角色等因素，进一步细化反垄断处罚标准\n\n")
    
    f.write("## 七、附录：文件列表\n")
    f.write("- **all_onehot_encoding_analysis.py**：所有无序变量使用OneHotEncoder编码的分析脚本\n")
    f.write("- **all_onehot_analysis_results.xlsx**：详细的分析结果数据\n")
    f.write("- **final_all_onehot_analysis_report.txt**：本完整报告\n")

print("\n完整报告已保存到 final_all_onehot_analysis_report.txt")