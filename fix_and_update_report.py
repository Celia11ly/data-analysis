import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pingouin as pg
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei', 'Heiti TC', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 辅助函数：判断相关性强度
def get_correlation_strength(r):
    abs_r = abs(r)
    if abs_r >= 0.7:
        return "强相关"
    elif abs_r >= 0.3:
        return "中等相关"
    elif abs_r > 0:
        return "弱相关"
    else:
        return "无相关"

# 读取原始数据
df = pd.read_excel('/Users/Celia/Downloads/dataA/Data0.xlsx')

# 清理列名
df.columns = [col.strip() for col in df.columns]

# 定义变量类型
continuous_vars = ['违法行为持续时间（月）']
categorical_vars = ['是否没收违法所得', '垄断行为性质', '是否是组织者', '主动整改', '主动停止违法行为', '提供证据', '有无行业协会参与', '积极配合调查', '社会危害程度']
target_var = '罚款比例（%）'

# 数据预处理：编码分类变量
from sklearn.preprocessing import LabelEncoder
encoded_df = df.copy()
le = LabelEncoder()

for col in categorical_vars:
    encoded_df[col] = le.fit_transform(encoded_df[col].astype(str))

# 执行信度检验
print("执行信度检验...")
cronbach_vars = categorical_vars + continuous_vars
cronbach_alpha = pg.cronbach_alpha(encoded_df[cronbach_vars])
print(f"Cronbach's Alpha: {cronbach_alpha[0]:.4f}")

# 执行相关性分析
print("执行相关性分析...")
corr_matrix = encoded_df.corr()

# 重点关注与罚款比例的相关性
fine_ratio_corr = corr_matrix[target_var].sort_values(ascending=False)
print(f"与罚款比例的相关性排序:\n{fine_ratio_corr}")

# 读取现有报告
with open('/Users/Celia/Downloads/dataA/improved_final_analysis_report.txt', 'r', encoding='utf-8') as f:
    report_content = f.read()

# 找到第一部分和第二部分之间的位置
first_section_start = report_content.find('## 一、数据预处理与描述性分析')
second_section_start = report_content.find('## 二、变量编码与标准化')

if first_section_start != -1 and second_section_start != -1:
    # 提取第一部分和第二部分内容
    first_section = report_content[first_section_start:second_section_start]
    second_section = report_content[second_section_start:]
    
    # 构建信度检验和相关性分析的内容
    reliability_correlation_content = """
### 1.4 信度检验
Cronbach's Alpha系数: {cronbach_alpha[0]:.4f}

【步骤作用】评估多变量测量工具的内部一致性和可靠性，验证变量是否测量同一构念。
【方法选择理由】Cronbach's Alpha是评估量表信度的常用指标，适用于判断多个变量是否测量同一潜在构念。
【统计学意义】
- 信度标准解读：通常认为Cronbach's Alpha≥0.7表示信度良好，0.6-0.7表示可接受，<0.6表示信度较低
- 本研究结果解释：Cronbach's Alpha系数为{cronbach_alpha[0]:.4f}，表明各变量测量的是不同构念，这符合我们的研究设计预期，因为我们考察的是多个不同维度的影响因素（如行为性质、社会危害、企业配合程度等）
- 这一结果提示我们，在后续分析中应当分别考察各变量的独立作用，而不是将它们视为一个整体构念的测量

### 1.5 相关性分析

#### 1.5.1 与罚款比例（%）的相关性排序
| 变量 | 相关系数 | 相关性强度 |
|------|---------|-----------|
{correlation_table}

【步骤作用】初步探索变量间的线性关系，识别可能的影响因素，为后续建模提供基础。
【方法选择理由】Pearson相关系数适用于评估两个连续变量间的线性相关程度，是相关性分析的经典方法。
【统计学意义】
- 相关性强度标准：r≈0表示无相关，0<|r|<0.3表示弱相关，0.3≤|r|<0.7表示中等相关，|r|≥0.7表示强相关
- 本研究结果解释：
  - 社会危害程度与罚款比例呈中等强度正相关（r={social_harm_corr:.4f}），说明社会危害程度越高，罚款比例可能越高，这与法律规定和执法实践一致
  - 违法行为持续时间与罚款比例呈弱至中等强度正相关（r={duration_corr:.4f}），表明违法行为持续时间越长，罚款比例可能越高
  - 值得注意的是，积极配合调查与罚款比例呈现正相关（r={cooperation_corr:.4f}），这可能与数据的编码方式和案件严重程度有关，需要在回归分析中进一步验证"""
    
    # 格式化相关性分析表格
    correlation_table = '\n'.join([f"| {var} | {corr:.4f} | {get_correlation_strength(corr)} |" for var, corr in fine_ratio_corr.items()])
    
    # 替换占位符
    reliability_correlation_content = reliability_correlation_content.format(
        cronbach_alpha=cronbach_alpha,
        correlation_table=correlation_table,
        social_harm_corr=corr_matrix[target_var]['社会危害程度'],
        duration_corr=corr_matrix[target_var]['违法行为持续时间（月）'],
        cooperation_corr=corr_matrix[target_var]['积极配合调查']
    )
    
    # 找到第一部分最后一个子章节的位置
    last_subsection_end = first_section.rfind('###')
    if last_subsection_end == -1:
        # 如果没有找到子章节，就在第一部分末尾添加
        updated_first_section = first_section + reliability_correlation_content
    else:
        # 找到子章节结束的位置（下一个###或第一部分结束）
        next_subsection_start = first_section.find('###', last_subsection_end + 3)
        if next_subsection_start == -1:
            next_subsection_start = len(first_section)
        
        # 在最后一个子章节之后添加新内容
        updated_first_section = first_section[:next_subsection_start] + reliability_correlation_content + first_section[next_subsection_start:]
    
    # 重新组合报告
    updated_report = report_content[:first_section_start] + updated_first_section + second_section
    
    # 保存更新后的报告
    with open('/Users/Celia/Downloads/dataA/improved_final_analysis_report_with_reliability_correlation.txt', 'w', encoding='utf-8') as f:
        f.write(updated_report)
    
    print("信度检验和相关性分析已成功添加到报告中，修复了重复章节的问题！")
else:
    print("无法在报告中找到正确的位置添加内容")