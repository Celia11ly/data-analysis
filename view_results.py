import pandas as pd
import os

# 设置显示选项
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 50)

# 文件路径
file_path = '/Users/Celia/Downloads/dataA/all_onehot_analysis_results.xlsx'

if os.path.exists(file_path):
    # 读取工作表名称
    excel_file = pd.ExcelFile(file_path)
    print('Excel文件包含的工作表：', excel_file.sheet_names)
    
    # 读取编码信息
    print('\n=== 编码信息 ===')
    encoding_df = pd.read_excel(file_path, sheet_name='编码信息')
    print(encoding_df[encoding_df['变量'] == '积极配合调查'])
    
    # 读取回归分析结果
    print('\n=== 显著变量回归结果（p<0.05）===')
    regression_df = pd.read_excel(file_path, sheet_name='回归分析结果')
    significant_vars = regression_df[regression_df['p值'] < 0.05]
    print(significant_vars)
    
    # 读取模型评估指标
    print('\n=== 模型评估指标 ===')
    # 尝试不同的工作表名，因为原代码可能使用了不同的名称
    model_sheets = [sheet for sheet in excel_file.sheet_names if '模型' in sheet or 'metrics' in sheet.lower()]
    if model_sheets:
        model_metrics = pd.read_excel(file_path, sheet_name=model_sheets[0])
        print(model_metrics)
    else:
        print('未找到模型评估相关工作表')
        
    # 查看简化报告文件
    report_file = '/Users/Celia/Downloads/dataA/final_all_onehot_analysis_report.txt'
    if os.path.exists(report_file):
        print('\n=== 简化报告文件已生成 ===')
else:
    print('Excel文件不存在')