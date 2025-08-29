import pandas as pd
import sys

# 读取Excel文件的所有工作表
try:
    excel_file = '/Users/Celia/Downloads/dataA/all_onehot_analysis_results.xlsx'
    print(f"读取文件: {excel_file}")
    
    # 读取所有工作表
    all_sheets = pd.read_excel(excel_file, sheet_name=None)
    
    print(f"发现{len(all_sheets)}个工作表")
    
    # 打印每个工作表的内容
    for sheet_name, df in all_sheets.items():
        print(f"\n\n\n=== 工作表: {sheet_name} ===")
        print(f"形状: {df.shape}")
        print("\n内容:")
        print(df.to_string())
        
    print("\n\n读取完成")
except Exception as e:
    print(f"读取文件时出错: {e}")
    sys.exit(1)