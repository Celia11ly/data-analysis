import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC", "Arial Unicode MS", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

data_path = "/Users/Celia/Downloads/dataA/Data0.xlsx"
output_dir = "/Users/Celia/Downloads/dataA/charts"
report_path = "/Users/Celia/Downloads/dataA/analysis_report.txt"

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 读取Excel数据
print("正在读取数据...")
df = pd.read_excel(data_path)
print("数据读取完成。")
print(f"数据形状: {df.shape}")
print(f"列名: {df.columns.tolist()}")

# 生成描述性统计
print("\n生成描述性统计...")
descriptive_stats = df.describe(include='all')
print("描述性统计完成。")

# 保存描述性统计到报告
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("中国反垄断法量裁不确定性分析报告\n")
    f.write("====================================\n\n")
    f.write("一、数据概览\n")
    f.write(f"数据形状: {df.shape}\n")
    f.write(f"列名: {df.columns.tolist()}\n\n")
    f.write("二、描述性统计\n")
    f.write(str(descriptive_stats) + "\n\n")
    f.write("三、变量分析\n")

# 分析每个变量并生成图表
print("\n分析变量并生成图表...")

# 1. 被解释变量：罚款比例

# 假设罚款比例列名为'罚款比例'
if '罚款比例（%）' in df.columns:
    # 绘制直方图
    plt.figure(figsize=(10, 6))
    sns.histplot(df['罚款比例（%）'], kde=True)
    plt.title('罚款比例（%）分布')
    plt.xlabel('罚款比例（%）')
    plt.ylabel('频数')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '罚款比例分布.png'))
    plt.close()

    # 添加到报告
    with open(report_path, 'a', encoding='utf-8') as f:
        f.write("1. 罚款比例\n")
        f.write(f"   均值: {df['罚款比例（%）'].mean():.4f}\n")
        f.write(f"   中位数: {df['罚款比例（%）'].median():.4f}\n")
        f.write(f"   标准差: {df['罚款比例（%）'].std():.4f}\n")
        f.write(f"   最小值: {df['罚款比例（%）'].min():.4f}\n")
        f.write(f"   最大值: {df['罚款比例（%）'].max():.4f}\n")
        f.write("   分布情况: 见图表'罚款比例分布.png'\n\n")

# 2. 违法行为持续时间（连续变量）
if '违法行为持续时间（月）' in df.columns:
    # 绘制直方图
    plt.figure(figsize=(10, 6))
    sns.histplot(df['违法行为持续时间（月）'], kde=True)
    plt.title('违法行为持续时间（月）分布')
    plt.xlabel('持续时间（月）')
    plt.ylabel('频数')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '违法行为持续时间分布.png'))
    plt.close()

    # 绘制与罚款比例的散点图
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='违法行为持续时间（月）', y='罚款比例（%）', data=df)
    plt.title('违法行为持续时间（月）与罚款比例（%）关系')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '违法行为持续时间与罚款比例关系.png'))
    plt.close()

    # 添加到报告
    with open(report_path, 'a', encoding='utf-8') as f:
        f.write("2. 违法行为持续时间\n")
        f.write(f"   均值: {df['违法行为持续时间（月）'].mean():.2f}\n")
        f.write(f"   中位数: {df['违法行为持续时间（月）'].median():.2f}\n")
        f.write(f"   标准差: {df['违法行为持续时间（月）'].std():.2f}\n")
        f.write(f"   最小值: {df['违法行为持续时间（月）'].min():.2f}\n")
        f.write(f"   最大值: {df['违法行为持续时间（月）'].max():.2f}\n")
        f.write("   分布情况: 见图表'违法行为持续时间分布.png'\n")
        f.write("   与罚款比例关系: 见图表'违法行为持续时间与罚款比例关系.png'\n\n")

# 分析其他变量（假设它们是分类变量）
other_vars = [col for col in df.columns if col not in ['罚款比例（%）', '违法行为持续时间（月）']]

for i, var in enumerate(other_vars, start=3):
    with open(report_path, 'a', encoding='utf-8') as f:
        f.write(f"{i}. {var}\n")

    # 检查变量类型
    if df[var].dtype == 'object' or df[var].nunique() < 10:
        # 分类变量：绘制柱状图
        plt.figure(figsize=(10, 6))
        sns.countplot(x=var, data=df)
        plt.title(f'{var}分布')
        plt.xlabel(var)
        plt.ylabel('频数')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{var}分布.png'))
        plt.close()

        # 绘制与罚款比例的箱线图
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=var, y='罚款比例（%）', data=df)
        plt.title(f'{var}与罚款比例（%）关系')
        plt.xlabel(var)
        plt.ylabel('罚款比例（%）')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{var}与罚款比例关系.png'))
        plt.close()

        # 添加到报告
        with open(report_path, 'a', encoding='utf-8') as f:
            f.write(f"   类型: 分类变量\n")
            f.write(f"   唯一值数量: {df[var].nunique()}\n")
            f.write(f"   分布情况: 见图表'{var}分布.png'\n")
            f.write(f"   与罚款比例关系: 见图表'{var}与罚款比例关系.png'\n\n")
    else:
        # 连续变量：绘制直方图和散点图
        plt.figure(figsize=(10, 6))
        sns.histplot(df[var], kde=True)
        plt.title(f'{var}分布')
        plt.xlabel(var)
        plt.ylabel('频数')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{var}分布.png'))
        plt.close()

        # 绘制与罚款比例的散点图
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=var, y='罚款比例（%）', data=df)
        plt.title(f'{var}与罚款比例（%）关系')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{var}与罚款比例关系.png'))
        plt.close()

        # 添加到报告
        with open(report_path, 'a', encoding='utf-8') as f:
            f.write(f"   类型: 连续变量\n")
            f.write(f"   均值: {df[var].mean():.2f}\n")
            f.write(f"   中位数: {df[var].median():.2f}\n")
            f.write(f"   标准差: {df[var].std():.2f}\n")
            f.write(f"   最小值: {df[var].min():.2f}\n")
            f.write(f"   最大值: {df[var].max():.2f}\n")
            f.write(f"   分布情况: 见图表'{var}分布.png'\n")
            f.write(f"   与罚款比例关系: 见图表'{var}与罚款比例关系.png'\n\n")

print("分析完成。报告和图表已保存到相应目录。")