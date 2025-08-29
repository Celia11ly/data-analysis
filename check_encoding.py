import pandas as pd
from sklearn.preprocessing import LabelEncoder

# 读取数据
df = pd.read_excel('/Users/Celia/Downloads/dataA/Data0.xlsx')

# 检查主动整改的编码
le_整改 = LabelEncoder()
le_整改.fit(df['主动整改'].astype(str))
print('主动整改的编码映射:')
for cls, code in zip(le_整改.classes_, le_整改.transform(le_整改.classes_)):
    print(f'  {cls}: {code}')

# 检查主动停止违法行为的编码
le_停止 = LabelEncoder()
le_停止.fit(df['主动停止违法行为'].astype(str))
print('\n主动停止违法行为的编码映射:')
for cls, code in zip(le_停止.classes_, le_停止.transform(le_停止.classes_)):
    print(f'  {cls}: {code}')

# 查看编码后的值与罚款比例的关系
df['主动整改_编码'] = le_整改.transform(df['主动整改'].astype(str))
df['主动停止违法行为_编码'] = le_停止.transform(df['主动停止违法行为'].astype(str))
print('\n编码后的值与罚款比例的关系（前10行）:')
print(df[['主动整改', '主动整改_编码', '主动停止违法行为', '主动停止违法行为_编码', '罚款比例（%）']].head(10))
print('\n各编码值对应的平均罚款比例:')
print('主动整改编码值的平均罚款比例:')
print(df.groupby('主动整改_编码')['罚款比例（%）'].mean())
print('\n主动停止违法行为编码值的平均罚款比例:')
print(df.groupby('主动停止违法行为_编码')['罚款比例（%）'].mean())