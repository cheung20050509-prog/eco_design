import sqlite3
import pandas as pd

conn = sqlite3.connect('stock_data.db')

# 查看 feature_600519 (茅台) 的完整列名
df = pd.read_sql('SELECT * FROM feature_600519 LIMIT 5', conn)
print('feature_600519 (茅台) 列名:')
print(df.columns.tolist())
print(f'\n样本数据:')
print(df.head())

# 查看 feature_data 表
df2 = pd.read_sql('SELECT * FROM feature_data LIMIT 5', conn)
print('\n\nfeature_data 列名:')
print(df2.columns.tolist())

# 查看有哪些股票
stocks = pd.read_sql('SELECT DISTINCT ts_code FROM feature_data', conn)
print(f'\n股票列表: {stocks["ts_code"].tolist()}')

conn.close()
