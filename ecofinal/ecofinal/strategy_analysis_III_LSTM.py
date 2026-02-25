# -*- coding: utf-8 -*-
"""
题目三：机器学习策略开发与回测
- LSTM 价格预测模型
- 交易策略回测
- 风险评估
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import sqlite3
import warnings
warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("题目三：LSTM 机器学习策略开发与回测")
print("=" * 60)

# ============================================================
# 1. 数据加载
# ============================================================
print("\n[1] 加载数据...")

conn = sqlite3.connect('stock_data.db')

stock_tables = {
    '600519.SH': 'feature_600519',  # 贵州茅台
    '300750.SZ': 'feature_300750',  # 宁德时代
    '601318.SH': 'feature_601318',  # 中国平安
    '000858.SZ': 'feature_000858',  # 五粮液
}

all_stock_data = {}
for code, table in stock_tables.items():
    df_stock = pd.read_sql(f'SELECT * FROM {table}', conn)
    df_stock['trade_date'] = pd.to_datetime(df_stock['trade_date'])
    df_stock = df_stock.sort_values('trade_date').reset_index(drop=True)
    all_stock_data[code] = df_stock
    print(f"  ✓ {code}: {len(df_stock)} 条记录")

conn.close()

# ============================================================
# 2. LSTM 模型定义
# ============================================================
print("\n[2] 定义 LSTM 模型...")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"  使用设备: {device}")

class LSTMPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.2):
        super(LSTMPredictor, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim,
                            num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        return self.fc(last_output)

# ============================================================
# 3. 数据准备与训练
# ============================================================
print("\n[3] 训练 LSTM 模型...")

SEQ_LENGTH = 20
PRED_DAYS = 5
TRAIN_RATIO = 0.8
EPOCHS = 50
FEATURE_COLS = ['MACD_DIF', 'MACD_DEA', 'MACD_Hist', 'RSI_14', 
                'BOLL_MID', 'BOLL_UP', 'BOLL_DN', 'turnover_rate']

def prepare_data(stock_df):
    data = stock_df.copy()
    data['target'] = data['close'].pct_change(PRED_DAYS).shift(-PRED_DAYS)
    
    available_cols = [c for c in FEATURE_COLS if c in data.columns]
    if len(available_cols) < 3:
        return None
    
    data = data.dropna(subset=['target'] + available_cols)
    if len(data) < SEQ_LENGTH + 50:
        return None
    
    features = data[available_cols].values
    target = data['target'].values
    dates = data['trade_date'].values
    
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    X, y, d = [], [], []
    for i in range(len(features_scaled) - SEQ_LENGTH):
        X.append(features_scaled[i:i+SEQ_LENGTH])
        y.append(target[i + SEQ_LENGTH])
        d.append(dates[i + SEQ_LENGTH])
    
    X, y = np.array(X), np.array(y)
    split_idx = int(len(X) * TRAIN_RATIO)
    
    return {
        'X_train': X[:split_idx], 'y_train': y[:split_idx],
        'X_test': X[split_idx:], 'y_test': y[split_idx:],
        'test_dates': d[split_idx:], 'input_dim': len(available_cols)
    }

def train_model(X_train, y_train, input_dim):
    X_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
    
    model = LSTMPredictor(input_dim=input_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        loss = criterion(model(X_tensor), y_tensor)
        loss.backward()
        optimizer.step()
    
    return model, loss.item()

# 训练所有股票
all_results = {}
model_metrics = []

for code, stock_df in all_stock_data.items():
    print(f"\n  训练: {code}")
    
    data = prepare_data(stock_df)
    if data is None:
        print(f"    数据不足，跳过")
        continue
    
    model, final_loss = train_model(data['X_train'], data['y_train'], data['input_dim'])
    
    # 预测
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(data['X_test'], dtype=torch.float32).to(device)
        predictions = model(X_test_tensor).cpu().numpy().flatten()
    
    # 评估
    y_test = data['y_test']
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    ic = np.corrcoef(predictions, y_test)[0, 1]
    direction_acc = np.mean(np.sign(predictions) == np.sign(y_test))
    
    print(f"    Loss: {final_loss:.6f} | IC: {ic:.4f} | 方向准确率: {direction_acc:.2%}")
    
    all_results[code] = {
        'predictions': predictions,
        'actual': y_test,
        'test_dates': data['test_dates'],
        'stock_df': stock_df
    }
    
    model_metrics.append({
        '股票': code, 'MSE': mse, 'MAE': mae, 'IC': ic, '方向准确率': direction_acc
    })

# ============================================================
# 4. 模型评估汇总
# ============================================================
print("\n" + "=" * 60)
print("[4] 模型评估汇总")
print("=" * 60)
metrics_df = pd.DataFrame(model_metrics)
print(metrics_df.to_string(index=False))

# ============================================================
# 5. 交易策略回测
# ============================================================
print("\n" + "=" * 60)
print("[5] 交易策略回测")
print("=" * 60)

INITIAL_CAPITAL = 1000000
BUY_THRESHOLD = 0.01
HOLD_DAYS = 5

def backtest(stock_df, predictions, test_dates):
    stock_df = stock_df.copy().sort_values('trade_date').reset_index(drop=True)
    date_to_idx = {d: i for i, d in enumerate(stock_df['trade_date'])}
    
    trades = []
    cash = INITIAL_CAPITAL
    position = 0
    entry_price = entry_date = None
    hold_count = 0
    equity_history = []
    
    for pred, date in zip(predictions, test_dates):
        date = pd.to_datetime(date)
        if date not in date_to_idx:
            continue
        
        idx = date_to_idx[date]
        if idx >= len(stock_df) - 1:
            continue
        
        current_price = stock_df.iloc[idx]['close']
        next_open = stock_df.iloc[idx + 1]['open']
        
        # 持仓检查
        if position > 0:
            hold_count += 1
            if hold_count >= HOLD_DAYS:
                profit_pct = current_price / entry_price - 1
                trades.append({
                    '买入日期': entry_date, '卖出日期': date,
                    '买入价': entry_price, '卖出价': current_price,
                    '收益率': profit_pct
                })
                cash += current_price * position
                position = 0
                hold_count = 0
        
        # 买入信号
        if position == 0 and pred > BUY_THRESHOLD:
            position = int(cash * 0.95 / next_open / 100) * 100
            if position > 0:
                cash -= next_open * position
                entry_price = next_open
                entry_date = date
                hold_count = 0
        
        equity_history.append(cash + position * current_price)
    
    return trades, equity_history

backtest_results = {}
all_trades = []

for code, result in all_results.items():
    trades, equity = backtest(result['stock_df'], result['predictions'], result['test_dates'])
    backtest_results[code] = {'trades': trades, 'equity': equity}
    
    if len(trades) > 0:
        trades_df = pd.DataFrame(trades)
        win_rate = (trades_df['收益率'] > 0).mean()
        avg_return = trades_df['收益率'].mean()
        
        print(f"\n{code}:")
        print(f"  交易次数: {len(trades)}")
        print(f"  胜率: {win_rate:.2%}")
        print(f"  平均收益: {avg_return:.2%}")
        
        if len(equity) > 0:
            total_return = (equity[-1] / INITIAL_CAPITAL - 1)
            print(f"  总收益率: {total_return:.2%}")
        
        for t in trades:
            t['股票'] = code
        all_trades.extend(trades)
    else:
        print(f"\n{code}: 无交易信号")

# ============================================================
# 6. 交易清单
# ============================================================
print("\n" + "=" * 60)
print("[6] 交易清单")
print("=" * 60)

if len(all_trades) > 0:
    trades_df = pd.DataFrame(all_trades)
    trades_df['收益率'] = trades_df['收益率'].apply(lambda x: f"{x:.2%}")
    print(trades_df[['股票', '买入日期', '卖出日期', '买入价', '卖出价', '收益率']].to_string(index=False))
    trades_df.to_csv('题目三_output/trade_list.csv', index=False, encoding='utf-8-sig')
    print("\n已保存: 题目三_output/trade_list.csv")
else:
    print("无交易记录")

# ============================================================
# 7. 风险评估
# ============================================================
print("\n" + "=" * 60)
print("[7] 风险评估指标")
print("=" * 60)

risk_metrics = []
for code, result in backtest_results.items():
    equity = result['equity']
    if len(equity) < 2:
        continue
    
    equity = np.array(equity)
    returns = np.diff(equity) / equity[:-1]
    
    total_return = equity[-1] / equity[0] - 1
    max_dd = np.min(equity / np.maximum.accumulate(equity) - 1)
    sharpe = (np.mean(returns) * 252 - 0.03) / (np.std(returns) * np.sqrt(252)) if np.std(returns) > 0 else 0
    
    trades = result['trades']
    win_rate = sum(1 for t in trades if t['收益率'] > 0) / len(trades) if trades else 0
    
    risk_metrics.append({
        '股票': code,
        '总收益率': f"{total_return:.2%}",
        '最大回撤': f"{max_dd:.2%}",
        '夏普比率': f"{sharpe:.2f}",
        '交易次数': len(trades),
        '胜率': f"{win_rate:.2%}"
    })

risk_df = pd.DataFrame(risk_metrics)
print(risk_df.to_string(index=False))
risk_df.to_csv('题目三_output/risk_metrics.csv', index=False, encoding='utf-8-sig')
print("\n已保存: 题目三_output/risk_metrics.csv")

print("\n" + "=" * 60)
print("运行完成!")
print("=" * 60)
