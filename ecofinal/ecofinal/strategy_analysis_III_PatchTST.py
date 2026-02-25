# -*- coding: utf-8 -*-
"""
题目三：机器学习策略开发与回测 (PatchTST + Kelly Criterion)
- 模型: PatchTST (Patch Time Series Transformer)
- 策略: 基于预测收益率 + 凯利公式仓位管理
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
import os
import sys

# 解决控制台输出乱码问题
sys.stdout.reconfigure(encoding='utf-8')

warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建输出目录
OUTPUT_DIR = '题目三_output_patchtst'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("题目三：机器学习策略开发与回测 (PatchTST 版)")
print(f"输出目录: {OUTPUT_DIR}/")
print("=" * 60)

# ============================================================
# 1. 数据加载 (含情绪指标)
# ============================================================
print("\n[1] 加载数据...")

conn = sqlite3.connect('stock_data.db')

all_stock_data = {}

# 尝试从 feature_data 表（包含整合后的宏观与情绪特征）读取
try:
    # 检查是否存在 feature_data 表
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='feature_data'")
    if cursor.fetchone():
        print("  正在从 feature_data 表读取数据...")
        df_all = pd.read_sql('SELECT * FROM feature_data', conn)
        df_all['trade_date'] = pd.to_datetime(df_all['trade_date'])
        
        codes = df_all['ts_code'].unique()
        print(f"  发现标的: {codes}")
        
        for code in codes:
            df_stock = df_all[df_all['ts_code'] == code].copy()
            df_stock = df_stock.sort_values('trade_date').reset_index(drop=True)
            
            # 简单预处理：填充
            df_stock = df_stock.fillna(method='ffill').fillna(method='bfill')
            
            all_stock_data[code] = df_stock
            print(f"  [OK] {code}: {len(df_stock)} 条记录")
    else:
        print("  [!] 未找到 feature_data 表，请先运行 data_analysis_III.ipynb 生成数据。")

except Exception as e:
    print(f"  数据加载出错: {e}")

conn.close()

# ============================================================
# 2. PatchTST 模型定义
# ============================================================
print("\n[2] 定义 PatchTST 模型...")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"  使用设备: {device}")

class PatchTST(nn.Module):
    def __init__(self, input_dim, seq_len, patch_len=8, stride=4, d_model=64, n_heads=4, n_layers=2, dropout=0.1):
        """
        PatchTST implementation adapted for Regression (Single Target Prediction)
        """
        super(PatchTST, self).__init__()
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.stride = stride
        self.input_dim = input_dim
        self.d_model = d_model
        
        # Calculate number of patches
        # (L - P) / S + 1
        self.num_patches = (seq_len - patch_len) // stride + 1
        # print(f"  Debug: Seq={seq_len}, Patch={patch_len}, Stride={stride} -> Num Patches={self.num_patches}")
        
        # Patch Projection: Project each patch of length P to d_model
        # Treating each variable independently initially (Channel Independence logic)
        # But we will combine them later for the specific target prediction
        self.patch_embed = nn.Linear(patch_len, d_model)
        
        # Positional Embedding: Learnable position for each patch
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, d_model))
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, 
                                                 dim_feedforward=d_model*4, dropout=dropout, 
                                                 batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Prediction Head
        # Flatten: (Batch * Input_Dim, Num_Patches, d_model) -> (Batch, Input_Dim * Num_Patches * d_model)
        # We need to map all features' patch info to 1 scalar output
        self.flatten_dim = input_dim * self.num_patches * d_model
        
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flatten_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        # x shape: (Batch, Seq_Len, Input_Dim)
        B, S, C = x.shape
        
        # 1. Permute to (Batch, Input_Dim, Seq_Len) for patching
        x = x.permute(0, 2, 1) 
        
        # 2. Patchify using unfold
        # (Batch, Input_Dim, Num_Patches, Patch_Len)
        x_patched = x.unfold(dimension=2, size=self.patch_len, step=self.stride)
        
        # 3. Channel Independence Trick: Flatten Batch and Channel dimensions
        # (Batch * Input_Dim, Num_Patches, Patch_Len)
        x_ci = x_patched.reshape(B * C, self.num_patches, self.patch_len)
        
        # 4. Patch Embedding & Positional Encoding
        emb = self.patch_embed(x_ci) # (B*C, N_P, d_model)
        emb = emb + self.pos_embed   # Broadcast add position
        
        # 5. Transformer Encoder
        enc_out = self.encoder(emb)  # (B*C, N_P, d_model)
        
        # 6. Reshape back to separate Batch and Channel
        # (Batch, Input_Dim, Num_Patches, d_model)
        enc_out = enc_out.reshape(B, C, self.num_patches, self.d_model)
        
        # 7. Prediction Head (Mixing all channels information)
        # Flatten everything except Batch
        out = self.head(enc_out)
        
        return out

# ============================================================
# 3. 数据准备与训练
# ============================================================
print("\n[3] 训练 PatchTST 模型...")

# PatchTST通常适合稍长一点的序列，这里增加seq_length
SEQ_LENGTH = 32  # 32天
PATCH_LEN = 8    # Patch长度
STRIDE = 4       # 步长 32 -> (32-8)/4 + 1 = 7 patches
PRED_DAYS = 5    # 预测未来 5 日
TRAIN_RATIO = 0.8
EPOCHS = 60      # 稍微增加训练轮数

# 更新特征列表以匹配 data_analysis_III.ipynb 生成的最新列名 (全小写)
FEATURE_COLS = ['dif', 'dea', 'macd', 'rsi', 
                'ma20', 'upper_bb', 'lower_bb', 'bb_width',
                'turnover_rate',
                'vix', 'sentiment_score', 'cn_us_spread', 'baidu_index']

def prepare_data(stock_df):
    data = stock_df.copy()
    
    # 预测目标：未来5日收益率
    data['target'] = data['close'].pct_change(PRED_DAYS).shift(-PRED_DAYS)
    
    available_cols = [c for c in FEATURE_COLS if c in data.columns]
    
    # Drop NAs
    data = data.dropna(subset=['target'] + available_cols)
    if len(data) < SEQ_LENGTH + 50:
        return None
    
    features = data[available_cols].values
    target = data['target'].values
    dates = data['trade_date'].values
    
    # 标准化
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
        'test_dates': d[split_idx:], 'input_dim': len(available_cols),
        'scaler': scaler
    }

def train_model(X_train, y_train, input_dim):
    X_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
    
    model = PatchTST(input_dim=input_dim, seq_len=SEQ_LENGTH, patch_len=PATCH_LEN, stride=STRIDE).to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)
    
    model.train()
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        loss = criterion(model(X_tensor), y_tensor)
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 10 == 0:
            pass # print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss.item():.6f}")
    
    return model, loss.item()

all_results = {}
model_metrics = []

for code, stock_df in all_stock_data.items():
    print(f"\n  训练: {code}")
    
    data = prepare_data(stock_df)
    if data is None:
        print(f"    数据不足，跳过")
        continue

    # 计算训练集的统计数据用于凯利公式
    # 我们需要预先评估一下训练集上的胜率
    
    model, final_loss = train_model(data['X_train'], data['y_train'], data['input_dim'])
    
    model.eval()
    with torch.no_grad():
        # 预测测试集
        X_test_tensor = torch.tensor(data['X_test'], dtype=torch.float32).to(device)
        predictions = model(X_test_tensor).cpu().numpy().flatten()
        
        # 预测训练集 (用于凯利公式参数估计)
        X_train_tensor = torch.tensor(data['X_train'], dtype=torch.float32).to(device)
        train_preds = model(X_train_tensor).cpu().numpy().flatten()

    y_test = data['y_test']
    y_train = data['y_train']
    
    # 评估指标
    ic = np.corrcoef(predictions, y_test)[0, 1]
    direction_acc = np.mean(np.sign(predictions) == np.sign(y_test))
    
    # 计算凯利公式参数 (基于训练集表现)
    # 定义一次"赢": 预测涨且实际涨 (做多获利)
    # 这里我们简化：如果预测 > 0，我们假设会买入。
    # 统计所有 预测 > 0 的样本中，实际 > 0 的比例 (胜率 P)
    # 以及 实际收益 / 亏损比 (赔率 b)
    
    long_signals = train_preds > 0.005 # 略微设置阈值
    if np.sum(long_signals) > 10:
        actual_returns = y_train[long_signals]
        wins = actual_returns[actual_returns > 0]
        losses = actual_returns[actual_returns <= 0]
        
        if len(wins) > 0 and len(losses) > 0:
            win_rate = len(wins) / len(actual_returns)
            avg_win = np.mean(wins)
            avg_loss = abs(np.mean(losses))
            win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0
            
            # 凯利公式 f = p/a - q/b ? 
            # 通用形式: f = p - (1-p)/b, 其中 b = 赔率 (获利/亏损)
            kelly_f = win_rate - (1 - win_rate) / win_loss_ratio if win_loss_ratio > 0 else 0
        else:
            kelly_f = 0
            win_rate = 0
            win_loss_ratio = 0
    else:
        kelly_f = 0
        win_rate = 0
        win_loss_ratio = 0
        
    print(f"    Loss: {final_loss:.6f} | IC: {ic:.4f} | 方向准确率: {direction_acc:.2%}")
    print(f"    Kelly参数(Train): 胜率={win_rate:.2%}, 盈亏比={win_loss_ratio:.2f} -> f={kelly_f:.2f}")

    all_results[code] = {
        'predictions': predictions,
        'actual': y_test,
        'test_dates': data['test_dates'],
        'stock_df': stock_df,
        'kelly_f': max(0, kelly_f) # 只有当f>0时才开仓
    }
    
    model_metrics.append({
        '股票': code, 'IC': ic, '方向准确率': direction_acc, 'Kelly_f': kelly_f
    })

# ============================================================
# 4. 交易策略回测 (凯利公式)
# ============================================================
print("\n" + "=" * 60)
print("[4] 交易策略回测 (基于 Kelly Criterion)")
print("=" * 60)

INITIAL_CAPITAL = 1000000
BUY_THRESHOLD = 0.01
HOLD_DAYS = 5
MAX_POSITION_PCT = 0.8 # 单只股票最大仓位限制 (风控)

def backtest_kelly(stock_df, predictions, test_dates, kelly_f):
    stock_df = stock_df.copy().sort_values('trade_date').reset_index(drop=True)
    date_to_idx = {d: i for i, d in enumerate(stock_df['trade_date'])}
    
    trades = []
    cash = INITIAL_CAPITAL
    position_value = 0
    position_shares = 0
    entry_price = entry_date = None
    hold_count = 0
    equity_history = []
    
    # 若计算出的 kelly_f 太激进，进行减半凯利 (Half Kelly) 以降低波动
    # 且设置上限
    target_pos_pct = min(kelly_f * 0.5, MAX_POSITION_PCT)
    if target_pos_pct < 0.05: target_pos_pct = 0 # 仓位太小就不做了
    
    # print(f"  策略配置: 目标仓位占比 = {target_pos_pct:.2%}")

    for pred, date in zip(predictions, test_dates):
        date = pd.to_datetime(date)
        if date not in date_to_idx:
            continue
        idx = date_to_idx[date]
        if idx >= len(stock_df) - 1:
            continue
        
        current_price = stock_df.iloc[idx]['close']
        next_open = stock_df.iloc[idx + 1]['open']
        
        # 1. 卖出逻辑 (持有期满)
        if position_shares > 0:
            hold_count += 1
            if hold_count >= HOLD_DAYS:
                # 卖出
                revenue = position_shares * current_price
                profit_pct = current_price / entry_price - 1
                
                trades.append({
                    '买入日期': entry_date, '卖出日期': date,
                    '买入价': entry_price, '卖出价': current_price,
                    '收益率': profit_pct,
                    '盈亏额': revenue - (position_shares * entry_price)
                })
                
                cash += revenue
                position_shares = 0
                position_value = 0
                hold_count = 0
        
        # 2. 买入逻辑
        if position_shares == 0 and pred > BUY_THRESHOLD and target_pos_pct > 0:
            # 使用凯利公式确定的资金比例
            invest_amount = (cash + position_value) * target_pos_pct
            
            # 如果 cash 不够 invest_amount (理论上不应该，因为全平仓了)，则用全部cash
            invest_amount = min(invest_amount, cash)
            
            shares = int(invest_amount / next_open / 100) * 100
            if shares > 0:
                cost = shares * next_open
                cash -= cost
                position_shares = shares
                entry_price = next_open
                entry_date = date
                hold_count = 0
        
        # 更新每日权益
        current_equity = cash + (position_shares * current_price)
        equity_history.append(current_equity)
        position_value = position_shares * current_price
    
    return trades, equity_history, target_pos_pct

backtest_results = {}
all_trades = []

for code, result in all_results.items():
    trades, equity, used_kelly = backtest_kelly(result['stock_df'], result['predictions'], 
                                              result['test_dates'], result['kelly_f'])
    backtest_results[code] = {'trades': trades, 'equity': equity}
    
    if len(trades) > 0:
        trades_df = pd.DataFrame(trades)
        win_rate = (trades_df['收益率'] > 0).mean()
        avg_return = trades_df['收益率'].mean()
        
        print(f"\n{code} (Target Pos: {used_kelly:.1%}):")
        print(f"  交易次数: {len(trades)}")
        print(f"  胜率: {win_rate:.2%}")
        print(f"  总收益率: {(equity[-1]/INITIAL_CAPITAL - 1):.2%}")
        
        for t in trades:
            t['股票'] = code
        all_trades.extend(trades)
    else:
        print(f"\n{code}: 无交易信号 (Kelly F={result['kelly_f']:.2f})")

# ============================================================
# 5. 结果保存与可视化
# ============================================================
if len(all_trades) > 0:
    trades_df = pd.DataFrame(all_trades)
    trades_df.to_csv(f'{OUTPUT_DIR}/trade_list_kelly.csv', index=False, encoding='utf-8-sig')

# 风险指标计算
risk_metrics = []
for code, result in backtest_results.items():
    equity = result['equity']
    if len(equity) < 2: continue
    
    equity_arr = np.array(equity)
    returns = np.diff(equity_arr) / equity_arr[:-1]
    
    total_ret = equity_arr[-1] / equity_arr[0] - 1
    max_dd = np.min(equity_arr / np.maximum.accumulate(equity_arr) - 1)
    sharpe = (np.mean(returns) * 252 - 0.03) / (np.std(returns) * np.sqrt(252)) if np.std(returns) > 0 else 0
    
    risk_metrics.append({
        '股票': code, '总收益率': f"{total_ret:.2%}", 
        '最大回撤': f"{max_dd:.2%}", '夏普比率': f"{sharpe:.2f}"
    })

pd.DataFrame(risk_metrics).to_csv(f'{OUTPUT_DIR}/risk_metrics_kelly.csv', index=False, encoding='utf-8-sig')
print(f"\n结果已保存至 {OUTPUT_DIR}/")

pd.DataFrame(risk_metrics).to_csv(f'{OUTPUT_DIR}/risk_metrics_kelly.csv', index=False, encoding='utf-8-sig')
print(f"\n结果已保存至 {OUTPUT_DIR}/")

# 高级绘图
import seaborn as sns
import matplotlib.ticker as mtick

# 1. 动态权益曲线 (归一化)
plt.figure(figsize=(12, 6))
for code, result in backtest_results.items():
    if len(result['equity']) > 0:
        equity = np.array(result['equity'])
        # 归一化为净值 (从1.0开始)
        normalized_equity = equity / equity[0]
        plt.plot(normalized_equity, label=f'{code}', linewidth=1.5)

plt.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='初始净值 (1.0)')
plt.title('PatchTST + Kelly 策略净值曲线 (Net Value Curve)', fontsize=14)
plt.xlabel('交易日 (Days)', fontsize=12)
plt.ylabel('策略净值 (Initial=1.0)', fontsize=12)
plt.legend(loc='upper left', frameon=True)
plt.grid(True, alpha=0.3)
plt.savefig(f'{OUTPUT_DIR}/equity_curve_kelly.png', dpi=150, bbox_inches='tight')
print(f"已保存: {OUTPUT_DIR}/equity_curve_kelly.png")

# 2. 收益率对比柱状图
plt.figure(figsize=(12, 6))
returns_data = []
for code, result in backtest_results.items():
    if len(result['equity']) > 0:
        total_ret = result['equity'][-1] / result['equity'][0] - 1
        returns_data.append({'股票': code, '总收益率': total_ret})

if returns_data:
    df_plot = pd.DataFrame(returns_data)
    # 按收益率排序
    df_plot = df_plot.sort_values('总收益率', ascending=False)
    
    colors = ['red' if x > 0 else 'green' for x in df_plot['总收益率']]
    bars = plt.bar(df_plot['股票'], df_plot['总收益率'], color=colors, alpha=0.8, edgecolor='black', width=0.6)
    plt.axhline(0, color='black', linestyle='-', linewidth=0.8)
    plt.title('策略总收益率对比 (PatchTST + Kelly)', fontsize=14)
    plt.ylabel('累计收益率', fontsize=12)
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    plt.xticks(rotation=45)
    
    for bar in bars:
        height = bar.get_height()
        xy_pos = (bar.get_x() + bar.get_width() / 2, height)
        # 调整标签位置以免遮挡
        xy_text = (0, 5) if height > 0 else (0, -15)
        plt.annotate(f'{height:.2%}', xy=xy_pos, xytext=xy_text,
                     textcoords="offset points", ha='center', va='bottom', fontsize=10, fontweight='bold')
                     
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/strategy_comparison_kelly.png', dpi=150, bbox_inches='tight')
    print(f"已保存: {OUTPUT_DIR}/strategy_comparison_kelly.png")

# 3. 持仓热力图
print("\n绘制持仓热力图...")
all_dates = set()
for code, result in backtest_results.items():
    if 'trades' in result:
        for trade in result['trades']:
            all_dates.add(pd.to_datetime(trade['买入日期']))
            all_dates.add(pd.to_datetime(trade['卖出日期']))

if len(all_dates) > 0:
    min_date = min(all_dates)
    max_date = max(all_dates)
    date_range = pd.date_range(start=min_date, end=max_date, freq='D')
    
    # 按照收益率排序股票显示顺序
    sorted_stocks = df_plot['股票'].tolist() if 'df_plot' in locals() else list(backtest_results.keys())
    
    position_matrix = pd.DataFrame(0, index=sorted_stocks, columns=date_range)
    
    for code, result in backtest_results.items():
        if code not in position_matrix.index: continue
        for trade in result['trades']:
            buy_date = pd.to_datetime(trade['买入日期'])
            sell_date = pd.to_datetime(trade['卖出日期'])
            # 标记持仓期
            # 注意: 切片是包含头尾的，但卖出当天往往不再持有过夜，可视情况调整
            # 这里简化处理: 标记整个区间 [买入, 卖出]
            if buy_date in position_matrix.columns and sell_date in position_matrix.columns:
                position_matrix.loc[code, buy_date:sell_date] = 1 
    
    # 绘图
    plt.figure(figsize=(15, 6))
    # 使用自定义颜色: 0=白色, 1=深绿色
    sns.heatmap(position_matrix, cmap=['#f7f7f7', '#2ca02c'], cbar=False, 
                linewidths=0.5, linecolor='lightgray')
                
    plt.title('策略持仓分布热力图 (绿色=持仓)', fontsize=14)
    plt.xlabel('日期', fontsize=12)
    plt.ylabel('股票 (按收益率排序)', fontsize=12)
    
    # 优化X轴标签，只显示月初
    # xticks = [d for d in date_range if d.day == 1]
    # plt.xticks(...) 
    # Seaborn heatmap会自动处理一些，但可能太密，此处默认即可
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/position_heatmap_kelly.png', dpi=150, bbox_inches='tight')
    print(f"已保存: {OUTPUT_DIR}/position_heatmap_kelly.png")
else:
    print("无交易记录，无法绘制热力图")

print("所有可视化完成.")
