import nbformat as nbf

nb = nbf.v4.new_notebook()

md_intro = """# 基于深度强化学习与动态仓位管理（凯利公式+海龟法则）的量化交易策略

本 Notebook 实现了期末设计的核心创新模型。
**核心创新点**：
1. 使用深度强化学习（PPO算法）作为交易智能体，提取非线性市场特征。
2. 引入**海龟交易法则**的 ATR 波动率缩放，在市场高风险时自动降仓。
3. 引入**凯利公式（Kelly Criterion）**，根据智能体近期的胜率和盈亏比，动态计算最优下注比例，严格控制回撤。
"""

code_setup = """import tushare as ts
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import ta
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 初始化 Tushare
ts.set_token('229e2c478deaef0ccf3030b42121cc7b5ba066dd3c9789b4835c943d')
pro = ts.pro_api()
print("环境导入成功！")
"""

md_data = """## 1. 数据获取与特征工程 (Feature Engineering)
获取股票数据，并计算技术指标（MACD, RSI, ATR等）。ATR（真实波动幅度）将用于海龟交易法则的仓位控制。"""

code_data = """def get_data_with_features(ts_code, start_date, end_date):
    df = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
    df = df.sort_values('trade_date').reset_index(drop=True)
    
    # 计算技术指标
    df['MACD'] = ta.trend.macd_diff(df['close'])
    df['RSI'] = ta.momentum.rsi(df['close'])
    df['ATR'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'])
    df['CCI'] = ta.trend.cci(df['high'], df['low'], df['close'])
    
    # 计算收益率
    df['return'] = df['close'].pct_change()
    
    df = df.dropna().reset_index(drop=True)
    return df

# 获取贵州茅台数据作为训练和测试
df_stock = get_data_with_features('600519.SH', '20180101', '20241231')
print(f"数据获取完成，共 {len(df_stock)} 条记录。")
df_stock.tail()
"""

md_env = """## 2. 构建融合仓位管理的强化学习环境 (Custom Gym Environment)
这是本设计的核心。在 `step` 函数中，我们将 RL 输出的动作与凯利公式、海龟法则结合。"""

code_env = """class StockTradingEnv(gym.Env):
    def __init__(self, df, initial_balance=100000):
        super(StockTradingEnv, self).__init__()
        self.df = df
        self.initial_balance = initial_balance
        
        # 动作空间：[-1, 1]，表示 AI 的买卖意愿强度
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        
        # 状态空间：[收益率, MACD, RSI, ATR, CCI, 当前持仓比例, 凯利建议仓位]
        self.features = ['return', 'MACD', 'RSI', 'ATR', 'CCI']
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(len(self.features) + 2,), dtype=np.float32)
        
    def reset(self, seed=None):
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        
        # 记录历史交易用于凯利公式
        self.trade_history = [] 
        self.entry_price = 0
        
        return self._get_obs(), {}
        
    def _get_obs(self):
        obs = self.df.loc[self.current_step, self.features].values.tolist()
        holding_ratio = (self.shares_held * self.df.loc[self.current_step, 'close']) / self.net_worth
        kelly_f = self._calculate_kelly()
        obs.extend([holding_ratio, kelly_f])
        return np.array(obs, dtype=np.float32)
        
    def _calculate_kelly(self):
        # 凯利公式：f = W - (1-W)/R
        if len(self.trade_history) < 5:
            return 0.5 # 初始默认半仓
            
        wins = [t for t in self.trade_history if t > 0]
        losses = [t for t in self.trade_history if t <= 0]
        
        W = len(wins) / len(self.trade_history)
        avg_win = np.mean(wins) if wins else 0.01
        avg_loss = abs(np.mean(losses)) if losses else 0.01
        
        R = avg_win / avg_loss if avg_loss > 0 else 1.0
        kelly_f = W - (1 - W) / R
        
        # 限制凯利仓位在 0 到 1 之间（A股不能做空，且不加杠杆）
        return np.clip(kelly_f, 0.0, 1.0)
        
    def step(self, action):
        current_price = self.df.loc[self.current_step, 'close']
        current_atr = self.df.loc[self.current_step, 'ATR']
        mean_atr = self.df['ATR'].mean()
        
        # 1. AI 原始意愿
        ai_action = action[0] 
        
        # 2. 海龟法则：波动率缩放 (ATR越大，允许的仓位越小)
        turtle_scalar = np.clip(mean_atr / (current_atr + 1e-5), 0.1, 1.0)
        
        # 3. 凯利公式：胜率盈亏比缩放
        kelly_fraction = self._calculate_kelly()
        
        # 综合目标仓位 (Target Position)
        # 如果 AI 想买 (ai_action > 0)，则结合风控计算仓位；如果想卖 (ai_action < 0)，则直接减仓
        if ai_action > 0:
            target_ratio = ai_action * turtle_scalar * kelly_fraction
        else:
            target_ratio = 0 # 简化处理：负数直接清仓
            
        # 执行交易
        target_value = self.net_worth * target_ratio
        current_value = self.shares_held * current_price
        
        # 记录交易盈亏用于凯利公式
        if target_ratio == 0 and self.shares_held > 0:
            trade_return = (current_price - self.entry_price) / self.entry_price
            self.trade_history.append(trade_return)
            if len(self.trade_history) > 20: # 只看最近20次交易
                self.trade_history.pop(0)
                
        if target_ratio > 0 and self.shares_held == 0:
            self.entry_price = current_price
            
        # 更新账户
        self.shares_held = target_value / current_price
        self.balance = self.net_worth - target_value
        
        # 步进
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        
        # 计算 Reward (收益率 - 惩罚回撤)
        next_price = self.df.loc[self.current_step, 'close']
        self.net_worth = self.balance + self.shares_held * next_price
        
        self.max_net_worth = max(self.max_net_worth, self.net_worth)
        drawdown = (self.max_net_worth - self.net_worth) / self.max_net_worth
        
        # 奖励函数：净值增加给奖励，回撤给惩罚
        reward = (self.net_worth - self.initial_balance) / self.initial_balance - drawdown * 0.5
        
        return self._get_obs(), reward, done, False, {'net_worth': self.net_worth}
"""

md_train = """## 3. 模型训练 (Training PPO Agent)
划分训练集和测试集，使用 PPO 算法训练智能体。"""

code_train = """# 划分数据集 (2018-2022 训练, 2023-2024 测试)
train_df = df_stock[df_stock['trade_date'] < '20230101'].reset_index(drop=True)
test_df = df_stock[df_stock['trade_date'] >= '20230101'].reset_index(drop=True)

# 包装环境
env_train = DummyVecEnv([lambda: StockTradingEnv(train_df)])

# 初始化 PPO 模型
model = PPO("MlpPolicy", env_train, learning_rate=0.0005, n_steps=2048, batch_size=64, verbose=0)

print("开始训练 DRL 智能体...")
model.learn(total_timesteps=20000)
print("训练完成！")
"""

md_test = """## 4. 回测与可视化 (Backtesting & Visualization)
在测试集上运行训练好的模型，并与“买入持有”策略进行对比。"""

code_test = """env_test = StockTradingEnv(test_df)
obs, _ = env_test.reset()
done = False

net_worths = [env_test.initial_balance]
buy_hold_worths = [env_test.initial_balance]
initial_price = test_df.loc[0, 'close']

while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, _, info = env_test.step(action)
    
    net_worths.append(info['net_worth'])
    
    # 买入持有策略净值
    current_price = test_df.loc[env_test.current_step, 'close']
    buy_hold_worths.append(env_test.initial_balance * (current_price / initial_price))

# 绘制资金曲线
plt.figure(figsize=(12, 6))
plt.plot(test_df['trade_date'], net_worths[:-1], label='DRL + 凯利/海龟策略', color='red', linewidth=2)
plt.plot(test_df['trade_date'], buy_hold_worths[:-1], label='买入持有策略 (基准)', color='blue', alpha=0.6)

# 简化X轴标签显示
plt.xticks(test_df['trade_date'][::40], rotation=45)
plt.title('深度强化学习与动态仓位管理策略回测 (贵州茅台 2023-2024)')
plt.xlabel('日期')
plt.ylabel('账户净值 (元)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('drl_kelly_backtest.png')
plt.show()

# 计算指标
drl_return = (net_worths[-1] - net_worths[0]) / net_worths[0] * 100
bh_return = (buy_hold_worths[-1] - buy_hold_worths[0]) / buy_hold_worths[0] * 100
print(f"DRL策略总收益率: {drl_return:.2f}%")
print(f"买入持有总收益率: {bh_return:.2f}%")
"""

nb.cells = [
    nbf.v4.new_markdown_cell(md_intro),
    nbf.v4.new_code_cell(code_setup),
    nbf.v4.new_markdown_cell(md_data),
    nbf.v4.new_code_cell(code_data),
    nbf.v4.new_markdown_cell(md_env),
    nbf.v4.new_code_cell(code_env),
    nbf.v4.new_markdown_cell(md_train),
    nbf.v4.new_code_cell(code_train),
    nbf.v4.new_markdown_cell(md_test),
    nbf.v4.new_code_cell(code_test)
]

with open('/root/eco_design/DRL_Trading_Strategy.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Notebook generated successfully at /root/eco_design/DRL_Trading_Strategy.ipynb")
