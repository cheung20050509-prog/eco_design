import os
import pandas as pd
import numpy as np
import tushare as ts
import ta
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 数据获取与特征工程
def get_data(ts_code, start_date, end_date, token):
    ts.set_token(token)
    pro = ts.pro_api()
    print(f"正在获取 {ts_code} 数据...")
    df = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
    df = df.sort_values('trade_date').reset_index(drop=True)
    
    print("计算技术指标 (SMA, MACD, RSI, ATR)...")
    # 趋势指标
    df['sma_5'] = ta.trend.sma_indicator(df['close'], window=5)
    df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
    df['macd'] = ta.trend.macd(df['close'])
    
    # 动量指标
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)
    
    # 波动率指标 (用于海龟交易法)
    df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
    
    # 填充缺失值
    df = df.fillna(method='bfill').fillna(method='ffill')
    return df

# 2. 定义强化学习交易环境 (结合凯利公式与海龟交易法)
class AdvancedTradingEnv(gym.Env):
    def __init__(self, df, initial_balance=100000):
        super(AdvancedTradingEnv, self).__init__()
        self.df = df
        self.initial_balance = initial_balance
        
        # 动作空间: [-1, 1] (AI 输出的原始买卖意愿)
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        
        # 状态空间: [余额, 持仓量, 收盘价, SMA5, SMA20, MACD, RSI, ATR]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        
        # 记录历史交易盈亏，用于计算凯利公式
        self.trades = [] 
        self.entry_price = 0
        
        self.history = []
        return self._get_obs(), {}

    def _get_obs(self):
        obs = np.array([
            self.balance,
            self.shares_held,
            self.df.loc[self.current_step, 'close'],
            self.df.loc[self.current_step, 'sma_5'],
            self.df.loc[self.current_step, 'sma_20'],
            self.df.loc[self.current_step, 'macd'],
            self.df.loc[self.current_step, 'rsi'],
            self.df.loc[self.current_step, 'atr']
        ], dtype=np.float32)
        return obs

    def _calculate_kelly(self):
        """计算凯利公式最优仓位比例 f* = W - (1-W)/R"""
        if len(self.trades) < 5:
            return 0.2 # 初始默认仓位 20%
            
        wins = [t for t in self.trades if t > 0]
        losses = [t for t in self.trades if t <= 0]
        
        W = len(wins) / len(self.trades) # 胜率
        
        if len(losses) == 0:
            return 0.5 # 全胜时，限制最大仓位为 50% 防范黑天鹅
            
        avg_win = np.mean(wins) if wins else 0
        avg_loss = abs(np.mean(losses))
        
        if avg_loss == 0:
            return 0.5
            
        R = avg_win / avg_loss # 盈亏比
        
        kelly_f = W - ((1 - W) / R)
        # 限制凯利比例在 0 到 0.5 之间 (半凯利，更稳健)
        return np.clip(kelly_f, 0.0, 0.5)

    def step(self, action):
        current_price = self.df.loc[self.current_step, 'close']
        current_atr = self.df.loc[self.current_step, 'atr']
        action = action[0]
        
        prev_net_worth = self.balance + self.shares_held * current_price
        
        # --- 核心创新：风控外脑 (凯利公式 + 海龟法则) ---
        kelly_f = self._calculate_kelly()
        
        # 海龟法则：根据 ATR 波动率计算 1 个 Unit 的资金量 (风险暴露 1%)
        turtle_unit_capital = (self.net_worth * 0.01) / (current_atr / current_price + 1e-5)
        max_turtle_capital = turtle_unit_capital * 4 # 海龟法则最多允许 4 个 Unit
        
        if action > 0: # AI 决定买入
            if self.shares_held == 0:
                self.entry_price = current_price
            
            # 1. 账户最大可买资金
            max_affordable = self.balance
            # 2. 凯利公式允许的最大资金
            kelly_capital = self.net_worth * kelly_f
            
            # 最终允许动用的资金 = min(账户余额, 凯利上限, 海龟上限)
            allowed_capital = min(max_affordable, kelly_capital, max_turtle_capital)
            
            # 结合 AI 的买入意愿 (action 0~1)
            actual_buy_capital = allowed_capital * action
            shares_to_buy = int(actual_buy_capital // current_price)
            
            self.balance -= shares_to_buy * current_price
            self.shares_held += shares_to_buy
            
        elif action < 0: # AI 决定卖出
            shares_to_sell = int(self.shares_held * abs(action))
            if shares_to_sell > 0:
                # 记录这笔交易的盈亏率，喂给凯利公式
                trade_return = (current_price - self.entry_price) / self.entry_price
                self.trades.append(trade_return)
                
            self.balance += shares_to_sell * current_price
            self.shares_held -= shares_to_sell
            if self.shares_held == 0:
                self.entry_price = 0
                
        # 更新净值
        self.net_worth = self.balance + self.shares_held * current_price
        self.max_net_worth = max(self.max_net_worth, self.net_worth)
        
        # 计算奖励 (Reward): 收益率 - 惩罚回撤
        step_return = (self.net_worth - prev_net_worth) / prev_net_worth
        drawdown = (self.max_net_worth - self.net_worth) / self.max_net_worth
        reward = step_return - 1.0 * drawdown # 强力惩罚回撤
        
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        
        self.history.append({
            'date': self.df.loc[self.current_step, 'trade_date'],
            'net_worth': self.net_worth,
            'action': action,
            'price': current_price,
            'kelly_f': kelly_f
        })
        
        return self._get_obs(), reward, done, False, {}

# 3. 训练与回测主函数
def main():
    token = '229e2c478deaef0ccf3030b42121cc7b5ba066dd3c9789b4835c943d'
    ts_code = '600519.SH' # 贵州茅台
    
    df = get_data(ts_code, '20200101', '20241231', token)
    
    # 划分训练集和测试集 (80% 训练, 20% 测试)
    split = int(len(df) * 0.8)
    train_df = df.iloc[:split].reset_index(drop=True)
    test_df = df.iloc[split:].reset_index(drop=True)
    
    print(f"\n训练集大小: {len(train_df)} 天, 测试集大小: {len(test_df)} 天")
    print("开始训练 DRL 智能体 (结合凯利公式与海龟法则)...")
    
    env = DummyVecEnv([lambda: AdvancedTradingEnv(train_df)])
    model = PPO("MlpPolicy", env, verbose=0, learning_rate=0.0005)
    model.learn(total_timesteps=30000)
    print("模型训练完成！")
    
    print("\n开始在测试集上进行回测...")
    test_env = AdvancedTradingEnv(test_df)
    obs, _ = test_env.reset()
    done = False
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = test_env.step(action)
        
    history = pd.DataFrame(test_env.history)
    
    # 计算收益率
    initial_price = test_df.loc[0, 'close']
    benchmark_net_worth = test_env.initial_balance * (test_df['close'] / initial_price)
    
    strat_ret = (test_env.net_worth - test_env.initial_balance) / test_env.initial_balance
    bench_ret = (benchmark_net_worth.iloc[-1] - test_env.initial_balance) / test_env.initial_balance
    
    print("\n=== 回测结果 ===")
    print(f"测试期: {test_df['trade_date'].iloc[0]} 至 {test_df['trade_date'].iloc[-1]}")
    print(f"买入持有 (基准) 收益率: {bench_ret*100:.2f}%")
    print(f"DRL+凯利+海龟 策略收益率: {strat_ret*100:.2f}%")
    print(f"最终净值: {test_env.net_worth:.2f}")
    
    # 绘制资金曲线
    plt.figure(figsize=(14, 7))
    plt.plot(pd.to_datetime(history['date']), history['net_worth'], label='DRL + Kelly + Turtle Strategy', color='red', linewidth=2)
    plt.plot(pd.to_datetime(test_df['trade_date'].iloc[1:]), benchmark_net_worth.iloc[1:], label='Buy & Hold Benchmark', color='blue', alpha=0.5)
    
    plt.title(f'Deep Reinforcement Learning Trading Strategy ({ts_code})')
    plt.xlabel('Date')
    plt.ylabel('Net Worth')
    plt.legend()
    plt.grid(True)
    
    out_dir = '/root/eco_design/output'
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, 'drl_kelly_turtle_result.png')
    plt.savefig(save_path)
    print(f"\n资金曲线图已保存至: {save_path}")

if __name__ == '__main__':
    main()
