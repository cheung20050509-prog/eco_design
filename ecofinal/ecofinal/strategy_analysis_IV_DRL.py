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
def get_and_process_data(ts_code, start_date, end_date, token):
    ts.set_token(token)
    pro = ts.pro_api()
    
    print(f"获取 {ts_code} 数据...")
    df = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
    df = df.sort_values('trade_date').reset_index(drop=True)
    
    # 添加技术指标 (Technical Analysis)
    print("计算技术指标...")
    # 趋势指标
    df['sma_5'] = ta.trend.sma_indicator(df['close'], window=5)
    df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
    df['macd'] = ta.trend.macd(df['close'])
    
    # 动量指标
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)
    
    # 波动率指标
    df['boll_h'] = ta.volatility.bollinger_hband(df['close'], window=20)
    df['boll_l'] = ta.volatility.bollinger_lband(df['close'], window=20)
    
    # 填充缺失值
    df = df.fillna(method='bfill').fillna(method='ffill')
    
    return df

# 2. 定义强化学习交易环境 (结合凯利公式思想)
class StockTradingEnv(gym.Env):
    def __init__(self, df, initial_balance=100000):
        super(StockTradingEnv, self).__init__()
        self.df = df
        self.initial_balance = initial_balance
        
        # 动作空间: [-1, 1]
        # -1 到 0: 卖出比例 (0=不卖, -1=全卖)
        # 0 到 1: 买入比例 (0=不买, 1=全买)
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        
        # 状态空间: [余额, 持仓量, 收盘价, SMA5, SMA20, MACD, RSI, BOLL_H, BOLL_L]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)
        
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        self.history = []
        return self._get_observation(), {}

    def _get_observation(self):
        obs = np.array([
            self.balance,
            self.shares_held,
            self.df.loc[self.current_step, 'close'],
            self.df.loc[self.current_step, 'sma_5'],
            self.df.loc[self.current_step, 'sma_20'],
            self.df.loc[self.current_step, 'macd'],
            self.df.loc[self.current_step, 'rsi'],
            self.df.loc[self.current_step, 'boll_h'],
            self.df.loc[self.current_step, 'boll_l']
        ], dtype=np.float32)
        return obs

    def step(self, action):
        current_price = self.df.loc[self.current_step, 'close']
        action = action[0]
        
        # 记录上一步的净值
        prev_net_worth = self.balance + self.shares_held * current_price
        
        # 执行交易动作
        if action > 0: # 买入
            # 结合凯利公式思想：不全仓买入，根据动作幅度决定买入比例
            buy_amount = self.balance * action
            shares_bought = buy_amount // current_price
            self.balance -= shares_bought * current_price
            self.shares_held += shares_bought
        elif action < 0: # 卖出
            sell_amount = self.shares_held * abs(action)
            shares_sold = int(sell_amount)
            self.balance += shares_sold * current_price
            self.shares_held -= shares_sold
            
        # 更新净值
        self.net_worth = self.balance + self.shares_held * current_price
        self.max_net_worth = max(self.max_net_worth, self.net_worth)
        
        # 计算奖励 (Reward)
        # 奖励 = 净值变化率 - 惩罚回撤
        step_return = (self.net_worth - prev_net_worth) / prev_net_worth
        drawdown = (self.max_net_worth - self.net_worth) / self.max_net_worth
        reward = step_return - 0.5 * drawdown # 惩罚回撤
        
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        
        # 记录历史
        self.history.append({
            'step': self.current_step,
            'date': self.df.loc[self.current_step, 'trade_date'],
            'net_worth': self.net_worth,
            'action': action,
            'price': current_price
        })
        
        return self._get_observation(), reward, done, False, {}

# 3. 训练与回测
def train_and_test():
    token = '229e2c478deaef0ccf3030b42121cc7b5ba066dd3c9789b4835c943d'
    ts_code = '600519.SH' # 贵州茅台
    
    # 获取数据
    df = get_and_process_data(ts_code, '20200101', '20241231', token)
    
    # 划分训练集和测试集 (80% 训练, 20% 测试)
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    test_df = df.iloc[split_idx:].reset_index(drop=True)
    
    print(f"训练集大小: {len(train_df)}, 测试集大小: {len(test_df)}")
    
    # 创建训练环境
    train_env = DummyVecEnv([lambda: StockTradingEnv(train_df)])
    
    # 训练 PPO 模型
    print("开始训练 PPO 强化学习模型...")
    model = PPO("MlpPolicy", train_env, verbose=0, learning_rate=0.0003)
    model.learn(total_timesteps=20000)
    print("模型训练完成！")
    
    # 测试模型
    print("开始在测试集上回测...")
    test_env = StockTradingEnv(test_df)
    obs, _ = test_env.reset()
    done = False
    
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = test_env.step(action)
        
    # 分析结果
    history_df = pd.DataFrame(test_env.history)
    
    # 计算基准收益 (买入持有)
    initial_price = test_df.loc[0, 'close']
    final_price = test_df.loc[len(test_df)-1, 'close']
    benchmark_return = (final_price - initial_price) / initial_price
    
    # 计算策略收益
    strategy_return = (test_env.net_worth - test_env.initial_balance) / test_env.initial_balance
    
    print("\n=== 回测结果 ===")
    print(f"测试期: {test_df['trade_date'].iloc[0]} 至 {test_df['trade_date'].iloc[-1]}")
    print(f"买入持有收益率: {benchmark_return*100:.2f}%")
    print(f"强化学习策略收益率: {strategy_return*100:.2f}%")
    print(f"最终净值: {test_env.net_worth:.2f}")
    
    # 绘制资金曲线
    plt.figure(figsize=(12, 6))
    plt.plot(history_df['net_worth'], label='DRL Strategy Net Worth', color='red')
    
    # 绘制基准曲线 (假设全仓买入)
    benchmark_net_worth = test_env.initial_balance * (test_df['close'] / initial_price)
    plt.plot(benchmark_net_worth.iloc[1:].values, label='Buy & Hold Benchmark', color='blue', alpha=0.6)
    
    plt.title('Deep Reinforcement Learning Trading Strategy vs Buy & Hold')
    plt.xlabel('Trading Days')
    plt.ylabel('Net Worth')
    plt.legend()
    plt.grid(True)
    
    output_dir = "/root/eco_design/ecofinal/ecofinal/题目三_output_drl"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'drl_equity_curve.png'))
    print(f"资金曲线已保存至 {output_dir}/drl_equity_curve.png")

if __name__ == "__main__":
    train_and_test()
