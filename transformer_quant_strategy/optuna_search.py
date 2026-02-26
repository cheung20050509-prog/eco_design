# -*- coding: utf-8 -*-
"""
Optuna超参数搜索脚本
====================
功能：
1. 使用Optuna自动搜索iTransformer模型超参数 + 策略超参数
2. 复用已有模块 (data_acquisition, feature_engineering, transformer_model, 
   trading_strategy, backtest_engine)
3. 以夏普比率为主要优化目标（兼顾年化收益率和最大回撤）

超参数搜索空间：
  模型超参数（7个）: seq_length, d_model, n_heads, n_layers, dropout, lr, batch_size
  策略超参数（5个）: buy_threshold, sell_threshold, stop_loss, take_profit, max_position
"""

import warnings
warnings.filterwarnings('ignore')

import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import torch

# 导入项目模块
from data_acquisition import DataAcquisition
from feature_engineering import FeatureEngineering
from transformer_model import TransformerPredictor
from trading_strategy import TradingStrategy
from backtest_engine import BacktestEngine

# 输出目录
OUTPUT_DIR = 'output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================================
# 数据准备（只做一次，所有trial共享）
# ============================================================
def prepare_data(stock_list, start_date='20180101', end_date='20251231'):
    """准备数据，返回 feature_data 和 feature_cols"""
    print("=" * 60)
    print("[数据准备] 获取并预处理数据（所有trial共享）")
    print("=" * 60)
    
    data_module = DataAcquisition()
    
    print("\n[1] 获取股票数据...")
    all_stock_data = data_module.fetch_stock_data(
        stock_list=stock_list,
        start_date=start_date,
        end_date=end_date
    )
    
    print("\n[2] 数据清洗...")
    cleaned_data = data_module.clean_data(all_stock_data)
    
    print("\n[3] 特征工程...")
    fe_module = FeatureEngineering()
    feature_data = fe_module.compute_all_features(cleaned_data)
    feature_cols = fe_module.get_feature_columns()
    
    print(f"\n  数据准备完成:")
    print(f"    股票数量: {len(feature_data['stock_code'].unique())}")
    print(f"    数据总量: {len(feature_data)} 条")
    print(f"    特征数量: {len(feature_cols)}")
    
    return feature_data, feature_cols


# ============================================================
# 单次trial评估函数
# ============================================================
def evaluate_trial(feature_data, feature_cols, model_params, strategy_params):
    """
    用给定超参数训练模型 + 回测，返回绩效指标
    
    参数:
        feature_data: 特征数据
        feature_cols: 特征列名
        model_params: 模型超参数字典
        strategy_params: 策略超参数字典
    
    返回:
        metrics: 绩效指标字典, 或 None（失败时）
    """
    try:
        # 1. 训练模型
        predictor = TransformerPredictor(
            seq_length=model_params['seq_length'],
            pred_length=5,
            d_model=model_params['d_model'],
            n_heads=model_params['n_heads'],
            n_layers=model_params['n_layers'],
            d_ff=model_params.get('d_ff', 4 * model_params['d_model']),
            dropout=model_params['dropout'],
            epochs=model_params['epochs'],
            learning_rate=model_params['learning_rate'],
            batch_size=model_params['batch_size'],
            use_itransformer=True
        )
        
        model_results = predictor.train_and_predict(
            feature_data=feature_data,
            feature_cols=feature_cols,
            target_col='future_return_5d',
            train_ratio=0.8
        )
        
        # 2. 生成交易信号
        strategy = TradingStrategy(
            initial_capital=1000000,
            max_position_ratio=strategy_params['max_position_ratio'],
            stop_loss_ratio=strategy_params['stop_loss_ratio'],
            take_profit_ratio=strategy_params['take_profit_ratio'],
            use_kelly=True,
            buy_threshold=strategy_params['buy_threshold'],
            sell_threshold=strategy_params['sell_threshold']
        )
        
        signals = strategy.generate_signals(
            predictions=model_results['predictions'],
            feature_data=feature_data
        )
        
        # 3. 回测
        backtest = BacktestEngine(initial_capital=1000000)
        backtest_results = backtest.run_backtest(signals=signals, price_data=feature_data)
        metrics = backtest.calculate_metrics(backtest_results)
        
        # 附加模型指标
        metrics['direction_accuracy'] = model_results.get('direction_accuracy', 0)
        metrics['mse'] = model_results.get('mse', float('inf'))
        
        return metrics
        
    except Exception as e:
        print(f"    [trial失败] {e}")
        return None


# ============================================================
# Optuna目标函数
# ============================================================
def create_objective(feature_data, feature_cols):
    """
    创建Optuna目标函数闭包
    
    参数:
        feature_data: 特征数据
        feature_cols: 特征列
    """
    
    def objective(trial):
        """Optuna目标函数 - 最大化夏普比率"""
        
        # ========== 模型超参数搜索空间 ==========
        d_model = trial.suggest_categorical('d_model', [512, 1024, 1536, 2048])
        n_heads = trial.suggest_categorical('n_heads', [8, 16, 32])
        n_layers = trial.suggest_int('n_layers', 4, 16)
        seq_length = trial.suggest_categorical('seq_length', [30, 60, 90, 120])
        dropout = trial.suggest_float('dropout', 0.05, 0.3)
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 5e-4, log=True)
        batch_size = trial.suggest_categorical('batch_size', [64, 128, 256, 512])
        d_ff_ratio = trial.suggest_categorical('d_ff_ratio', [2, 4])  # d_ff = d_ff_ratio * d_model
        d_ff = d_ff_ratio * d_model
        epochs = trial.suggest_int('epochs', 200, 800, step=10)
        
        # 确保 d_model 能被 n_heads 整除
        if d_model % n_heads != 0:
            raise optuna.TrialPruned()
        
        # ========== 策略超参数搜索空间 ==========
        buy_threshold = trial.suggest_float('buy_threshold', 0.003, 0.02)
        sell_threshold = trial.suggest_float('sell_threshold', -0.02, -0.002)
        stop_loss_ratio = trial.suggest_float('stop_loss_ratio', 0.02, 0.08)
        take_profit_ratio = trial.suggest_float('take_profit_ratio', 0.04, 0.15)
        max_position_ratio = trial.suggest_float('max_position_ratio', 0.10, 0.40)
        
        model_params = {
            'seq_length': seq_length,
            'd_model': d_model,
            'n_heads': n_heads,
            'n_layers': n_layers,
            'd_ff': d_ff,
            'dropout': dropout,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'epochs': epochs,
        }
        
        strategy_params = {
            'buy_threshold': buy_threshold,
            'sell_threshold': sell_threshold,
            'stop_loss_ratio': stop_loss_ratio,
            'take_profit_ratio': take_profit_ratio,
            'max_position_ratio': max_position_ratio,
        }
        
        trial_num = trial.number + 1
        print(f"\n{'='*50}")
        print(f"  Trial {trial_num}")
        print(f"{'='*50}")
        print(f"  模型: d={d_model}, d_ff={d_ff}, heads={n_heads}, layers={n_layers}, "
              f"seq={seq_length}, dropout={dropout:.2f}, lr={learning_rate:.6f}, bs={batch_size}, epochs={epochs}")
        print(f"  策略: buy={buy_threshold:.4f}, sell={sell_threshold:.4f}, "
              f"sl={stop_loss_ratio:.3f}, tp={take_profit_ratio:.3f}, "
              f"pos={max_position_ratio:.2f}")
        
        t0 = time.time()
        
        # 评估
        metrics = evaluate_trial(feature_data, feature_cols, model_params, strategy_params)
        
        elapsed = time.time() - t0
        
        if metrics is None:
            raise optuna.TrialPruned()
        
        sharpe = metrics.get('sharpe_ratio', -999)
        annual_ret = metrics.get('annual_return', -999)
        max_dd = metrics.get('max_drawdown', 999)
        win_rate = metrics.get('win_rate', 0)
        
        print(f"\n  Trial {trial_num} 结果 ({elapsed:.0f}s):")
        print(f"    年化收益率: {annual_ret:.2%}")
        print(f"    最大回撤: {max_dd:.2%}")
        print(f"    夏普比率: {sharpe:.4f}")
        print(f"    胜率: {win_rate:.2%}")
        
        # 记录额外指标
        trial.set_user_attr('annual_return', annual_ret)
        trial.set_user_attr('max_drawdown', max_dd)
        trial.set_user_attr('win_rate', win_rate)
        trial.set_user_attr('elapsed_time', elapsed)
        
        # 过滤明显很差的结果（年化亏损超过10%直接剪枝）
        if annual_ret < -0.10:
            raise optuna.TrialPruned()
        
        # 综合目标: 最大化夏普比率（主要）+ 年化收益（次要）- 回撤惩罚
        # Optuna默认minimize，所以取负数
        score = sharpe * 0.5 + annual_ret * 0.3 - max_dd * 0.2
        return -score
    
    return objective


# ============================================================
# 结果汇总与保存
# ============================================================
def summarize_results(study, output_dir='output'):
    """
    汇总Optuna搜索结果
    """
    print("\n" + "=" * 60)
    print("  Optuna 超参数搜索结果汇总")
    print("=" * 60)
    
    best = study.best_trial
    
    print(f"\n  最佳Trial: #{best.number + 1}")
    print(f"  综合得分: {-best.value:.4f}")
    print(f"  年化收益: {best.user_attrs.get('annual_return', 'N/A')}")
    print(f"  最大回撤: {best.user_attrs.get('max_drawdown', 'N/A')}")
    print(f"  夏普比率: 由综合得分推算")
    print(f"  胜率: {best.user_attrs.get('win_rate', 'N/A')}")
    
    print(f"\n  最佳模型超参数:")
    model_keys = ['d_model', 'd_ff', 'd_ff_ratio', 'n_heads', 'n_layers', 'seq_length', 
                  'dropout', 'learning_rate', 'batch_size']
    for k in model_keys:
        if k in best.params:
            print(f"    {k}: {best.params[k]}")
    
    print(f"\n  最佳策略超参数:")
    strategy_keys = ['buy_threshold', 'sell_threshold', 'stop_loss_ratio',
                     'take_profit_ratio', 'max_position_ratio']
    for k in strategy_keys:
        if k in best.params:
            print(f"    {k}: {best.params[k]}")
    
    # 保存所有trial结果
    trials_data = []
    for t in study.trials:
        if t.state == optuna.trial.TrialState.COMPLETE:
            row = {'trial': t.number + 1, 'score': -t.value}
            row.update(t.params)
            row.update(t.user_attrs)
            trials_data.append(row)
    
    if trials_data:
        df = pd.DataFrame(trials_data)
        df = df.sort_values('score', ascending=False)
        save_path = os.path.join(output_dir, 'optuna_all_trials.csv')
        df.to_csv(save_path, index=False, encoding='utf-8-sig')
        print(f"\n  所有trial结果已保存: {save_path}")
        
        # 打印Top-5
        print(f"\n  Top-5 Trial:")
        print(f"  {'Trial':>6} | {'Score':>8} | {'年化收益':>10} | {'最大回撤':>10} | {'胜率':>8}")
        print(f"  {'-'*55}")
        for _, row in df.head(5).iterrows():
            ar = row.get('annual_return', 0)
            md = row.get('max_drawdown', 0)
            wr = row.get('win_rate', 0)
            print(f"  #{int(row['trial']):>5} | {row['score']:>8.4f} | "
                  f"{ar:>9.2%} | {md:>9.2%} | {wr:>7.2%}")
    
    # 保存最佳参数为JSON
    best_config = {
        'model': {k: best.params[k] for k in model_keys if k in best.params},
        'strategy': {k: best.params[k] for k in strategy_keys if k in best.params},
        'performance': {
            'score': -best.value,
            'annual_return': best.user_attrs.get('annual_return'),
            'max_drawdown': best.user_attrs.get('max_drawdown'),
            'win_rate': best.user_attrs.get('win_rate'),
        }
    }
    
    json_path = os.path.join(output_dir, 'optuna_best_params.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(best_config, f, indent=2, ensure_ascii=False)
    print(f"  最佳参数已保存: {json_path}")
    
    return best_config


# ============================================================
# 主入口
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='Optuna超参数搜索')
    parser.add_argument('--trials', type=int, default=30, 
                        help='搜索次数 (默认30)')
    parser.add_argument('--timeout', type=int, default=0,
                        help='超时时间秒 (默认0=不限)')
    parser.add_argument('--stocks', type=int, default=100,
                        help='使用前N只股票 (默认100, 与main.py一致)')
    args = parser.parse_args()
    
    print("=" * 60)
    print("  Optuna 超参数自动搜索")
    print("=" * 60)
    print(f"  搜索次数: {args.trials}")
    print(f"  训练轮数: 500 epochs (与main.py一致)")
    print(f"  股票数量: {args.stocks}")
    print(f"  超时: {'不限' if args.timeout == 0 else f'{args.timeout}s'}")
    print(f"  设备: {device}")
    
    # ---------- 股票池（100只，与main.py一致）----------
    all_stocks = [
        # ========== 消费板块（15只）==========
        '600519.SH', '000858.SZ', '000568.SZ', '600887.SH', '601888.SH',
        '000895.SZ', '603288.SH', '002304.SZ', '000661.SZ', '600809.SH',
        '603369.SH', '600132.SH', '002714.SZ', '600600.SH', '000876.SZ',
        # ========== 金融板块（15只）==========
        '601318.SH', '600036.SH', '000001.SZ', '601166.SH', '600030.SH',
        '601688.SH', '601398.SH', '601939.SH', '600016.SH', '601601.SH',
        '601628.SH', '600837.SH', '601211.SH', '000776.SZ', '601377.SH',
        # ========== 制造/科技板块（20只）==========
        '000333.SZ', '000651.SZ', '002415.SZ', '600031.SH', '600588.SH',
        '002230.SZ', '601012.SH', '300750.SZ', '002352.SZ', '601816.SH',
        '000725.SZ', '002371.SZ', '600690.SH', '000100.SZ', '601138.SH',
        '600406.SH', '300014.SZ', '002594.SZ', '601766.SH', '600104.SH',
        # ========== 医药板块（15只）==========
        '600276.SH', '000538.SZ', '300760.SZ', '603259.SH', '000963.SZ',
        '600196.SH', '002007.SZ', '300122.SZ', '600085.SH', '002001.SZ',
        '000423.SZ', '600436.SH', '300015.SZ', '002603.SZ', '300347.SZ',
        # ========== 能源/材料（15只）==========
        '600900.SH', '600309.SH', '601857.SH', '600028.SH', '601088.SH',
        '600585.SH', '002460.SZ', '600346.SH', '601899.SH', '600547.SH',
        '002466.SZ', '600188.SH', '601985.SH', '601225.SH', '600989.SH',
        # ========== 地产/基建/交运（10只）==========
        '600048.SH', '001979.SZ', '601668.SH', '601390.SH', '600115.SH',
        '601111.SH', '600029.SH', '601006.SH', '600009.SH', '601288.SH',
        # ========== 通信/传媒（10只）==========
        '600050.SH', '601728.SH', '000063.SZ', '002475.SZ', '600183.SH',
        '002236.SZ', '300413.SZ', '603501.SH', '002049.SZ', '600745.SH',
    ]
    
    stock_list = all_stocks[:args.stocks]
    
    # ---------- 准备数据 ----------
    feature_data, feature_cols = prepare_data(
        stock_list=stock_list,
        start_date='20180101',
        end_date='20251231'
    )
    
    # ---------- 创建Optuna Study（SQLite持久化，中断可恢复）----------
    db_path = os.path.join(OUTPUT_DIR, 'optuna_study.db')
    storage = f'sqlite:///{db_path}'
    sampler = TPESampler(seed=42, n_startup_trials=5)
    
    study = optuna.create_study(
        direction='minimize',  # 最小化负得分 = 最大化得分
        sampler=sampler,
        study_name='iTransformer_quant_strategy',
        storage=storage,
        load_if_exists=True,  # 如果已有study则恢复继续
    )
    
    # 显示已有进度
    completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    if completed > 0:
        print(f"\n  [恢复模式] 已有 {completed} 个已完成trial，继续搜索...")
        print(f"  当前最佳得分: {-study.best_value:.4f}")
    
    # ---------- 运行搜索 ----------
    objective = create_objective(feature_data, feature_cols)
    
    timeout = args.timeout if args.timeout > 0 else None
    
    print(f"\n{'='*60}")
    print(f"  开始搜索 ({args.trials} trials)...")
    print(f"{'='*60}")
    
    study.optimize(
        objective,
        n_trials=args.trials,
        timeout=timeout,
        gc_after_trial=True,  # 每次trial后GC释放显存
    )
    
    # ---------- 汇总结果 ----------
    best_config = summarize_results(study, OUTPUT_DIR)
    
    print(f"\n{'='*60}")
    print(f"  搜索完成！")
    print(f"  完成trial数: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
    print(f"  剪枝trial数: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
    print(f"{'='*60}")
    
    # ---------- 提示用户如何使用最佳参数 ----------
    print(f"\n  使用最佳参数重新运行完整实验:")
    print(f"    1. 查看: cat {OUTPUT_DIR}/optuna_best_params.json")
    print(f"    2. 更新 main.py 中的参数")
    print(f"    3. 运行: ./run.sh")


if __name__ == '__main__':
    main()
