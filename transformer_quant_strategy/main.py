# -*- coding: utf-8 -*-
"""
基于Transformer模型的A股量化交易策略设计与效果分析
========================================================
论文题目: 基于Transformer模型的A股量化交易策略设计与效果分析

主程序入口 - 完整运行数据获取、模型训练、策略回测

作者: [学生姓名]
学号: [学号]
日期: 2026年2月
"""

import warnings
warnings.filterwarnings('ignore')

import argparse

# 导入各模块
from data_acquisition import DataAcquisition
from feature_engineering import FeatureEngineering
from transformer_model import TransformerPredictor
from trading_strategy import TradingStrategy
from backtest_engine import BacktestEngine
from visualization import Visualization

import pandas as pd
import numpy as np
import os

# 创建输出目录
OUTPUT_DIR = 'output'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_hyperparameter_optimization(feature_data, feature_cols, n_trials=30):
    """运行Optuna超参数优化"""
    print("\n" + "=" * 50)
    print("[超参数优化模式] 使用Optuna寻找最佳超参数")
    print("=" * 50)
    
    from hyperparameter_tuning import HyperparameterTuner
    
    tuner = HyperparameterTuner(
        feature_data=feature_data,
        feature_cols=feature_cols,
        train_ratio=0.7,
        val_ratio=0.15
    )
    
    # 优化模型参数
    best_params = tuner.optimize_model_params(n_trials=n_trials, timeout=1200)
    
    # 保存最佳参数
    params_df = pd.DataFrame([best_params])
    params_df.to_csv(f'{OUTPUT_DIR}/best_hyperparameters.csv', index=False)
    print(f"\n最佳超参数已保存至 {OUTPUT_DIR}/best_hyperparameters.csv")
    
    return best_params

def main():
    """主函数：完整运行量化交易策略"""
    
    print("=" * 70)
    print("基于Transformer模型的A股量化交易策略设计与效果分析")
    print("=" * 70)
    
    # ============================================================
    # 第一部分：数据获取与清洗
    # ============================================================
    print("\n" + "=" * 50)
    print("[第一部分] 数据获取与清洗")
    print("=" * 50)
    
    # 选取沪深300成分股中的代表性股票（扩展到100只蓝筹股，全面覆盖A股核心资产）
    stock_list = [
        # ========== 消费板块（15只）==========
        '600519.SH',  # 贵州茅台
        '000858.SZ',  # 五粮液
        '000568.SZ',  # 泸州老窖
        '600887.SH',  # 伊利股份
        '601888.SH',  # 中国中免
        '000895.SZ',  # 双汇发展
        '603288.SH',  # 海天味业
        '002304.SZ',  # 洋河股份
        '000661.SZ',  # 长春高新
        '600809.SH',  # 山西汾酒
        '603369.SH',  # 今世缘
        '600132.SH',  # 重庆啤酒
        '002714.SZ',  # 牧原股份
        '600600.SH',  # 青岛啤酒
        '000876.SZ',  # 新希望
        # ========== 金融板块（15只）==========
        '601318.SH',  # 中国平安
        '600036.SH',  # 招商银行
        '000001.SZ',  # 平安银行
        '601166.SH',  # 兴业银行
        '600030.SH',  # 中信证券
        '601688.SH',  # 华泰证券
        '601398.SH',  # 工商银行
        '601939.SH',  # 建设银行
        '600016.SH',  # 民生银行
        '601601.SH',  # 中国太保
        '601628.SH',  # 中国人寿
        '600837.SH',  # 海通证券
        '601211.SH',  # 国泰君安
        '000776.SZ',  # 广发证券
        '601377.SH',  # 兴业证券
        # ========== 制造/科技板块（20只）==========
        '000333.SZ',  # 美的集团
        '000651.SZ',  # 格力电器
        '002415.SZ',  # 海康威视
        '600031.SH',  # 三一重工
        '600588.SH',  # 用友网络
        '002230.SZ',  # 科大讯飞
        '601012.SH',  # 隆基绿能
        '300750.SZ',  # 宁德时代
        '002352.SZ',  # 顺丰控股
        '601816.SH',  # 京沪高铁
        '000725.SZ',  # 京东方A
        '002371.SZ',  # 北方华创
        '600690.SH',  # 海尔智家
        '000100.SZ',  # TCL科技
        '601138.SH',  # 工业富联
        '600406.SH',  # 国电南瑞
        '300014.SZ',  # 亿纬锂能
        '002594.SZ',  # 比亚迪
        '601766.SH',  # 中国中车
        '600104.SH',  # 上汽集团
        # ========== 医药板块（15只）==========
        '600276.SH',  # 恒瑞医药
        '000538.SZ',  # 云南白药
        '300760.SZ',  # 迈瑞医疗
        '603259.SH',  # 药明康德
        '000963.SZ',  # 华东医药
        '600196.SH',  # 复星医药
        '002007.SZ',  # 华兰生物
        '300122.SZ',  # 智飞生物
        '600085.SH',  # 同仁堂
        '002001.SZ',  # 新和成
        '000423.SZ',  # 东阿阿胶
        '600436.SH',  # 片仔癀
        '300015.SZ',  # 爱尔眼科
        '002603.SZ',  # 以岭药业
        '300347.SZ',  # 泰格医药
        # ========== 能源/材料（15只）==========
        '600900.SH',  # 长江电力
        '600309.SH',  # 万华化学
        '601857.SH',  # 中国石油
        '600028.SH',  # 中国石化
        '601088.SH',  # 中国神华
        '600585.SH',  # 海螺水泥
        '002460.SZ',  # 赣锋锂业
        '600346.SH',  # 恒力石化
        '601899.SH',  # 紫金矿业
        '600547.SH',  # 山东黄金
        '002466.SZ',  # 天齐锂业
        '600188.SH',  # 兖矿能源
        '601985.SH',  # 中国核电
        '601225.SH',  # 陕西煤业
        '600989.SH',  # 宝丰能源
        # ========== 地产/基建/交运（10只）==========
        '600048.SH',  # 保利发展
        '001979.SZ',  # 招商蛇口
        '601668.SH',  # 中国建筑
        '601390.SH',  # 中国中铁
        '600115.SH',  # 中国东航
        '601111.SH',  # 中国国航
        '600029.SH',  # 南方航空
        '601006.SH',  # 大秦铁路
        '600009.SH',  # 上海机场
        '601288.SH',  # 农业银行
        # ========== 通信/传媒（10只）==========
        '600050.SH',  # 中国联通
        '601728.SH',  # 中国电信
        '000063.SZ',  # 中兴通讯
        '002475.SZ',  # 立讯精密
        '600183.SH',  # 生益科技
        '002236.SZ',  # 大华股份
        '300413.SZ',  # 芒果超媒
        '603501.SH',  # 韦尔股份
        '002049.SZ',  # 紫光国微
        '600745.SH',  # 闻泰科技
    ]
    
    data_module = DataAcquisition()
    
    # 获取数据 - 7年数据(2018-2025)，确保所有股票都有完整数据
    print("\n[1.1] 获取股票日度数据...")
    all_stock_data = data_module.fetch_stock_data(
        stock_list=stock_list,
        start_date='20180101',
        end_date='20251231'
    )
    
    # 数据清洗
    print("\n[1.2] 数据清洗...")
    cleaned_data = data_module.clean_data(all_stock_data)
    
    # 数据描述性统计
    print("\n[1.3] 数据描述性统计...")
    stats = data_module.descriptive_statistics(cleaned_data)
    stats.to_csv(f'{OUTPUT_DIR}/data_statistics.csv', index=True, encoding='utf-8-sig')
    print(f"  统计结果已保存至 {OUTPUT_DIR}/data_statistics.csv")
    
    # ============================================================
    # 第二部分：特征工程
    # ============================================================
    print("\n" + "=" * 50)
    print("[第二部分] 特征工程")
    print("=" * 50)
    
    fe_module = FeatureEngineering()
    
    # 计算技术指标
    print("\n[2.1] 计算技术指标...")
    feature_data = fe_module.compute_all_features(cleaned_data)
    
    # 特征说明
    print("\n[2.2] 特征列表:")
    feature_cols = fe_module.get_feature_columns()
    for i, col in enumerate(feature_cols, 1):
        print(f"  {i}. {col}")
    
    # 保存特征数据
    feature_data.to_csv(f'{OUTPUT_DIR}/feature_data.csv', index=False, encoding='utf-8-sig')
    print(f"\n  特征数据已保存至 {OUTPUT_DIR}/feature_data.csv")
    
    # ============================================================
    # 第三部分：iTransformer模型训练 (2024 ICLR)
    # ============================================================
    print("\n" + "=" * 50)
    print("[第三部分] iTransformer模型训练与预测")
    print("=" * 50)
    print("  核心创新: 在股票维度做Attention，学习多股票关联关系")
    print("    - 消费板块联动 (茅台-五粮液)")
    print("    - 金融板块协同 (平安-招行)")
    print("    - 跨板块领先滞后关系")
    
    transformer_module = TransformerPredictor(
        seq_length=60,        # 60个交易日 (约3个月)
        pred_length=5,        # 预测未来5个交易日
        d_model=2048,         # 大模型维度，充分利用A800 80G显存
        n_heads=16,           # 注意力头数 (head_dim=128)
        n_layers=10,          # 10层Transformer
        d_ff=8192,            # FFN隐层维度 = 4×d_model (标准比例)
        dropout=0.25,         # 略增dropout防过拟合
        epochs=500,           # 训练轮数
        learning_rate=0.0001, # 学习率
        batch_size=256,       # 大批量充分利用显存
        use_itransformer=True # 启用iTransformer多股票关联模式
    )
    
    # 训练模型
    print("\n[3.1] 训练Transformer模型...")
    model_results = transformer_module.train_and_predict(
        feature_data=feature_data,
        feature_cols=feature_cols,
        target_col='future_return_5d',  
        train_ratio=0.8  
    )
    
    # 模型评估
    print("\n[3.2] 模型评估指标:")
    print(f"  - MSE: {model_results['mse']:.6f}")
    print(f"  - MAE: {model_results['mae']:.6f}")
    print(f"  - 方向准确率: {model_results['direction_accuracy']:.2%}")
    
    # ============================================================
    # 第四部分：量化交易策略设计
    # ============================================================
    print("\n" + "=" * 50)
    print("[第四部分] 量化交易策略设计")
    print("=" * 50)
    
    strategy_module = TradingStrategy(
        initial_capital=1000000,    # 初始资金100万
        max_position_ratio=0.20,    # 单只股票最大仓位20%（更分散以降低风险）
        stop_loss_ratio=0.04,       # 止损4%（更严格快速止损）
        take_profit_ratio=0.06,     # 止盈6%（早点获利了结）
        use_kelly=True,             # 使用凯利公式
        buy_threshold=0.008,        # 买入阈值0.8%（更多交易机会）
        sell_threshold=-0.005       # 卖出阈值-0.5%
    )
    
    # 生成交易信号
    print("\n[4.1] 生成交易信号...")
    signals = strategy_module.generate_signals(
        predictions=model_results['predictions'],
        feature_data=feature_data
    )
    
    # ============================================================
    # 第五部分：策略回测
    # ============================================================
    print("\n" + "=" * 50)
    print("[第五部分] 策略回测与绩效评估")
    print("=" * 50)
    
    backtest_module = BacktestEngine(initial_capital=1000000)
    
    # 运行回测
    print("\n[5.1] 运行策略回测...")
    backtest_results = backtest_module.run_backtest(
        signals=signals,
        price_data=feature_data
    )
    
    # 计算绩效指标
    print("\n[5.2] 绩效指标计算...")
    metrics = backtest_module.calculate_metrics(backtest_results)
    
    print("\n" + "-" * 40)
    print("本文策略绩效指标:")
    print("-" * 40)
    print(f"  年化收益率: {metrics['annual_return']:.2%}")
    print(f"  最大回撤: {metrics['max_drawdown']:.2%}")
    print(f"  夏普比率: {metrics['sharpe_ratio']:.2f}")
    print(f"  胜率: {metrics['win_rate']:.2%}")
    print(f"  盈亏比: {metrics['profit_loss_ratio']:.2f}")
    print(f"  总交易次数: {metrics['total_trades']}")
    
    # 对比策略
    print("\n[5.3] 对比策略回测...")
    
    # 买入持有策略
    bh_results = backtest_module.buy_and_hold(feature_data)
    bh_metrics = backtest_module.calculate_metrics(bh_results)
    
    # 传统均线策略
    ma_results = backtest_module.ma_strategy(feature_data)
    ma_metrics = backtest_module.calculate_metrics(ma_results)
    
    # 汇总对比
    comparison_df = pd.DataFrame({
        '策略': ['本文Transformer策略', '传统均线策略', '买入持有策略'],
        '年化收益率(%)': [
            metrics['annual_return'] * 100,
            ma_metrics['annual_return'] * 100,
            bh_metrics['annual_return'] * 100
        ],
        '最大回撤(%)': [
            metrics['max_drawdown'] * 100,
            ma_metrics['max_drawdown'] * 100,
            bh_metrics['max_drawdown'] * 100
        ],
        '夏普比率': [
            metrics['sharpe_ratio'],
            ma_metrics['sharpe_ratio'],
            bh_metrics['sharpe_ratio']
        ],
        '胜率(%)': [
            metrics['win_rate'] * 100,
            ma_metrics['win_rate'] * 100,
            bh_metrics['win_rate'] * 100
        ]
    })
    
    print("\n策略对比表:")
    print(comparison_df.to_string(index=False))
    comparison_df.to_csv(f'{OUTPUT_DIR}/strategy_comparison.csv', index=False, encoding='utf-8-sig')
    
    # ============================================================
    # 第六部分：可视化
    # ============================================================
    print("\n" + "=" * 50)
    print("[第六部分] 结果可视化")
    print("=" * 50)
    
    viz_module = Visualization(output_dir=OUTPUT_DIR)
    
    # 绘制累计收益曲线
    print("\n[6.1] 绘制累计收益曲线...")
    viz_module.plot_equity_curves(
        results_dict={
            '本文Transformer策略': backtest_results,
            '传统均线策略': ma_results,
            '买入持有策略': bh_results
        }
    )
    
    # 绘制预测vs实际对比图
    print("[6.2] 绘制预测效果图...")
    viz_module.plot_prediction_comparison(model_results)
    
    # 绘制交易信号图
    print("[6.3] 绘制交易信号图...")
    viz_module.plot_trading_signals(signals, feature_data)
    
    # 绘制回撤曲线
    print("[6.4] 绘制回撤曲线...")
    viz_module.plot_drawdown(backtest_results)
    
    # 绘制特征重要性（如果有）
    print("[6.5] 绘制模型注意力权重...")
    viz_module.plot_attention_weights(model_results)
    
    # ============================================================
    # 保存结果
    # ============================================================
    print("\n" + "=" * 50)
    print("[完成] 结果已保存")
    print("=" * 50)
    
    # 保存交易记录
    if 'trade_log' in backtest_results:
        backtest_results['trade_log'].to_csv(
            f'{OUTPUT_DIR}/trade_log.csv', 
            index=False, 
            encoding='utf-8-sig'
        )
        print(f"  交易记录: {OUTPUT_DIR}/trade_log.csv")
    
    # 保存绩效指标
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(f'{OUTPUT_DIR}/performance_metrics.csv', index=False, encoding='utf-8-sig')
    print(f"  绩效指标: {OUTPUT_DIR}/performance_metrics.csv")
    
    print(f"\n所有图表已保存至 {OUTPUT_DIR}/ 目录")
    
    print("\n" + "=" * 70)
    print("程序运行完成！")
    print("=" * 70)
    
    return {
        'metrics': metrics,
        'comparison': comparison_df,
        'model_results': model_results
    }


if __name__ == '__main__':
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Transformer量化交易策略')
    parser.add_argument('--optimize', action='store_true', help='运行Optuna超参数优化')
    parser.add_argument('--trials', type=int, default=30, help='Optuna试验次数')
    args = parser.parse_args()
    
    if args.optimize:
        # 先运行数据获取和特征工程，然后进行超参数优化
        print("=" * 70)
        print("运行超参数优化模式")
        print("=" * 70)
        
        # 数据获取
        stock_list = [
            '600519.SH', '000858.SZ', '601318.SH', '600036.SH', '000333.SZ',
        ]
        data_module = DataAcquisition()
        all_stock_data = data_module.fetch_stock_data(
            stock_list=stock_list,
            start_date='20200101',
            end_date='20251231'
        )
        cleaned_data = data_module.clean_data(all_stock_data)
        
        # 特征工程
        fe_module = FeatureEngineering()
        feature_data = fe_module.compute_all_features(cleaned_data)
        feature_cols = fe_module.get_feature_columns()
        
        # 超参数优化
        best_params = run_hyperparameter_optimization(feature_data, feature_cols, args.trials)
        
        print("\n使用最佳参数重新运行完整实验...")
        print("请使用以下参数更新main.py并重新运行:")
        for k, v in best_params.items():
            print(f"  {k}: {v}")
    else:
        results = main()
