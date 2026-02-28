# -*- coding: utf-8 -*-
"""
LSTM Baseline for A-share Quant Strategy
=======================================
- Uses an independent SQLite database under baselines/
- Reuses data cleaning, feature engineering, strategy, and backtest modules
- Trains LSTM on the same feature set and split protocol
"""

import os
import sys
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TQ_ROOT = PROJECT_ROOT / "transformer_quant_strategy"
sys.path.append(str(TQ_ROOT))

from data_acquisition import DataAcquisition
from feature_engineering import FeatureEngineering
from trading_strategy import TradingStrategy
from backtest_engine import BacktestEngine


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output_lstm"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = BASE_DIR / "stock_data_lstm.db"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


STOCK_LIST = [
    '600519.SH', '000858.SZ', '000568.SZ', '600887.SH', '601888.SH', '000895.SZ', '603288.SH',
    '002304.SZ', '000661.SZ', '600809.SH', '603369.SH', '600132.SH', '002714.SZ', '600600.SH', '000876.SZ',
    '601318.SH', '600036.SH', '000001.SZ', '601166.SH', '600030.SH', '601688.SH', '601398.SH', '601939.SH',
    '600016.SH', '601601.SH', '601628.SH', '600837.SH', '601211.SH', '000776.SZ', '601377.SH',
    '000333.SZ', '000651.SZ', '002415.SZ', '600031.SH', '600588.SH', '002230.SZ', '601012.SH', '002074.SZ',
    '002352.SZ', '601100.SH', '000725.SZ', '002371.SZ', '600690.SH', '000100.SZ', '600406.SH', '300014.SZ',
    '002594.SZ', '601766.SH', '600104.SH',
    '600276.SH', '000538.SZ', '601607.SH', '000963.SZ', '600196.SH', '002007.SZ', '300122.SZ', '600085.SH',
    '002001.SZ', '000423.SZ', '600436.SH', '300015.SZ', '002603.SZ', '300347.SZ',
    '600900.SH', '600309.SH', '601857.SH', '600028.SH', '601088.SH', '600585.SH', '002460.SZ', '600346.SH',
    '601899.SH', '600547.SH', '002466.SZ', '600188.SH', '601985.SH', '601225.SH', '601898.SH',
    '600048.SH', '001979.SZ', '601668.SH', '601390.SH', '600115.SH', '601111.SH', '600029.SH', '601006.SH',
    '600009.SH', '601288.SH',
    '600050.SH', '600522.SH', '000063.SZ', '002475.SZ', '600183.SH', '002236.SZ', '300413.SZ', '002049.SZ',
    '600745.SH'
]


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class LSTMRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last).squeeze(-1)


def build_sequences(
    feature_data: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    seq_length: int,
    train_end_date: pd.Timestamp,
    val_end_date: pd.Timestamp,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list, list]:
    x_train, y_train = [], []
    x_val, y_val = [], []
    x_test, y_test = [], []
    meta_val, meta_test = [], []

    for stock_code in sorted(feature_data['stock_code'].unique()):
        stock_df = feature_data[feature_data['stock_code'] == stock_code].sort_values('date').reset_index(drop=True)
        if len(stock_df) <= seq_length:
            continue

        x_raw = stock_df[feature_cols].values
        y_raw = stock_df[target_col].values
        dates = pd.to_datetime(stock_df['date']).values

        for idx in range(len(stock_df) - seq_length):
            x_seq = x_raw[idx: idx + seq_length]
            y_target = y_raw[idx + seq_length]
            target_date = pd.to_datetime(dates[idx + seq_length])

            if target_date <= train_end_date:
                x_train.append(x_seq)
                y_train.append(y_target)
            elif target_date <= val_end_date:
                x_val.append(x_seq)
                y_val.append(y_target)
                meta_val.append((target_date, stock_code))
            else:
                x_test.append(x_seq)
                y_test.append(y_target)
                meta_test.append((target_date, stock_code))

    return (
        np.array(x_train), np.array(y_train),
        np.array(x_val), np.array(y_val),
        np.array(x_test), np.array(y_test),
        meta_val, meta_test,
    )


def zscore_by_date(pred_df: pd.DataFrame) -> pd.DataFrame:
    df = pred_df.copy()

    def _z(g: pd.DataFrame) -> pd.DataFrame:
        std = g['predicted'].std()
        mean = g['predicted'].mean()
        if std is None or std < 1e-12:
            g['predicted'] = 0.0
        else:
            g['predicted'] = (g['predicted'] - mean) / std
        return g

    return df.groupby('date', group_keys=False).apply(_z)


def apply_train_only_preprocessing(
    feature_data: pd.DataFrame,
    feature_cols: List[str],
    train_end_date: pd.Timestamp,
) -> pd.DataFrame:
    """
    严格防泄露预处理：
    1) 仅使用训练集统计量处理 NaN/Inf
    2) 不使用验证/测试集信息参与统计
    """
    df = feature_data.copy()
    df['date'] = pd.to_datetime(df['date'])

    train_mask = df['date'] <= train_end_date
    train_df = df.loc[train_mask]

    # 仅对模型输入特征做处理，目标列不做未来信息相关变换
    for col in feature_cols:
        if col not in df.columns:
            continue

        train_series = train_df[col].replace([np.inf, -np.inf], np.nan)
        train_median = train_series.median()
        if pd.isna(train_median):
            train_median = 0.0

        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        df[col] = df[col].fillna(train_median)

    return df


def main():
    set_seed(42)
    print("=" * 70)
    print("LSTM Baseline (独立数据库版)")
    print("=" * 70)
    print(f"[数据库] {DB_PATH}")
    print(f"[设备] {DEVICE}")

    da = DataAcquisition(db_path=str(DB_PATH))
    raw_data = da.fetch_stock_data(
        stock_list=STOCK_LIST,
        start_date='20160101',
        end_date='20251231',
        force_refresh=False,
    )
    clean_data = da.clean_data(raw_data)

    fe = FeatureEngineering()
    feature_data = fe.compute_all_features(clean_data)
    feature_cols = fe.get_feature_columns()

    all_dates = sorted(pd.to_datetime(feature_data['date'].unique()))
    train_idx = int(len(all_dates) * 0.70)
    val_idx = int(len(all_dates) * 0.85)
    train_end_date = all_dates[train_idx - 1]
    val_end_date = all_dates[val_idx - 1]

    # 与 iTransformer 一致的严格防泄露口径：
    # 预处理统计量仅来自训练集
    feature_data = apply_train_only_preprocessing(
        feature_data=feature_data,
        feature_cols=feature_cols,
        train_end_date=train_end_date,
    )

    seq_length = 60
    target_col = 'future_return_5d'

    (
        x_train, y_train,
        x_val, y_val,
        x_test, y_test,
        meta_val, meta_test,
    ) = build_sequences(
        feature_data=feature_data,
        feature_cols=feature_cols,
        target_col=target_col,
        seq_length=seq_length,
        train_end_date=train_end_date,
        val_end_date=val_end_date,
    )

    if len(x_train) == 0 or len(x_test) == 0:
        raise RuntimeError("样本构建失败，请检查数据区间与特征可用性")

    scaler = StandardScaler()
    x_train_2d = x_train.reshape(-1, x_train.shape[-1])
    scaler.fit(x_train_2d)

    def transform_x(x: np.ndarray) -> np.ndarray:
        x2d = x.reshape(-1, x.shape[-1])
        x2d = scaler.transform(x2d)
        return x2d.reshape(x.shape)

    x_train = transform_x(x_train)
    x_val = transform_x(x_val)
    x_test = transform_x(x_test)

    train_loader = DataLoader(
        TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)),
        batch_size=512,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(x_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32)),
        batch_size=1024,
        shuffle=False,
    )

    model = LSTMRegressor(input_dim=x_train.shape[-1], hidden_dim=128, num_layers=2, dropout=0.2).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.02)
    criterion = nn.MSELoss()

    best_val = float('inf')
    best_state = None
    patience, patience_counter = 10, 0

    for epoch in range(1, 81):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                pred = model(xb)
                val_losses.append(criterion(pred, yb).item())

        train_loss = float(np.mean(train_losses)) if train_losses else 0.0
        val_loss = float(np.mean(val_losses)) if val_losses else 0.0
        print(f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        test_pred = model(torch.tensor(x_test, dtype=torch.float32, device=DEVICE)).cpu().numpy()

    pred_df = pd.DataFrame({
        'date': [m[0] for m in meta_test],
        'stock_code': [m[1] for m in meta_test],
        'predicted': test_pred,
        'actual': y_test,
    })
    pred_df['date'] = pd.to_datetime(pred_df['date'])

    mse = float(np.mean((pred_df['predicted'] - pred_df['actual']) ** 2))
    mae = float(np.mean(np.abs(pred_df['predicted'] - pred_df['actual'])))
    direction_acc = float((np.sign(pred_df['predicted']) == np.sign(pred_df['actual'])).mean())

    print("\n[LSTM测试集预测指标]")
    print(f"MSE: {mse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"Direction Accuracy: {direction_acc:.2%}")

    pred_df_for_strategy = zscore_by_date(pred_df[['date', 'stock_code', 'predicted']])

    strategy = TradingStrategy(
        initial_capital=1000000,
        max_position_ratio=0.373,
        stop_loss_ratio=0.027,
        take_profit_ratio=0.14,
        use_kelly=True,
        buy_threshold=1.258,
        sell_threshold=-0.895,
    )

    signals = strategy.generate_signals(predictions=pred_df_for_strategy, feature_data=feature_data)

    backtest = BacktestEngine(initial_capital=1000000)
    backtest_results = backtest.run_backtest(signals=signals, price_data=feature_data)
    perf = backtest.calculate_metrics(backtest_results)

    print("\n[LSTM策略回测指标]")
    for k, v in perf.items():
        if isinstance(v, float):
            print(f"{k}: {v:.6f}")
        else:
            print(f"{k}: {v}")

    pred_df.to_csv(OUTPUT_DIR / "lstm_predictions_test.csv", index=False)
    signals.to_csv(OUTPUT_DIR / "lstm_signals.csv", index=False)
    backtest_results['equity_curve'].to_csv(OUTPUT_DIR / "lstm_equity_curve.csv", index=False)
    if not backtest_results['trade_log'].empty:
        backtest_results['trade_log'].to_csv(OUTPUT_DIR / "lstm_trade_log.csv", index=False)

    metrics_df = pd.DataFrame([
        {
            'model': 'LSTM',
            'mse': mse,
            'mae': mae,
            'direction_accuracy': direction_acc,
            **perf,
        }
    ])
    metrics_df.to_csv(OUTPUT_DIR / "lstm_performance_metrics.csv", index=False)

    print(f"\n结果已保存到: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
