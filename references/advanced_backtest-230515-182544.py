import datetime
import sys

import yfinance as yf
import pandas as pd

pd.set_option('display.max_rows',None)
pd.set_option('display.max_columns',None)
pd.set_option('display.width',None)

def backtest(df, initial_capital, candle_len, candle_direction,
             profit_target, stop_loss, holding_day,
             sma_direction, sma_len, std_ratio_threshold):

    df['sma'] = df['Close'].rolling(sma_len).mean()
    df['std'] = df['Close'].rolling(sma_len).std()
    df['std_raito'] = (df['sma'] - df['Close']) / df['std']

    ##### initialization #####
    open_date    = datetime.datetime.now().date()
    open_price   = 0
    num_of_share = 0
    net_profit   = 0
    num_of_trade = 0

    lot_size     = 100
    last_realized_capital = initial_capital

    equity_value = 0
    realized_pnl   = 0
    unrealized_pnl = 0

    commission = 0
    commission_rate = 0.03 * 0.01
    min_commission  = 3
    platform_fee    = 15

    for i, row in df.iterrows():
        now_date  = i.date()
        now_open  = row['Open']
        now_high  = row['High']
        now_low   = row['Low']
        now_close = row['Close']

        now_candle    = round(now_close - now_open,2)
        now_sma       = row['sma']
        now_std_raito = row['std_raito']

        ##### commission #####
        if num_of_share > 0:
            commission = (now_close * num_of_share) * commission_rate
            if commission < min_commission: commission = min_commission
            commission += platform_fee
            commission = 2 * commission
        else:
            commission = 0

        ##### equity value #####
        unrealized_pnl = num_of_share * (now_close - open_price) - commission
        equity_value   = last_realized_capital + unrealized_pnl
        net_profit     = round(equity_value - initial_capital,2)

        if candle_direction == 'positive':
            trade_logic        = now_candle > candle_len
        elif candle_direction == 'negative':
            trade_logic        = now_candle < -1 * candle_len

        if sma_direction == 'above':
            trade_logic = trade_logic and (now_close > now_sma) and now_std_raito < -1 * std_ratio_threshold
        elif sma_direction == 'below':
            trade_logic = trade_logic and (now_close < now_sma) and now_std_raito > std_ratio_threshold

        close_logic        = (now_date - open_date).days >= holding_day
        profit_target_cond = now_close - open_price > profit_target
        stop_loss_cond     = open_price - now_close > stop_loss
        last_index_cond    = i == df.index[-1]
        min_cost_cond      = last_realized_capital > now_close * lot_size

        ##### open position #####
        if num_of_share == 0 and not last_index_cond and min_cost_cond and trade_logic:

            num_of_lot = last_realized_capital // (lot_size * now_close)
            num_of_share = num_of_lot * lot_size

            open_price = now_close
            open_date  = now_date

        ##### close position #####
        elif num_of_share > 0 and (profit_target_cond or stop_loss_cond or last_index_cond or close_logic):

            realized_pnl = unrealized_pnl
            last_realized_capital += realized_pnl

            num_of_trade += 1

            num_of_share = 0

    return net_profit, num_of_trade

if __name__ == '__main__':

    initial_capital = 200000

    ticker = yf.Ticker('0388.HK')
    df = ticker.history(start='2022-01-01', end='2022-12-31')
    df = df[['Open','High','Low','Close','Volume']]
    df = df[df['Volume'] > 0]

    candle_direction_list    = ['positive', 'negative']
    candle_len_list          = [5, 10, 15]
    sma_len_list             = [10, 20, 50]
    sma_direction_list       = ['above', 'below', 'whatever']
    std_ratio_threshold_list = [0.5, 1]
    profit_target_list       = [4, 8]
    stop_loss_list           = [20, 25]
    holding_day_list         = [3, 5, 10]

    result_dict = {}
    result_dict['net_profit'] = []
    result_dict['num_of_trade'] = []
    result_dict['std_ratio_threshold'] = []
    result_dict['sma_len'] = []
    result_dict['sma_direction'] = []
    result_dict['candle_direction'] = []
    result_dict['holding_day'] = []
    result_dict['stop_loss'] = []
    result_dict['profit_target'] = []
    result_dict['candle_len'] = []

    for std_ratio_threshold in std_ratio_threshold_list:
        for sma_len in sma_len_list:
            for sma_direction in sma_direction_list:
                for candle_direction in candle_direction_list:
                    for holding_day in holding_day_list:
                        for stop_loss in stop_loss_list:
                            for profit_target in profit_target_list:
                                for candle_len in candle_len_list:
                                    net_profit, num_of_trade = backtest(df, initial_capital,
                                             candle_len, candle_direction,
                                             profit_target, stop_loss, holding_day,
                                             sma_direction, sma_len,
                                             std_ratio_threshold)
                                    print('net_profit:', net_profit)
                                    print('num_of_trade:', num_of_trade)
                                    print('std_ratio_threshold:', std_ratio_threshold)
                                    print('sma_len:', sma_len)
                                    print('candle_len:', candle_len)
                                    print('sma_direction:', sma_direction)
                                    print('candle_direction:', candle_direction)
                                    print('holding_day:', holding_day)
                                    print('profit_target:', profit_target)
                                    print('stop_loss:', stop_loss)
                                    print('---------------------------')
                                    result_dict['net_profit'].append(net_profit)
                                    result_dict['num_of_trade'].append(num_of_trade)
                                    result_dict['std_ratio_threshold'].append(std_ratio_threshold)
                                    result_dict['sma_len'].append(sma_len)
                                    result_dict['sma_direction'].append(sma_direction)
                                    result_dict['candle_direction'].append(candle_direction)
                                    result_dict['holding_day'].append(holding_day)
                                    result_dict['stop_loss'].append(stop_loss)
                                    result_dict['profit_target'].append(profit_target)
                                    result_dict['candle_len'].append(candle_len)

    result_df = pd.DataFrame(result_dict)
    result_df = result_df.sort_values(by='net_profit', ascending=False)
    print(result_df)
