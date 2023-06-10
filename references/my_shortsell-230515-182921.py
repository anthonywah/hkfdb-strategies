import time
import datetime

import os
import sys
import multiprocessing as mp

import hkfdb
import yfinance as yf

import pandas as pd
import numpy as np

import plotguy
import itertools

import pandas_ta as ta

pd.set_option('display.max_rows',None)
pd.set_option('display.max_columns',None)
pd.set_option('display.width',None)

client = hkfdb.Database('test_token_123')

data_folder = 'data'
secondary_data_folder = 'secondary_data'
backtest_output_folder = 'backtest_output'
signal_output_folder = 'signal_output'

if not os.path.isdir(data_folder): os.mkdir(data_folder)
if not os.path.isdir(secondary_data_folder): os.mkdir(secondary_data_folder)
if not os.path.isdir(backtest_output_folder): os.mkdir(backtest_output_folder)
if not os.path.isdir(signal_output_folder): os.mkdir(signal_output_folder)

py_filename = os.path.basename(__file__).replace('.py','')

def backtest(para_combination):

    para_dict       = para_combination['para_dict']
    sec_profile     = para_combination['sec_profile']
    start_date      = para_combination['start_date']
    end_date        = para_combination['end_date']
    reference_index = para_combination['reference_index']
    freq            = para_combination['freq']
    file_format     = para_combination['file_format']
    df              = para_combination['df']
    intraday        = para_combination['intraday']
    output_folder   = para_combination['output_folder']
    data_folder     = para_combination['data_folder']
    run_mode        = para_combination['run_mode']
    summary_mode    = para_combination['summary_mode']
    py_filename     = para_combination['py_filename']

    ##### stra specific #####
    code                = para_combination['code']
    profit_target       = para_combination['profit_target']
    stop_loss           = para_combination['stop_loss']
    holding_day         = para_combination['holding_day']

    ssr_len                = para_combination['ssr_len']
    ssr_zscore_threshold   = para_combination['ssr_zscore_threshold']
    ssr_zscore_sign        = para_combination['ssr_zscore_sign']
    zscore_cross_direction = para_combination['zscore_cross_direction']

    ##### sec_profile #####

    market          = sec_profile['market']
    sectype         = sec_profile['sectype']
    initial_capital = sec_profile['initial_capital']
    lot_size_dict   = sec_profile['lot_size_dict']
    lot_size        = lot_size_dict[code]

    if sectype == 'STK':
        if market == 'HK':
            commission_rate = sec_profile['commission_rate']
            min_commission  = sec_profile['min_commission']
            platform_fee    = sec_profile['platform_fee']
        if market == 'US':
            commission_each_stock   = sec_profile['commission_each_stock']
            min_commission          = sec_profile['min_commission']
            platform_fee_each_stock = sec_profile['platform_fee_each_stock']
            min_platform_fee        = sec_profile['min_platform_fee']


    ##### stra specific #####
    df['ssr_avg'] = df['real_ssr'].rolling(ssr_len).mean()
    df['ssr_std'] = df['real_ssr'].rolling(ssr_len).std()
    df['ssr_zscore'] = (df['real_ssr'] - df['ssr_avg']) / df['ssr_std']
    df['ssr_zscore-1'] = df['ssr_zscore'].shift(1)

    if ssr_zscore_sign == 'negative': ssr_zscore_threshold = -1 * ssr_zscore_threshold

    if zscore_cross_direction == 'cross_over':
        df['trade_logic'] = (df['ssr_zscore'] > ssr_zscore_threshold) & (df['ssr_zscore-1'] <= ssr_zscore_threshold)
    elif zscore_cross_direction == 'cross_under':
        df['trade_logic'] = (df['ssr_zscore'] < ssr_zscore_threshold) & (df['ssr_zscore-1'] >= ssr_zscore_threshold)


    ##### initialization #####

    df['action'] = ''
    df['num_of_share'] = 0

    df['open_price'] = np.NaN
    df['close_price'] = np.NaN

    df['realized_pnl'] = np.NaN
    df['unrealized_pnl'] = 0
    df['net_profit'] = 0

    df['equity_value'] = initial_capital
    df['mdd_dollar'] = 0
    df['mdd_pct'] = 0

    df['commission'] = 0
    df['logic'] = None

    open_date    = datetime.datetime.now().date()
    open_price   = 0
    num_of_share = 0
    net_profit   = 0
    num_of_trade = 0

    last_realized_capital = initial_capital

    equity_value = 0
    realized_pnl   = 0
    unrealized_pnl = 0

    commission = 0

    for i, row in df.iterrows():
        now_date  = i.date()
        now_open  = row['open']
        now_high  = row['high']
        now_low   = row['low']
        now_close = row['close']

        ##### stra specific #####
        trade_logic = row['trade_logic']

        ##### commission #####
        if sectype == 'STK':
            if market == 'HK':
                if num_of_share > 0:
                    commission = (now_close * num_of_share) * commission_rate
                    if commission < min_commission: commission = min_commission
                    commission += platform_fee
                    commission = 2 * commission
                else:
                    commission = 0
            elif market == 'US':
                if num_of_share > 0:
                    commission = num_of_share * commission_each_stock
                    if commission < min_commission: commission = min_commission
                    platform_fee = num_of_share * platform_fee_each_stock
                    if platform_fee < min_platform_fee: platform_fee = min_platform_fee
                    commission += platform_fee
                    commission = 2 * commission
                else:
                    commission = 0

        ##### equity value #####
        unrealized_pnl = num_of_share * (now_close - open_price) - commission
        equity_value   = last_realized_capital + unrealized_pnl
        net_profit     = round(equity_value - initial_capital,2)

        if trade_logic: df.at[i, 'logic'] = 'trade_logic'

        if run_mode == 'backtest':

            close_logic        = num_of_share != 0 and (now_date - open_date).days >= holding_day
            profit_target_cond = num_of_share != 0 and now_close - open_price > profit_target * open_price * 0.01
            stop_loss_cond     = num_of_share != 0 and open_price - now_close > stop_loss * open_price * 0.01
            last_index_cond    = i == df.index[-1]
            min_cost_cond      = last_realized_capital > now_close * lot_size

            ##### open position #####
            if num_of_share == 0 and not last_index_cond and min_cost_cond and trade_logic:

                num_of_lot = last_realized_capital // (lot_size * now_close)
                num_of_share = num_of_lot * lot_size

                open_price = now_close
                open_date  = now_date

                df.at[i, 'action'] = 'open'
                df.at[i, 'open_price'] = open_price

            ##### close position #####
            elif num_of_share > 0 and (profit_target_cond or stop_loss_cond or last_index_cond or close_logic):

                realized_pnl = unrealized_pnl
                unrealized_pnl = 0
                last_realized_capital += realized_pnl

                num_of_trade += 1

                num_of_share = 0

                if close_logic: df.at[i, 'logic'] = 'close_logic'

                if last_index_cond: df.at[i, 'action'] = 'last_index'
                if close_logic: df.at[i, 'action'] = 'close_logic'
                if profit_target_cond: df.at[i, 'action'] = 'profit_target'
                if stop_loss_cond: df.at[i, 'action'] = 'stop_loss'

                df.at[i, 'close_price']  = now_close
                df.at[i, 'realized_pnl'] = realized_pnl
                df.at[i, 'commission']   = commission

        ### record at last ###
        df.at[i, 'equity_value'] = equity_value
        df.at[i, 'num_of_share'] = num_of_share
        df.at[i, 'unrealized_pnl'] = unrealized_pnl
        df.at[i, 'net_profit'] = net_profit

    if summary_mode and run_mode == 'backtest':
        df = df[df['action'] != '']

    save_path = plotguy.generate_filepath(para_combination)
    print(save_path)

    if file_format == 'csv':
        df.to_csv(save_path)
    elif file_format == 'parquet':
        df.to_parquet(save_path)


def get_hist_data(code_list, start_date, end_date, freq,
                  data_folder, file_format, update_data,
                  market):

    start_date_int = int(start_date.replace('-',''))
    end_date_int   = int(end_date.replace('-',''))

    df_dict ={}
    for code in code_list:

        file_path = os.path.join(data_folder, code + '_' + freq + '.' + file_format)

        if os.path.isfile(file_path) and not update_data:
            if file_format == 'csv':
                df = pd.read_csv(file_path)
                df['datetime'] = pd.to_datetime(df['datetime'], format='%Y-%m-%d')
                df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
                df = df.set_index('datetime')
            elif file_format == 'parquet':
                df = pd.read_parquet(file_path)
                df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')

            print(datetime.datetime.now(), 'successfully read data', code)
        else:
            if market == 'HK':
                df = client.get_hk_stock_ohlc(code, start_date_int, end_date_int, freq, price_adj=True, vol_adj=True)
                df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')

            elif market == 'US':
                ticker = yf.Ticker(code)
                df = ticker.history(start=start_date, end=end_date)
                df = df[['Open','High','Low','Close','Volume']]
                df = df[df['Volume'] > 0]
                df.columns = map(str.lower, df.columns)
                df = df.rename_axis('datetime')
                df['date'] = df.index.date
                df = df[['date', 'open', 'high', 'low', 'close', 'volume']]

            time.sleep(1)
            if file_format == 'csv':
                df.to_csv(file_path)
            elif file_format == 'parquet':
                df.to_parquet(file_path)
            print(datetime.datetime.now(), 'successfully get data from data source', code)

        df['pct_change'] = df['close'].pct_change()

        df_dict[code] = df

    return df_dict

def get_secondary_data(df_dict, secondary_data_folder, start_date, end_date, update_data):
    start_date_int = int(start_date.replace('-',''))
    end_date_int = int(end_date.replace('-',''))

    for code, df in df_dict.items():

        file_path = os.path.join(secondary_data_folder, code + '_shortsell.parquet')

        if os.path.isfile(file_path) and not update_data:
            df2 = pd.read_parquet(file_path)
            print(datetime.datetime.now(), 'successfully read shortsell data', code)
        else:
            df2 = client.get_hk_shortsell_by_code(code)
            time.sleep(1)
            df2.to_parquet(file_path)
            print(datetime.datetime.now(), 'successfully get shortsell data', code)

        df2 = df2[['ss_dollars']]
        df2['ss_dollars'] = df2['ss_dollars'].shift(1)

        file_path = os.path.join(secondary_data_folder, code + 'marketcap.parquet')

        if os.path.isfile(file_path) and not update_data:
            df3 = pd.read_parquet(file_path)
            print(datetime.datetime.now(), 'successfully read marketcap data', code)
        else:
            df3 = client.get_hk_market_cap_hist_by_code(code)
            time.sleep(1)
            df3.to_parquet(file_path)
            print(datetime.datetime.now(), 'successfully get marketcap data', code)

        df3 = df3[['market_cap_mil']]

        df = pd.concat([df, df2, df3], axis=1)

        df = df[df.index >= start_date]
        df = df[df.index <= end_date]

        df = df[df['close'].notna()]

        ### rear_ssr in pct
        df['real_ssr'] = df['ss_dollars'] / (df['market_cap_mil'] * 10000)

        df_dict[code] = df

    return df_dict

def get_sec_profile(code_list, market, sectype, initial_capital):

    sec_profile = {}
    lot_size_dict = {}

    if market == 'HK':
        if sectype == 'STK':
            info = client.get_basic_hk_stock_info()
            for code in code_list:
                lot_size = int(info[info['code'] == code]['lot_size'])
                lot_size_dict[code] = lot_size
        else:
            for code in code_list:
                lot_size_dict[code] = 1

    elif market == 'US':
        for code in code_list:
            lot_size_dict[code] = 1

    sec_profile['market'] = market
    sec_profile['sectype'] = sectype
    sec_profile['initial_capital'] = initial_capital
    sec_profile['lot_size_dict'] = lot_size_dict

    if sectype == 'STK':
        if market == 'HK':
            sec_profile['commission_rate'] = 0.03 * 0.01
            sec_profile['min_commission'] = 3
            sec_profile['platform_fee'] = 15
        if market == 'US':
            sec_profile['commission_each_stock']   = 0.0049
            sec_profile['min_commission']          = 0.99
            sec_profile['platform_fee_each_stock'] = 0.005
            sec_profile['min_platform_fee']        = 1

    return sec_profile


def get_all_para_combination(para_dict, df_dict, sec_profile, start_date, end_date,
                             data_folder, signal_output_folder, backtest_output_folder,
                             run_mode, summary_mode, freq, py_filename):

    para_values = list(para_dict.values())
    para_keys = list(para_dict.keys())
    para_list = list(itertools.product(*para_values))

    print('number of combination:', len(para_list))

    intraday = True if freq != '1D' else False
    output_folder = backtest_output_folder if run_mode == 'backtest' else signal_output_folder

    all_para_combination = []

    for reference_index in range(len(para_list)):
        para = para_list[reference_index]
        code = para[0]
        df = df_dict[code]
        para_combination = {}
        for i in range(len(para)):
            key = para_keys[i]
            para_combination[key] = para[i]

        para_combination['para_dict'] = para_dict
        para_combination['sec_profile'] = sec_profile
        para_combination['start_date'] = start_date
        para_combination['end_date'] = end_date
        para_combination['reference_index'] = reference_index
        para_combination['freq'] = freq
        para_combination['file_format'] = file_format
        para_combination['df'] = df
        para_combination['intraday'] = intraday
        para_combination['output_folder'] = output_folder
        para_combination['data_folder'] = data_folder
        para_combination['run_mode'] = run_mode
        para_combination['summary_mode'] = summary_mode
        para_combination['py_filename'] = py_filename

        all_para_combination.append(para_combination)

    return all_para_combination


if __name__ == '__main__':

    start_date  = '2018-01-01'
    end_date    = '2022-12-31'
    freq        = '1D'
    market      = 'HK'
    sectype     = 'STK'
    file_format = 'parquet'

    initial_capital = 200000

    update_data = False
    run_mode = 'backtest'
    summary_mode = False
    read_only = False
    number_of_core = 60
    mp_mode = True

    df = client.get_hk_index_const('hang_seng_index', True).head(10)
    code_list = df['code'].to_list() + ['00388']
    #code_list = ['00388']

    para_dict = {
        'code'                   : code_list,
        'profit_target'          : [5, 10, 15, 20, 25, 30],
        'stop_loss'              : [2.5, 5, 7.5, 10],
        'holding_day'            : [5, 10, 20],
        'ssr_len'                : [50, 100, 250],
        'ssr_zscore_threshold'   : [1, 1.5, 2, 2.5, 3],
        'ssr_zscore_sign'        : ['positive', 'negative'],
        'zscore_cross_direction' : ['cross_over', 'cross_under']

    }


    ########################################################################
    ########################################################################

    df_dict = get_hist_data(code_list, start_date, end_date, freq,
                            data_folder, file_format, update_data,
                            market)

    df_dict = get_secondary_data(df_dict, secondary_data_folder, start_date, end_date, update_data)

    sec_profile = get_sec_profile(code_list, market, sectype, initial_capital)

    all_para_combination = get_all_para_combination(para_dict, df_dict, sec_profile, start_date, end_date,
                             data_folder, signal_output_folder, backtest_output_folder,
                             run_mode, summary_mode, freq, py_filename)

    if not read_only:
        if mp_mode:
            pool = mp.Pool(processes=number_of_core)
            pool.map(backtest, all_para_combination)
            pool.close()
        else:
            for para_combination in all_para_combination:
               backtest(para_combination)

    plotguy.generate_backtest_result(
        all_para_combination=all_para_combination,
        number_of_core=number_of_core
    )

    app = plotguy.plot(
        mode='equity_curves',
        all_para_combination=all_para_combination
    )

    app.run_server(port=8900)

