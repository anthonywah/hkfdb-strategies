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

import holidays
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

def row_backtest(row, para_combination, last_index):

    ##### variables from previous row #####
    open_date             = row_backtest.open_date
    open_price            = row_backtest.open_price
    num_of_share          = row_backtest.num_of_share
    net_profit            = row_backtest.net_profit
    num_of_trade          = row_backtest.num_of_trade
    last_realized_capital = row_backtest.last_realized_capital
    equity_value          = row_backtest.equity_value
    unrealized_pnl        = row_backtest.unrealized_pnl
    commission            = row_backtest.commission

    ##################################################
    para_dict       = para_combination['para_dict']
    sec_profile     = para_combination['sec_profile']
    start_date      = para_combination['start_date']
    end_date        = para_combination['end_date']
    reference_index = para_combination['reference_index']
    freq            = para_combination['freq']
    intraday        = para_combination['intraday']
    output_folder   = para_combination['output_folder']
    data_folder     = para_combination['data_folder']
    py_filename     = para_combination['py_filename']
    holiday_list    = para_combination['holiday_list']
    run_mode        = para_combination['run_mode']

    ##### stra specific #####
    code                = para_combination['code']
    profit_target       = para_combination['profit_target']
    stop_loss           = para_combination['stop_loss']
    holding_day         = para_combination['holding_day']

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

    now_date = row.name.date()
    now_time = row.name.time()
    now_open = row['open']
    now_high = row['high']
    now_low = row['low']
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
    if num_of_share == 0: open_price = 0
    unrealized_pnl = num_of_share * (now_close - open_price) - commission
    equity_value = last_realized_capital + unrealized_pnl
    net_profit = round(equity_value - initial_capital, 2)

    logic = 'trade_logic' if trade_logic else None

    if run_mode == 'backtest':

        close_logic = num_of_share != 0 and np.busday_count(open_date, now_date, holidays=holiday_list) >= holding_day and \
                      (now_time >= datetime.time(13, 00) or now_time <= datetime.time(15, 45))

        profit_target_cond = num_of_share != 0 and now_close - open_price > profit_target * open_price * 0.01
        stop_loss_cond = num_of_share != 0 and open_price - now_close > stop_loss * open_price * 0.01
        last_index_cond = row.name == last_index
        min_cost_cond = last_realized_capital > now_close * lot_size

        ##### open position #####
        if num_of_share == 0 and not last_index_cond and min_cost_cond and trade_logic:

            num_of_lot = last_realized_capital // (lot_size * now_close)
            num_of_share = num_of_lot * lot_size

            open_price = now_close
            open_date = now_date

            action = 'open'

            close_price  = np.NaN
            realized_pnl = np.NaN

        ##### close position #####
        elif num_of_share > 0 and (profit_target_cond or stop_loss_cond or last_index_cond or close_logic):

            close_price = now_close
            realized_pnl = unrealized_pnl
            unrealized_pnl = 0
            last_realized_capital += realized_pnl

            num_of_trade += 1

            num_of_share = 0

            if close_logic: logic = 'close_logic'

            if last_index_cond:    action = 'last_index'
            if close_logic:        action = 'close_logic'
            if profit_target_cond: action = 'profit_target'
            if stop_loss_cond:     action = 'stop_loss'

            open_price = np.NaN

        else:
            close_price  = np.NaN
            action       = ''
            realized_pnl = np.NaN
            if num_of_share == 0:
                open_price = np.NaN



    row_backtest.open_date             = open_date
    row_backtest.open_price            = open_price
    row_backtest.num_of_share          = num_of_share
    row_backtest.net_profit            = net_profit
    row_backtest.num_of_trade          = num_of_trade
    row_backtest.last_realized_capital = last_realized_capital
    row_backtest.equity_value          = equity_value
    row_backtest.unrealized_pnl        = unrealized_pnl
    row_backtest.commission            = commission

    return action, num_of_share, open_price, close_price, realized_pnl, unrealized_pnl, net_profit, equity_value, 0, 0, commission, logic

def backtest(para_combination):

    df              = para_combination['df']
    summary_mode    = para_combination['summary_mode']
    run_mode        = para_combination['run_mode']
    file_format     = para_combination['file_format']
    initial_capital = para_combination['sec_profile']['initial_capital']

    del para_combination['df']

    ##### stra specific #####
    pct_change_sign      = para_combination['pct_change_sign']
    pct_change_threshold = para_combination['pct_change_threshold']

    if pct_change_sign == 'positive':
        df['trade_logic'] = (df['pct_change'] > pct_change_threshold * 0.01) & (df.index.time == datetime.time(15, 45))
    elif pct_change_sign == 'negative':
        df['trade_logic'] = (df['pct_change'] < -1 * pct_change_threshold * 0.01) & (df.index.time == datetime.time(15, 45))

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

    row_backtest.open_date    = datetime.datetime.now().date()
    row_backtest.open_price   = 0
    row_backtest.num_of_share = 0
    row_backtest.net_profit   = 0
    row_backtest.num_of_trade = 0
    row_backtest.last_realized_capital = initial_capital
    row_backtest.equity_value = 0
    row_backtest.realized_pnl   = 0
    row_backtest.unrealized_pnl = 0
    row_backtest.commission = 0

    last_index = df.index[-1]

    new_columns = df.columns[-12:]
    df[new_columns] = df.apply(row_backtest, args= (para_combination, last_index), axis=1, result_type='expand')

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
    
    def get_average(row):
        if row['avg_price_per_cbbc_bought'] != 0 and row['avg_price_per_cbbc_sold'] != 0:
            return (row['avg_price_per_cbbc_bought'] + row['avg_price_per_cbbc_sold']) / 2
        else:
            return max(row['avg_price_per_cbbc_bought'], row['avg_price_per_cbbc_sold'])

    start_year = start_date[0:4]
    end_year   = end_date[0:4]
    year_diff = int(end_year) - int(start_year)

    for code, df in df_dict.items():
        
        if code == '02800':
            dl_code = 'HSI'
        else:
            dl_code = code
            
        df2_list = []

        for i in range(year_diff + 1):
            dl_year = str(int(start_year) + i)
            dl_start_date = int(dl_year + '0101')
            dl_end_date = int(dl_year + '1231')

            file_path = os.path.join(secondary_data_folder, code + '_' + str(dl_start_date) + '_' + str(dl_end_date) + '.parquet')

            try:

                if os.path.isfile(file_path) and not update_data:
                    df2 = pd.read_parquet(file_path)
                    print(datetime.datetime.now(), ' successfully read secondary data', code, dl_start_date)
                else:
                    df2 = client.get_hk_cbbc(dl_code, dl_start_date, dl_end_date)
                    time.sleep(1)
                    df2.to_parquet(file_path)
                    print(datetime.datetime.now(), ' successfully get secondary data', code, dl_start_date)

                df2_list.append(df2)

            except:
                print(datetime.datetime.now(), ' error in  getting secondary data', code, dl_start_date)
                time.sleep(1)

        df2 = pd.concat(df2_list)
        
        df2['average_price'] = df2.apply(get_average, axis=1)
        df2['cbbc_mak_cap'] = df2['average_price'] * df2['num_of_cbbc_in_mkt']

        df2_bull = df2[df2['bull_bear'] == 'Bull']
        df2_bear  = df2[df2['bull_bear'] == 'Bear']

        df2_bull = df2_bull.groupby('date').agg({'cbbc_mak_cap':'sum'})
        df2_bear  = df2_bear.groupby('date').agg({'cbbc_mak_cap':'sum'})

        df2_bull = df2_bull.rename(columns= {'cbbc_mak_cap': 'bull_mak_cap'})
        df2_bear  = df2_bear.rename(columns= {'cbbc_mak_cap': 'bear_mak_cap'})

        df2 = pd.concat([df2_bull, df2_bear], axis=1)

        df2['cbbc_mak_cap_ratio'] = df2['bull_mak_cap'] / (df2['bull_mak_cap'] + df2['bear_mak_cap'])

        df2 = df2[['cbbc_mak_cap_ratio']]

        df = pd.concat([df, df2], axis=1)

        df['cbbc_mak_cap_ratio'] = df['cbbc_mak_cap_ratio'].shift(1)
        df['cbbc_mak_cap_ratio-1'] = df['cbbc_mak_cap_ratio'].shift(1)

        df = df[df.index >= start_date]
        df = df[df.index <= end_date]

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
                             run_mode, summary_mode, freq, py_filename, market):

    para_values = list(para_dict.values())
    para_keys = list(para_dict.keys())
    para_list = list(itertools.product(*para_values))

    print('number of combination:', len(para_list))

    intraday = True if freq != '1D' else False
    output_folder = backtest_output_folder if run_mode == 'backtest' else signal_output_folder

    start_year = int(start_date[0:4])
    end_year   = int(end_date[0:4])

    if market == 'HK':
        holiday_list = hkfdb.get_hk_holiday_and_expiry_date(start_year, end_year, format='dt')['public_holiday']
    elif market == 'US':
        year_list = range(start_year, end_year + 1)
        holiday_list = list(holidays.US(year_list))

    all_para_combination = []

    for reference_index in range(len(para_list)):
        para = para_list[reference_index]
        code = para[0]
        df = df_dict[code]
        para_combination = {}
        for i in range(len(para)):
            key = para_keys[i]
            para_combination[key] = para[i]

        para_combination['holiday_list'] = holiday_list
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
    freq        = '15T'
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
    code_list = df['code'].to_list() + ['00388', '02800']
    #code_list = ['00388']

    para_dict = {
        'code'                   : code_list,
        'profit_target'          : [10],
        'stop_loss'              : [5],
        'holding_day'            : [5],
        'pct_change_sign'        : ['positive','negative'],
        'pct_change_threshold'   : [0.2, 0.4, 0.6, 0.8],
    }


    ########################################################################
    ########################################################################

    df_dict = get_hist_data(code_list, start_date, end_date, freq,
                            data_folder, file_format, update_data,
                            market)

    # df_dict = get_secondary_data(df_dict, secondary_data_folder, start_date, end_date, update_data)

    sec_profile = get_sec_profile(code_list, market, sectype, initial_capital)

    all_para_combination = get_all_para_combination(para_dict, df_dict, sec_profile, start_date, end_date,
                             data_folder, signal_output_folder, backtest_output_folder,
                             run_mode, summary_mode, freq, py_filename, market)

    start_time = datetime.datetime.now()

    if not read_only:
        if mp_mode:
            pool = mp.Pool(processes=number_of_core)
            pool.map(backtest, all_para_combination)
            pool.close()
        else:
            for para_combination in all_para_combination:
               backtest(para_combination)

    print('backtest finished, time used: ', datetime.datetime.now() - start_time)

    start_time = datetime.datetime.now()

    plotguy.generate_backtest_result(
        all_para_combination=all_para_combination,
        number_of_core=number_of_core
    )

    print('generate_backtest_result finished, time used: ', datetime.datetime.now() - start_time)

    app = plotguy.plot(
        mode='equity_curves',
        all_para_combination=all_para_combination
    )

    app.run_server(port=8901)

