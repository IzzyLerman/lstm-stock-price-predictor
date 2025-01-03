import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import httpx
import psycopg2
from datetime import datetime, timedelta
import os
import copy

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calc_avg_gain_loss(ohlcs: list, today: int, n: int) -> tuple:
    '''
    Calculate average gain and average loss for the past n days for each stock in ohlcs
    '''
    assert(n <= today)
    gains, losses = [], []
    start = today-n+1
    for idx in range(start+1, today+1):
        p_diff = 100*(ohlcs[idx][:,3] - ohlcs[idx-1][:,3])/ohlcs[idx-1][:,3]
        gains.append(copy.deepcopy(p_diff))
        gains[-1][p_diff < 0] = 0

        losses.append(copy.deepcopy(p_diff))
        losses[-1][p_diff > 0] = 0
        

    avg_gain = np.mean(gains, axis=0)
    avg_loss = np.mean(losses, axis=0)
    return avg_gain, avg_loss

def get_t_day_ohlcs(roots: list, t: int, date: datetime) -> np.array:
    '''
    input: list of roots (length n), t days, date
    output: t nx5 ndarrays (open, high, low, close, volume) for each root, for each date
    '''
    end_date = date + timedelta(days=t)
    if end_date > datetime.now():
        end_date = datetime.now()
    st_str = datetime.strftime(date, '%Y%m%d')
    end_str = datetime.strftime(end_date, '%Y%m%d')
    n = len(roots)
    ohlcs = []
    for root in roots:
        if root == 'META' and end_date < datetime(2021, 8, 1):
            c_root = 'FB'
        else:
            c_root = root
        biweek = []
        BASE_URL = "http://127.0.0.1:25510/v2" 
        params = {
            'root': c_root,
            'start_date': st_str,
            'end_date': end_str,
        }
        url = BASE_URL + '/hist/stock/eod'
        try:
            response = httpx.get(url, params=params)
            response.raise_for_status()  # Raise an HTTPError for bad responses
            data = response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error occurred: {e}")
            return
        except httpx.RequestError as e:
            logger.error(f"Request error occurred: {e}")
            return
        except Exception as e:
            logger.error(f"An error occurred: {e}")
            return
        # first create tx5 array of ohlc for this stock
        idx = 0
        for i in range(t):
            if idx >= len(data['response']):
                biweek.append([np.nan, np.nan, np.nan, np.nan, np.nan])
                continue
            r = data['response'][idx]
            curr_date = date + timedelta(days=i)
            if str(r[16]) != datetime.strftime(curr_date, '%Y%m%d'):
                # Look for the next date in the same index
                biweek.append([np.nan, np.nan, np.nan, np.nan, np.nan])
                continue
            biweek.append([r[2], r[3], r[4], r[5], r[6]])
            # Look for the next date in the next index
            idx += 1
        biweek = np.transpose(np.array(biweek))

        # convert to 5xt dataframe and ffill/bfill
        df = pd.DataFrame(biweek, columns=[i for i in range(t)])
        df.ffill(inplace=True, axis='columns')
        df.bfill(inplace=True, axis='columns')
        biweek = df.to_numpy()
        assert(np.shape(biweek) == (5, t))
        ohlcs.append(biweek)
    stacked_ohlcs = np.stack(ohlcs, axis=0)
    # Convert n 5 x t arrays to t n x 5 arrays
    stacked_ohlcs = np.transpose(stacked_ohlcs, (2, 0, 1))
    assert(np.shape(stacked_ohlcs) == (t, n, 5)) 
    ohlcs = [np.squeeze(x) for x in np.vsplit(stacked_ohlcs, t)] 
    return ohlcs



        

def get_data_from_period(roots: list, st: str, end: str) -> None:
    '''
    Update the database with OHLC data and indicators for the given roots in the given period.
    Date format: 'YYYYMMDD'
    '''
    ohlcs = []
    feature_series = []
    n = len(roots)
    psql_password = os.getenv('POSTGRES_PASSWORD')

    BASE_URL = "http://127.0.0.1:25510/v2"
    url = BASE_URL + '/hist/stock/eod'

    conn = psycopg2.connect(database='postgres',host='localhost',user='postgres',password=psql_password,port='5432')
    cur = conn.cursor()


    start_date = datetime.strptime(st, '%Y%m%d')
    end_date = datetime.strptime(end, '%Y%m%d')
    period_length = (end_date-start_date).days
    curr_date = start_date 
    dates = []

    idx = 0
    for idx in range(period_length+1):
        curr_date = start_date + timedelta(days=idx)
        curr_date_str = curr_date.strftime('%Y%m%d')
        # Every 120 days, 120 get days of OHLC for each root
        # list of nx5 ndarrays
        print(f'Getting data for {curr_date_str}, idx: {idx}')
        if idx % 120 == 0:
            period = get_t_day_ohlcs(roots, t=120, date=curr_date)
            if period is None:
                print('Error: Couldn\'t retrive data for this period')
                return
            ohlcs += period



        # Calculate SMA, EMA, RSI, MACD, %K, %D, ROC, Williams R%
        # Indicators are stored as -1 if there is not enough data (MACD needs 26 days)
        today = ohlcs[idx]
        dates.append(curr_date_str)
        if idx < 25:
            features = np.column_stack([today, np.full((len(roots), 8), -1)])
        else:
            sma_14 = np.mean([ohlcs[i][:,3] for i in range(idx-13,idx+1)],axis=0)

            #initialize first values for some features
            if(idx == 25):
                ema_12 = np.mean([ohlcs[i][:,3] for i in range(idx-11,idx+1)],axis=0)
                ema_14 = sma_14
                ema_26 = np.mean([ohlcs[i][:,3] for i in range(idx-25,idx+1)],axis=0)
                ag, al = calc_avg_gain_loss(ohlcs, idx, n=14)

            else:
                #EMA
                ema_12 = 2/(12+1)*(today[:,3]-ema_12)+ema_12
                ema_14 = 2/(14+1)*(today[:,3]-ema_14)+ema_14
                ema_26 = 2/(26+1)*(today[:,3]-ema_26)+ema_26

                #Average gain and loss
                gain = 100*(today[:,3] - ohlcs[idx-1][:,3])/ohlcs[idx-1][:,3]
                loss = copy.deepcopy(gain)
                gain[gain<0] = 0
                loss[loss>0] = 0
                ag = (ag*13 + gain)/14
                al = (al*13 + loss)/14

            #Replace 0s with small values for stability
            al[al == 0] = 1/(np.power(10,6))
            rs = np.divide(ag,-al)
            rsi = 100 - 100*np.reciprocal(1+rs)

            #Highest high and lowest low
            highs = np.max([ohlcs[i][:,1] for i in range(idx-13,idx+1)],axis=0)
            lows = np.min([ohlcs[i][:,2] for i in range(idx-13,idx+1)],axis=0)
            

            macd = ema_12 - ema_26
            stochastic_k = (today[:,3]-lows)/ (highs-lows)
            if idx < 27:
                stochastic_d = np.full((n,), -1)
            else:
                sto_k_3 = np.vstack([feature_series[idx-2][:,9], feature_series[idx-1][:,9], stochastic_k])
                stochastic_d = np.mean(sto_k_3, axis=0)
            roc = 100*np.divide(today[:,3]-ohlcs[idx-13][:,3],ohlcs[idx-13][:,3])
            williams_r = 100 *np.divide(highs-today[:,3],highs-lows)
            
            features = np.column_stack((today, sma_14, ema_14, rsi, macd, stochastic_k, stochastic_d, roc, williams_r))
        assert(np.shape(features) == (n, 13))
            
        feature_series.append(features)
        # Every 14 days, send 14xnx15 array to postgres
        # (date, root, open, high, low, close, volume, SMA, EMA, RSI, MACD, %K, %D, ROC, Williams R%)
        if idx % 14 == 13:
            for i in range(idx-13, idx+1):
                for j in range(len(roots)):
                    stock = feature_series[i][j]
                    cur.execute('INSERT INTO features (date, root, open, high, low, close, volume, sma, ema, rsi, macd, stochastic_k, stochastic_d, roc, williams_r) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)',
                                (dates[i], roots[j], stock[0], stock[1], stock[2], stock[3], stock[4], stock[5], stock[6], stock[7], stock[8], stock[9], stock[10], stock[11], stock[12]))

            conn.commit()
    cur.close()
    conn.close()

    
    #Display features for roots[0]
    features_first_stock = np.array([features[0] for features in feature_series])
    feature_names = [
        'Open', 'High', 'Low', 'Close', 'Volume', 'SMA', 'EMA', 'RSI', 'MACD',
        'Stochastic %K', 'Stochastic %D', 'ROC', 'Williams %R'
    ]
    fig, axes = plt.subplots(len(feature_names), 1, figsize=(10, 20), sharex=True)
    fig.suptitle(f'Features Over Time for {roots[0]}')
    for i, ax in enumerate(axes):
        ax.plot(features_first_stock[:, i])
        ax.set_ylabel(feature_names[i])
        ax.grid(True)
    axes[-1].set_xlabel('Days')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()






if __name__ == '__main__':
    #roots = ['AAPL', 'MSFT']
    roots = [
        'AAPL', 'NVDA', 'MSFT', 'GOOG', 'AMZN', 'META', 'TSLA', 'AVGO', 'BRK.B', 'WMT',
        'LLY', 'JPM', 'V', 'MA', 'XOM', 'ORCL', 'UNH', 'COST', 'PG', 'HD',
        'NFLX', 'JNJ', 'BAC', 'ABBV', 'CRM', 'KO', 'CVX', 'TMUS', 'MRK', 'CSCO',
        'WFC', 'NOW', 'AXP', 'MCD', 'PEP', 'IBM', 'MS', 'DIS', 'TMO', 'ABT',
        'AMD', 'ADBE', 'PM', 'ISRG', 'GE', 'GS', 'INTU', 'CAT', 'PLTR', 'QCOM'
    ]
    get_data_from_period(roots, '20210101', '20250101')