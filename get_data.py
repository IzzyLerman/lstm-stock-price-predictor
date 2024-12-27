import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import httpx
import psycopg2
from datetime import datetime, timedelta

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_daily_data(date: str, root: str) -> None:
    '''
    Update the database with OHLC data and indicators for the root stock at the given date.
    Date format: 'YYYYMMDD'
    '''
    BASE_URL = "http://127.0.0.1:25510/v2" 
    params = {
        'root': root,
        'start_date': date,
        'end_date': date,
    }
    url = BASE_URL + '/hist/stock/eod'
    response = httpx.get(url, params=params)
    try:
        data = response.json()
    except:
        print(response.text)
        return

    if 'response' not in data:
        print(f'No data for {date}')
        return

    r = data['response'][0]
    st_open = r[2]
    st_high = r[3]
    st_low = r[4]
    st_close = r[5]
    st_volume = r[6]

    conn = psycopg2.connect(database='postgres',
                            host='localhost',
                            user='postgres',
                            password='',
                            port='5432')
    cur = conn.cursor()
    cur.execute('INSERT INTO features (date, root, open, high, low, close, volume) VALUES (%s, %s, %s, %s, %s, %s, %s)',
                (date, root, st_open, st_high, st_low, st_close, st_volume))

    conn.commit()
    cur.close()
    conn.close()

def get_t_day_ohlcs(roots: list, t: int, date: datetime) -> np.array:
    '''
    input: list of roots (length n), t days, date
    output: t nx5 ndarrays (open, high, low, close, volume) for each root, for each date
    '''
    end_date = date + timedelta(days=t)
    st_str = datetime.strftime(date, '%Y%m%d')
    end_str = datetime.strftime(end_date, '%Y%m%d')
    n = len(roots)
    ohlcs = []
    for root in roots:
        biweek = []
        BASE_URL = "http://127.0.0.1:25510/v2" 
        params = {
            'root': root,
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
            #print( root, r[16],  datetime.strftime(curr_date, '%Y%m%d'))
            if str(r[16]) != datetime.strftime(curr_date, '%Y%m%d'):
                #print(f'No data for {curr_date}')
                # Look for the next date in the same index
                biweek.append([np.nan, np.nan, np.nan, np.nan, np.nan])
                continue
            biweek.append([r[2], r[3], r[4], r[5], r[6]])
            # Look for the next date in the next index
            idx += 1
            #print(f'Got data for {curr_date}')
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
    return [np.squeeze(x) for x in np.vsplit(stacked_ohlcs, t)]



        

def get_data_from_period(roots: list, st: str, end: str) -> None:
    '''
    Update the database with OHLC data and indicators for the given roots in the given period.
    Date format: 'YYYYMMDD'
    '''
    ohlcs = []

    BASE_URL = "http://127.0.0.1:25510/v2"
    url = BASE_URL + '/hist/stock/eod'


    start_date = datetime.strptime(st, '%Y%m%d')
    end_date = datetime.strptime(end, '%Y%m%d')

    curr_date = start_date 
    while curr_date <= end_date:
        curr_date_str = curr_date.strftime('%Y%m%d')

        idx = curr_date-start_date
        # Every 14 days, get 14 days of OHLC for each root
        # list of nx5 ndarrays
        if idx % 14 == 0:
            ohlcs.append(get_t_day_ohlcs(roots, t=14, date=curr_date))


        # Calculate indicators based on past 14 days for each root
        # Cache simple average, average gain/loss, ema_12 and ema_26
        # highest high, and lowest low



        # Every 14 days, send 14xnx9 array to postgres
        # (close, SMA, EMA, RSI, MACD, %K, %D, ROC, Williams R%)
        if curr_date-start_date % 14 == 0:
            pass





if __name__ == '__main__':
    ohlcs = get_t_day_ohlcs(['AAPL', 'MSFT'], 14, datetime.strptime('20210101', '%Y%m%d'))
    for i in ohlcs:
        #print(i)
        pass