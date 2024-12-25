import requests
import matplotlib.pyplot as plt
import httpx
import psycopg2


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
    


if __name__ == '__main__':
    for i in range(1, 32):
        date = f'202101{i:02d}'
        get_daily_data(date, 'AAPL')