import sklearn
import psycopg2
from datetime import datetime, timedelta
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def load_data() -> np.array:
    '''
    Load features from postgres and normalize
    returns: list of t nx9 ndarrays
    '''
    start_date = '20210101'
    psql_password = os.getenv('POSTGRES_PASSWORD')
    conn = psycopg2.connect(database='postgres',host='localhost',user='postgres',password=psql_password,port='5432')
    cur = conn.cursor()
    try:
        cur.execute('SELECT root FROM features WHERE date = %s;',(start_date,))
        roots = cur.fetchall()
        roots = [root[0] for root in roots]
    except Exception as e:
        print(e)
        conn.rollback()
    conn.commit()
    cur.close()
    conn.close()

if __name__ == '__main__':
    load_data()

