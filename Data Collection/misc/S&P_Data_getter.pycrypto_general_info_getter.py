import pandas as pd
import numpy as np
import requests
tickers = ['eth','ada','btc','xmr','xrp','ltc']
coin_info = {}
for ticker in tickers:
    tmp = requests.get('https://coinmetrics.io/data/'+ticker+'.csv')
    df = pd.read_csv(pd.compat.StringIO(tmp.text))
    df = df.reset_index()
    df.columns = np.append(df.columns[1:].values,[''])
    df = df.iloc[:,:-1]
    coin_info[ticker] = df
