import numpy as np
import pandas as pd
import json
import datetime
import requests

date_range = pd.date_range(start = '2018-08-23', end = pd.datetime.now().date().isoformat(), freq = 'd')
for idx1, date in enumerate(date_range):
    if date =='2018-12-16':
        break
    else:
        pass
    date1 = date_range[idx1].date().isoformat()
    date2 = date_range[idx1 + 1].date().isoformat()
    tmp = requests.get('https://api.coindesk.com/charts/data?data=close&startdate='+date1+'&enddate='+date2+'&exchanges=bpi&dev=1&index=USD')
    tmp = pd.DataFrame(json.loads(tmp.content[3:-2])['bpi'])
    tmp.iloc[:,0] = tmp.iloc[:,0].apply(lambda x: datetime.datetime.utcfromtimestamp(round(x-3)/1000) )
    tmp.columns = ['date','price']
    tmp.date = pd.to_datetime(tmp.date)
    tmp = tmp[tmp.iloc[:,0].apply(lambda x : x.date().isoformat()==date1)]
    
    tmp.to_csv('C:\\Users\\secret\\Desktop\\Python\\Sublime\\Dissertation\\Data Storage\\minute_data\\bitcoin_'+date1+'.csv')
