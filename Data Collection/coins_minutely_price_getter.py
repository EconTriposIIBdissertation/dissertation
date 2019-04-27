import json
import datetime
import requests
import pandas as pd


coins = ['BTC','ETH','XMR','LTC','XRP','ADA']
for coin in coins:
	tmp = requests.get('https://production.api.coindesk.com/v1/currency/'+coin+'/graph?start_date=2018-05-23&end_date=2018-12-25&interval=1-mins&convert=USD&ohlc=false')
	tmp = pd.DataFrame(tmp.json()['data']['graph'][coin]['to']['USD']['chartData'])
	tmp.iloc[:,0] = tmp.iloc[:,0].apply(lambda x: datetime.datetime.utcfromtimestamp(round(x-3)/1000) )
	tmp.columns = ['date','price']
	tmp.date = pd.to_datetime(tmp.date)
	# tmp = tmp[tmp.iloc[:,0].apply(lambda x : x.date().isoformat()==date1)]  
	tmp.to_csv('C:\\Users\\secret\\Desktop\\Python\\Sublime\\Dissertation\\Data Storage\\minute_data\\minutelyprice_'+coin+'.csv')
