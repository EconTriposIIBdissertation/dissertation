from pytrends.request import TrendReq
import pandas as pd
import datetime


coins = ['Bitcoin','Ethereum', 'Ripple', 'Litecoin', 'Monero', 'Cardano']

for coin in coins:
    pytrend = TrendReq(hl='de-CH')

    pytrend.build_payload(kw_list=[coin.lower()], timeframe='today 3-m', geo='', gprop='')

    # Interest Over Time
    interest_over_time_df = pytrend.interest_over_time()
    date_num = str(datetime.datetime.now().day)+'_'+ str(datetime.datetime.now().month)
    interest_over_time_df.to_csv(r'C:\Users\secret\Desktop\Python\Sublime\Dissertation\Data Storage\Gtrend_'+coin.lower()+date_num+'.csv')

		
