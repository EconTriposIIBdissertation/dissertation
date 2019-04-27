import requests 
import re
import datetime
import pandas as pd

coin_ids = [1,437,3,436]
coins = ['Bitcoin','Ethereum','Litecoin', 'Monero']
coins = [x.lower() for x in coins]
charts = ['network-hashrate', 'difficulty','exchange-rate']
chart_dfs = {}
for (coin,coin_id) in zip(coins,coin_ids):
    chart_dfs[coin] = {}
    for chart_name in charts:
        if chart_name == 'network-hashrate':
            print('getting hashrate data')
            r1 = requests.get('https://www.coinwarz.com/network-hashrate-charts/'+coin+'-network-hashrate-chart')
            t = re.findall('[0-9]{96}',r1.text)[0]
            r2= requests.get('https://www.coinwarz.com/ajax/networkhashratechartdata?coinId='+str(coin_id)+'&t='+str(t))
            df = pd.DataFrame(r2.json())
            df.iloc[:,0] = df.iloc[:,0].apply(lambda x: datetime.datetime.utcfromtimestamp(x/1000) )
        elif chart_name =='difficulty':
            print('getting difficulty data')
            r1 = requests.get('https://www.coinwarz.com/difficulty-charts/'+coin+'-difficulty-chart')
            t = re.findall('[0-9]{96}',r1.text)[0]
            r2= requests.get('https://www.coinwarz.com/ajax/diffchartdata?coinId='+str(coin_id)+'&t='+str(t))
            df = pd.DataFrame(r2.json())
            df.iloc[:,0] = df.iloc[:,0].apply(lambda x: datetime.datetime.utcfromtimestamp(x/1000) )
        elif chart_name =='exchange-rate':
            print('getting ex-rate data')
            r1 = requests.get('https://www.coinwarz.com/exchange-charts/'+coin+'-exchange-rate-chart')
            t = re.findall('[0-9]{96}',r1.text)[0]
            r2= requests.get('https://www.coinwarz.com/ajax/exchangechartdata?coinId='+str(coin_id)+'&t='+str(t))
            df = pd.DataFrame(r2.json()[0]['ExchangeSnapshots'])
            df.iloc[:,0] = df.iloc[:,0].apply(lambda x: datetime.datetime.utcfromtimestamp(x/1000) )
        else:
            pass
        chart_dfs[coin][chart_name] = df

date_num = str(datetime.datetime.now().day)+'_'+ str(datetime.datetime.now().month)

for coin in coins:
    for chart_name in charts:
        chart_dfs[coin][chart_name].to_csv('C:\\Users\\secret\\Desktop\\Python\\Sublime\\Dissertation\\Data Storage\\Price_diff_hash_data\\3month_'+coin+'_'+chart_name+'_'+date_num+'.csv')

###Cardano
rADA = requests.get('https://www.coingecko.com/price_charts/975/usd/max.json')
ada = pd.DataFrame(rADA.json()['stats'])
ada.iloc[:,0] = ada.iloc[:,0].apply(lambda x: datetime.datetime.utcfromtimestamp(round(x-3)/1000) )

date_num = str(datetime.datetime.now().day)+'_'+ str(datetime.datetime.now().month)
ada.to_csv('C:\\Users\\secret\\Desktop\\Python\\Sublime\\Dissertation\\Data Storage\\Price_diff_hash_data\\max_'+'cardano'+'_'+'exchange-rate'+'_'+date_num+'.csv')


### Ripple
rXRP= requests.get('https://www.coingecko.com/price_charts/44/usd/max.json')
ripple = pd.DataFrame(rXRP.json()['stats'])
ripple.iloc[:,0] = ripple.iloc[:,0].apply(lambda x: datetime.datetime.utcfromtimestamp(round(x-3)/1000) )

date_num = str(datetime.datetime.now().day)+'_'+ str(datetime.datetime.now().month)
ripple.to_csv('C:\\Users\\secret\\Desktop\\Python\\Sublime\\Dissertation\\Data Storage\\Price_diff_hash_data\\max_'+'ripple'+'_'+'exchange-rate'+'_'+date_num+'.csv')
