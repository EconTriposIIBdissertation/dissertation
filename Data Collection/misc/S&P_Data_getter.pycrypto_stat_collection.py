import requests
import pandas as pd
import numpy as np
import bs4
import datetime
import schedule
import time
from sqlalchemy import create_engine
import re

coins = ['Bitcoin','Ethereum', 'Ripple', 'Litecoin', 'Monero', 'Cardano']
def collect_stats(*args):

    coin_stat_dict = {}

    for coin in coins:
        coin_stat_dict[coin] = {}
        
        response = requests.get('https://bitinfocharts.com/'+coin.lower()+'/')

        soup = bs4.BeautifulSoup(response.text)

        body_soup = soup.findAll('div', id='main_body')[0]

        data = []
        table = body_soup.find('table')

        rows = table.find_all('tr')
        for row in rows:
            cols = row.find_all('td')
            cols = [ele.text.strip() for ele in cols]
            data.append([ele for ele in cols]) # Get rid of empty values

        for tup in data:
            if re.match('Total '+ coin, tup[0], flags = re.IGNORECASE): # Does total coin stock data exist
                #save total currency stock
                total_coin = int(re.findall('([0-9,]+)',tup[1])[0].replace(',',''))
                coin_stat_dict[coin]['total_coin'] = total_coin
                
            if re.match('Market Capitalization',tup[0],flags=re.IGNORECASE): # Does Market Cap data exist
                market_cap = float(re.findall('([0-9,]+)', tup[1])[0].replace(',',''))
                coin_stat_dict[coin]['market_cap'] = market_cap
                
            if re.match('Tweets/day', tup[0],flags=re.IGNORECASE): # Does tweet data exist
                tweets_pday = int(re.findall('([0-9,]+)', tup[1])[0].replace(',',''))
                coin_stat_dict[coin]['tweets_pday'] = tweets_pday
            
            if re.match('Transactions last 24h', tup[0],flags=re.IGNORECASE): # Does transactions data exist
                trans_pday = int(re.findall('([0-9,]+)', tup[1])[0].replace(',',''))
                coin_stat_dict[coin]['trans_pday'] = trans_pday
            
            if re.match('Hashrate', tup[0],flags=re.IGNORECASE): # Does hashrate data exist
                hashrate_avg = float(re.findall('([0-9,]+)', tup[1])[0].replace(',',''))
                coin_stat_dict[coin]['hashrate_avg'] = hashrate_avg
                
            if re.match('Reward per block', tup[0],flags=re.IGNORECASE): # Does reward data exist
                block_reward = float(re.findall('([0-9,]+)', tup[1])[0].replace(',',''))
                coin_stat_dict[coin]['block_reward'] = block_reward
            
            if re.match('Difficulty', tup[0],flags=re.IGNORECASE): # Does hashrate data exist
                difficulty = float(re.findall('([0-9,]+)', tup[1])[0].replace(',',''))
                coin_stat_dict[coin]['difficulty'] = difficulty
                
            coin_stat_dict[coin]['time'] = pd.to_datetime(datetime.datetime.now().isoformat())
            
        

    
    engine = create_engine('mysql+mysqldb://root:redcard@127.0.0.1/tweet_store', echo=False)
    coin_statsdf = pd.DataFrame.from_dict(coin_stat_dict,orient='index')
    coin_statsdf.index.name = 'name'
    coin_statsdf = coin_statsdf.reset_index()
    coin_statsdf.to_sql('Crypto_daily_stats',engine, if_exists='append',index = False)

schedule.every().day.at("23:59").do(collect_stats,'It is 23:59')

while True:
    schedule.run_pending()
    time.sleep(60) # wait one minute
