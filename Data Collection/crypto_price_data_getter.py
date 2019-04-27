import pandas as pd
import requests
import bs4
import datetime

coins = ['Bitcoin','Ethereum', 'Ripple', 'Litecoin', 'Monero', 'Cardano']

coins = [coin.lower() for coin in coins]

for coin in coins:
	response = requests.get('https://coinmarketcap.com/currencies/'+coin+'/historical-data/')

	soup = bs4.BeautifulSoup(response.text)
	table = soup.find('table')

	rows = table.find_all('tr')
	data =[]
	for row in rows:
	    cols = row.find_all('td')
	    cols = [ele.text.strip() for ele in cols]
	    data.append([ele for ele in cols]) # add data to dict

	df =pd.DataFrame(data,columns=['Date','Open','High','Low','Close','Volume','Market_Cap'])
	df = df.iloc[1:,:]
	date_num = str(datetime.datetime.now().day)+'_'+ str(datetime.datetime.now().month)
	df.to_csv('C:\\Users\\secret\\Desktop\\Python\\Sublime\\Dissertation\\Data Storage\\'+coin+date_num+'.csv')
