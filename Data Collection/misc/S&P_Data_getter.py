import requests
import bs4
import pandas as pd
import datetime


response = requests.get('https://uk.investing.com/currencies/eur-usd-historical-data')

soup = bs4.BeautifulSoup(response.text)
body_soup = soup.findAll('div',id='historicalContainer')[0]
data = []
table = body_soup.find('table')

rows = table.find_all('tr')
for row in rows:
    cols = row.find_all('td')
    cols = [ele.text.strip() for ele in cols]
    data.append([ele for ele in cols]) # add data to dict

df =pd.DataFrame(data,columns=['Date','Open','High','Low','Close','Volume'])
df = df.iloc[1:,:]

date_num = str(datetime.datetime.now().day)+'_'+ str(datetime.datetime.now().month)
df.to_csv(r'C:\Users\secret\Desktop\Python\Sublime\Dissertation\Data Storage\S&P500'+date_num+'.csv')
