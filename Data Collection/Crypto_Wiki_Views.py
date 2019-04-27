import requests
import datetime
import time
import json
#initialize list to hold currencies to track
Currencies = ['Bitcoin','Ethereum','Ripple','Litecoin', 'Monero','Cryptocurrency']
#initialise dict to hold wiki views of crypto currency for each day
wiki_views = {}
#set up dict to hold dict of date: views for each currency
for currency in Currencies:
    wiki_views[currency] = {}


#continue data collection indefiinitely
while True:
    #for each currency
    for currency in Currencies:
        if datetime.datetime.now().hour %23 ==0:
            #query wiki api for views of wiki page
            try:
                response = requests.get('https://en.wikipedia.org/w/api.php?format=json&titles='+currency+'&action=query&prop=pageviews&pvipdays=1')
                if response.status_code == 200:
                    page_views_dict = list(response.json()['query']['pages'].values())[0]['pageviews']
                    wiki_views[currency][list(page_views_dict)[0]] = list(page_views_dict.values())[0]
            except (requests.exceptions.Timeout,requests.exceptions.ConnectionError, requests.exceptions.HTTPError):
                time.sleep(30)
            
            try:
                with open(r'C:\Users\secret\Desktop\Python\Sublime\Dissertation\Data Storage'+'\\'+ currency+'_wiki_views.txt','r') as f:
                    old_file = json.load(f)
                old_file.update(wiki_views[currency])
                
                with open(r'C:\Users\secret\Desktop\Python\Sublime\Dissertation\Data Storage'+'\\'+ currency+'_wiki_views.txt','w') as f:
                    json.dump(old_file,f)
                print('Updated Data')
            except FileNotFoundError:
                with open(r'C:\Users\secret\Desktop\Python\Sublime\Dissertation\Data Storage'+'\\'+ currency+'_wiki_views.txt','w') as f:
                    json.dump(wiki_views[currency],f)
                print('Updated Data') 
                
                
    time.sleep(60)
