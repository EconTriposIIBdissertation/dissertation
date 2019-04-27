#! python3

import requests
import json
import pandas as pd
import numpy as np
import praw
import datetime
import time
from sqlalchemy import create_engine
from urllib.parse import urlparse
import sqlalchemy

CLIENT_ID = 'tBTveYnvHH1xsw'
CLIENT_SECRET = 'WDIOGdOp3vWUH40FnxeMilEzt2U'
REDIRECT_URI = 'https://localhost:8889/reddit'



#initialize list to hold currencies to track
Currencies = ['Bitcoin','Ethereum','Ripple','Litecoin', 'Monero','Cryptocurrency','Cardano']
Currencies = [x.lower() for x in Currencies]
#Perform data extraction indefinitely until interrupt
while True:
    #Given we are at the end of a day
    cur_time = datetime.datetime.now()
    if (cur_time.hour ==23) & (cur_time.minute >=30):
        #Create the reddit api accessor
        reddit = praw.Reddit(client_id=CLIENT_ID,
                     client_secret=CLIENT_SECRET,
                     user_agent='api_accessor',
                     username='secret',
                     password='redcard')
        
        #Iterate over currencies
        for currency in Currencies:
            #acess the top subreddits of the day
            try:
                top_subreddit = reddit.subreddit(currency).top('day')
                #initialise dict to store data
                topics_dict = { "title":[],
                                "score":[],
                                "id":[], "url":[],
                                "comms_num": [],
                                "created": [],
                                "body":[]}
                #add subreddit data to dictionary for the currency
                for submission in top_subreddit:
                    topics_dict["title"].append(submission.title)
                    topics_dict["score"].append(submission.score)
                    topics_dict["id"].append(submission.id)
                    topics_dict["url"].append(submission.url)
                    topics_dict["comms_num"].append(submission.num_comments)
                    topics_dict["created"].append(submission.created)
                    topics_dict["body"].append(submission.selftext)

                #turn data to dataframe
                df = pd.DataFrame(topics_dict)

                ###Fix some of the data
                try:
                    #Fix url
                    def extract_netloc(url):
                        return urlparse(url).netloc
                    df['url'] = df['url'].apply(extract_netloc)
                except:
                    df['url'] = 'error'

                #fix the date
                df['created'] = df['created'].apply(datetime.datetime.fromtimestamp)
                
                #fix text lengths
                df['body'] = df['body'].apply(lambda x: x[:2000])
                df['title'] = df['title'].apply(lambda x: x[:400])
                
                #fix index name to be compatible with mysql
                df.index.name = 'idx'

                ###Add the data to an SQL database
                #Connect to the SQL database
                engine = sqlalchemy.create_engine('mysql+mysqldb://root:redcard@127.0.0.1/tweet_store?charset=utf8mb4',echo=False)
                #Check the Table exists
                exec_stmt = str("CREATE TABLE IF NOT EXISTS crypto_reddits_" + str(currency) + " (idx INTEGER, body VARCHAR(2000), comms_num INTEGER, created DATETIME, id VARCHAR(6), score INTEGER, title VARCHAR(400), url VARCHAR(80)) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;")
                result = engine.execute(exec_stmt)
                try:
                    result.close()
                except:
                    pass
                #Add the data to the database
                df.to_sql('crypto_reddits_'+str(currency), con=engine, if_exists='append')
            except Exception as e:
                print(str(e))
        
        #wait until it is a new day to repeat process  
        while cur_time.hour ==23:
            time.sleep(60)
    else:
        pass
    time.sleep(60)
