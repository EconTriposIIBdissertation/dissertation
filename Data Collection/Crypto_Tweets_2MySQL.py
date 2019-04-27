import numpy as np
import pandas as pd
import tweepy
import json
import subprocess
import datetime
import MySQLdb
import sqlalchemy
import time

from ssl import SSLError
from requests.exceptions import Timeout, ConnectionError
from urllib3.exceptions import ReadTimeoutError

 

def Stream_Twitter_Data_MYSQL(C_Key, C_Secret, A_Token, A_Token_Secret, Max_Tweets= np.inf, Filters = ["Bitcoin"], Table_Name = "tmp_tweets", *args, **kwargs):

    """
   
    C_Key,
        String - Twitter Consumer Key
    C_Secret, 
        String - Twitter Consumer Secret
    A_Token,
        String - Twitter Access Token
    A_Token_Secret, 
        String - Twitter Access Token Secret
    Max_Tweets= 100,
        INT - Number of tweets to extract
    Filters = None, 
        List(String) - What words to filter on
        Default - ["Trump"]
    Table_Name = None,
        String - The name of your new table (default is tmp_tweets)
    New_Table_Columns = "(date DATETIME, username VARCHAR(20), tweet VARCHAR(280))",
        List(String) - SQL format tuple of string pairs for column name and type e.g. ['time DATETIME', 'age INT(2)']'
    Tweet_Data_Parts = None
        List(String/List(String)) - Parts of the tweet json (according to twitter) to extract e.g. [{"user":"screen_name"}, text'] is default
        Time is automatically added in to database
    Temporary = True,
        Bool - Store Tweets in the Database temporarily or permanently
        Default = True
    
    """
    #### Check if Mysql server is running
    # exit_code = subprocess.check_call(["mysql", "server" ,"start"], shell =True)
    # if exit_code ==0:
    #     pass

    # else:
    #     raise Warning("Mysql server did not start, may want to start server manually")


    # time.sleep(5)

    auth = tweepy.OAuthHandler(consumer_key=C_Key, consumer_secret=C_Secret)
    auth.set_access_token(A_Token, A_Token_Secret)

    db_connection = MySQLdb.connect("127.0.0.1","root", "redcard", "tweet_store", charset = 'utf8mb4')
    cursor = db_connection.cursor()


    if Max_Tweets ==np.inf:
        tweet_add_milestone = 5000
    else:
        tweet_add_milestone = int(Max_Tweets/5)

     # ## Define a class to listen to the twitter API
    # If we want to use twitter data and/or a database other than the default then define this custom listener:
      
    class Stream_Listener(tweepy.StreamListener):
            def __init__(self, api=None, Max_Tweets_=None, Table_Name_=None):
                super().__init__()
                self.num_tweets = 0
                self.max_tweets = Max_Tweets_        
                self.table_name = Table_Name_
                
                # For creating, create table if not default
                # Below line is to hide your warning 
                cursor.execute("SET sql_notes = 0; ")
                # create table here....
            
                exec_stmt = str("CREATE TABLE IF NOT EXISTS " + self.table_name + " (date DATETIME, username VARCHAR(20), text VARCHAR(280), follower_count INTEGER, num_statuses INTEGER, retweet_count INTEGER, fav_count INTEGER, tweet_lang VARCHAR(15)) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;")  
                
                cursor.execute(exec_stmt)
                db_connection.commit()
            #From every tweet recieved extract the relevant data
            def on_data(self, data):
                if self.num_tweets < self.max_tweets:
                    try:
                        all_data = json.loads(data)
                        cur_time = datetime.datetime.strptime(all_data["created_at"], "%a %b %d %H:%M:%S %z %Y")
                        tweet = all_data["text"]
                        username = all_data["user"]["screen_name"]
                        follower_count = int(all_data['user']['followers_count'])
                        num_statuses = int(all_data['user']['statuses_count'])
                        retweet_count = int(all_data['retweet_count'])
                        fav_count = int(all_data['favorite_count'])
                        tweet_lang = all_data['lang']
                        
                        #insert data into MYSQL database
                        exec_stmt = str("INSERT INTO " + self.table_name + " (date, username, text, follower_count, num_statuses, retweet_count, fav_count, tweet_lang) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)")
                                    
                        cursor.execute(exec_stmt, (cur_time, username, tweet, follower_count, num_statuses, retweet_count, fav_count, tweet_lang))
                        #commit the changes into the databse
                        db_connection.commit()
                        #Give periodic feedback on how the streaming is going
                        if self.num_tweets%tweet_add_milestone == 0 or self.num_tweets ==0:    
                            print("Successfully added tweet number:", self.num_tweets +1)
                        self.num_tweets +=1

                        return True
                    except:
                        print('Encountered error but continuing')
                        time.sleep(5)
                        pass
                #After a certain number of tweets finish writing to the database
                else:
                    print("Finished writing to table:", self.table_name)
                    
                    return False 
            #Handle errors in streaming                        
            def on_error(self, status):
                print("Error Code:", status)
                if status ==420:
                    print('Slowing down as being rate limited')
                    time.sleep(10)
                    return True

                else:
                    return True


            def on_timeout(self):
                """Called when stream connection times out"""
                time.sleep(10)
                return True

            def on_limit(self, track):
                """Called when a limitation notice arrives"""
                time.sleep(2)
                print('Being limited and slowing down')
                return True


        #Initialise the stream listener
    listener = Stream_Listener(Max_Tweets_ = Max_Tweets, Table_Name_ = Table_Name)    
    #Authenticate the listener
    data_stream = tweepy.Stream(auth, listener)
    
    #Add filters
    data_stream.filter(track = Filters,async = False)

    #Zombie connection to stop wifi connection causing errors
    # create a zombie
    # while not data_stream.running:
    #     try:
    #         #wait a bit so as not to overload the server
    #         time.sleep(10)
    #         # start stream synchronously
    #         logging.info("Started listening to twitter stream...")
    #         data_stream.filter(track = Filters, async=False)
    #     except (Timeout, SSLError, ReadTimeoutError, ConnectionError) as e:
    #         logging.warning("Network error occurred. Keep calm and carry on.", str(e))
    #     except Exception as e:
    #         logging.error("Unexpected error!", e)
    #     finally:
    #         logging.info("Stream has crashed. System will restart twitter stream!")
    # logging.critical("Somehow zombie has escaped...!")
    

    # subprocess.check_call(["mysql.server", "stop"])

    print("Database Server Successfully written to")

    # # First create the engine to connect to the database
    # engine = sqlalchemy.create_engine('mysql+mysqldb://root:redcard@127.0.0.1/tweet_store')
    # #Set up a metadata object to track table metadata
    # meta_data = sqlalchemy.MetaData()
    # tweet_table = sqlalchemy.Table(Table_Name, meta_data, autoload=True, autoload_with=engine)
    # #Establish the database connection
    # connection = engine.connect()

    #Establish Connection
    db_connection = MySQLdb.connect("127.0.0.1","root", "redcard", "tweet_store", charset = 'utf8mb4')
    db_cursor = db_connection.cursor()
    db_cursor.execute('SELECT COUNT(*) FROM '+Table_Name)
    totaltweets = db_cursor.fetchall()[0]

    return totaltweets

#Define what currencies we want to track
Currencies = ['Bitcoin','Ethereum','Ripple','Litecoin', 'Monero', 'Cardano']

Filters = [['#'+str(currency), '#'+str(currency)+'s',str(currency), str(currency)+'s'] for currency in Currencies]
Filters = [fil for cur_fils in Filters for fil in cur_fils]
Filters += ['BTC','#BTC','LTC','#LTC','ETH','#ETH','ZEC','#ZEC', 'XRP','#XRP', 'XMR','#XMR']

#Set up authentication access codes
C_Key = 'ILYTd5Abkw85OKTd5sAbSpPdC'
C_Secret='1hHTjqyq5Kitt3b5gka6uPMkvKzuz5wO7C63HcRLP5v2mHUtz6'

A_Token='2758135350-UfWPEgJPQJTCHvCQRQEzcavyl45mcwwLkRnWWBi'
A_Token_Secret='LKa0IlzGhWdLKgzhATUtFx0kM5AGG6lAWBUoOEP1lKj9g'



while True:
    tweets_so_far = Stream_Twitter_Data_MYSQL(C_Key, C_Secret, A_Token, A_Token_Secret, Max_Tweets= np.inf, Filters = Filters, Table_Name = "Crypto_tweets")
    print('Restarting stream from',tweets_so_far,'tweets')
    #Clear MySQL connections
    
    #Delete PID file after
    import os
    try:
        os.remove(r'C:\ProgramData\MySQL\MySQL Server 5.7\Uploads\tmp.txt')
    except:
        pass

    db_connection = MySQLdb.connect("127.0.0.1","root", "redcard", "tweet_store", charset = 'utf8mb4')
    db_cursor = db_connection.cursor()
    db_cursor.execute("SELECT concat(id,',') FROM information_schema.processlist WHERE USER='root' AND TIME > 200 INTO OUTFILE 'C:/ProgramData/MySQL/MySQL Server 5.7/Uploads/tmp.txt'")
    db_connection.commit()

    with open(r'C:\ProgramData\MySQL\MySQL Server 5.7\Uploads\tmp.txt') as f:
        pids = list(f.readline().split(','))

    pids = [int(x) for x in pids[:-1]]

    for pid in pids:
        db_cursor.execute("KILL "+ str(pid))

    db_connection.commit()

    #Delete PID file after
    try:
        os.remove(r'C:\ProgramData\MySQL\MySQL Server 5.7\Uploads\tmp.txt')
    except:
        pass



# db_con = Stream_Twitter_Data_MYSQL(C_Key, C_Secret, A_Token, A_Token_Secret, Max_Tweets= np.inf, Filters = Filters, Table_Name = "Crypto_tweets")


# class Twitter_Listener(object):
#     def Start_listening(self):
#         try:
#             db_con = Stream_Twitter_Data_MYSQL(C_Key, C_Secret, A_Token, A_Token_Secret, Max_Tweets= np.inf, Filters = Filters, Table_Name = "Crypto_tweets")
#         except:
#             time.sleep(10)
#             self.Start_listening()


# db_con = Twitter_Listener().Start_listening()
    
