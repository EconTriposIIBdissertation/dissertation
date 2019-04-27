import pandas as pd 
import numpy as np 
import MySQLdb
import praw
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from praw.models import MoreComments
import datetime


def Analyse_threads_sentiment():

	db_connection = MySQLdb.connect("127.0.0.1","root", "redcard", "tweet_store", charset = 'utf8mb4')

	#import all reddit data for each currency
	reddits = []
	Currencies = ['Bitcoin','Ethereum','Ripple','Litecoin', 'Monero', 'Cardano']
	for currency in Currencies:
	    tmp = pd.read_sql('SELECT * FROM crypto_reddits_%s;'%currency.lower(), db_connection)
	    reddits.append(tmp)

	##Set up reddit connection
	CLIENT_ID = 'tBTveYnvHH1xsw'
	CLIENT_SECRET = 'WDIOGdOp3vWUH40FnxeMilEzt2U'
	REDIRECT_URI = 'https://localhost:8889/reddit'
	reddit = praw.Reddit(client_id=CLIENT_ID,
	                     client_secret=CLIENT_SECRET,
	                     user_agent='api_accessor',
	                     username='secret',
	                     password='redcard')
	##Initialise list to store all new redit data for each currency
	reddit_data = []
	## Initialise sentiment analyser
	sentiment_analyser = SentimentIntensityAnalyzer()
	#iterate over currencies
	for idx, currency in enumerate(Currencies):
		##Initialise list to store all new redit data for specific currency
		currency_reddit_data = []
		#remove duplicate id's
		reddits[idx] = reddits[idx].groupby('id').first()

		#for each unique id
		for iteration,unique_id in enumerate(reddits[idx].index):
			#find the thread
			submission = reddit.submission(id=unique_id)
			#quickly save the sentiment of the title
			title_sentiment = sentiment_analyser.polarity_scores(submission.title)
			#for each comment analyse sentiment + record (un)weighted score
			#initialise a list to store values
			comment_variables = []
			for comment in submission.comments.list():
				if isinstance(comment, MoreComments):	
					continue
				sentiment = sentiment_analyser.polarity_scores(comment.body)
				score = comment.score

				comment_variables.append([sentiment['compound'],sentiment['pos'],sentiment['neg'],sentiment['neu'], score])

			#convert into dataframe for easier analysis 
			tmp_df = pd.DataFrame(comment_variables)
			#some threads can have no comments
			if not tmp_df.empty:
				tmp_df.columns = ['compound','pos','neg','neu','score']
				#Convert all comment scores into single thread score
				thread_UW_sent = tmp_df['compound'].mean()
				thread_median_sent = tmp_df['compound'].median()
				#Weighted mean score requires weighting sum:
				tmp_df['weights'] = (tmp_df['score'] +1)/tmp_df['score'].sum()

				thread_W_sent = (tmp_df['weights'] *tmp_df['compound']).sum()
			else:
				(thread_UW_sent, thread_W_sent, thread_median_sent) = (np.nan, np.nan, np.nan)


			#store appropriate data
			if iteration % 10 == 0:
				print(iteration)
			new_data_entry = list(reddits[idx].loc[unique_id,:][['created','score','title','comms_num']].values) + [unique_id, title_sentiment, thread_UW_sent, thread_W_sent, thread_median_sent]



			# add the entry to appropriate currency
			currency_reddit_data.append(new_data_entry)



		# add currency data to all currencies
		reddit_data.append(currency_reddit_data)


def Analyse_date_sentiment():

	db_connection = MySQLdb.connect("127.0.0.1","root", "redcard", "tweet_store", charset = 'utf8mb4')

	#import all reddit data for each currency
	reddits = []
	Currencies = ['Bitcoin','Ethereum','Ripple','Litecoin', 'Monero', 'Cardano']
	for currency in Currencies:
	    tmp = pd.read_sql('SELECT * FROM crypto_reddits_%s;'%currency.lower(), db_connection)
	    reddits.append(tmp)

	##Set up reddit connection
	CLIENT_ID = 'tBTveYnvHH1xsw'
	CLIENT_SECRET = 'WDIOGdOp3vWUH40FnxeMilEzt2U'
	REDIRECT_URI = 'https://localhost:8889/reddit'
	reddit = praw.Reddit(client_id=CLIENT_ID,
	                     client_secret=CLIENT_SECRET,
	                     user_agent='api_accessor',
	                     username='secret',
	                     password='redcard')
	##Initialise list to store all new redit data for each currency
	reddit_data = []
	## Initialise sentiment analyser
	sentiment_analyser = SentimentIntensityAnalyzer()
	#iterate over currencies
	for idx, currency in enumerate(Currencies):
		##Initialise dict to store all new redit data for specific currency according to a date
		currency_reddit_data = {}

		#remove duplicate id's
		reddits[idx] = reddits[idx].groupby('id').first()

		#for each unique id
		for iteration,unique_id in enumerate(reddits[idx].index):
			#find the thread
			submission = reddit.submission(id=unique_id)
			# #quickly save the sentiment of the title - this is irrelevant data in current algorithm but may be useful if needed
			# title_sentiment = sentiment_analyser.polarity_scores(submission.title)
			
			## for each comment analyse sentiment + record (un)weighted score
			# deal with comment replies also
			submission.comments.replace_more(limit=None)
			for comment in submission.comments.list():
				
				#check we won't throw errors if we encounter a morecomments object
				if isinstance(comment, MoreComments):	
					continue
				
				#analyse for sentiment
				sentiment = sentiment_analyser.polarity_scores(comment.body)
				score = comment.score
				comment_date = datetime.datetime.fromtimestamp(comment.created).date().isoformat()
				
				#check if a record of data for given date exists in our collection dictionary
				if not comment_date in currency_reddit_data:
					currency_reddit_data[comment_date] = []
				
				#add data to associated date
				comment_variables = [sentiment['compound'],sentiment['pos'],sentiment['neg'],sentiment['neu'], score]
				currency_reddit_data[comment_date].append(comment_variables)
				
			# Give updates on progress
			if iteration % 10 == 0:
				print('Thread iteration:', iteration,'for currency:', currency)

		#convert each date's data into dataframe for easier analysis 
		for k,v in currency_reddit_data.items():
			
			tmp_df = pd.DataFrame.from_records(v)
		
			tmp_df.columns = ['compound','pos','neg','neu','score']
			## Convert all comment scores into single thread score
			# unweighted and median sentiment
			thread_UW_sent = tmp_df['compound'].mean()
			thread_median_sent = tmp_df['compound'].median()
			
			# Weighted mean score requires weighting sum:
			tmp_df['weights'] = (tmp_df['score'] +1)/tmp_df['score'].sum()
			thread_W_sent = (tmp_df['weights'] *tmp_df['compound']).sum()

			# Unweighted positive, negative and neutral sentiments
			pos_sent = tmp_df['pos'].mean()
			neu_sent = tmp_df['neu'].mean()
			neg_sent = tmp_df['neg'].mean()

			# Net upvotes for date
			total_score = tmp_df['score'].sum()

			currency_reddit_data[k] = [thread_UW_sent, thread_median_sent, thread_W_sent, pos_sent, neu_sent, neg_sent, total_score]

		# add currency data to all currencies
		reddit_data.append(currency_reddit_data)

	return reddit_data
