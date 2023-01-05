#import snscrape as sntwitter
import pandas as pd
import numpy as np
import os
import snscrape.modules.twitter as sntwitter

# Creating list to append tweet data to
tweets_list2 = []
ii=0
# Using TwitterSearchScraper to scrape data and append tweets to list
for i, tweet in enumerate(sntwitter.TwitterSearchScraper('covid-19 lang:id (-filter:replies -filter:retweet) until:2021-12-31 since:2021-12-01').get_items()):

    if tweet.user.followersCount >=1000 and tweet.likeCount >= 100 and tweet.retweetCount >= 100 and tweet.replyCount >= 100:
        ii+=1
        print(ii,tweet.date,tweet.user.followersCount,tweet.likeCount,tweet.retweetCount,tweet.replyCount)
        tweets_list2.append(
            [tweet.date, tweet.id, tweet.rawContent, tweet.user.username])    

# Creating a dataframe from the tweets list above
tweets_df2 = pd.DataFrame(tweets_list2, columns=[
                          'Datetime', 'Tweet Id', 'Text', 'Username'])

os.system('clear')
tweets_df2.to_csv('dataset.csv',index = True)
print(tweets_df2['Text'])