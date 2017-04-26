#Import the necessary methods from tweepy library
import tweepy
from tweepy import OAuthHandler
import csv
import datetime

class TweetCollector(object):
    @staticmethod
    def getTweet():
        access_token = "412017355-iZP4Irs384FswKlGG4FZ2uOrCDLHkAugUlJA65rB"
        access_token_secret = "plEbvyYbyHePmd9u7SddTTpJYGvsuULXqrMNxzwEVWUDM"
        consumer_key = "G7clrfKyn0jXWABMCJu3X8afD"
        consumer_secret = 	"7I0scApAk0tcnEfRMqGj4GDQ06DJK64MZwC4rZewHLRjQUzEJJ"

        auth = OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token,access_token_secret)
        api = tweepy.API(auth)

        for tweet in tweepy.Cursor(api.search,
                q="#fakenews -RT filter:links",
                count=100, lang="en").items():
            return tweet
