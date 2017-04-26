import pandas as pd
from pandas import DataFrame
from pandas import Series
import re

import matplotlib
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import tokenize

s1 = Series.from_csv('/Users/cprinz/Developer/MIS375_TwitterProject/fakenews_2-25.csv')
s2 = Series.from_csv('/Users/cprinz/Developer/MIS375_TwitterProject/fakenews_2-26.csv')
s3 = Series.from_csv('/Users/cprinz/Developer/MIS375_TwitterProject/fakenews_2-27.csv')
s4 = Series.from_csv('/Users/cprinz/Developer/MIS375_TwitterProject/fakenews_2-28.csv')

all_tweets = pd.concat([s1,s2,s3,s4])

twitter_handle_re = re.compile(r'@([A-Za-z0-9_]+)')

mention_counts = Series()
for item in all_tweets:
    mentions = twitter_handle_re.findall(item)
    for mention in mentions:
        if mention in mention_counts.keys():
            mention_counts[mention] += 1
        else:
            mention_counts[mention] = 1

mention_counts.sort(ascending = False)
#print mention_counts

mention_counts.plot()
