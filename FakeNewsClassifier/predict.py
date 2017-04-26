import tensorflow as tf
import model
import json
from model import NewsClassifier
from TweetCollector import TweetCollector
import urllib
import re
import newspaper
from newspaper import Article
import tweepy
import nltk
from nltk import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn import model_selection
import numpy as np

pattern = re.compile(r'http(?:s)?:\/\/(?:www\.)?t\.co\/([a-zA-Z0-9_]+)')

def GetArticle(tweet):
    status = tweet.text.encode('utf8')
    link = re.findall(r'(http(?:s)?:\/\/(?:www\.)?t\.co\/[a-zA-Z0-9_]+)',status)[0]
    print status
    print link
    a = Article(link)
    a.download()
    a.parse()
    text = a.text.encode('utf8')
    return text

def Clean(text):
    tokenizer = RegexpTokenizer(r'\w+')
    lemma = WordNetLemmatizer()
    stop = set(stopwords.words('english'))
    tokens = tokenizer.tokenize(text)
    cleaned = []
    for token in tokens:
        if token not in stop:
            token = token.lower()
            token = lemma.lemmatize(token)
            cleaned.append(token)
    clean_text = ' '.join(cleaned)
    return clean_text

def EncodeText(text):
    layer = np.zeros((1,len(word2index)),dtype=float)
    for word in text.split(' '):
        if word in word2index.keys():
            layer[0][word2index[word]] = 1
    return layer


word2index = json.load(open('vocab_index.json','r'))

n_input = len(word2index)
n_hidden_1 = 10
n_hidden_2 = 5
n_classes = 2
tweet = TweetCollector.getTweet()
text = GetArticle(tweet)
clean_text = Clean(text)
input_tensor = tf.placeholder(tf.float32,[None, n_input], name='input')
classification = {0:"True",1:"Fake"}

weights = {
    'h1': tf.Variable(tf.random_normal([n_input,n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1,n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2,n_classes]))
}

biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess,'tmp/model.ckpt')
    prediction = sess.run(NewsClassifier.multilayer_perceptron(input_tensor,weights,biases), feed_dict={input_tensor:EncodeText(clean_text)})
    softmax = tf.nn.softmax(prediction)
    print '0: True\n1:Fake'
    print sess.run(tf.arg_max(softmax,1))
