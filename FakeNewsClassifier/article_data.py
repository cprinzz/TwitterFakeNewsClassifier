import pandas as pd
from pandas import DataFrame
import xml
from xml import etree
from xml.etree import ElementTree as ET
import os
import nltk
from nltk import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn import model_selection

path = '/Users/cprinz/Developer/2017/MIS375_TwitterProject/buzzfeed-webis-fake-news-corpus-2016/'

def xml2df(xml_data):
    all_records = []
    headers = ['title','mainText','orientation','veracity', 'portal',
                'uri', 'link_count', 'paragraph_count', 'quote_count']

    for f in os.listdir(path):
        if f.find('.xml') != -1:
            article = {}
            article['link_count'] = 0
            article['quote_count'] = 0
            article['paragraph_count'] = 0
            tree = ET.parse(path+'/'+f)
            root = tree.getroot()
            for i, child in enumerate(root):
                if child.tag == 'title':
                    article[child.tag] = child.text
                elif child.tag == 'mainText':
                    article[child.tag] = child.text
                elif child.tag == 'orientation':
                    article[child.tag] = child.text
                elif child.tag == 'veracity':
                    article[child.tag] = child.text
                elif child.tag == 'portal':
                    article[child.tag] = child.text
                elif child.tag == 'uri':
                    article[child.tag] = child.text
                elif child.tag == 'links':
                    article['link_count'] += 1
                elif child.tag == 'paragraph':
                    article['paragraph_count'] += 1
                elif child.tag == 'quotes':
                    article['quote_count'] += 1
            all_records.append(article)
        else: continue

    return DataFrame(all_records, columns = headers)

def classify(veracity):
    if veracity == 'mostly true':
        return 0
    elif veracity == 'mixture of true and false':
        return 0
    elif veracity == 'mostly false':
        return 1
    elif veracity == 'no factual content':
        return 1
    return -1

def clean(text):
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

def CreateDataset():
    df = xml2df(path)
    df['target'] = df['veracity'].map(classify)
    df.dropna(inplace=True)
    df['text_clean'] = df['mainText'].map(clean)
    return df

def DatasetToCSV(df, name):
    df.to_csv(name, encoding='utf8')

def GetFake():
    df = xml2df(path)
    df['target'] = df['veracity'].map(classify)
    df.dropna(inplace=True)
    df['text_clean'] = df['mainText'].map(clean)
    print df[df['target'] == 1]['uri'][:10]



DatasetToCSV(CreateDataset(),'news_data3.csv')
