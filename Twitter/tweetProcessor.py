import re

emoticons_str = r"""
    (?:
    [:;=] #eyes
    [0o\-]? #noses
    [D\(\)\]\[\\\OPp] #mouths
    )
"""


regex_str = [
    emoticons_str,
    r'(?:\#[\w_]+[\w\-]+)', #hashtags
    r'(?:\@[[\w_]+[\w\-]+[\w\.]+)', #mentions
    r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', # URLs
    r'(?:(?:\d+,?)+(?:\.?\d+)?)', # numbers
    r"(?:[a-z][a-z'\-_]+[a-z])", # words with - and '
    r'(?:[\w_]+)', # other words
    r'(?:\S)' # anything else
]

emoticons_re = re.compile(r'^' + emoticons_str + '$', re.VERBOSE | re.IGNORECASE)
tokens_re = re.compile(r'('+'|'.join(regex_str)+')', re.VERBOSE | re.IGNORECASE)

def tokenize(s):
    return re.findall(tokens_re, s)

def preprocess(s, lowercase = False):
    tokens = tokenize(s)
    if lowercase:
        tokens = [token if emoticons_re.search(token) else token.lower for token in tokens]
    return tokens



