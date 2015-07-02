'''
Created on 02/07/2015

@author: Mota
'''

import nltk
from nltk.stem.porter import PorterStemmer

stem_flag = False

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(text):
    tokens = nltk.word_tokenize(text)
    global stem_flag
    if stem_flag:
        stems = stem_tokens(tokens, PorterStemmer())
        return stems
    else:
        return tokens

def stemTokens():
    global stem_flag
    stem_flag = True