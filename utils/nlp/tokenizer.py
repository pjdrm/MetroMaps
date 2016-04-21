'''
Created on 02/07/2015

@author: Mota
'''

import nltk
from nltk.stem.porter import PorterStemmer

stem_flag = False
stemMap = {}

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stem_item = stemmer.stem(item)
        if not item in stemMap:
            stemMap[item] = stem_item 
        stemmed.append(stem_item)
    return stemmed

def tokenize(text):
    tokens = nltk.word_tokenize(text)
    global stem_flag
    if stem_flag:
        stems = stem_tokens(tokens, PorterStemmer())
        return stems
    else:
        return tokens
    
def writeStemMap(filePath):
    smStr = ""
    for word in stemMap:
        smStr += word + " -> " + stemMap[word] + "\n"
    with open(filePath, "w+") as smFile:
        smFile.write(smStr)

def stemTokens():
    global stem_flag
    stem_flag = True