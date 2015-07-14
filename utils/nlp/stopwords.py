'''
Created on 13/07/2015

@author: Mota
'''
import nltk

# TODO: load stopwords list only once
def removeStopWords(text, customSWFile):
    nltkStopWords = set(nltk.corpus.stopwords.words("english"))
    with open(customSWFile) as f:
        customSW = [x.strip('\n') for x in f.readlines()]
    nltkStopWords.update(customSW)
    new_txt = ' '.join([word for word in text.split(' ') if word not in nltkStopWords]) 
    return new_txt
