'''
Created on 02/07/2015

@author: Mota
'''
import string
import os
import utils.nlp.tokenizer as tokenizer

from sklearn.feature_extraction.text import TfidfVectorizer

'''
Keywords are obtained based of tfidf score.
The topN words from each document are returned
'''
def getkeywords(path, n, customSWFile):
    token_dict = {}
    for subdir, dirs, files in os.walk(path):
        for file in files:
            file_path = subdir + os.path.sep + file
            #shakes = codecs.open(file_path, 'r', encoding='utf-8')
            with open (file_path, "r") as docFile:
                    text = docFile.read()
            token_dict[file] = text
            
    #this can take some time
    tf = TfidfVectorizer(tokenizer=tokenizer.tokenize, stop_words='english')
    tfidf_matrix = tf.fit_transform(token_dict.values())
    feature_names = tf.get_feature_names() 
    dense = tfidf_matrix.todense()
    keywords = []
    for i in range(0, len(dense)-1):
        doc = dense[i].tolist()[0]
        doc_scores = [pair for pair in zip(range(0, len(doc)), doc) if pair[1] > 0]
        sorted_doc_scores = sorted(doc_scores, key=lambda t: t[1] * -1)
        for word, score in [(feature_names[word_id], score) for (word_id, score) in sorted_doc_scores][:n]:
            #print('{0: <20} {1}'.format(word, score))
            keywords.append(word)
    return set(keywords)
