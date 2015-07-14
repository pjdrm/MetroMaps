'''
Created on 14/07/2015

@author: Mota
'''
import string
import glob
import utils.nlp.tokenizer as tokenizer
import utils.nlp.tfidf as tfidf
import utils.nlp.stopwords as stopwords

class TextPreprocessing(object):
    '''
    classdocs
    
    Performs the following preprocessing tasks:
        - steming
        - stop word removal
        - remove punctuation
        - lower case
    These tasks can be enabled/disabled through the input_preprocessing part of the .yalm config file
    It also generates the whitelist.txt
    '''
    def __init__(self, input_preprocessing_configs):
        self.input_preprocessing_configs = input_preprocessing_configs
        self.domain  = input_preprocessing_configs['domain']
        self.n  = input_preprocessing_configs['n']
        self.customSWFile = input_preprocessing_configs['custom_stop_words']
        self.raw_text = self.domain + "/data/rawtext/"
        self.white_list = self.domain + '/data/whitelist.txt'
        
    def stem_input(self, docStr):
        stemDoc = tokenizer.tokenize(docStr)
        return " ".join(stemDoc)
                
    def remove_stop_words(self, docStr):
        no_sw_doc = stopwords.removeStopWords(docStr, self.input_preprocessing_configs['custom_stop_words'])
        return no_sw_doc
    
    def remove_punctuation(self, docStr):
        return docStr.translate(None, string.punctuation+'0123456789')
    
    def lower_case(self, docStr):
        return docStr.lower()
                
    '''
    Generated the list of words that can appear on a metro station (whitelist)
    '''
    def mkwhitelist(self, outPath):
        keywords = tfidf.getkeywords(self.raw_text, self.n, self.customSWFile)
        with open(outPath, 'w') as file:
            for keyword in keywords:
                file.write("{}\n".format(keyword))
        
    def run(self):
        for doc in glob.glob(self.raw_text + "/*"):
            with open (doc, "r+") as docFile:
                docStr = docFile.read()
                docStr = filter(lambda x: x in string.printable, docStr)
                
                if(self.input_preprocessing_configs['lower_case']):
                    docStr = self.lower_case(docStr)
                    
                if(self.input_preprocessing_configs['remove_punctuation']):
                    docStr = self.remove_punctuation(docStr)
                    
                if(self.input_preprocessing_configs['steming']):
                    tokenizer.stemTokens()
                    docStr = self.stem_input(docStr)
        
                if(self.input_preprocessing_configs['remove_stopwords']):
                    docStr = self.remove_stop_words(docStr)
                    
                docFile.seek(0)
                docFile.write(docStr)
                docFile.truncate()
            
        if(self.input_preprocessing_configs['gen_whitelist']):
            self.mkwhitelist(self.white_list)
        