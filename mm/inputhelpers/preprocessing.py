'''
Created on 14/07/2015

@author: Mota
'''
import json
import re
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
        
        if(self.input_preprocessing_configs['expand_contractions']):
            with open(input_preprocessing_configs['contractions_dic_file']) as contractions_dic_file:
                self.contractions_dic =  json.load(contractions_dic_file)
        
    def stem_input(self, docStr):
        stemDoc = tokenizer.tokenize(docStr)
        return " ".join(stemDoc)
                
    def remove_stop_words(self, docStr):
        no_sw_doc = stopwords.removeStopWords(docStr, self.input_preprocessing_configs['custom_stop_words'])
        return no_sw_doc
    
    def remove_punctuation(self, docStr):
        return docStr.encode('utf-8').translate(None, string.punctuation+'0123456789')
    
    def lower_case(self, docStr):
        return docStr.lower()
    
    def expand(self, docStr):
        contractions_re = re.compile('(%s)' % '|'.join(self.contractions_dic.keys()))
        def replace(match):
            return self.contractions_dic[match.group(0)]
        
        return contractions_re.sub(replace, docStr)
                
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
            doc = doc.replace("\\", "/")
            doc_name = doc.split('/')[-1]
            with open (doc, "r+") as docFile, open(self.input_preprocessing_configs['domain'] + '/data/swtext/' + doc_name, "w") as docWithSW:
                docStr = docFile.read()
                docStr = filter(lambda x: x in string.printable, docStr)
                
                if(self.input_preprocessing_configs['expand_contractions']):
                    docStr = self.expand(docStr)
            
                if(self.input_preprocessing_configs['lower_case']):
                    docStr = self.lower_case(docStr)
                    
                if(self.input_preprocessing_configs['remove_punctuation']):
                    docStr = self.remove_punctuation(docStr)
                    
                if(self.input_preprocessing_configs['steming']):
                    tokenizer.stemTokens()
                    docStr = self.stem_input(docStr)
        
                '''
                Major Hack: generating metro stations with RAKE requires the text with stop words,
                since the multi-word candidate generation is based on them.
                '''
                docWithSW.write(docStr)
                
                if(self.input_preprocessing_configs['remove_stopwords']):
                    docStr = self.remove_stop_words(docStr)
                    
                docFile.seek(0)
                docFile.write(docStr)
                docFile.truncate()
          
        #printing stem mapping
        tokenizer.writeStemMap("stemMap.txt")
           
        if(self.input_preprocessing_configs['gen_whitelist']):
            self.mkwhitelist(self.white_list)
        