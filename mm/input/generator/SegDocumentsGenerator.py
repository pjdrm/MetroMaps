'''
Created on 02/07/2015

Produces the necessary files for input to the MetroMaps
from segmented documents.

whitelist: generated from topN tfidf words in each document

@author: Mota
'''

import glob
import json
import string
import re
import os
import utils.myutils

class SegDocumentsGenerator(object):
    '''
    classdocs
    '''
    def __init__(self, input_gen_config_dic):
        self.docsDir = input_gen_config_dic['input_directory']
        self.domain  = input_gen_config_dic['domain']
        
    def run(self):
        utils.myutils.rmDir(self.domain)
        os.makedirs(self.domain + "/data/")
        os.makedirs(self.domain + "/data/rawtext/")
        os.makedirs(self.domain + "/data/swtext")
        os.makedirs(self.domain + "/out/")
        os.makedirs(self.domain + "/out/final")
        dicts = []
        docStr = ''
        i = 1
        
        for doc in glob.glob(self.docsDir+"/*"):
            docStr = ""
            with open (doc, "r") as docFile:
                docStr += docFile.read()
            docStr = filter(lambda x: x in string.printable, docStr)
            segments = re.compile("==========\n").split(docStr)
            cleanSeg = segments[len(segments)-1].replace("==========", "")
            segments[len(segments)-1] = cleanSeg
            for j in range(1, len(segments)):
                with open(self.domain + "/data/rawtext/"+ str(i) + ".txt", "w") as text_file:
                    text_file.write(segments[j])
                dicts.append({"timestamp": i, "id": i, "name": str(i) + ".txt"})
                i += 1
            
        with open(self.domain + '/data/doc_meta.json', 'w') as outfile:
            json.dump(dicts, outfile)
            
        print "Done generating the " + self.domain + " domain"
        
def construct(config):
    return SegDocumentsGenerator(config)  
        