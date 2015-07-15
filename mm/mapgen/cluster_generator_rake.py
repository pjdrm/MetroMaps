'''
Created on 14/07/2015

@author: Mota
'''
import logging
import json
import rake
import metro_station_generator
import utils.nlp.stopwords as stopwords

class ClusterGeneratorRAKE(metro_station_generator.MetroStationGenerator):
    '''
    Generates cluster candidates (aka metro stations) based on the RAKE keywords extraction algorithm
    '''
    def __init__(self, config):
        super(ClusterGeneratorRAKE, self).__init__(config)
        self.minchars = int(config['minchars'])
        self.maxwords = int(config['maxwords'])
        self.minfreq = int(config['minfreq'])

        self.input_JSON = config['input_json']
        self.output_JSON = config['output_json']
        self.docDir = config['domain'] + "/data/swtext/"
        with open(self.input_JSON) as f_in:
            self.timeslices = json.load(f_in)
            self.num_timeslices = len(self.timeslices)
    
    def mergeDocs(self, timeslice):
        mergedDocs = ""
        for timelice_el in timeslice['doc_data']:
            docPath = self.docDir + timelice_el['doc_metadata']['name']
            with open (docPath, "r") as docFile:
                mergedDocs += docFile.read() + '\n'
        return mergedDocs
    
    def genDicFromClust(self, cluster):
        clust_dic = {}
        clust_dic['k'] = 1
        clust_tokens = []
        for el in cluster:
            clust_tokens.append(el)
        clust_dic['cluster_tokens'] = clust_tokens
        return [clust_dic]
    
    def run(self):
        logging.debug('RAKE based Metro Station Generator: run begin')
        self.timeslice_clusters = {}
        for i in range(self.num_timeslices):
            current_timeslice = self.timeslices[i]
            mergedDocs = self.mergeDocs(current_timeslice)
            rake_object = rake.Rake(stopwords.swList(), self.minchars, self.maxwords, self.minfreq)
            cluster = rake_object.run(mergedDocs)
            clust = [tup[0] for tup in cluster]
            self.timeslice_clusters[i] = self.genDicFromClust(clust)
            
def construct(config):
    return ClusterGeneratorRAKE(config)  