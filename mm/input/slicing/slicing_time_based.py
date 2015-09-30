
import math
import json
import os.path
import os
import logging
from mm.input.slicing import slicer_factory
from itertools import combinations

class SlicingHandler(slicer_factory.SlicingHandlerGenerator):
    def __init__(self, legacy_helper_config_dict):
        super(SlicingHandler, self).__init__(legacy_helper_config_dict)
        self.global_tokens = self.data['global_tokens']
        self.global_counts = self.data['global_counts']
        self.num_clusters = int(legacy_helper_config_dict['num_timeslices'])
        self.max_token_counts, self.num_docs_with_term = self.token_stats(self.doc_counts)
    
    def slice(self):
        docs = sorted(self.doc_metadata, key= lambda x: int(x['timestamp']))
        clusters = [[] for i in range(self.num_clusters)]
        ''' assign clusters to documents '''
        docs_per_cluster = len(docs) / self.num_clusters
        logging.debug('Expected docs per cluster: %i' % docs_per_cluster)
        for i, doc in enumerate(docs):
        
            clusters[min(i / docs_per_cluster, self.num_clusters - 1)].append(doc)
        logging.debug('All clusters have %i docs; last cluster has %i docs' %(len(clusters[0]), len(clusters[-1])))
        self.write(clusters)

def construct(config):
    return SlicingHandler(config)  




            







        
        







