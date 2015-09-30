'''
Provides the interface for the specific DBSCAN algorithm
to be used for clustering.

To use this algorithm, in the .yaml configuration write the name of this module.
(slicing: type: slicing_dbscan)

@author: Mota
'''
import numpy as np
from sklearn.cluster import DBSCAN
import mm.input.slicing.clustering.slicing_cluster_based as slicing_cluster_based
from sklearn.metrics.pairwise import cosine_similarity

class SlicingDBSCAN(slicing_cluster_based.SlicingClusterBased):
    def __init__(self, legacy_helper_config_dict):
        super(SlicingDBSCAN, self).__init__(legacy_helper_config_dict)
        self.eps = 0.1
        self.min_samples = 2
        self.metric = cosine_similarity
        
    def dbscan(self, samples):
        db = DBSCAN(eps = self.eps, min_samples = self.min_freq_in_doc, metric = self.metric).fit(samples)
        return db.labels_
    
    def run(self):
        return self.dbscan(self.cluster_elms)
        
def construct(config):
    return SlicingDBSCAN(config) 
