'''
Provides the interface for the specific Kmeans algorithm
to be used for clustering.

To use this algorithm, in the .yaml configuration write the name of this module.
(slicing: type: slicing_kmeans)

@author: Mota
'''
import numpy as np
from sklearn.cluster import KMeans
import mm.input.slicing.clustering.slicing_cluster_based as slicing_cluster_based

class SlicingKmeans(slicing_cluster_based.SlicingClusterBased):
    def __init__(self, slicer_configs):
        super(SlicingKmeans, self).__init__(slicer_configs)
        
    def kmeans(self, samples):
        km = KMeans(n_clusters=self.num_clusters)
        km.fit(samples)
        return km.labels_
    
    def run(self):
        return self.kmeans(self.cluster_elms)
        
def construct(config):
    return SlicingKmeans(config) 