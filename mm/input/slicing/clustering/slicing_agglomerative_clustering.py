'''
Provides the interface for the specific Agglomerative-Clustering algorithm
to be used for clustering.

To use this algorithm, in the .yaml configuration write the name of this module.
(slicing: type: slicing_agglomerative_clustering)

@author: Mota
'''
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import mm.input.slicing.clustering.slicing_cluster_based as slicing_cluster_based

class SlicingAgglomerativeClustering(slicing_cluster_based.SlicingClusterBased):
    def __init__(self, slicer_configs):
        super(SlicingAgglomerativeClustering, self).__init__(slicer_configs)
        self.linkage = 'ward'
        
    def agglomerative_clustering(self, samples):
        ac = AgglomerativeClustering(linkage = self.linkage, n_clusters=7)
        ac.fit(samples)
        return ac.labels_
    
    def run(self):
        return self.agglomerative_clustering(self.cluster_elms)
        
def construct(config):
    return SlicingAgglomerativeClustering(config)
