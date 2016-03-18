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
from math import exp
from sklearn import metrics
import mm.input.slicing.clustering.utils.similairty_metrics as similairty_metrics

#var = 100

class SlicingAgglomerativeClustering(slicing_cluster_based.SlicingClusterBased):
    def __init__(self, slicer_configs):
        super(SlicingAgglomerativeClustering, self).__init__(slicer_configs)
        self.linkage = slicer_configs["clustering"]["linkage"]
        self.metric = slicer_configs["clustering"]["metric"]
        self.desc = "Agglomerative linkage: " + self.linkage + " metric: " + self.metric
        if self.metric == "gaussian":
            self.var = slicer_configs["clustering"]["var"]
            self.desc += " var: " + str(self.var)
            similairty_metrics.var = self.var
        
    def agglomerative_clustering(self, samples):
        affinityArg = self.metric
        if self.metric == "gaussian":
            affinityArg = similairty_metrics.gaussianSimGraph
            
        ac = AgglomerativeClustering(linkage = self.linkage, n_clusters=self.num_clusters, affinity = affinityArg)
        ac.fit(samples)
        return ac.labels_
    
    def run(self):
        return self.agglomerative_clustering(self.cluster_elms)
    
def construct(config):
    return SlicingAgglomerativeClustering(config)
