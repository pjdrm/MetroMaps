'''
Provides the interface for the specific Spectral Clustering algorithm.

To use this algorithm, in the .yaml configuration write the name of this module.
(slicing: type: slicing_spectral)

@author: Mota
'''
import numpy as np
import mm.input.slicing.clustering.slicing_cluster_based as slicing_cluster_based
from math import exp
from sklearn.cluster import spectral_clustering
from sklearn import metrics
import pylab
from scipy import spatial
import mm.input.slicing.clustering.utils.similairty_metrics as similairty_metrics

class SlicingSpectral(slicing_cluster_based.SlicingClusterBased):
    def __init__(self, slicer_configs):
        super(SlicingSpectral, self).__init__(slicer_configs)
        self.metric = slicer_configs["clustering"]["metric"]
        self.desc = "Spectral metric: " + self.metric
        if self.metric == "gaussian":
            self.var = slicer_configs["clustering"]["var"]
            self.desc += " var: " + str(self.var)
            similairty_metrics.var = self.var
        
    def spectral(self, sim_graph):
        labels = spectral_clustering(sim_graph, n_clusters=self.num_clusters, eigen_solver='arpack')
        return labels
    
    def run(self):
        sim_graph = similairty_metrics.genSimGraph(self.cluster_elms, self.metric)
        try:
            labels = self.spectral(sim_graph)
        except Exception:
            labels = [0]*len(self.cluster_elms)
            pass
        return labels
    
def construct(config):
    return SlicingSpectral(config) 