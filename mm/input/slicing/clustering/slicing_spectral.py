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

class SlicingSpectral(slicing_cluster_based.SlicingClusterBased):
    def __init__(self, slicer_configs):
        super(SlicingSpectral, self).__init__(slicer_configs)
        ranges = slicer_configs["clustering"]["varRange"]
        self.varRange = range(ranges[0], ranges[1])
        
    def genSimGraph(self, cluster_elms, var):
        sim_graph = np.zeros([cluster_elms.shape[0], cluster_elms.shape[0]])
        for i, j in np.ndindex(sim_graph.shape):
            sim = self.gaussianSim(cluster_elms[i], cluster_elms[j], var)
            if sim >= 0:
                sim_graph[i, j] = sim
        return sim_graph
            
    def gaussianSim(self, xi, xj, var):
        return exp((-1 * np.linalg.norm(xi - xj) ** 2) / (2 * var))
        
        
    def spectral(self, sim_graph):
        labels = spectral_clustering(sim_graph, n_clusters=self.num_clusters, eigen_solver='arpack')
        return labels
    
    def run(self):
        riBest = -1.0
        labels = None
        for var in self.varRange:
            sim_graph = self.genSimGraph(self.cluster_elms, var)
            try:
                labels = self.spectral(sim_graph)
                ri = metrics.adjusted_rand_score(self.true_labels, labels)
                if ri >= riBest:
                    riBest = ri
                    bestLabels = labels
            except Exception: 
                pass
        return bestLabels
        
def construct(config):
    return SlicingSpectral(config) 