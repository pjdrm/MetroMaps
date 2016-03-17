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

var = 100

class SlicingAgglomerativeClustering(slicing_cluster_based.SlicingClusterBased):
    def __init__(self, slicer_configs):
        super(SlicingAgglomerativeClustering, self).__init__(slicer_configs)
        self.linkage = slicer_configs["clustering"]["linkage"]
        ranges = slicer_configs["clustering"]["varRange"]
        self.varRange = range(ranges[0], ranges[1])
        
    def agglomerative_clustering(self, samples):
        ac = AgglomerativeClustering(linkage = self.linkage, n_clusters=self.num_clusters, affinity = "cosine")
        ac.fit(samples)
        return ac.labels_
    
    def run(self):
        '''
        riBest = -1.0
        labels = None
        bestVar = 0.0
        for variance in self.varRange:
            global var
            var = variance
            try:
                labels = self.agglomerative_clustering(self.cluster_elms)
                ri = metrics.adjusted_rand_score(self.true_labels, labels)
                if ri >= riBest:
                    riBest = ri
                    bestLabels = labels
                    bestVar = var
            except Exception: 
                pass
        print "Var %f" % bestVar
        return bestLabels
        '''
        return self.agglomerative_clustering(self.cluster_elms)
        
def gaussianSim(xi, xj):
    return exp((-1 * np.linalg.norm(xi - xj) ** 2) / (2 * var))
    
def genSimGraph(cluster_elms):
    sim_graph = np.zeros([cluster_elms.shape[0], cluster_elms.shape[0]])
    for i, j in np.ndindex(sim_graph.shape):
        sim = gaussianSim(cluster_elms[i], cluster_elms[j])
        if sim >= 0:
            sim_graph[i, j] = sim
    return sim_graph
    
def construct(config):
    return SlicingAgglomerativeClustering(config)
