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
from sklearn import metrics

class SlicingDBSCAN(slicing_cluster_based.SlicingClusterBased):
    def __init__(self, slicer_configs):
        super(SlicingDBSCAN, self).__init__(slicer_configs)
        self.eps = slicer_configs["clustering"]["eps"]
        self.min_samples = slicer_configs["clustering"]["min_samples"]
        self.metric = "cosine"
        
    def dbscan(self, minPts, eps, samples):
        db = DBSCAN(algorithm='brute', eps = eps, min_samples = minPts, metric = self.metric).fit(samples)
        return db.labels_
    
    def run(self):
        riBest = -1.0
        labels = None
        minPtsArray = range(1, 107)
        epsArray = [x * 0.1 for x in range(0, 10)][1:]
        bestEps = ""
        bestMinPts = ""
        for minPts in minPtsArray: 
            for eps in epsArray:
                labels = self.dbscan(minPts, eps, self.cluster_elms)
                ri = metrics.adjusted_rand_score(self.true_labels, labels)
                if ri >= riBest:
                    bestEps = eps
                    bestMinPts = minPts
                    riBest = ri
                    bestLabels = labels
        
        print "MinPts %d Eps %f" % (bestMinPts, bestEps)  
        return bestLabels
        
def construct(config):
    return SlicingDBSCAN(config) 
