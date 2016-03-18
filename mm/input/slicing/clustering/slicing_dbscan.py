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
import mm.input.slicing.clustering.utils.similairty_metrics as similairty_metrics

class SlicingDBSCAN(slicing_cluster_based.SlicingClusterBased):
    def __init__(self, slicer_configs):
        super(SlicingDBSCAN, self).__init__(slicer_configs)
        self.eps = slicer_configs["clustering"]["eps"]
        self.min_samples = slicer_configs["clustering"]["min_samples"]
        self.metric = slicer_configs["clustering"]["metric"]
        self.desc = "dbscan minPts: " + str(self.min_samples) + " eps: " + str(self.eps) + " metric: " + self.metric
        if self.metric == "gaussian":
            self.var = slicer_configs["clustering"]["var"]
            self.desc += " var: " + str(self.var)
            similairty_metrics.var = self.var
        
    def dbscan(self, minPts, eps, samples):
        met = self.metric
        if self.metric == "gaussian":
            met = similairty_metrics.gaussianSim
            
        db = DBSCAN(algorithm='brute', eps = eps, min_samples = minPts, metric = met).fit(samples)
        return db.labels_
    
    def run(self):
        return self.dbscan(self.min_samples, self.eps, self.cluster_elms)
        
def construct(config):
    return SlicingDBSCAN(config) 
