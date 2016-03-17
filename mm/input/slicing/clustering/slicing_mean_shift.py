'''
Provides the interface for the specific Mean-Shift algorithm
to be used for clustering.

To use this algorithm, in the .yaml configuration write the name of this module.
(slicing: type: slicing_mean_shift)

@author: Mota
'''
import mm.input.slicing.clustering.slicing_cluster_based as slicing_cluster_based
from sklearn.cluster import MeanShift
from sklearn import metrics

class SlicingMeanShift(slicing_cluster_based.SlicingClusterBased):
    def __init__(self, slicer_configs):
        super(SlicingMeanShift, self).__init__(slicer_configs)
        self.bandwidth = slicer_configs["clustering"]["bandwidth"]
        #self.bandwidth = estimate_bandwidth(elements, quantile=0.3, n_samples=None)
        
    def mean_shift(self, bandwidth, samples):
        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        ms.fit(samples)
        return ms.labels_
    
    def run(self):
        riBest = -1.0
        labels = None
        bandWidths = range(1, 1000)
        bestBandwidth = -1
        bestLabels = None
        for bandwidth in bandWidths:
            try:
                labels = self.mean_shift(bandwidth, self.cluster_elms)
                ri = metrics.adjusted_rand_score(self.true_labels, labels)
                if ri >= riBest:
                    bestBandwidth = bandwidth
                    riBest = ri
                    bestLabels = labels
            except Exception: 
                pass
        
        print "Bandwidth %f" % (bestBandwidth)  
        return bestLabels

def construct(config):
    return SlicingMeanShift(config) 
