'''
Provides the interface for the specific Mean-Shift algorithm
to be used for clustering.

To use this algorithm, in the .yaml configuration write the name of this module.
(slicing: type: slicing_mean_shift)

@author: Mota
'''
import mm.input.slicing.clustering.slicing_cluster_based as slicing_cluster_based
from sklearn.cluster import MeanShift

class SlicingMeanShift(slicing_cluster_based.SlicingClusterBased):
    def __init__(self, legacy_helper_config_dict):
        super(SlicingMeanShift, self).__init__(legacy_helper_config_dict)
        self.bandwidth = 14
        #self.bandwidth = estimate_bandwidth(elements, quantile=0.3, n_samples=None)
        
    def mean_shift(self, samples):
        ms = MeanShift(bandwidth=self.bandwidth, bin_seeding=True)
        ms.fit(samples)
        return ms.labels_
    
    def run(self):
        return self.mean_shift(self.cluster_elms)

def construct(config):
    return SlicingMeanShift(config) 
