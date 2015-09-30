'''
Provides the interface for the specific Affinity-Propagtion algorithm
to be used for clustering.

To use this algorithm, in the .yaml configuration write the name of this module.
(slicing: type: slicing_affinity_propagation)

@author: Mota
'''
import numpy as np
from sklearn.cluster import AffinityPropagation
import mm.input.slicing.clustering.slicing_cluster_based as slicing_cluster_based

class SlicingAffinityPropagation(slicing_cluster_based.SlicingClusterBased):
    def __init__(self, legacy_helper_config_dict):
        super(SlicingAffinityPropagation, self).__init__(legacy_helper_config_dict)
        self.damping = 0.5
        self.preference = -100
        
    def affinity_propagation(self, samples):
        af = AffinityPropagation(damping = self.damping, preference = self.preference).fit(samples)
        return af.labels_
    
    def run(self):
        return self.affinity_propagation(self.cluster_elms)

def construct(config):
    return SlicingAffinityPropagation(config) 