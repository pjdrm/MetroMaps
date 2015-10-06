'''
Provides the interface for the specific Girvan-Newman algorithm
to be used for word community detection.

To use this algorithm, in the .yaml configuration write the name of this module.
(slicing: type: slicing_girvan_newman)

@author: Mota
'''
import snap
import mm.input.slicing.graph.slicing_graph_based as slicing_graph_based

class SlicingGirvanNewman(slicing_graph_based.SlicingGraphBased):
    def __init__(self, legacy_helper_config_dict):
        super(SlicingGirvanNewman, self).__init__(legacy_helper_config_dict)
        self.g = self.createGraph()
    
    def girvan_newman(self):
        CmtyV = snap.TCnComV()
        snap.CommunityGirvanNewman(self.g, CmtyV)
        community_list = []
        for C in CmtyV:
            cluster_k = []
            for Node in C:
                token = self.token_to_word[Node]
                cluster_k += [token]
            cluster_d = {'cluster_tokens': cluster_k, 'k': self.k}
            community_list += [cluster_d]
        return community_list
    
    def run(self):
        return self.girvan_newman()
    
def construct(config):
    return SlicingGirvanNewman(config)  
