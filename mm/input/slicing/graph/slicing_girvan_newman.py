'''
Provides the interface for the specific Girvan-Newman algorithm
to be used for word community detection.

To use this algorithm, in the .yaml configuration write the name of this module.
(slicing: type: slicing_girvan_newman)

@author: Mota
'''
import os
import snap
import mm.input.slicing.graph.slicing_graph_based as slicing_graph_based

class SlicingGirvanNewman(slicing_graph_based.SlicingGraphBased):
    def __init__(self, slicer_configs):
        super(SlicingGirvanNewman, self).__init__(slicer_configs)
        self.g = self.createGraph()
        self.debugDir = 'resources/slicing_results/girvan_newman/'
        self.debugFile = self.debugDir + self.wc_des + ".txt"
    
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
        communities = self.girvan_newman()
        if not os.path.exists(self.debugDir):
            os.makedirs(self.debugDir)
        self.print_communities(communities, self.debugFile)
        return communities
    
def construct(config):
    return SlicingGirvanNewman(config)  
