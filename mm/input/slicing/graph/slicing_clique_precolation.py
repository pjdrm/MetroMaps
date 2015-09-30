'''
Provides the interface for the specific Clique-Precolation algorithm
to be used for word community detection.

To use this algorithm, in the .yaml configuration write the name of this module.
(slicing: type: slicing_clique_precolation)
@author: Mota
'''
import snap
import mm.input.slicing.graph.slicing_graph_based as slicing_graph_based

class SlicingCP(slicing_graph_based.SlicingGraphBased):
    def __init__(self, legacy_helper_config_dict):
        super(SlicingCP, self).__init__(legacy_helper_config_dict)
        self.g = self.createGraph()
    
    def best_clique_precolation(self):
        maxComm = 0
        bestK = 0
        for k in range(200):
            self.k = k
            communities = self.clique_percolation()
            #self.printComms(communities)
            #print "number of communities with k = %d: %d" % (k, len(communities))
            if len(communities) > maxComm:
                maxComm = len(communities)
                bestK = k
        #print "Max number of found communities: %d" % (maxComm)
        self.k = bestK
        communities = self.clique_percolation()
        return communities
    
    def clique_percolation(self):
        Communities = snap.TIntIntVV()
        snap.TCliqueOverlap_GetCPMCommunities(self.g, self.k, Communities)
        community_list = []
        for C in Communities:
            cluster_k = []
            for Node in C:
                token = self.id_to_token[Node]
                cluster_k += [token]
            cluster_d = {'cluster_tokens': cluster_k, 'k': self.k}
            community_list += [cluster_d]
        return community_list
    
    def run(self):
        return self.clique_percolation()
        
    
def construct(config):
    return SlicingCP(config)  