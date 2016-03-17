'''
Provides the interface for the specific Clique-Precolation algorithm
to be used for word community detection.

To use this algorithm, in the .yaml configuration write the name of this module.
(slicing: type: slicing_clique_precolation)
@author: Mota
'''
import os
import snap
import mm.input.slicing.graph.slicing_graph_based as slicing_graph_based

class SlicingCP(slicing_graph_based.SlicingGraphBased):
    def __init__(self, slicer_configs):
        super(SlicingCP, self).__init__(slicer_configs)
        self.g = self.createGraph()
        self.debugDir = 'resources/slicing_results/cliqueprecolation/'
        self.debugFile = self.debugDir + "clique_precolation.txt"
        self.desc = "clique_precolation " + self.desc
    
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
                token = self.token_to_word[Node]
                cluster_k += [token]
            cluster_d = {'cluster_tokens': cluster_k, 'k': self.k}
            community_list += [cluster_d]
        return community_list
    
    def run(self):
        communities = self.clique_percolation()
        if not os.path.exists(self.debugDir):
            os.makedirs(self.debugDir)
        self.print_communities(communities, self.debugFile)
        return communities
        
    
def construct(config):
    return SlicingCP(config)  