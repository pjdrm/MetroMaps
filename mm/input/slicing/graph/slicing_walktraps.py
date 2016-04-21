'''
Provides the interface for the specific Walktraps algorithm
to be used for word community detection.

To use this algorithm, in the .yaml configuration write the name of this module.
(slicing: type: slicing_walktraps)

The Walktraps algorithm is sensible to weighted edges. These weights
are generated by classes in the weight package. The specific
weighting scheme can be configured in "slicing: weight_calculator:"
attribute of the yaml configuration file. 

@author: Mota
'''
import mm.input.slicing.graph.slicing_graph_based as slicing_graph_based
from mm.input.slicing.graph.weight.factory import factory
from wrapper.iGraphWrapper import iGraphWrapper
import os

class SlicingWalktraps(slicing_graph_based.SlicingGraphBased):
    def __init__(self, slicer_configs):
        super(SlicingWalktraps, self).__init__(slicer_configs)
        self.igraphWrapper = iGraphWrapper(self)
        self.g = self.igraphWrapper.createGraph()
        self.weightcalc = factory(slicer_configs, self.igraphWrapper)
        self.weightcalc.calculateWeights()
        self.wc_des = slicer_configs["graph_community"]['weight_calculator']
        self.debugDir = 'resources/slicing_results/walktraps/'
        self.debugFile = self.debugDir + self.wc_des + ".txt"
        self.steps = slicer_configs["steps"]
        self.desc = "walktraps steps " + str(self.steps) + " weight: " + self.wc_des + " " + self.desc 
        
    def walktraps(self):
        vertexCluster =  self.g.community_walktrap(weights="weight", steps=self.steps).as_clustering()
        self.graph2Txt(self.g, self.igraphWrapper.node_to_token_dic, vertexCluster.membership)
        return self.igraphWrapper.getCommunities(vertexCluster)
    
    def run(self):
        '''
        maxSize = 0;
        for i in range(1,50):
            communities = self.walktraps(i)
            lenComm = len(communities)
            if lenComm > maxSize:
                maxSize = lenComm
                bestComm = communities
        '''
        communities = self.walktraps()
            
        if not os.path.exists(self.debugDir):
            os.makedirs(self.debugDir)
        self.print_communities(communities, self.debugFile)
        return communities
    
def construct(config):
    return SlicingWalktraps(config)  
