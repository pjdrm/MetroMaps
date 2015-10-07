'''
Assigns weights to the edges equal to the number
of times two words co-occurred together in different documents.

@author: Mota
'''
class WeightCount(object):

    def __init__(self, igraphWrapper):
        self.igraphWrapper = igraphWrapper
    
    def calculateWeights(self):
        for edgeid, edge in enumerate(self.igraphWrapper.graph_slicer.g.get_edgelist()):
            weight = self.igraphWrapper.graph_slicer.g.es[edgeid]['count']
            self.igraphWrapper.graph_slicer.g.es[edgeid]['weight'] = weight
            
def construct(graph_slicer):
    return WeightCount(graph_slicer) 
            
        