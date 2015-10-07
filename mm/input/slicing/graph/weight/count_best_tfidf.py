'''
Assigns weight to the edges equal to the sum of the best
tfidf scores, in some document, plus the number of times 
they co-occurred in different documents.

@author: Mota
'''
class WeightCountBestTFIDF(object):

    def __init__(self, igraphWrapper):
        self.igraphWrapper = igraphWrapper
        self.token_tfidfscores_avg = self.bestTFIDF(self.igraphWrapper.token_tfidfscores)
        
    def bestTFIDF(self, token_tfidfscores):
        for token_id in token_tfidfscores:
            token_tfidfscores[token_id] = max(token_tfidfscores[token_id])
        return token_tfidfscores
    
    def calculateWeights(self):
        for edgeid, edge in enumerate(self.igraphWrapper.graph_slicer.g.get_edgelist()):
            token1 = self.igraphWrapper.getTokenid(edge[0])
            token2 = self.igraphWrapper.getTokenid(edge[1])
            weight = self.igraphWrapper.graph_slicer.g.es[edgeid]['count'] + self.token_tfidfscores_avg[token1] + self.token_tfidfscores_avg[token2]
            self.igraphWrapper.graph_slicer.g.es[edgeid]['weight'] = weight
            
def construct(graph_slicer):
    return WeightCountBestTFIDF(graph_slicer) 
