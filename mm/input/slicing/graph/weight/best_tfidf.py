'''
Assigns weight to the edges equal to the sum of the best
tfidf scores, in some document, of the corresponding words.

@author: Mota
'''
class WeightBestTFIDF(object):

    def __init__(self, graph_slicer):
        self.graph_slicer = graph_slicer
        self.token_tfidfscores_avg = self.bestTFIDF(self.graph_slicer.token_tfidfscores)
        
    def bestTFIDF(self, token_tfidfscores):
        for token_id in token_tfidfscores:
            token_tfidfscores[token_id] = max(token_tfidfscores[token_id])
        return token_tfidfscores
    
    def calculateWeights(self):
        for edgeid, edge in enumerate(self.graph_slicer.g.get_edgelist()):
            token1 = self.graph_slicer.getTokenid(edge[0])
            token2 = self.graph_slicer.getTokenid(edge[1])
            weight = self.token_tfidfscores_avg[token1] + self.token_tfidfscores_avg[token2]
            self.graph_slicer.g.es[edgeid]['weight'] = weight
            
def construct(graph_slicer):
    return WeightBestTFIDF(graph_slicer)
