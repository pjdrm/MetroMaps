'''
Created on 01/10/2015

@author: Mota
'''
import networkx as nx
import networkx.algorithms.community.asyn_lpa as asyn_lpa
import mm.input.slicing.graph.slicing_graph_based as slicing_graph_based
from itertools import combinations

class SlicingAsynLPA(slicing_graph_based.SlicingGraphBased):
    def __init__(self, legacy_helper_config_dict):
        super(SlicingAsynLPA, self).__init__(legacy_helper_config_dict)
        self.g = self.createGraph()
        
    def createGraph(self):
        g = nx.Graph()
        #node_degrees = {}
        for doc_id in self.doc_keys:
            token_ids = []
            token_tfidf = {}
            for token_id in self.doc_counts[doc_id]:
                if self.doc_counts[doc_id][token_id] < self.min_freq_in_doc:
                    continue
                token_tfidf[token_id] = self.tfidf(token_id, doc_id)
            token_ids = sorted(token_tfidf, key=lambda x : -token_tfidf[x])[:self.max_tokens]
          
            for (node_1, node_2) in combinations(token_ids,2):
                intnode1 = int(node_1)
                intnode2 = int(node_2)
                if not g.has_node(intnode1):
                    g.add_node(intnode1)
                if not g.has_node(intnode2):    
                    g.add_node(intnode2)
                if not g.has_edge(intnode1, intnode2):
                    g.add_edge(intnode1, intnode2, weight = 1.0)
                else:
                    w = g[intnode1][intnode2]['weight']
                    g[intnode1][intnode2]['weight'] = w + 1.0
                #node_degrees[int(node_1)] = node_degrees.get(int(node_1), 0) + 1
                #node_degrees[int(node_2)] = node_degrees.get(int(node_2), 0) + 1
        
        toRemove = []
        for n in g.nodes():
            for e in g[n]:
                if g[n][e]['weight'] == 1.0:
                    toRemove.append([n, e])
        g.remove_edges_from(toRemove)

        return g
        
    def asyn_lpa(self):
        communities = asyn_lpa.asyn_lpa_communities(self.g)
        c = 0
        for community in communities:
            c += 1
        print "number of communities: %d" %c
        return 0
    
    def run(self):
        return self.asyn_lpa()
    
def construct(config):
    return SlicingAsynLPA(config)  
