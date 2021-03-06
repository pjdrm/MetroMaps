'''
Created on 06/10/2015

@author: Mota
'''
from itertools import combinations
from itertools import chain
import igraph as ig

class iGraphWrapper(object):
    
    def __init__(self, igraph_graph_slicer):
        self.graph_slicer = igraph_graph_slicer
        self.token_to_node_dic = {}
        self.node_to_token_dic = {}
        self.token_tfidfscores = {}
        
    def getTokenid(self, nodeIndex):
        return self.node_to_token_dic[nodeIndex]
    
    def addTokenid(self, tokenid, nodeIndex):
        self.node_to_token_dic[nodeIndex] = tokenid
    
    def getNodeid(self, tokenid):
        return self.token_to_node_dic[tokenid]
    
    def addNodeIndex(self, tokenid, nodeIndex):
        self.token_to_node_dic[tokenid] = nodeIndex
        
    def deleteNode(self, nodeId, graph):
        nnodes = graph.vcount()
        self.token_to_node_dic.pop(self.node_to_token_dic[nodeId], None)
        graph.delete_vertices([nodeId])
        nodes_to_shift = range(nodeId+1, nnodes)
        for node in nodes_to_shift:
            token = self.node_to_token_dic[node]
            self.token_to_node_dic[token] = self.token_to_node_dic[token] - 1
            self.node_to_token_dic[node-1] = token
        self.node_to_token_dic.pop(nnodes-1, None)
        
    def deleteNodes(self, to_del_nodes, graph):
        del_counter = 0
        for node in to_del_nodes:
            self.deleteNode(node - del_counter, graph)
            del_counter += 1
            
        
        
    def hasEdge(self, intnode1, intnode2, edgeList):
        for edge in edgeList:
            if (intnode1, intnode2) == edge:
                return True
            if (intnode2, intnode1) == edge:
                return True
        return False
        
    '''
    Some of the community detection algorithms are part of the igraph package.
    Therefore, this class overrides the super method createGraph
    to generate the appropriate type of graph.
    '''
    def createGraph(self):
        token_ids_list = []
        for doc_id in self.graph_slicer.doc_keys:
            token_ids = []
            token_tfidf = {}
            for token_id in self.graph_slicer.doc_counts[doc_id]:
                if self.graph_slicer.doc_counts[doc_id][token_id] < self.graph_slicer.min_freq_in_doc:
                    continue
                
                tfidf_score = self.graph_slicer.tfidf(token_id, doc_id)
                token_tfidf[token_id] = tfidf_score
                      
                if not int(token_id) in self.token_tfidfscores:
                    self.token_tfidfscores[int(token_id)] = []
                self.token_tfidfscores[int(token_id)].append(tfidf_score)
                
                
            token_ids = sorted(token_tfidf, key=lambda x : -token_tfidf[x])[:self.graph_slicer.max_tokens]
            
            len_token_ids = len(token_ids)
            if len_token_ids == 0 or len_token_ids == 1:
                continue
            token_ids_list.append(token_ids)
            
        n_nodes = len(set(chain(*token_ids_list)))
        g = ig.Graph(n_nodes)
        nodeIndex = 0
        for token_ids in token_ids_list:
            for (token_1, token_2) in combinations(token_ids,2):
                int_token1 = int(token_1) 
                int_token2 = int(token_2)
                
                if not int_token1 in self.token_to_node_dic:
                    self.addTokenid(int_token1, nodeIndex)
                    self.addNodeIndex(int_token1, nodeIndex)
                    intnode1 = nodeIndex
                    nodeIndex += 1
                else:
                    intnode1 = self.getNodeid(int_token1)
                    
                if not int_token2 in self.token_to_node_dic:
                    self.addTokenid(int_token2, nodeIndex)
                    self.addNodeIndex(int_token2, nodeIndex)
                    intnode2 = nodeIndex
                    nodeIndex += 1
                else:
                    intnode2 = self.getNodeid(int_token2)
                    
                if not self.hasEdge(intnode1, intnode2, g.get_edgelist()):
                    g.add_edges([(intnode1, intnode2)])
                    edgeid = g.get_eid(intnode1, intnode2)
                    g.es[edgeid]['count'] = 1.0
                else:
                    edgeid = g.get_eid(intnode1, intnode2)
                    w = g.es[edgeid]['count']
                    g.es[edgeid]['count'] = w + 1.0     
        self.graph_slicer.logTFIDFWordScores("./resources/debug/tfidfScores.txt")
        return g
    
    '''
    Filters communities that have less elements than the minSize
    parameter.
    
    Note: this method was initially used for filtering communities with
    size 1. Since the graph to be analyzed has the appropriate
    number of nodes (no extra and isolated nodes) this seems not to
    be a problem.
    '''
    def filterCommunities(self, communities_list, minSize):
        comms_filtered = []
        for comm in communities_list:
            if len(comm['cluster_tokens']) <  minSize:
                continue
            comms_filtered.append(comm)
        return comms_filtered
    
    def getCommunities(self, vertexCluster):
        node_membership = vertexCluster.membership
        communities_list = [{'cluster_tokens' : [], 'k' : 5} for i in range(max(node_membership)+1)]
        for node_index, com_index in enumerate(node_membership):
            communities_list[com_index]['cluster_tokens'].append(self.graph_slicer.token_to_word[self.node_to_token_dic[node_index]])
            
        comms_filtered = self.filterCommunities(communities_list, 2)
        if len(comms_filtered) == 0:
            comms_filtered = communities_list
        return comms_filtered