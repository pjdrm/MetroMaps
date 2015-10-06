'''
Base class for slicing procedures that are
based on graph community detection algorithms.

The idea is to find word communities from a word co-occurrence
graph and then assign documents to the discovered communities.
Documents assigned to the same community will be belong to the same "slice",
from which word clusters will be extracted in the next step of the pipeline. 
@author: Mota
'''
import snap
from itertools import combinations
from mm.input.slicing import slicer_factory

class SlicingGraphBased(slicer_factory.SlicingHandlerGenerator):
    def __init__(self, legacy_helper_config_dict):
        super(SlicingGraphBased, self).__init__(legacy_helper_config_dict)
        self.doc_keys = self.data["doc_counts"].keys()
        self.token_to_word = {v: k for k, v in self.data["global_tokens"].items()}
        self.k = 5
        self.max_tokens = 50
    
    '''
    Creates a co-occurrence graph from the set of documents. The graph is
    to be used by algorithms on the snap package.
    
    It should be noted that no weight on the edges is calculated. In practice
    it is not relevant as the algorithm in snap dont use weights anyway.
    '''  
    def createGraph(self):
        g = snap.TUNGraph.New()
        node_degrees = {}
        for doc_id in self.doc_keys:
            token_ids = []
            token_tfidf = {}
            for token_id in self.doc_counts[doc_id]:
                if self.doc_counts[doc_id][token_id] < self.min_freq_in_doc:
                    continue
                token_tfidf[token_id] = self.tfidf(token_id, doc_id)
            token_ids = sorted(token_tfidf, key=lambda x : -token_tfidf[x])[:self.max_tokens]
          
            for (node_1, node_2) in combinations(token_ids,2):
                if not g.IsNode(int(node_1)):
                    g.AddNode(int(node_1))
                if not g.IsNode(int(node_2)):    
                    g.AddNode(int(node_2))
                g.AddEdge(int(node_1), int(node_2))
                node_degrees[int(node_1)] = node_degrees.get(int(node_1), 0) + 1
                node_degrees[int(node_2)] = node_degrees.get(int(node_2), 0) + 1
        
        '''
        counter = 0
        #gLoaded = snap.LoadEdgeList(snap.PNGraph, "graph_file.txt", 0, 1)
        for NI in g.Nodes():
            print "node: %d, out-degree %d, in-degree %d" % ( NI.GetId(), NI.GetOutDeg(), NI.GetInDeg())
            counter +=1
        print "number of nodes: %d" % (counter)
        '''
        return g
    
    def slice(self):
        communities = self.run()
        for cluster in communities:
            print "%d %s" % (len(cluster["cluster_tokens"]), cluster["cluster_tokens"])
        mapping = self.mapSegsComm(communities, self.token_to_word, self.data["doc_counts"])
        self.write(mapping)
    
    def printComms(self, communitites):
        print "Words in some community"
        tokenSet = set()
        for comm in communitites:
            for token in comm["cluster_tokens"]:
                tokenSet.add(token)
        print tokenSet
        print "Total tokens: %d\nNumber of communities: %d " % (len(tokenSet), len(communitites))
      
    '''
    Implements a method for assigning documents to a particular community.
    Each document has a score regarding a community and is assigned
    to the one with the hieghst score.
    '''          
    def mapSegsComm(self, communities, id_to_token, doc_counts):
        mapping = [ [] for com in communities]
        for doc_id in sorted(doc_counts.keys(), key=lambda x: int(x)):
            bestScore = 0.0
            bestComm = 0
            for commIndex, comm in enumerate(communities):
                score = self.commScore(comm, doc_counts[doc_id], id_to_token)
                if score >= bestScore:
                    bestScore = score
                    bestComm = commIndex
            mapping[bestComm].append({"timestamp" : int(doc_id), "id" : int(doc_id), "name" : doc_id + ".txt"})
        return mapping
    
    '''
    Score function between a document and a community.
    It is based on the number of tokens that are common to
    the document and a community.
    
    The score is normalized by the number of elements in the community.
    The idea is to prevent that communities with a high number of elements
    automatically have a higher score.
    '''
    def commScore(self, comm, doc_counts, id_to_token):
        score = 0.0
        for token_id in doc_counts:
            token = id_to_token[int(token_id)]
            if token in comm["cluster_tokens"]:
                score += 1
        normalized_score = score / len(comm["cluster_tokens"])
        return normalized_score
