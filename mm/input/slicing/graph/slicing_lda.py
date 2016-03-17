'''
Provides the interface for the specific LDA algorithm
to be used for word community detection, in this case viewed as a
topic modeling task.

To use this algorithm, in the .yaml configuration write the name of this module.
(slicing: type: slicing_lda)

@author: Mota
'''
import os
import snap
import mm.input.slicing.graph.slicing_graph_based as slicing_graph_based
import numpy as np
from gensim.models import ldamodel as ldamodel

class SlicingLDA(slicing_graph_based.SlicingGraphBased):
    def __init__(self, slicer_configs):
        super(SlicingLDA, self).__init__(slicer_configs)
        self.g = self.createGraph()
        self.debugDir = 'resources/slicing_results/lda/'
        self.debugFile = self.debugDir + "lda.txt"
        self.nTopics = slicer_configs["clustering"]["k"]
        self.corpus = self.createSparseElements()
        self.id2Word = self.token_to_word
        self.wordsPerTopic = slicer_configs["graph_community"]["wordsPerTopic"]
        self.desc = "LDA wordsPerTopic: " + str(self.wordsPerTopic) + " " + self.desc
        
    def lda(self):
        lda = ldamodel.LdaModel(corpus=self.corpus, id2word=self.id2Word, num_topics=self.nTopics, update_every=1, chunksize=11, passes=10)
        topicWordsList = lda.print_topics(self.nTopics, self.wordsPerTopic)
        wordComm = []
        for topicWords in topicWordsList:
            words = topicWords.split(" + ")
            wordComm.append([w.split("*")[1] for w in words])
            
        community_list = []
        for C in wordComm:
            cluster_d = {'cluster_tokens': C, 'k': self.k}
            community_list += [cluster_d]
        return community_list
    
    def createSparseElements(self):
        elements = []
        doc_ids = range(1, len(self.doc_keys) + 1)
        for doc_id in doc_ids:
            sparseVec = []
            for token_id, count in self.doc_counts[str(doc_id)].iteritems():
                tfidf_score = self.tfidf(token_id, str(doc_id))
                entry = (int(token_id), tfidf_score)
                sparseVec.append(entry)
            sparseVec = sorted(sparseVec, key=lambda tup: tup[0])
            elements.append(sparseVec)
        return elements
    
    def run(self):
        communities = self.lda()
        if not os.path.exists(self.debugDir):
            os.makedirs(self.debugDir)
        self.print_communities(communities, self.debugFile)
        return communities
        
def construct(config):
    return SlicingLDA(config)
