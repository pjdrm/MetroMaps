'''
Created on 21/07/2015

@author: Mota
'''
import numpy as np
import json
import glob
import math
import logging
import os
import slicer_factory
from sklearn.cluster import MeanShift, estimate_bandwidth, KMeans, AffinityPropagation, DBSCAN, AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models.doc2vec import LabeledSentence, Doc2Vec

'''
class LabeledLineSentence(object):
    def __init__(self, dirPath):
        self.dirPath = dirPath
        
    def __iter__(self):
        def getint(name):
            basename = name.split('\\')[-1]
            num = basename.split('.')[0]
            return int(num)
        files = glob.glob(self.dirPath+"/*")
        files.sort(key=getint)
        uid = 0
        for filename in files:
            doc = ""
            for line in open(filename):
                line = line.strip('\n\r')
                doc += line + " "
            doc = doc[:-1]
            yield LabeledSentence(doc.split(), ['SENT_%s' % uid])
            uid += 1
'''

def gen_labled_sent(dirPath):
    def getint(name):
        basename = name.split('\\')[-1]
        num = basename.split('.')[0]
        return int(num)
    sentences = []
    files = glob.glob(dirPath+"/*")
    files.sort(key=getint)
    uid = 0
    for filename in files:
        doc = ""
        for line in open(filename):
            line = line.strip('\n\r')
            doc += line + " "
        doc = doc[:-1]
        sentences.append(LabeledSentence(doc.split(), ['SENT_%s' % uid]))
        uid += 1
    return sentences
            
class SlicingClusterBased(slicer_factory.SlicingHandlerGenerator):
    '''
    classdocs
    '''
    def __init__(self, legacy_helper_config_dict):
        super(SlicingClusterBased, self).__init__(legacy_helper_config_dict)
        self.doc_keys = self.data["doc_counts"].keys()
        self.global_counts = self.data["global_counts"]
        self.vocab_size = len(self.global_counts.keys())
        self.max_token_counts, self.num_docs_with_term = self.token_stats(self.doc_counts)
        self.pos_doc_dic = {}
        self.cluster_elms = self.createElements()
        #self.token_pos_dic, self.vocab_size, self. n_docs = self.getTokenPosDic()
       
    def kmeans(self, samples):
        num_clusters = 7
        km = KMeans(n_clusters=num_clusters)
        km.fit(samples)
        return km.labels_
    
    def mean_shift(self, samples):
        #bandwidth = estimate_bandwidth(elements, quantile=0.3, n_samples=None)
        bandwidth = 14
        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        ms.fit(samples)
        return ms.labels_
    
    def affinity_propagation(self, samples):
        af = AffinityPropagation(damping = 0.5, preference=-100).fit(samples)
        return af.labels_
    
    def dbscan(self, samples):
        db = DBSCAN(eps=0.1, min_samples=2, metric=cosine_similarity).fit(samples)
        return db.labels_
    
    def agglomerative_clustering(self, samples):
        ac = AgglomerativeClustering(linkage='ward', n_clusters=7)
        ac.fit(samples)
        return ac.labels_
        
    def slice(self):
        elements = self.cluster_elms
        #elements = self.createElemntsDoc2Vec()
        cluster_labels = self.kmeans(elements)
        print cluster_labels
        
        labels_unique = np.unique(cluster_labels)
        n_clusters = len(labels_unique)
        
        clusters = [[] for i in range(n_clusters)]
        for i, cluster_label in enumerate(cluster_labels):
            clusters[cluster_label].append(self.getDoc(i))
        
        self.write(clusters)
            

        
    def createElements(self):
        elements = np.zeros(shape=(self.num_docs, self.vocab_size))
        i = 0
        for doc_id in self.doc_keys:
            self.pos_doc_dic[i] = doc_id
            for token_id, count in self.doc_counts[doc_id].iteritems():
                tfidf_score = self.tfidf(token_id, doc_id)
                elements[i][int(token_id)-1] = tfidf_score
            i += 1
        return elements
    
    def createElemntsDoc2Vec(self):
        model = Doc2Vec(alpha=0.025, min_alpha=0.025)  # use fixed learning rate
        #segments = [s for s in LabeledLineSentence("domains/avl/data/rawtext")]
        segments = gen_labled_sent("domains/avl/data/rawtext")
        model.build_vocab(segments)
        for epoch in range(10):
            print "epoch: %d" % epoch
            model.train(segments)
            model.alpha -= 0.002  # decrease the learning rate
            model.min_alpha = model.alpha  # fix the learning rate, no decay
        print model.docvecs.__dict__.keys()
        elements = [model.docvecs[key] for key in  model.docvecs.__dict__['doctags'].keys()]
        return elements
    
    def getDoc(self, pos):
        doc_id = int(self.pos_doc_dic[pos])
        for doc in self.doc_metadata:
            if doc["id"] == doc_id:
                return doc
    
def eval(labels, reference):
    correct = 0.0
    for label, ref in zip(labels, reference):
        if label == ref:
            correct += 1.0
    return correct / len(labels)
 
def construct(config):
    return SlicingClusterBased(config)  
   
#test = SlicingClusterBased("/tmp/legacy_handler_out.json")
#test.slice()
    
    
                