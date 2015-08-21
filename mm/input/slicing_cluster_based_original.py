'''
Created on 21/07/2015

@author: Mota
'''
import numpy as np
import json
import glob
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
            
class SlicingClusterBased(object):
    '''
    classdocs
    '''


    def __init__(self, doc_json_path):
        with open(doc_json_path) as in_json:
            self.docs_data = json.load(in_json)
            
        self.token_pos_dic, self.vocab_size, self. n_docs = self.getTokenPosDic()
       
    def getTokenPosDic(self):
        token_pos_dic = {}
        i = 0
        n_docs = 0
        for docs in self.docs_data:
            for doc_data in docs["doc_data"]:
                n_docs += 1
                for token in doc_data["tokens"]:
                    word = token["plaintext"]
                    if not token_pos_dic.get(word, False):
                        token_pos_dic[word] = i
                        i += 1
        return token_pos_dic, i, n_docs
        
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
        elements = self.createElements()
        #elements = self.createElemntsDoc2Vec()
        labels = self.kmeans(elements)
        print labels
        
        labels_unique = np.unique(labels)
        n_clusters_ = len(labels_unique)
        print("number of estimated clusters : %d" % n_clusters_)
        print("accuracy : %f" % eval(labels, [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6]))
        
        '''
        for i in range(len(elements)):
            print "segment %d" % (i + 1)
            for j in range(len(elements)):
                if i == j:
                    continue
                print "%d: %f" % (j + 1, cosine_similarity(elements[i], elements[j])[0])
            print "----------"
        '''
            

        
    def createElements(self):
        elements = np.zeros(shape=(self.n_docs, self.vocab_size))
        i = 0
        for docs in self.docs_data:
            for doc_data in docs["doc_data"]:
                for token in doc_data["tokens"]:
                    pos = self.token_pos_dic[token["plaintext"]]
                    elements[i][pos] = token["tfidf"]
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
    
def eval(labels, reference):
    correct = 0.0
    for label, ref in zip(labels, reference):
        if label == ref:
            correct += 1.0
    return correct / len(labels)
    
test = SlicingClusterBased("/tmp/legacy_handler_out.json")
test.slice()
    
    
                