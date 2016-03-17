'''
Base class for slicing procedures that are
based on clustering algorithms.

Documents that are clustered together will be in the same
"slice", from which word clusters will be extracted in the next step of the pipeline.

@author: Mota
'''
from gensim.models.doc2vec import Doc2Vec, LabeledSentence
from mm.input.slicing import slicer_factory
import numpy as np
import glob
from random import shuffle

class SlicingClusterBased(slicer_factory.SlicingHandlerGenerator):
    def __init__(self, slicer_configs):
        super(SlicingClusterBased, self).__init__(slicer_configs)
        self.doc_keys = self.data["doc_counts"].keys()
        self.global_counts = self.data["global_counts"]
        self.vocab_size = len(self.global_counts.keys())
        self.pos_doc_dic = {}
        self.cluster_elms = self.createElements()
        #self.cluster_elms = self.createElemntsDoc2Vec()
        self.num_clusters = slicer_configs["clustering"]["k"]
        self.true_labels = slicer_configs["true_labels"]
        #self.token_pos_dic, self.vocab_size, self. n_docs = self.getTokenPosDic()
        
    def slice(self):
        cluster_labels = self.run()
        
        labels_unique = np.unique(cluster_labels)
        n_clusters = len(labels_unique)
        
        clusters = [[] for i in range(n_clusters)]
        for i, cluster_label in enumerate(cluster_labels):
            clusters[cluster_label].append(self.getDoc(i))
        
        self.write(clusters)
        return clusters
        
    '''
    Computes the bag of words features representation of the
    documents based on the tfidf score.
    '''
    def createElements(self):
        elements = np.zeros(shape=(self.num_docs, self.vocab_size))
        i = 0
        for doc_id in self.doc_keys:
            self.pos_doc_dic[int(doc_id) - 1] = int(doc_id)
            for token_id, count in self.doc_counts[doc_id].iteritems():
                tfidf_score = self.tfidf(token_id, doc_id)
                elements[int(doc_id)-1][int(token_id)-1] = tfidf_score
            i += 1
        return elements
    
    def gen_labled_sent(self, dirPath):
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

    '''
    Computes the Doc2Vec features representation of the
    documents (extension of the Google's word2Vec algorithm).
    '''
    def createElemntsDoc2Vec(self, alpha = 871, alphaDec = 1.5):
        
        #model_dm = Doc2Vec(min_count=1, window=15, alpha=alpha, min_alpha=0.025, sample=1e-3, negative=5, workers=3)  # use fixed learning rate
        model_dbow = Doc2Vec(dm=0, min_count=1, window=15, alpha=0.0025, min_alpha=0.025, sample=1e-3, negative=5, workers=3)  # use fixed learning rate
        #segments = [s for s in LabeledLineSentence("domains/avl/data/rawtext")]
        segments = self.gen_labled_sent("domains/SSSS/data/rawtext")
        #model_dm.build_vocab(segments)
        model_dbow.build_vocab(segments)
        for epoch in range(10):
            print "doc2Vec training epoch %d" % epoch
            shuffle(segments)
            '''
            model_dm.train(segments)
            model_dm.alpha -= alphaDec  # decrease the learning rate
            model_dm.min_alpha = model_dm.alpha  # fix the learning rate, no decay
            '''
            
            model_dbow.train(segments)
            model_dbow.alpha -= alphaDec  # decrease the learning rate
            model_dbow.min_alpha = model_dbow.alpha  # fix the learning rate, no decay
            
        #elements_dm = [model_dm.docvecs[key] for key in  model_dm.docvecs.__dict__['doctags'].keys()]
        elements_dbow = [model_dbow.docvecs[key] for key in  model_dbow.docvecs.__dict__['doctags'].keys()]
        #elements = np.hstack((elements_dm, elements_dbow))
        return np.array(elements_dbow)
    
    def getDoc(self, pos):
        doc_id = int(self.pos_doc_dic[pos])
        for doc in self.doc_metadata:
            if doc["id"] == doc_id:
                return doc
            
    def eval(self, labels, reference):
        correct = 0.0
        for label, ref in zip(labels, reference):
            if label == ref:
                correct += 1.0
        return correct / len(labels)
