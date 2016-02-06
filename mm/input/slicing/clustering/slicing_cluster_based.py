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

class SlicingClusterBased(slicer_factory.SlicingHandlerGenerator):
    def __init__(self, slicer_configs):
        super(SlicingClusterBased, self).__init__(slicer_configs)
        self.doc_keys = self.data["doc_counts"].keys()
        self.global_counts = self.data["global_counts"]
        self.vocab_size = len(self.global_counts.keys())
        self.pos_doc_dic = {}
        self.cluster_elms = self.createElements()
        #elements = self.createElemntsDoc2Vec()
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
            self.pos_doc_dic[i] = doc_id
            for token_id, count in self.doc_counts[doc_id].iteritems():
                tfidf_score = self.tfidf(token_id, doc_id)
                elements[i][int(token_id)-1] = tfidf_score
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
    def createElemntsDoc2Vec(self):
        model = Doc2Vec(alpha=0.025, min_alpha=0.025)  # use fixed learning rate
        #segments = [s for s in LabeledLineSentence("domains/avl/data/rawtext")]
        segments = self.gen_labled_sent("domains/avl/data/rawtext")
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
            
    def eval(self, labels, reference):
        correct = 0.0
        for label, ref in zip(labels, reference):
            if label == ref:
                correct += 1.0
        return correct / len(labels)
