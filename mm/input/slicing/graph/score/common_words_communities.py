'''
Created on 01/02/2016

@author: Mota
'''

class CommonWordsComm(object):
           
    def score(self, comm, graph, docId):
        return self.commonWordsScore(comm, graph, docId)    
    
    '''
    Score function between a document and a community.
    It is based on the number of tokens that are common to
    the document and a community.
    
    The score is normalized by the number of elements in the community.
    The idea is to prevent that communities with a high number of elements
    automatically have a higher score.
    '''
    def commonWordsScore(self, comm, graph, doc_id):
        id_to_token = graph.token_to_word
        doc_counts = graph.data["doc_counts"][doc_id]
        score = 0.0
        contributingWords = ""
        for token_id in doc_counts:
            token = id_to_token[int(token_id)]
            if token in comm["cluster_tokens"]:
                score += 1
                contributingWords += token + ", "
        normalized_score = score / len(comm["cluster_tokens"])
        return (normalized_score, contributingWords[:-2])
    
def construct():
    return CommonWordsComm()