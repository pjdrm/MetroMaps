'''
Created on 01/02/2016

@author: Mota
'''

class CommonWordsTFIDF(object):
           
    def score(self, comm, graph, doc_id):
        return self.commonWordsScoreTfidf(comm, graph, doc_id)
    
    '''
    Score function between a document and a community.
    It is based on the number of tokens that are common to
    the document and a community.
    
    The score is normalized by the number of elements in the community.
    The idea is to prevent that communities with a high number of elements
    automatically have a higher score.
    '''
    def commonWordsScoreTfidf(self, comm, graph, doc_id):
        id_to_token = graph.token_to_word
        doc_counts = graph.data["doc_counts"][doc_id]
        score = 0.0
        contributingWords = ""
        totalPossibleScore = 0.0
        for token_id in doc_counts:
            token = id_to_token[int(token_id)]
            tfidfScore = graph.tfidf(token_id, doc_id)
            totalPossibleScore += tfidfScore
            if token in comm["cluster_tokens"]:
                score += tfidfScore 
                contributingWords += token + ", "
        normalized_score = score / totalPossibleScore
        return normalized_score, contributingWords[:-2]
    
def construct():
    return CommonWordsTFIDF()