'''
Created on 20/08/2015

@author: Mota
'''
import math
import logging
import os
import json
from Cython.Compiler.Main import module_name_pattern

class SlicingHandlerGenerator(object):
    def __init__(self, legacy_helper_config_dict):
        self.data = {}
        with open(legacy_helper_config_dict['mm_standard_input']) as in_json, open(legacy_helper_config_dict['doc_metadata']) as doc_meta_json:
            self.data = json.load(in_json)
            self.doc_metadata = json.load(doc_meta_json)
        self.output_dir = legacy_helper_config_dict['output_dir']
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.output_json = legacy_helper_config_dict['output_json']
        self.repr_tokens = self.data['representative_tokens']
        self.doc_counts = self.data['doc_counts']
        self.num_docs = len(self.doc_counts)
        self.max_token_counts, self.num_docs_with_term = self.token_stats(self.doc_counts)
        self.min_freq_in_doc = int(legacy_helper_config_dict.get('min_freq_in_doc'))
        
    def token_stats(self, doc_counts):
            token_max = {} # maps token_id -> max doc frequency
            num_docs_with_term = {}
            for doc_id,doc_data in doc_counts.iteritems():
                for token_id,count in doc_data.iteritems():
                    current_max = token_max.get(token_id, -1)
                    num_docs_with_term[token_id] = num_docs_with_term.get(token_id, 0) + 1
                    if count > current_max:
                        token_max[token_id] = count

            return token_max, num_docs_with_term
        
    def tfidf(self, token_id, doc_id):
        if self.num_docs_with_term.get(token_id, 1) <= 1:
            '''do check so we do not divide by 0'''
            return 0
        in_doc_count = self.doc_counts.get(str(doc_id),{}).get(token_id,0)
        tf = math.log(float(in_doc_count + 1.0))  #0.5 + 0.5 * in_doc_count / float(max_count)
        idf = math.log(float(self.num_docs)/float(self.num_docs_with_term[str(token_id)]))
        return tf * idf
    
    ''' Helper Functions for Writing '''
    def fakeDate(self, cluster_number, doc_number, time=False):
        ''' TODO: handle this more nicely '''
        if not (cluster_number >= 0 and cluster_number < 12):
            cluster_number = 12 # temporary hack!
        assert(doc_number >= 0)

        if doc_number >= 30:
            doc_number = 29

        t = "%s%02i%02i" %("2013", cluster_number+1, doc_number+1)
        if time:
            t += '0000'
        return t

    def write_docs_in_cluster(self, docs_in_cluster, ostream, cluster_index):
        doc_ids_in_cluster = [str(doc.get('id')) for doc in docs_in_cluster]
        logging.debug('Files in cluster %i: %s-%s' % (cluster_index, doc_ids_in_cluster[0], doc_ids_in_cluster[-1]))
        for doc_i, doc in enumerate(docs_in_cluster):
            doc_id = doc['id']
            doc_name = doc.get('name', 'untitled')
            doc_link = doc.get('link', '#')
            doc_data = self.doc_counts.get(str(doc_id), None)
            if not doc_data:
                logging.warning('Skipping doc counts of doc id %s (%s)' % (doc_id, doc_name))
                continue

            doc_tokens = []
            tfidf_sum = 0
            for token_id, token_count in doc_data.iteritems():
                tfidf_score = self.tfidf(token_id, doc_id)
                doc_tokens += [(token_id, tfidf_score)]
                tfidf_sum += tfidf_score
            tfidf_avg = float(tfidf_sum) / float(len(doc_data))
            doc_tokens.sort(key=lambda x: x[1],reverse=True)

            ''' 1. write doc header
                2. write doc tokens
                3. write two new lines'''
            ostream.write('%s\t%f\n' % (doc_id, tfidf_avg))
            ostream.write('%s\t%s\t%s\t%s\n' % (doc_id, self.fakeDate(cluster_index, doc_i, True), doc_link, doc_name))

            for token in doc_tokens:
                ostream.write('%s\t%s\n' % (token[0], token[1]))
            ostream.write('\n')
            
    def write(self, clusters):
        for i, cluster in enumerate(clusters):
            startDate = self.fakeDate(i, 0)
            endDate = self.fakeDate(i, len(cluster) - 1)
            

            ostream = open(os.path.join(self.output_dir, '%s-%s' % (startDate, endDate)),'w')
            ostream.write('%s\n%i\n%s %s\n\n' % (startDate, len(cluster), startDate, endDate))
            self.write_docs_in_cluster(cluster, ostream, i)
            ostream.close()

        json_data = {}
        clusters_data = []
        for i,cluster in enumerate(clusters):
            cluster_dict = {}
            cluster_start_date = self.fakeDate(i,0)
            cluster_end_date = self.fakeDate(i,len(cluster)-1)
            document_json_data = []
            for doc_i, doc_d in enumerate(cluster):
                doc_id = doc_d['id']
                doc_entry = {'doc_metadata': doc_d}
                try:
                    doc_data = self.doc_counts[str(doc_id)]
                except KeyError:
                    print doc_d
                    print self.doc_counts
                    print 'Error with key %s' % str(doc_id)


                tokens = []
                for token,count in doc_data.iteritems():
                    token_score = self.tfidf(token, doc_id)
                    token_doc_count = count
                    token_id = token

                    token_dict = {'id':token_id,'tfidf':token_score,'token_doc_count':count}
                    tokens.append(token_dict) 
                    token_syns = self.repr_tokens[token_id]
                    best_syn = max(token_syns, key=lambda x: token_syns[x])
                    token_dict['plaintext'] = best_syn

                doc_entry['tokens'] = tokens
                document_json_data.append(doc_entry)

            cluster_dict['cluster_start_date'] = cluster_start_date
            cluster_dict['cluster_end_date'] = cluster_end_date
            cluster_dict['index'] = i
            cluster_dict['doc_data'] = document_json_data

            clusters_data.append(cluster_dict)

        with open(self.output_json,'w') as output_json:
            json.dump(clusters_data, output_json)   
            logging.info('Scoring JSON info written to %s'%self.output_json)

def isGraphAlg(module_name):
    graph_algs = {
        "slicing_girvan_newman" : True,
        "slicing_clique_precolation" : True,
        "slicing_bigclam" : True,
        "slicing_cnm" : True
    }
    return graph_algs.get(module_name, False)

def isClusterAlg(module_name):
    graph_algs = {
        "slicing_kmeans" : True,
        "slicing_mean_shift" : True,
        "slicing_affinity_propagation" : True,
        "slicing_dbscan" : True,
        "slicing_agglomerative_clustering" : True
    }
    return graph_algs.get(module_name, False)

def factory(configs):
    metro_station_generator_configs = configs
    input_generator_name = metro_station_generator_configs['type']
    fromlist = [0]
    if isGraphAlg(input_generator_name):
        input_generator_name = "graph."+input_generator_name
        
    if isClusterAlg(input_generator_name):
        input_generator_name = "clustering."+input_generator_name
    
    generator_module = __import__(input_generator_name, globals=globals(), fromlist = fromlist)
    return generator_module.construct(metro_station_generator_configs)
