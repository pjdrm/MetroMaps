'''
Created on 12/10/2015

@author: Mota
'''
import argparse
import logging
import yaml.composer
import mmrun
from sklearn import metrics
import json
import operator
import eval_metrics

def sliceTester(configs, test_configs):
    slicing_true_labels = [int(x) for x in test_configs['slicing_true_labels'].split(',')]
    
    
    mmrun.Run_init()
    mmrun.Run_input_generator(configs)
    mmrun.Run_input_preprocessing(configs)
    mmrun.Run_input_handler(configs)
    results_rand_index = {}
    results_f1 = {}
    results_slice_labels = {}
    if "slicing_igraph" in test_configs:
        for  graph_alg in test_configs["slicing_igraph"]["algorithms"]:
            slicing_configs = configs.get('slicing')
            slicing_configs["type"] = graph_alg
            for weight_scheme in test_configs["slicing_igraph"]["weight_schemes"]:
                slicing_configs["weight_calculator"] = weight_scheme
                print "Testing %s" % graph_alg + " " + weight_scheme
                slicing_clusters = mmrun.Run_slicing_handler(configs)
                slicing_labels = [None]*len(slicing_true_labels)
                label = 0
                for slice_cluster in slicing_clusters:
                    for el in slice_cluster:
                        index = el['timestamp'] - 1
                        slicing_labels[index] = label
                    label += 1
                
                alg = graph_alg + " " + weight_scheme
                results_rand_index[alg] = metrics.adjusted_rand_score(slicing_true_labels, slicing_labels)
                results_f1[alg] = eval_metrics.f_measure(slicing_true_labels, slicing_labels)
                results_slice_labels[alg] = ', '.join(str(e) for e in slicing_labels)
      
    if "slicing_other" in test_configs:       
        for  cluster_alg in test_configs["slicing_other"]["algorithms"]:
            slicing_configs = configs.get('slicing')
            slicing_configs["type"] = cluster_alg
            print "Testing %s" % cluster_alg
            slicing_clusters = mmrun.Run_slicing_handler(configs)
            slicing_labels = [None]*len(slicing_true_labels)
            label = 0
            for slice_cluster in slicing_clusters:
                for el in slice_cluster:
                    index = el['timestamp'] - 1
                    slicing_labels[index] = label
                label += 1
                       
            rand_index = metrics.adjusted_rand_score(slicing_true_labels, slicing_labels)
            results_rand_index[cluster_alg] = rand_index
            results_f1[cluster_alg] = eval_metrics.f_measure(slicing_true_labels, slicing_labels)
            results_slice_labels[cluster_alg] = ', '.join(str(e) for e in slicing_labels)
            
            
    sorted_results = sorted(results_rand_index.items(), key=operator.itemgetter(1), reverse=True)
    strREsults = ""
    for result in sorted_results:
        strREsults += result[0] + " Rand Index: " + str(result[1]) + " F1: " + str(results_f1[result[0]]) + '\t[' + results_slice_labels[result[0]] +  ']\n'
    with open("resources/tests/results.txt","w") as results_file:    
        results_file.write(strREsults)
    print "Finished tests"

def main(config_file, test_configs, defaults="mm/default.yaml"):
    config_dict = {}

    logging.basicConfig(format='%(levelname)s %(asctime)s %(message)s', level=logging.DEBUG)
    with open(defaults) as df:
        try: 
            config_dict = yaml.load(df)
        except yaml.composer.ComposerError:
            logging.error('ERROR in yaml-reading the default config file')
            raise
    with open(config_file) as cf:
        try: 
            new_config = yaml.load(cf)
            config_dict = new_config
            #for section in sections:
            #    sec_dict = new_config.get(section, {})
            #    config_dict.get(section).update(sec_dict)
        except yaml.composer.ComposerError:
            logging.error('ERROR in reading the input config file')
            raise
    log_level = {'error':logging.ERROR, 'debug':logging.DEBUG}.get(config_dict.get('global',{}).get('log_level'), logging.DEBUG)

    logging.basicConfig(level=log_level)

    logging.debug('final configuration: \n%s' % (str(yaml.dump(config_dict))))
    with open(test_configs) as test_configs_file:    
        test_configs_dic = json.load(test_configs_file)
    sliceTester(config_dict, test_configs_dic)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Run Metromaps by specifying a config file')
    parser.add_argument('config_file', help='See default.yaml for configuration options')
    parser.add_argument('test_configs')
    parser.add_argument('--defaults', default='mm/default.yaml', help='the default values get preloaded from this yaml configuration file')
    args = parser.parse_args()
    main(args.config_file, args.test_configs, args.defaults)
