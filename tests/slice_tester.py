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
from  mm.input.slicing.slicer_factory import isGraphAlg
from  mm.input.slicing.slicer_factory import factory as slicer_factory
import eval_metrics

'''
def sliceTester(configs, test_configs):
    slicing_true_labels = [int(x) for x in test_configs['slicing_true_labels'].split(',')]
    results_rand_index = {}
    results_f1 = {}
    results_acc = {}
    results_slice_labels = {}
    for n in [1]:
        configs.get('input_preprocessing')["n"] = 15
        mmrun.Run_init()
        mmrun.Run_input_generator(configs)
        mmrun.Run_input_preprocessing(configs)
        mmrun.Run_input_handler(configs)
        slicing_configs = configs.get('slicing')
        slicing_configs["true_labels"] = slicing_true_labels
        slicing_configs["steps"] = 19
        if "slicing_igraph" in test_configs:
            for  graph_alg in test_configs["slicing_igraph"]["algorithms"]:
                slicing_configs["type"] = graph_alg
                for score_f in test_configs["score_function"]:
                    slicing_configs["graph_community"]["score_function"] = score_f
                    for weight_scheme in test_configs["slicing_igraph"]["weight_schemes"]:
                        slicing_configs["graph_community"]["weight_calculator"] = weight_scheme
                        print "Testing %s" % graph_alg + " " + weight_scheme  + " " + str(n)
                        slicing_clusters = mmrun.Run_slicing_handler(configs)
                        slicing_labels = [None]*len(slicing_true_labels)
                        label = 0
                        for slice_cluster in slicing_clusters:
                            for el in slice_cluster:
                                index = el['timestamp'] - 1
                                slicing_labels[index] = label
                            label += 1
                        
                        alg = graph_alg + " " + weight_scheme +  " " + score_f  + " top-N " + str(n)       
                        rand_index = metrics.adjusted_rand_score(slicing_true_labels, slicing_labels)
                        results_rand_index[alg] = rand_index
                        results_f1[alg] = eval_metrics.f_measure(slicing_true_labels, slicing_labels)
                        results_acc[alg] = eval_metrics.accuracy(slicing_true_labels, slicing_labels)
                        results_slice_labels[alg] = ', '.join(str(e) for e in slicing_labels)
          
        if "slicing_other" in test_configs:       
            for cluster_alg in test_configs["slicing_other"]["algorithms"]:
                score_funcs = [""]
                if isGraphAlg(cluster_alg):
                    score_funcs = test_configs["score_function"]
                for score_f in score_funcs:
                    slicing_configs["graph_community"]["score_function"] = score_f
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
                
                    alg = cluster_alg + " " + score_f  + " top-N " + str(n)    
                    rand_index = metrics.adjusted_rand_score(slicing_true_labels, slicing_labels)
                    results_rand_index[alg] = rand_index
                    results_f1[alg] = eval_metrics.f_measure(slicing_true_labels, slicing_labels)
                    results_acc[alg] = eval_metrics.accuracy(slicing_true_labels, slicing_labels)
                    results_slice_labels[alg] = ', '.join(str(e) for e in slicing_labels)
            
            
    sorted_results = sorted(results_rand_index.items(), key=operator.itemgetter(1), reverse=True)
    strREsults = ""
    for result in sorted_results:
        strREsults += result[0] + " " + " Rand Index: " + str(result[1]) + " F1: " + str(results_f1[result[0]]) + " Acc: " + str(results_acc[result[0]]) + '\t[' + results_slice_labels[result[0]] +  ']\n'
    with open("resources/tests/results.txt","w") as results_file:    
        results_file.write(strREsults)
    print "Finished tests"
'''
    
def sliceTester(configsYaml, test_configs):
    resultsDic = {}
    resultsDic["ari"] = {}
    resultsDic["f1"] = {}
    resultsDic["acc"] = {}
    true_clusters = [int(x) for x in test_configs['slicing_true_labels'].split(',')]
    for alg in test_configs["algorithms"]:
        if test_configs["algorithms"][alg]["run"] == "False":
            continue
        if isGraphAlg(alg):
            testsGraph(alg, configsYaml, test_configs, true_clusters, resultsDic)
            
    sorted_results = sorted(resultsDic["ari"].items(), key=operator.itemgetter(1), reverse=True)
    strREsults = ""
    for result in sorted_results:
        strREsults += (result[0] + " " + " ARI: " + str(result[1]) + 
                                         " F1: " + str(resultsDic["f1"][result[0]]) + 
                                         " Acc: " + str(resultsDic["acc"][result[0]]) + 
                                         '\t[' + resultsDic[result[0]] +  ']\n')
                                        
    with open("resources/tests/results.txt","w") as results_file:    
        results_file.write(strREsults)
    print "Finished tests"
            
def runTest(configsYaml, true_clusters, resultsDic, nRuns = 1):
    slicer = slicer_factory(configsYaml["slicing"])
    print "Testing %s" % slicer.desc
    totalARI = 0.0
    totalF1 = 0.0
    totalAcc = 0.0
    for i in range(nRuns):
        slicing_clusters = slicer.slice()
        hyp_clusters = toArrayClusters(slicing_clusters)
        resultsHandler(slicer.desc, hyp_clusters, true_clusters, resultsDic)
        totalARI += resultsDic["ari"][slicer.desc]
        totalF1 += resultsDic["f1"][slicer.desc]
        totalAcc += resultsDic["acc"][slicer.desc]
    resultsDic["ari"][slicer.desc] = totalARI / float(nRuns)
    resultsDic["f1"][slicer.desc] = totalF1 / float(nRuns)
    resultsDic["acc"][slicer.desc] = totalAcc / float(nRuns)
            
def testsGraph(alg, configsYaml, test_configs, true_clusters, resultsDic):
    configsYaml.get('slicing')["type"] = alg
    n_vals = eval(test_configs["algorithms"][alg]["n"])
    for n in n_vals:
        configsYaml["input_preprocessing"]["n"] = n
        configsYaml["slicing"]["n"] = n
        mmrun.Run_init()
        mmrun.Run_input_generator(configsYaml)
        mmrun.Run_input_preprocessing(configsYaml)
        mmrun.Run_input_handler(configsYaml)
        for scoreFunc in test_configs["algorithms"][alg]["score_function"]:
            configsYaml.get('slicing')["graph_community"]["score_function"] = scoreFunc
            if "weight_schemes" in  test_configs["algorithms"][alg]:
                testWeightedGraph(alg, configsYaml, test_configs, true_clusters, resultsDic)
            elif alg == "slicing_lda":
                testLDA(alg, configsYaml, test_configs, true_clusters, resultsDic)
            else:
                runTest(configsYaml, true_clusters, resultsDic)
            
def testWeightedGraph(alg, configsYaml, test_configs, true_clusters, resultsDic):
    for weight_scheme in test_configs["algorithms"][alg]["weight_schemes"]:
        configsYaml["slicing"]["graph_community"]["weight_calculator"] = weight_scheme
        if alg == "slicing_walktraps":
            testWalktraps(alg, configsYaml, test_configs, true_clusters, resultsDic)
        else:
            runTest(configsYaml, true_clusters, resultsDic)
            
def testLDA(alg, configsYaml, test_configs, true_clusters, resultsDic):
    for wordsPerTopic in eval(test_configs["algorithms"][alg]["wordsPerTopic"]):
        configsYaml["slicing"]["graph_community"]["wordsPerTopic"] = wordsPerTopic
        runTest(configsYaml, true_clusters, resultsDic, test_configs["algorithms"][alg]["nRuns"])
                
def testWalktraps(alg, configsYaml, test_configs, true_clusters, resultsDic):
    for steps in eval(test_configs["algorithms"][alg]["steps"]):
        configsYaml.get('slicing')["steps"] = steps
        runTest(configsYaml, true_clusters, resultsDic)

def toArrayClusters(slicing_clusters):
    clustArray = [None]*sum([len(x) for x in slicing_clusters])
    label = 0
    for slice_cluster in slicing_clusters:
        for el in slice_cluster:
            index = el['timestamp'] - 1
            clustArray[index] = label
        label += 1
    return clustArray
                
def resultsHandler(alg_desc, hyp_clusters, true_clusters, resultsDic):
    resultsDic["ari"][alg_desc] = metrics.adjusted_rand_score(true_clusters, hyp_clusters)
    resultsDic["f1"][alg_desc] = eval_metrics.f_measure(true_clusters, hyp_clusters)
    resultsDic["acc"][alg_desc] = eval_metrics.accuracy(true_clusters, hyp_clusters)
    resultsDic[alg_desc] = ', '.join(str(e) for e in hyp_clusters)

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
