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
import matplotlib.pyplot as plt
from matplotlib.pyplot import legend, savefig

def sliceTester(configsYaml, test_configs):
    resultsDic = {}
    resultsDic["ari"] = {}
    resultsDic["f1"] = {}
    resultsDic["acc"] = {}
    true_clusters = [int(x) for x in test_configs['slicing_true_labels'].split(',')]
    for alg in test_configs["algorithms"]:
        configsYaml.get('slicing')["type"] = alg
        if test_configs["algorithms"][alg]["run"] == "False":
            continue
        if isGraphAlg(alg):
            testsGraph(alg, configsYaml, test_configs, true_clusters, resultsDic)
        else:
            testsCluster(alg, configsYaml, test_configs, true_clusters, resultsDic)
            
    sorted_results = sorted(resultsDic["ari"].items(), key=operator.itemgetter(1), reverse=True)
    strREsults = ""
    for result in sorted_results:
        strREsults += (result[0] + " " + " ARI: " + str(result[1]) + 
                                         " F1: " + str(resultsDic["f1"][result[0]]) + 
                                         " Acc: " + str(resultsDic["acc"][result[0]]) + 
                                         ' [' + resultsDic[result[0]] +  ']\n')
                                        
    resultsFile = "resources/tests/results.txt"
    
    with open(resultsFile,"w") as results_file:    
        results_file.write(strREsults)
    
    if test_configs["plots"]["run"] == "True":
        plotResult(test_configs["plots"]["xLabel"], eval(test_configs["plots"]["yLabels"]), resultsFile)
        
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
    
def testsCluster(alg, configsYaml, test_configs, true_clusters, resultsDic):
    if alg == "slicing_dbscan":
        testDBScan(alg, configsYaml, test_configs, true_clusters, resultsDic)
    elif alg == "slicing_affinity_propagation":
        testAffinityProp(alg, configsYaml, test_configs, true_clusters, resultsDic)
        
    elif alg == "slicing_agglomerative_clustering":
        testAgglomerative(alg, configsYaml, test_configs, true_clusters, resultsDic)
    elif alg == "slicing_mean_shift":
        testMeanShift(alg, configsYaml, test_configs, true_clusters, resultsDic)
    elif alg == "slicing_spectral":
        testSpectral(alg, configsYaml, test_configs, true_clusters, resultsDic)
    else:
        runTest(configsYaml, true_clusters, resultsDic)
        
def testDBScan(alg, configsYaml, test_configs, true_clusters, resultsDic):
    for metric in test_configs["algorithms"][alg]["metric"]:
        for eps in eval(test_configs["algorithms"][alg]["eps"]):
            for minPts in eval(test_configs["algorithms"][alg]["minPts"]):
                configsYaml["slicing"]["clustering"]["eps"] = eps
                configsYaml["slicing"]["clustering"]["min_samples"] = minPts
                configsYaml["slicing"]["clustering"]["metric"] = metric
                if metric == "gaussian":
                    for var in eval(test_configs["algorithms"][alg]["vars"]):
                        configsYaml["slicing"]["clustering"]["var"] = var
                        runTest(configsYaml, true_clusters, resultsDic)
                else:
                    runTest(configsYaml, true_clusters, resultsDic)
                
def testAffinityProp(alg, configsYaml, test_configs, true_clusters, resultsDic):
    for damping in eval(test_configs["algorithms"][alg]["damping"]):
        configsYaml["slicing"]["clustering"]["damping"] = damping
        for preference in eval(test_configs["algorithms"][alg]["preference"]):
            configsYaml["slicing"]["clustering"]["preference"] = preference
            runTest(configsYaml, true_clusters, resultsDic)
            
def testAgglomerative(alg, configsYaml, test_configs, true_clusters, resultsDic):
    for metric in test_configs["algorithms"][alg]["metric"]:
        configsYaml["slicing"]["clustering"]["metric"] = metric
        for linkage in test_configs["algorithms"][alg]["linkage"]:
            if linkage == "ward":
                configsYaml["slicing"]["clustering"]["metric"] = "euclidean"
            configsYaml["slicing"]["clustering"]["linkage"] = linkage
            if metric == "gaussian":
                for var in eval(test_configs["algorithms"][alg]["vars"]):
                    configsYaml["slicing"]["clustering"]["var"] = var
                    runTest(configsYaml, true_clusters, resultsDic)
            else:
                runTest(configsYaml, true_clusters, resultsDic)
                
def testMeanShift(alg, configsYaml, test_configs, true_clusters, resultsDic):
    for bandwidth in eval(test_configs["algorithms"][alg]["bandwidth"]):
        configsYaml["slicing"]["clustering"]["bandwidth"] = bandwidth
        runTest(configsYaml, true_clusters, resultsDic)
        
def testSpectral(alg, configsYaml, test_configs, true_clusters, resultsDic):
    for metric in test_configs["algorithms"][alg]["metric"]:
        configsYaml["slicing"]["clustering"]["metric"] = metric
        if metric == "gaussian":
            for var in eval(test_configs["algorithms"][alg]["vars"]):
                configsYaml["slicing"]["clustering"]["var"] = var
                runTest(configsYaml, true_clusters, resultsDic)
        else:
            runTest(configsYaml, true_clusters, resultsDic)
            
def testsGraph(alg, configsYaml, test_configs, true_clusters, resultsDic):
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
    
def plotResult(xLabel, yLabels, resultsFilPath):
    with open(resultsFilPath) as resultsFile:
        results = resultsFile.readlines()
    resultsDic = {}
    for result in results:
        x = getVals(result, [xLabel])[0]
        ys = getVals(result, yLabels)
        resultsDic[x] = ys
        
    sorted_results =sorted(resultsDic.items(), key=lambda t: t[0])
    xVals = []
    yVals_List = [[] for i in range(len(yLabels))]
    for res in sorted_results:
        xVals.append(res[0])
        i = 0
        for y in res[1]:
            yVals_List[i].append(y)
            i += 1
    i = 0
    plots = []
    legends = []
    for yVals in yVals_List:
        pl, = plt.plot(xVals, yVals, linewidth=1.0)
        plots.append(pl)
        legends.append(yLabels[i])
        i += 1
    legend(plots, legends, loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=len(yLabels))
    plt.grid(True)
    plt.xlabel(xLabel, fontsize=14)
    savefig('resultsPlot.png', bbox_inches='tight')
        
def getVals(result, labels):
    vals = []
    for label in labels:
        vals.append(float(result.split(label + ": ")[1].split(" ")[0].strip()))
    return vals    

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
