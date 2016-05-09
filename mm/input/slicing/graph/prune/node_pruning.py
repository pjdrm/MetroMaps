'''
Created on 06/05/2016

@author: Mota
'''
import numpy as np
import scipy.stats as stats

def prune(pruneConfigs, graph, igraphWrapper):
    if pruneConfigs != None:
        prune_type = pruneConfigs["node"].keys()[0]
        if  prune_type == "normal_dist":
            threshold = pruneConfigs["node"]["normal_dist"]
            normalDistPruning(graph, threshold, igraphWrapper)
        elif prune_type == "normal_dist_low_tail":
            threshold = pruneConfigs["node"]["normal_dist_low_tail"]
            normalDistPruningLowTail(graph, threshold, igraphWrapper)
        elif prune_type == "topN":
            threshold = pruneConfigs["node"]["topN"]
            topNPruning(graph, threshold, igraphWrapper)
        return prune_type + " " + str(threshold)
    else:
        return ""
            
'''
Prunes the nodes on the lower and upper tail of
a normal distribution.

probArea: the probability area to keep
'''
def normalDistPruning(graph, probArea, igraphWrapper):
    percentil = probArea + (1 - probArea)/2.0
    node_degrees = graph.degree()
    mean = np.mean(node_degrees)
    std =  np.std(node_degrees)
    pUp = stats.norm(mean, std).ppf(percentil)
    pLow = stats.norm(mean, std).ppf((1.0 - probArea)/2.0)
    to_del_nodes = []
    for nodeId, deg in enumerate(node_degrees):
        if deg >= pLow and deg <= pUp:
            continue
        to_del_nodes.append(nodeId)
    
    igraphWrapper.deleteNodes(to_del_nodes, graph)

'''
Prunes the nodes on the lower tail of
a normal distribution.

tailProbArea: the probability area to discard
'''   
def normalDistPruningLowTail(graph, tailProbArea, igraphWrapper):
    node_degrees = graph.degree()
    mean = np.mean(node_degrees)
    std =  np.std(node_degrees)
    pLow = stats.norm(mean, std).ppf(tailProbArea)
    to_del_nodes = []
    for nodeId, deg in enumerate(node_degrees):
        if deg >= pLow:
            continue
        to_del_nodes.append(nodeId)
    
    igraphWrapper.deleteNodes(to_del_nodes, graph)

'''
Prunes the nodes that are not in the top-N highest degree values.

n: number of nodes to keep
'''     
def topNPruning(graph, n, igraphWrapper):
    node_degrees = graph.degree()
    np_degrees = np.array(node_degrees)
    to_keep_nodes = np.argsort(-np_degrees)[:n]
    to_del_nodes = set(range(len(node_degrees))) - set(to_keep_nodes)
    igraphWrapper.deleteNodes(to_del_nodes, graph)