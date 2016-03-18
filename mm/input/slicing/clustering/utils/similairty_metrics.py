'''
Created on 18/03/2016

@author: Mota
'''
from math import exp
import numpy as np
from scipy import spatial

var = 100

def gaussianSim(xi, xj):
    return exp((-1 * np.linalg.norm(xi - xj) ** 2) / (2 * var))

def gaussianSimGraph(cluster_elms):
    sim_graph = np.zeros([cluster_elms.shape[0], cluster_elms.shape[0]])
    for i, j in np.ndindex(sim_graph.shape):
        sim = gaussianSim(cluster_elms[i], cluster_elms[j])
        if sim >= 0:
            sim_graph[i, j] = sim
    return sim_graph

def genSimGraph(cluster_elms, metric):
        sim_graph = np.zeros([cluster_elms.shape[0], cluster_elms.shape[0]])
        for i, j in np.ndindex(sim_graph.shape):
            if metric == "gaussian":
                sim = gaussianSim(cluster_elms[i], cluster_elms[j])
            elif metric == "cosine":
                sim = spatial.distance.cosine(cluster_elms[i], cluster_elms[j])
            elif metric == "euclidean":
                sim = spatial.distance.euclidean(cluster_elms[i], cluster_elms[j])
                
            if sim >= 0:
                sim_graph[i, j] = sim
                
        return sim_graph 
