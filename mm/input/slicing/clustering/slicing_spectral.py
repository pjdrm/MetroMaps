'''
Provides the interface for the specific Spectral Clustering algorithm.

To use this algorithm, in the .yaml configuration write the name of this module.
(slicing: type: slicing_spectral)

@author: Mota
'''
import numpy as np
import mm.input.slicing.clustering.slicing_cluster_based as slicing_cluster_based
from math import exp
from sklearn.cluster import spectral_clustering
from sklearn import metrics
import pylab
from scipy import spatial

class SlicingSpectral(slicing_cluster_based.SlicingClusterBased):
    def __init__(self, slicer_configs):
        super(SlicingSpectral, self).__init__(slicer_configs)
        ranges = slicer_configs["clustering"]["varRange"]
        self.varRange = range(ranges[0], ranges[1])
        
    def genSimGraph(self, cluster_elms, var):
        sim_graph = np.zeros([cluster_elms.shape[0], cluster_elms.shape[0]])
        for i, j in np.ndindex(sim_graph.shape):
            sim = self.gaussianSim(cluster_elms[i], cluster_elms[j], var)
            if sim >= 0:
                sim_graph[i, j] = sim
                
        return sim_graph
            
    def gaussianSim(self, xi, xj, var):
        return exp((-1 * np.linalg.norm(xi - xj) ** 2) / (2 * var))
        #return spatial.distance.euclidean(xi, xj)
        
        
    def spectral(self, sim_graph):
        labels = spectral_clustering(sim_graph, n_clusters=self.num_clusters, eigen_solver='arpack')
        return labels
    
    def plotHM(self, elms, var, trueLabels):
        orderedElms = np.zeros([len(elms), len(elms[0])])
        orderDic = {}
        for i in range(len(elms)):
            clusterID = trueLabels[i]
            if clusterID not in orderDic:
                orderDic[clusterID] = []
            orderDic[clusterID].append(elms[i])
            
        i = 0
        prevClusterIndex = 0
        clusterIndexes = []
        for cluster in orderDic:
            clusterIndexes.append(len(orderDic[cluster]) + prevClusterIndex)
            prevClusterIndex += len(orderDic[cluster])
            for clusterElm in orderDic[cluster]:
                orderedElms[i] = clusterElm
                i += 1
                
        sim_graph = self.genSimGraph(orderedElms, var)
        fig = pylab.figure(figsize=(8,8))
        axmatrix = fig.add_axes([0.3,0.1,0.6,0.6])
        im = axmatrix.matshow(sim_graph, aspect='auto', cmap=pylab.cm.YlGnBu)
        labels = ["C" + str(i) for i in range(1, len(clusterIndexes)+1)]
        axmatrix.set_xticks(clusterIndexes[:-1])
        axmatrix.get_xaxis().set_tick_params(direction='out')
        #axmatrix.set_xticklabels(labels)
        axmatrix.set_yticks(clusterIndexes[:-1])
        axmatrix.get_yaxis().set_tick_params(direction='out')
        #axmatrix.set_yticklabels(labels)
        axcolor = fig.add_axes([0.91,0.1,0.02,0.6])
        pylab.colorbar(im, cax=axcolor)
        fig.show()
        fig.savefig('colormap.png')
    
    def run(self):
        riBest = -1.0
        labels = None
        bestVar = 0.0
        for var in self.varRange:
            sim_graph = self.genSimGraph(self.cluster_elms, var)
            try:
                labels = self.spectral(sim_graph)
                ri = metrics.adjusted_rand_score(self.true_labels, labels)
                if ri >= riBest:
                    riBest = ri
                    bestLabels = labels
                    bestVar = var
            except Exception: 
                pass
        print "Best Var %f" % (bestVar)
        self.plotHM(self.cluster_elms, bestVar, self.true_labels)
        return bestLabels
        
def construct(config):
    return SlicingSpectral(config) 