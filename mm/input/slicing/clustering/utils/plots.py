'''
Created on 18/03/2016

@author: Mota
'''
import numpy as np
import pylab
import mm.input.slicing.clustering.utils.similairty_metrics as similairty_metrics
import pickle
import matplotlib.pyplot as plt

def plotHM(elms, trueLabels, metric, var=100):
    similairty_metrics.var = var
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
            
    sim_graph = similairty_metrics.genSimGraph(orderedElms, metric)
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
    
def plotCLustrs(true_labels, hyp_labels, outDir):
    dist = 5
    colors = []
    palete = ['red','green','blue', 'yellow', 'orange', 'purple', 'black', 'cyan', 'pink', 'grey', 'brown', 'white', 'chartreuse', 'salmon', 'orchid', 'gold', 'lightgrey', 'skyblue', 'khaki', 'navy', 'lightgreen', 'tomato', 'crimson', 'violet', 'cadetblue', 'teal', 'darkgreen' ]
    colorIndex = 0
    pointsDic = {}
    labelsDic = {}
    
    for i, tl in enumerate(true_labels):
        if tl not in labelsDic:
            labelsDic[tl] = {}
            labelsDic[tl]["dist"] = dist
            labelsDic[tl]["color"] = palete[colorIndex]
            colorIndex += 1
            dist += 5
        point = np.random.randn(1, 2)[0] + labelsDic[tl]["dist"]
        pointsDic[i] = (point, labelsDic[tl]["color"])
    
    output = open('pointDic.pkl', 'wb')
    pickle.dump(pointsDic, output)
    output.close()
    
    '''
    pkl_file = open('pointDic.pkl', 'rb')
    pointsDic = pickle.load(pkl_file)
    '''
    
    f = plt.figure(1)
    x = []
    y = []
    colors = []
    for p in pointsDic:
        x.append(pointsDic[p][0][0])
        y.append(pointsDic[p][0][1])
        colors.append(pointsDic[p][1])
        
    plt.tick_params(
        axis='both',
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        left='off',      # ticks along the bottom edge are off
        right='off',
        top='off',
        labelbottom='off',
        labeltop='off',
        labelright='off',
        labelleft='off')
    plt.scatter(x, y, s=100, c=colors)
    f.savefig(outDir + "/true_clusters.png")
    
    f2 = plt.figure(2)
    colors = []
    hlabelsDic = {}
    colorIndex = 0
    for i, hl in enumerate(hyp_labels):
        if hl not in hlabelsDic:
            hlabelsDic[hl] = palete[colorIndex]
            colorIndex += 1
        colors.append(hlabelsDic[hl])
        
    plt.tick_params(
        axis='both',
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        left='off',      # ticks along the bottom edge are off
        right='off',
        top='off',
        labelbottom='off',
        labeltop='off',
        labelright='off',
        labelleft='off')
    plt.scatter(x, y, s=100, c=colors)
    plt.scatter(x, y, s=100, c=colors)
    f2.savefig(outDir + "/hyp_clusters.png")

plotCLustrs([2, 0, 2, 0, 1, 2, 0, 1, 2, 0, 0, 2, 0, 0, 1, 1, 2, 2, 2, 2, 2, 2, 2], [2, 3, 1, 0, 4, 1, 2, 3, 1, 2, 0, 2, 2, 2, 2, 4, 2, 1, 2, 3, 1, 1, 0], "C:\Users\Mota\workspace\MetroMaps")

