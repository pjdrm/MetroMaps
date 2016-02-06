'''
Base class for slicing procedures that are
based on graph community detection algorithms.

The idea is to find word communities from a word co-occurrence
graph and then assign documents to the discovered communities.
Documents assigned to the same community will be belong to the same "slice",
from which word clusters will be extracted in the next step of the pipeline. 
@author: Mota
'''
import snap
from itertools import combinations
from mm.input.slicing import slicer_factory
from igraph import plot
from colour import Color
from math import ceil
from matplotlib import pyplot
import matplotlib as mpl
import matplotlib.image as mpimg
import numpy as np
import os
import score.score_function_factory as sff
from random import randint
from gensim.models.word2vec import score_cbow_pair

class SlicingGraphBased(slicer_factory.SlicingHandlerGenerator):
    def __init__(self, slicer_configs):
        super(SlicingGraphBased, self).__init__(slicer_configs)
        self.doc_keys = self.data["doc_counts"].keys()
        self.token_to_word = {v: k for k, v in self.data["global_tokens"].items()}
        self.k = 5
        self.max_tokens = int(slicer_configs['maxNodes'])
        self.doc_com_score = sff.scoreFuncFactory(slicer_configs)
    
    '''
    Creates a co-occurrence graph from the set of documents. The graph is
    to be used by algorithms on the snap package.
    
    It should be noted that no weight on the edges is calculated. In practice
    it is not relevant as the algorithm in snap dont use weights anyway.
    '''  
    def createGraph(self):
        g = snap.TUNGraph.New()
        node_degrees = {}
        for doc_id in self.doc_keys:
            token_ids = []
            token_tfidf = {}
            for token_id in self.doc_counts[doc_id]:
                #if self.doc_counts[doc_id][token_id] < self.min_freq_in_doc:
                if self.num_docs_with_term.get(token_id, 1) < self.min_freq_in_doc:
                    continue
                token_tfidf[token_id] = self.tfidf(token_id, doc_id)
            token_ids = sorted(token_tfidf, key=lambda x : -token_tfidf[x])[:self.max_tokens]
          
            for (node_1, node_2) in combinations(token_ids,2):
                if not g.IsNode(int(node_1)):
                    g.AddNode(int(node_1))
                if not g.IsNode(int(node_2)):    
                    g.AddNode(int(node_2))
                g.AddEdge(int(node_1), int(node_2))
                node_degrees[int(node_1)] = node_degrees.get(int(node_1), 0) + 1
                node_degrees[int(node_2)] = node_degrees.get(int(node_2), 0) + 1
        
        '''
        counter = 0
        #gLoaded = snap.LoadEdgeList(snap.PNGraph, "graph_file.txt", 0, 1)
        for NI in g.Nodes():
            print "node: %d, out-degree %d, in-degree %d" % ( NI.GetId(), NI.GetOutDeg(), NI.GetInDeg())
            counter +=1
        print "number of nodes: %d" % (counter)
        '''
        return g
    
    def slice(self):
        communities = self.run()
        
        '''
        for cluster in communities:
            print "%d %s" % (len(cluster["cluster_tokens"]), cluster["cluster_tokens"])
        '''
        mapping = self.mapSegsComm(communities, self.token_to_word, self.data["doc_counts"])
        self.write(mapping)
        return mapping
    
    def printComms(self, communitites):
        print "Words in some community"
        tokenSet = set()
        for comm in communitites:
            for token in comm["cluster_tokens"]:
                tokenSet.add(token)
        print tokenSet
        print "Total tokens: %d\nNumber of communities: %d " % (len(tokenSet), len(communitites))
      
    '''
    Implements a method for assigning documents to a particular community.
    Each document has a score regarding a community and is assigned
    to the one with the highest score.
    '''          
    def mapSegsComm(self, communities, id_to_token, doc_counts):
        mapping = [ [] for com in communities]
        f = open(self.debugFile,'a')
        for doc_id in sorted(doc_counts.keys(), key=lambda x: int(x)):
            bestScore = 0.0
            bestComm = 0
            f.write(doc_id+"\n")
            for commIndex, comm in enumerate(communities):
                score, contributingWords = self.commScore(comm, doc_id)
                f.write(str(commIndex) + ": " + str(score) + " words: " + contributingWords + "\n")
                if score >= bestScore:
                    bestScore = score
                    bestComm = commIndex
            mapping[bestComm].append({"timestamp" : int(doc_id), "id" : int(doc_id), "name" : doc_id + ".txt"})
        f.close()
        return mapping
    
    '''
    Score function between a document and a community.
    The used function is defined in the yaml file (slicing/score_function)
    '''
    def commScore(self, comm, doc_id):
        return self.doc_com_score.score(comm, self, doc_id)
        
    '''
    Function that plots a word co-occurrence graph.
    Edges are colored according to their corresponding weights.
    
    Note: only work for graph object from the igraph package.
    '''
    def plotGraph(self, graph, nodesCommMembership, node_to_token_dic, figFilePath):            
        edges = []
        for edge in graph.es():
            if nodesCommMembership[edge.tuple[0]] != nodesCommMembership[edge.tuple[1]]:
                edges.append(edge)
        graph.delete_edges(edges)
        
        '''
        Creating gradient color scheme for the graph edges. Green edges
        have the minimum found weight and red edges the maximum.
        '''
        n_colors = 20
        green = Color("green")
        colors = list(green.range_to(Color("red"), n_colors))
        weights = graph.es['weight']
        maxW = max(weights)
        minW = min(weights)
        binSize = (maxW - minW)/n_colors
        edgeColors = []
        for w in weights:
            colorBin = int(ceil((w-minW)/binSize)-1)
            if colorBin == -1:
                colorBin = 0
            color = colors[colorBin]
            edgeColors.append(color.get_rgb())
            
        if os.path.isfile(figFilePath):
            os.remove(figFilePath)
        fig = pyplot.figure(figsize=(15, 15))
        axplot = fig.add_axes([0.15, 0.07, 0.7, 0.06], label='Edge weight value')
        cmap = mpl.colors.ListedColormap([color.get_rgb() for color in colors])
        norm = mpl.colors.Normalize(vmin=minW, vmax=maxW)
        colorBar = mpl.colorbar.ColorbarBase(axplot, cmap=cmap,
                                   norm=norm,
                                   orientation='horizontal')
        colorBar.set_label('Edge weight value')
        ax = colorBar.ax
        text = ax.xaxis.label
        font = mpl.font_manager.FontProperties(size=18)
        text.set_font_properties(font)
        
        degrees = graph.degree()
        #nodeLabels = [self.token_to_word[node_to_token_dic[nodeIndex]] + " (" + str(degree) + ")"  for nodeIndex, degree in enumerate(degrees)]
        nodeLabels = [self.token_to_word[node_to_token_dic[nodeIndex]]  for nodeIndex, degree in enumerate(degrees)]
        visual_style = {}
        
        
        bins = np.linspace(0, max(degrees), n_colors)  
        digitized_degrees =  np.digitize(degrees, bins)
        graph.vs["color"] = [colors[x-1] for x in digitized_degrees]
        N = graph.vcount()
        #visual_style["layout"] = graph.layout_fruchterman_reingold(weights=graph.es["weight"], maxiter=4000, area=N**3, repulserad=N**3)
        visual_style["layout"] = graph.layout("kk")
        
        visual_style["vertex_label"] = nodeLabels
        visual_style["vertex_label_size"] = 12
        visual_style["vertex_size"] = 6
        color_list = ['red',  'blue', 'cyan', 'purple', 'white', 'black', '#84D3FD', '#16597D', '#9F5EC5', '#4E1C6B', '#D76602', '#8E3B35', '#F0DE16', '#FF2841', '#ADFF2F', '#20B2AA', '#9370DB', '#FFDEAD', '#808000', '#AFEEEE', '#8B4513', '#87CEEB', '#EE82EE', '#FF6347', '#008080', '#FF7F50', '#DC143C']
        vertex_color = [color_list[x] for x in nodesCommMembership]
        visual_style["vertex_color"] = vertex_color
        
        color_count = {}
        for vertex in vertex_color:
            if not vertex in color_count:
                color_count[vertex] = 0
            color_count[vertex] += 1
        index = 0
        for color in color_count:
            tempIndex = color_count[color]
            color_count[color] = index
            index += tempIndex
        vertexOrder = []
        for vertex in vertex_color:
            vertexOrder.append(color_count[vertex])
            color_count[vertex] += 1
        #visual_style["vertex_order"] = vertexOrder
        
        visual_style["vertex_label_dist"] = 1
        visual_style["edge_color"] = edgeColors
        plot(graph, figFilePath, margin = 40, **visual_style)
        img = mpimg.imread(figFilePath)
        axplot = fig.add_axes([0.01, 0.15, 1, 0.8])
        axplot.axis('off')
        axplot.imshow(img)
        fig.savefig(figFilePath)
