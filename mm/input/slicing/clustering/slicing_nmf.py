'''
Created on Feb 9, 2016

@author: pjdrm
'''
import numpy as np
import mm.input.slicing.clustering.slicing_cluster_based as slicing_cluster_based
from sklearn.utils import check_arrays
from sklearn.utils import warn_if_not_float
from sklearn.utils.sparsefuncs_fast import inplace_csr_row_normalize_l1
from sklearn.utils.sparsefuncs_fast import inplace_csr_row_normalize_l2
from sklearn.utils.extmath import row_norms
from sklearn.cluster import KMeans
from scipy import sparse
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from operator import div

class SlicingNMF(slicing_cluster_based.SlicingClusterBased):
    def __init__(self, slicer_configs):
        super(SlicingNMF, self).__init__(slicer_configs)
        self.cluster_elms = self.loadClusterElms(self.cluster_elms)
        self.desc = "NMF"
        
    def loadClusterElms(self, clusterElem):
        return sp.csr_matrix(self.cluster_elms,dtype='float64') 
    
    def nmf_clustering(self, norm = "l2", seed="random", post = "direct", gt=None):
        """
        NMF with Euclidean distance as the cost function.
        For comments on input parameters, please refer to conmf.conmf().
        """
        k = self.num_clusters
        data_norm = self.norm_data(self.cluster_elms, "l2")
        
        # print "Running NMF on a matrix with size ",data.shape
        #nmf_model = nimfa.mf(data_norm, method = "nmf", max_iter = 200, min_residuals = 0.001,n_run =1, rank = k, update = 'euclidean', objective = 'div')
        W,H = self.factorize(data_norm, seed, post , norm, gt, k) #W is m*k, H is k*n
        
        targets = self.get_targets(W.T,post) # clustering results. 
        
        return targets
    
    def run(self):
        return self.nmf_clustering()
    
    def norm_data(self, data,norm):
        """
        norm = 'l1', 'l2' or 'l0' or 'l2+l0'...
            'l0': normalize the data matrix as the sum of all entries=1
        """
        data_norm = data
        norms = norm.split("+")# the sequence of norm
        for norm in norms: 
            if norm in ['l1','l2']:
                data_norm = self.normalize(data_norm,norm,axis=1,copy=True) # donot change the original input data
            if norm == "l0":
                _sum = data.sum()
                data_norm = data_norm/_sum
    
        return data_norm
    
    def normalize(self, X, norm='l2', axis=1, copy=True):
        """Normalize a dataset along any axis
    
        Parameters
        ----------
        X : array or scipy.sparse matrix with shape [n_samples, n_features]
            The data to normalize, element by element.
            scipy.sparse matrices should be in CSR format to avoid an
            un-necessary copy.
    
        norm : 'l1' or 'l2', optional ('l2' by default)
            The norm to use to normalize each non zero sample (or each non-zero
            feature if axis is 0).
    
        axis : 0 or 1, optional (1 by default)
            axis used to normalize the data along. If 1, independently normalize
            each sample, otherwise (if 0) normalize each feature.
    
        copy : boolean, optional, default is True
            set to False to perform inplace row normalization and avoid a
            copy (if the input is already a numpy array or a scipy.sparse
            CSR matrix and if axis is 1).
    
        See also
        --------
        :class:`sklearn.preprocessing.Normalizer` to perform normalization
        using the ``Transformer`` API (e.g. as part of a preprocessing
        :class:`sklearn.pipeline.Pipeline`)
        """
        if norm not in ('l1', 'l2'):
            raise ValueError("'%s' is not a supported norm" % norm)
    
        if axis == 0:
            sparse_format = 'csc'
        elif axis == 1:
            sparse_format = 'csr'
        else:
            raise ValueError("'%d' is not a supported axis" % axis)
    
        X = check_arrays(X, sparse_format=sparse_format, copy=copy)[0]
        warn_if_not_float(X, 'The normalize function')
        if axis == 0:
            X = X.T
    
        if sparse.issparse(X):
            if norm == 'l1':
                inplace_csr_row_normalize_l1(X)
            elif norm == 'l2':
                inplace_csr_row_normalize_l2(X)
        else:
            if norm == 'l1':
                norms = np.abs(X).sum(axis=1)
                norms[norms == 0.0] = 1.0
            elif norm == 'l2':
                norms = row_norms(X)
                norms[norms == 0.0] = 1.0
            X /= norms[:, np.newaxis]
    
        if axis == 0:
            X = X.T
    
        return X
    
    def get_targets(self, H, method="k-means", k = 21): # k is #cluster. H is the k*n coefficient matrix
        if method.lower() not in ["k-means","kmeans","direct"]:
            print "Error! The method of post: data_load.get_targets is wrong!"
        col = H.shape[1]
        if method == "k-means" or method=="kmeans":
            H = H.todense()
            H[np.isnan(H)]=0
            H = self.normalize(H.T,'l2',axis=1,copy=False)
            
            km_model = KMeans(n_clusters=k, init='random', max_iter=500, n_init = 10, n_jobs = 1, verbose = False)
            km_model.fit(H)
            return km_model.labels_
        if method == "direct":
            if sp.issparse(H)==True:
                H = H.todense()
            H = np.array(H)
            targets = []
            for i in range(0,col):
                h_i = H[:,i]
                col_i = h_i.tolist()
                cluster = col_i.index(max(col_i))
                targets.append(cluster)
        return np.array(targets)
    
    def factorize(self, data, seed, post, norm, gt, rank, max_iter=200):
        """
        The factorization of NMF, data = W*H. 
        The input gt (groundtruth) is only for monitoring performance of each iteration.
        
        Note: since calculating the cost function is too slow, we can only use the number of iteration as the stopping critera for efficiency issue. 
        Return: W (m*k) and H (k*n) matrix.
        """   
        V = data
        W, H = self.initialize(V, rank, seed=seed, norm=norm)
        iter = 0    
        while iter <= max_iter:
            targets = self.get_targets(W.T,post)
            """
            #Add a function of counting #items in each cluster
            clusters = np.unique(targets)
            count_arr = [0 for i in range(0,len(clusters))]
            for c in targets:
                count_arr[c]+=1
            print sorted(count_arr)
            """
            if gt!=None:
                A = eval_metrics.accuracy(gt,targets)
                F1 = eval_metrics.f_measure(gt,targets)
                #print "Iter = %d, Acc = %f, F1 = %f" %(iter,A,F1)
            
            W, H = self.euclidean_update(V, W, H, norm)
            W, H = self._adjustment(W, H)
            iter += 1
    
        return W, H
    
    def initialize(self, V, rank, seed='random', norm='l2'):
        W,H = self.initialize_random(V, rank)
        if seed == "k-means":
            kmeans_results = self.kmeans(V.astype('float64'), rank, norm)
            labels = kmeans_results[0]
            H = kmeans_results[1]
            W = self.labels_to_matrix(labels, W)
        return W,H
    
    def initialize_random(self, V, rank):
        """
        Randomly initiate W and H matrix to run NMF for V
        
        :param V: Data matrix to run NMF for.
        :type V:  class:`scipy.sparse` of format csr, csc, coo, bsr, dok, lil, dia or :class:`numpy.matrix`
        :rank:    number of latent factors in NMF estimation, i.e. the rank of W and H matrix.
        :type rank: int.
        """
        max_V = V.data.max()
        W = np.mat(np.random.RandomState().uniform(0, max_V, (V.shape[0], rank)))    
        H = np.mat(np.random.RandomState().uniform(0, max_V, (rank, V.shape[1])))
        
        W = sp.csr_matrix(W)
        H = sp.csr_matrix(H)
        
        return W, H
    
    def kmeans(self, data, k, norm="l2", n_init = 1):
        """
        data: matrix, #item * #feature
        """
        if norm == None:
            km_model = KMeans(n_clusters=k, init='random', max_iter=500, n_init = n_init, n_jobs = 1, verbose = False)
            km_model.fit(data)
            return km_model.labels_, km_model.cluster_centers_
            
        data_norm = self.norm_data(data, norm)
        km_model = KMeans(n_clusters=k, init='random', max_iter=500, n_init = n_init, n_jobs = 1, verbose = False)
        km_model.fit(data_norm)
        # km_model.cluster_centers_ is k*N of <type 'numpy.ndarray'>
        # H: converted km_model.cluster_centers_ to csr_matrix, shape: k*N
        H = csr_matrix(km_model.cluster_centers_)
        H = H.todense()
        H = H + 0.1 # Add a small number to each element of the centroid matrix
        H_norm = self.norm_data(H, norm)
        
        return km_model.labels_, H_norm
    
    def labels_to_matrix(self, labels, W):
        """
        Change the input W matrix s.t. the largest element of each item vector is its cluster assignment (as in labels).
        
        :param labels: the cluster assignment of items.
        :type labels: list. 
        :param W: the item latent matrix (m*k)
        :type W: class:`scipy.sparse` of format csr, csc, coo, bsr, dok, lil, dia or :class:`numpy.matrix`
        """
        for i in range(0,len(labels)):
            j = labels[i]
            row_i = W.getrow(i)
            if row_i.getnnz() > 0:
                max_index = row_i.indices[np.argmax(row_i.data)]
                max_value = np.max(row_i.data)
                #swap the value of W[i,j] and W[i,max_index]
                t = W[i,j]
                W[i,j] = max_value
                W[i,max_index] = t
            else:
                W[i,j] = 1.0
        return W
    
    def euclidean_update(self, V, W, H, norm):
        """Update basis and mixture matrix based on Euclidean distance multiplicative update rules."""
        
        up = self.dot(W.T, V)
        down = self.dot(self.dot(W.T, W), H)
        elop_div = self.elop(up, down, div)
        
        H = self.multiply(H, elop_div)
        W = self.multiply(W, self.elop(self.dot(V, H.T), self.dot(W, self.dot(H, H.T)), div))
        
        return W, H
    
    def dot(self, X, Y):
        """
        Compute dot product of matrices :param:`X` and :param:`Y`.
        
        :param X: First input matrix.
        :type X: :class:`scipy.sparse` of format csr, csc, coo, bsr, dok, lil, dia or :class:`numpy.matrix`
        :param Y: Second input matrix. 
        :type Y: :class:`scipy.sparse` of format csr, csc, coo, bsr, dok, lil, dia or :class:`numpy.matrix` 
        """
        if sp.isspmatrix(X) and sp.isspmatrix(Y):
            return X * Y
        elif sp.isspmatrix(X) or sp.isspmatrix(Y):
            # avoid dense dot product with mixed factors
            return sp.csr_matrix(X) * sp.csr_matrix(Y)
        else:
            return np.asmatrix(X) * np.asmatrix(Y)
        
    def multiply(self, X, Y):
        """
        Compute element-wise multiplication of matrices :param:`X` and :param:`Y`.
        
        :param X: First input matrix.
        :type X: :class:`scipy.sparse` of format csr, csc, coo, bsr, dok, lil, dia or :class:`numpy.matrix`
        :param Y: Second input matrix. 
        :type Y: :class:`scipy.sparse` of format csr, csc, coo, bsr, dok, lil, dia or :class:`numpy.matrix` 
        """
        if sp.isspmatrix(X) and sp.isspmatrix(Y):
            return X.multiply(Y)
        elif sp.isspmatrix(X) or sp.isspmatrix(Y):
            return self._op_spmatrix(X, Y, np.multiply) 
        else:
            with self.warnings.catch_warnings():
                self.warnings.simplefilter('ignore')
                return np.multiply(np.mat(X), np.mat(Y))
        
    def elop(self, X, Y, op):
        """
        Compute element-wise operation of matrix :param:`X` and matrix :param:`Y`.
        
        :param X: First input matrix.
        :type X: :class:`scipy.sparse` of format csr, csc, coo, bsr, dok, lil, dia or :class:`numpy.matrix`
        :param Y: Second input matrix.
        :type Y: :class:`scipy.sparse` of format csr, csc, coo, bsr, dok, lil, dia or :class:`numpy.matrix`
        :param op: Operation to be performed. 
        :type op: `func` 
        """
        try:
            zp1 = op(0, 1) if sp.isspmatrix(X) else op(1, 0)
            zp2 = op(0, 0) 
            zp = zp1 != 0 or zp2 != 0
        except:
            zp = 0
        if sp.isspmatrix(X) or sp.isspmatrix(Y):
            return self._op_spmatrix(X, Y, op) if not zp else self._op_matrix(X, Y, op)
        else:
            try:
                X[X == 0] = np.finfo(X.dtype).eps
                Y[Y == 0] = np.finfo(Y.dtype).eps
            except ValueError:
                return op(np.mat(X), np.mat(Y))
            return op(np.mat(X), np.mat(Y))
        
    def _op_spmatrix(self, X, Y, op):
        """
        Compute sparse element-wise operation for operations preserving zeros.
        
        :param X: First input matrix.
        :type X: :class:`scipy.sparse` of format csr, csc, coo, bsr, dok, lil, dia or :class:`numpy.matrix`
        :param Y: Second input matrix.
        :type Y: :class:`scipy.sparse` of format csr, csc, coo, bsr, dok, lil, dia or :class:`numpy.matrix`
        :param op: Operation to be performed. 
        :type op: `func` 
        """
        # distinction as op is not necessarily commutative
        return self.__op_spmatrix(X, Y, op) if sp.isspmatrix(X) else self._op_spmatrix(Y, X, op)
    
    def __op_spmatrix(self, X, Y, op):
        """
        Compute sparse element-wise operation for operations preserving zeros.
        
        :param X: First input matrix.
        :type X: :class:`scipy.sparse` of format csr, csc, coo, bsr, dok, lil, dia
        :param Y: Second input matrix.
        :type Y: :class:`numpy.matrix`
        :param op: Operation to be performed. 
        :type op: `func` 
        """
        assert X.shape == Y.shape, "Matrices are not aligned."
        eps = np.finfo(Y.dtype).eps if not 'int' in str(Y.dtype) else 0
        Xx = X.tocsr()
        r, c = Xx.nonzero()
        R = op(Xx[r,c], Y[r,c]+eps)
        R = np.array(R)
        assert 1 in R.shape, "Data matrix in sparse should be rank-1."
        R = R[0, :] if R.shape[0] == 1 else R[:, 0]
        return sp.csr_matrix((R, Xx.indices, Xx.indptr), Xx.shape)
    
    def _adjustment(self, W, H):
        """Adjust small values to factors to avoid numerical underflow."""
        H = max(H, np.finfo(H.dtype).eps)
        W = max(W, np.finfo(W.dtype).eps)
            
        return W, H
        
def construct(config):
    return SlicingNMF(config) 
