'''
Provides the interface for the specific Bigclam algorithm
to be used for word community detection.

To use this algorithm, in the .yaml configuration write the name of this module.
(slicing: type: slicing_bigclam)

@author: Mota
'''
import snap
import os
import subprocess
import mm.input.slicing.graph.slicing_graph_based as slicing_graph_based

class SlicingBigClam(slicing_graph_based.SlicingGraphBased):
    def __init__(self, legacy_helper_config_dict):
            super(SlicingBigClam, self).__init__(legacy_helper_config_dict)
            self.g = self.createGraph()
    
    '''
    The actual Bigclam code is in an executable in the resources/bigclam directory. This
    function call that executable and processes its corresponding output (resources/bigclam/cmtyvv.txt).
    '''
    def bigclam(self):    
        def call_process(args, binary_name):
            
            # Open a stream for the standard output
            _stdout_fname = binary_name.split(".exe")[0]+'.stdout'
            disk_stdout = open(_stdout_fname, 'w')
            # Write the command lines we are running
            disk_stdout.write("Command line: %s\n"%repr(list(args)))
            disk_stdout.flush()
    
            ret = None
            try:
                ret = subprocess.call(args)
            finally:
                print open(binary_name.split(".exe")[0]+'.stdout').read()         
                #print "call_process time elapsed (s):", time.time() - time_start
                disk_stdout.write('\n\n\n==========\n')
                disk_stdout.write("Return value: %s\n"%ret)
                disk_stdout.flush()
            os.rename("cmtyvv.txt", "resources/bigclam/cmtyvv.txt")
                
        def processBigclamOutput(output_file):
            with open(output_file) as f:
                communities = f.readlines()
            community_list = []
            for community in communities:
                cluster_k = []
                for Node in community.split():
                    token = self.id_to_token[int(Node)]
                    cluster_k += [token]
                cluster_d = {'cluster_tokens': cluster_k}
                community_list += [cluster_d]
            return community_list
    
        output_file = "resources/bigclam/cmtyvv.txt"
        input_file = "resources/bigclam/input_graph.txt"
        snap.SaveEdgeList_PUNGraph(self.g, input_file, "Save as tab-separated list of edges")
        _binary = 'resources/bigclam/bigclam.exe'
        _threads = None
        q = -1   # Auto-detect q
        _q_doc = """Number of communities to detect (-1 autodetect, but within the given range)"""
        q_max = None  # default 100
        q_min = None  # default 5
        q_trials = None
        _q_trials_doc = """How many trials for the number of communities"""
        alpha = None
        _alpha_doc = "Alpha for backtracking line search"
        beta = None
        _beta_doc = "Beta for backtracking line search"

        args = [_binary]
        args.append('-i:%s' % input_file)
        if _threads:          args.append('-nt:%s'%str(_threads))
        if q is not None:     args.append('-c:%d'%q)
        if q_min:             args.append('-mc:%d'%q_max)
        if q_max:             args.append('-xc:%d'%q_min)
        if q_trials:          args.append('-nc:%d'%q_trials)
        if alpha is not None: args.append('-sa:%f'%alpha)
        if beta  is not None: args.append('-sb:%f'%beta)
        call_process(args, _binary)
        communities = processBigclamOutput(output_file)
        return communities
    
    def run(self):
        return self.bigclam()
    
def construct(config):
    return SlicingBigClam(config)
