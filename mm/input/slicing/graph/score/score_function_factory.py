'''
Created on 01/02/2016

@author: Mota
'''
def scoreFuncFactory(slicer_configs):
    score_func =  slicer_configs['score_function']
    fromlist = [0]
    score_func_module = __import__(score_func, globals=globals(), fromlist = fromlist)
    return score_func_module.construct()
