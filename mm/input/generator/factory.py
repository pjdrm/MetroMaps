'''
Created on 07/07/2015

Class for generating the different possible input generators.
Note: the 'type' parameter must match the class name.

@author: Mota
'''

def Generate(configs):
    ''' Specify a dictionary with configurations '''
    input_generator_configs = configs
    input_generator_name = input_generator_configs['type']
    generator_module = __import__(input_generator_name, globals=globals())
    return generator_module.construct(input_generator_configs)