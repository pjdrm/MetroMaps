'''
Created on 15/07/2015

@author: Mota
'''
import logging
import json
import os

class MetroStationGenerator(object):
    '''
    Class that the different implementations of metro stations generators must inherit.
    Provides the common implementation of writing the obtained metro stations 
    '''


    def __init__(self, config):
        self.output_JSON = config['output_json']
        self.out_legacy_dir = config.get('out_legacy_dir')
        if not os.path.exists(self.out_legacy_dir):
            logging.info('Created directory %s' % self.out_legacy_dir)
            os.makedirs(self.out_legacy_dir)
        
    def write(self):
        if not self.timeslice_clusters:
            logging.error('Run has not been run yet or there are no timeslices available')
        else:
            with open(self.output_JSON, 'w') as outjson:
                json.dump(self.timeslice_clusters, outjson)

            if self.out_legacy_dir: 
                for i in range(len(self.timeslices)):
                    timeslice_start_date = self.timeslices[i]['cluster_start_date']
                    timeslice_end_date = self.timeslices[i]['cluster_end_date']
                    filename = 'clusters_%s_%s' % (timeslice_start_date, timeslice_end_date)
                    with open(os.path.join(self.out_legacy_dir, filename), 'w') as legacy_out_cluster:
                        for cluster in self.timeslice_clusters[i]:
                            tokens = cluster['cluster_tokens']
                            tokens_joined = ', '.join(tokens)
                            num_tokens = len(tokens)
                            text = 'Cluster: %i %s\n' % (num_tokens, tokens_joined)
                            legacy_out_cluster.write(text.encode('utf-8'))
        logging.info('Clusters written to %s' % self.output_JSON)
        logging.info('Legacy clusters written to %s' % self.out_legacy_dir)
        
def factory(configs):
    metro_station_generator_configs = configs
    input_generator_name = metro_station_generator_configs['type']
    generator_module = __import__(input_generator_name, globals=globals())
    return generator_module.construct(metro_station_generator_configs)