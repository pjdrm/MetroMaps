#! /usr/bin/env python2.7

import argparse
import mm.inputhelpers.factory
import mm.inputhelpers.preprocessing
import mm.input
import mm.mapgen
import mm.viz
import logging
import yaml.composer
import os
import mm.input.generator.factory
import mm.input.slicing.slicer_factory

def rmDir(top):
    for root, dirs, files in os.walk(top, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))

def Run_init():
    if os.path.isdir("/tmp"):
        rmDir("/tmp")
    else:
        os.makedirs("/tmp")

def Run_input_generator(configs):
    input_generator_configs = configs.get('input_generator')
    if (input_generator_configs.get('mode')):
        logging.debug(input_generator_configs)
        logging.info("Running input generator handler")
        ig_handler = mm.input.generator.factory.Generate(input_generator_configs)
        ig_handler.run()
    else:
        logging.info("Skipping input generator")
        
def Run_input_preprocessing(configs):
    input_preprocessing_configs = configs.get('input_preprocessing')
    if (input_preprocessing_configs.get('mode')):
        logging.debug(input_preprocessing_configs)
        logging.info("Running input preprocessing")
        handler_preprocessing = mm.inputhelpers.preprocessing.TextPreprocessing(input_preprocessing_configs)
        handler_preprocessing.run()
    else:
        logging.info("Skipping preprocessing")
        
    
def Run_input_handler(configs):
    input_helper_configs = configs.get('input_helper')
    if (input_helper_configs.get('mode')):
        logging.debug(input_helper_configs)
        logging.info("Running input handler")
        handler_input = mm.inputhelpers.factory.ReadConfig(configs.get('input_helper'))
        handler_input.run()
        handler_input.save()
    else:
        logging.info("Skipping input handler")


def Run_slicing_handler(configs):
    legacy_configs = configs.get('slicing')
    if (legacy_configs.get('mode')):
        logging.info("Converting to legacy format")
        legacy_handler = mm.input.slicing.slicer_factory.factory(legacy_configs)
        legacy_handler.slice()
        logging.info("Legacy format written to %s" %(legacy_configs.get('output_dir')))

def Run_clustering_handler(configs):
    clustering_configs = configs.get('clustering',{})
    if (clustering_configs.get('mode')):
        logging.info("Running clustering handler")
        metro_station_generator_handler = mm.mapgen.metro_station_generator.factory(configs.get('clustering'))
        metro_station_generator_handler.run()
        metro_station_generator_handler.write()

def Run_map_generator(configs):
    map_gen_configs = configs.get('mapgen')
    if (map_gen_configs.get('mode')):
        logging.info("Running map generation")
        mapgen_handler = mm.mapgen.legacy_generator.LegacyGenerator(map_gen_configs)
        mapgen_handler.run()
    else:
        logging.info('Skipping map generator')

def Run_visualization(configs):
    viz_configs = configs.get('vizbuilder')
    if (viz_configs.get('mode')):
        logging.info("Running visualization")
        viz_handler = mm.viz.ReadConfig(viz_configs)
        viz_handler.run()
    else:
        logging.info('Skipping viz generator')
    
def Run(configs):
    Run_init()
    Run_input_generator(configs)
    Run_input_preprocessing(configs)
    Run_input_handler(configs)    
    Run_slicing_handler(configs)
    Run_clustering_handler(configs)
    Run_map_generator(configs)
    Run_visualization(configs)


def main(config_file, defaults="mm/default.yaml"):
    config_dict = {}

    logging.basicConfig(format='%(levelname)s %(asctime)s %(message)s', level=logging.DEBUG)
    with open(defaults) as df:
        try: 
            config_dict = yaml.load(df)
        except yaml.composer.ComposerError:
            logging.error('ERROR in yaml-reading the default config file')
            raise
    sections = config_dict.keys()
    with open(config_file) as cf:
        try: 
            new_config = yaml.load(cf)
            config_dict = new_config
            #for section in sections:
            #    sec_dict = new_config.get(section, {})
            #    config_dict.get(section).update(sec_dict)
        except yaml.composer.ComposerError:
            logging.error('ERROR in reading the input config file')
            raise
    log_level = {'error':logging.ERROR, 'debug':logging.DEBUG}.get(config_dict.get('global',{}).get('log_level'), logging.DEBUG)

    logging.basicConfig(level=log_level)

    logging.debug('final configuration: \n%s' % (str(yaml.dump(config_dict))))
    Run(config_dict)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Run Metromaps by specifying a config file')
    parser.add_argument('config_file', help='See default.yaml for configuration options')
    parser.add_argument('--defaults', default='mm/default.yaml', help='the default values get preloaded from this yaml configuration file')
    args = parser.parse_args()
    main(args.config_file, args.defaults)
    
    
