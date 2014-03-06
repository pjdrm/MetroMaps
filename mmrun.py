#!/usr/local/bin/python2.7

import argparse
import mm.inputhelpers
import mm.inputhelpers.factory
import mm.input
import logging
import pyyaml



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

def Run_legacy_handler(configs):
    legacy_configs = configs.get('legacy_helper')
    if (legacy_configs.get('mode')):
        logging.info("Converting to legacy format")

        legacy_handler = mm.input.LegacyHandler(legacy_configs)
        legacy_handler.write()
        logging.info("Legacy format available in %s" %(configs.get('legacy_helper').get('output_dir')))

def Main(configs):
    Run_input_handler(configs)    
    Run_legacy_handler(configs)
    # Run_clustering_handler(configs)





if __name__=='__main__':
    logging.basicConfig(format='%(levelname)s %(asctime)s %(message)s', level=logging.DEBUG)
    parser = argparse.ArgumentParser(description='Run Metromaps by specifying a config file (e.g. default.ini)')
    parser.add_argument('config_file', help='See default.ini for configurations')
    args = parser.parse_args()
    config_dict = {}
    with open(args.config_file) as cf:
        config_dict = yaml.load(cf)
    Main(config_dict)
    
