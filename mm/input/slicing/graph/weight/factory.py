def factory(configs, graph_slicer):
    metro_station_generator_configs = configs
    input_generator_name = metro_station_generator_configs['weight_calculator']    
    generator_module = __import__(input_generator_name, globals=globals())
    return generator_module.construct(graph_slicer)