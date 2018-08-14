SegmentLinkingAVL
=========

- Code for the experiments in the poster paper "Efficient navigation in learning materials: an empirical study on the linking process" published in AEID 2018
- Corpus with human judgements on topic segment similarity in the AVL domain

Running the system:
=========

python ./tests/slice_tester.py <config.yaml> <tests_config.json>

- Configuration files examples can be found in AVL_trees_dataset_configs.zip, namely:

<config.yaml> - avl.yaml
<tests_config.json> - avl_tests.json

- After running the program the output will be in the file ./resources/tests/results.txt

Setting up the system:
=========

1) Build a single file with all segments from all documents separated with the character sequence "==========" (see avl_segments.txt, in the AVL_trees_dataset_configs.zip, for an example)
2) Specify the path to the file in <config.yaml>, in the "input_generator" section, variable "input_directory" (see avl.yaml for an example)
3) Build a label sequence with the ground truth of the clustering of the segments. The label sequence must match the sequence of segments from step 1.
4) Specify the ground truth in <tests_config.json> file (see avl_tests.json for an example).
