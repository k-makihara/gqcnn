#!/bin/bash

#python tools/hyperparam_search.py /home/deepstation/Downloads/scale_test/dexnet_150_10000images_seed1_pos050 --train_configs cfg/train_dex-net_2.0_param.yaml

#python tools/hyperparam_search.py /home/deepstation/Downloads/scale_test/egad_150_10000images_seed1_pos050 --train_configs cfg/train_dex-net_2.0_param.yaml

python tools/hyperparam_search.py /home/deepstation/Downloads/scale_test/gfdb_030_150_v2_10000images_seed1_pos050 --train_configs cfg/train_dex-net_2.0_param.yaml

#python tools/hyperparam_search.py /home/deepstation/grasp-fractal/gqcnn-sim/3rdparty/dexnet/gfdb_006_150_x10_10000images_seed1_pos050 --train_configs cfg/train_dex-net_2.0_param.yaml

#python tools/hyperparam_search.py /home/deepstation/grasp-fractal/gqcnn-sim/3rdparty/dexnet/gfdb_070_150_x10_10000images_seed1_pos050 --train_configs cfg/train_dex-net_2.0_param.yaml

#python tools/hyperparam_search.py /home/deepstation/grasp-fractal/gqcnn-sim/3rdparty/dexnet/primitive_10000images_seed1_pos050 --train_configs cfg/train_dex-net_2.0_param.yaml