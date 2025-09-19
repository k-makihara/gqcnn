#!/bin/bash

python tools/train.py /home/deepstation/Downloads/scale_test/dexnet_150_10000images_seed2_pos050 --config_filename cfg/train_opt_param_dexnet.yaml --name Dexnet-fiximage-posneg-seed2
python tools/train.py /home/deepstation/Downloads/scale_test/dexnet_150_10000images_seed3_pos050 --config_filename cfg/train_opt_param_dexnet.yaml --name Dexnet-fiximage-posneg-seed3

python tools/train.py /home/deepstation/Downloads/scale_test/egad_150_10000images_seed2_pos050 --config_filename cfg/train_opt_param_egad.yaml --name EGAD-fiximage-posneg-seed2
python tools/train.py /home/deepstation/Downloads/scale_test/egad_150_10000images_seed3_pos050 --config_filename cfg/train_opt_param_egad.yaml --name EGAD-fiximage-posneg-seed3

python tools/train.py /home/deepstation/grasp-fractal/gqcnn-sim/3rdparty/dexnet/gfdb_006_150_x10_10000images_seed2_pos050 --config_filename cfg/train_opt_param_gfdb_006.yaml --name GFDB-006-fiximage-posneg-seed2
python tools/train.py /home/deepstation/grasp-fractal/gqcnn-sim/3rdparty/dexnet/gfdb_006_150_x10_10000images_seed3_pos050 --config_filename cfg/train_opt_param_gfdb_006.yaml --name GFDB-006-fiximage-posneg-seed3

python tools/train.py /home/deepstation/Downloads/scale_test/gfdb_030_150_v2_10000images_seed2_pos050 --config_filename cfg/train_opt_param_gfdb_030.yaml --name GFDB-030-fiximage-posneg-seed2
python tools/train.py /home/deepstation/Downloads/scale_test/gfdb_030_150_v2_10000images_seed3_pos050 --config_filename cfg/train_opt_param_gfdb_030.yaml --name GFDB-030-fiximage-posneg-seed3

python tools/train.py /home/deepstation/grasp-fractal/gqcnn-sim/3rdparty/dexnet/gfdb_070_150_x10_10000images_seed2_pos050 --config_filename cfg/train_opt_param_gfdb_070.yaml --name GFDB-070-fiximage-posneg-seed2
python tools/train.py /home/deepstation/grasp-fractal/gqcnn-sim/3rdparty/dexnet/gfdb_070_150_x10_10000images_seed3_pos050 --config_filename cfg/train_opt_param_gfdb_070.yaml --name GFDB-070-fiximage-posneg-seed3

python tools/train.py /home/deepstation/grasp-fractal/gqcnn-sim/3rdparty/dexnet/primitive_10000images_seed2_pos050 --config_filename cfg/train_opt_param_primitive.yaml --name Primitive-fiximage-posneg-seed2
python tools/train.py /home/deepstation/grasp-fractal/gqcnn-sim/3rdparty/dexnet/primitive_10000images_seed3_pos050 --config_filename cfg/train_opt_param_primitive.yaml --name Primitive-fiximage-posneg-seed3


