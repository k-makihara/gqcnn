#!/bin/bash

#python tools/train.py /home/deepstation/Downloads/scale_test/dexnet_egad_150 --config_filename cfg/train_dex-net_2.0.yaml --name Dexnet-EGAD-150

#python tools/train.py /home/deepstation/Downloads/scale_test/dexnet_gfdb006_150 --config_filename cfg/train_dex-net_2.0.yaml --name Dexnet-GFDB006-150

#python tools/train.py /home/deepstation/Downloads/scale_test/dexnet_gfdb030_150 --config_filename cfg/train_dex-net_2.0.yaml --name Dexnet-GFDB030-150

#python tools/train.py /home/deepstation/Downloads/scale_test/dexnet_gfdb070_150 --config_filename cfg/train_dex-net_2.0.yaml --name Dexnet-GFDB070-150

#python tools/train.py /home/deepstation/Downloads/scale_test/dexnet_gfdb006030070_150 --config_filename cfg/train_dex-net_2.0.yaml --name Dexnet-GFDB006030070-150

#python tools/train.py /home/deepstation/Downloads/scale_test/dexnet_gfdb006030070full_150 --config_filename cfg/train_dex-net_2.0.yaml --name Dexnet-GFDB006030070full-150

#python tools/train.py /home/deepstation/Downloads/scale_test/dexnet_primitive_150 --config_filename cfg/train_dex-net_2.0.yaml --name Dexnet-Primitive-150

#python tools/train.py /home/deepstation/grasp-fractal/gqcnn-sim/3rdparty/dexnet/gfdb_006_150_x10 --config_filename cfg/train_dex-net_2.0.yaml --name GFDB-006-150-x10

#python tools/train.py /home/deepstation/grasp-fractal/gqcnn-sim/3rdparty/dexnet/gfdb_030_150_x10 --config_filename cfg/train_dex-net_2.0.yaml --name GFDB-030-150-x10

python tools/train.py /home/deepstation/grasp-fractal/gqcnn-sim/3rdparty/dexnet/gfdb_070_150_x10 --config_filename cfg/train_dex-net_2.0.yaml --name GFDB-070-150-x10

python tools/train.py /home/deepstation/Downloads/scale_test/gfdb006030070_150 --config_filename cfg/train_dex-net_2.0.yaml --name GFDB-006-030-070-150-full

#python tools/merge_dataset.py \
#    --base-dirs /home/deepstation/grasp-fractal/gqcnn-sim/3rdparty/dexnet/gfdb_006_150_x10 /home/deepstation/grasp-fractal/gqcnn-sim/3rdparty/dexnet/gfdb_030_150_x10 /home/deepstation/grasp-fractal/gqcnn-sim/3rdparty/dexnet/gfdb_070_150_x10 \
#    --fractions 1.0 1.0 1.0\
#    --output-dir /home/deepstation/Downloads/scale_test/gfdb006030070x10_150

#python tools/train.py /home/deepstation/Downloads/scale_test/gfdb006030070x0_150 --config_filename cfg/train_dex-net_2.0.yaml --name GFDB-006-030-070-150-fullx10