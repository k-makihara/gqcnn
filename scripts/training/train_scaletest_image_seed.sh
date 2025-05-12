#!/bin/bash

for i in {1..5}; do
    path="/home/deepstation/Downloads/scale_test/dexnet_150_010_seed${i}" 
    python tools/train.py /home/deepstation/Downloads/scale_test/dexnet_150_010_seed${i} --config_filename cfg/train_dex-net_2.0.yaml --name Dexnet-150-10-seed${i}
    python tools/train.py /home/deepstation/Downloads/scale_test/dexnet_150_020_seed$i --config_filename cfg/train_dex-net_2.0.yaml --name Dexnet-150-20-seed$i
    python tools/train.py /home/deepstation/Downloads/scale_test/dexnet_150_030_seed$i --config_filename cfg/train_dex-net_2.0.yaml --name Dexnet-150-30-seed$i
    python tools/train.py /home/deepstation/Downloads/scale_test/dexnet_150_040_seed$i --config_filename cfg/train_dex-net_2.0.yaml --name Dexnet-150-40-seed$i
    python tools/train.py /home/deepstation/Downloads/scale_test/dexnet_150_050_seed$i --config_filename cfg/train_dex-net_2.0.yaml --name Dexnet-150-50-seed$i
    python tools/train.py /home/deepstation/Downloads/scale_test/dexnet_150_060_seed$i --config_filename cfg/train_dex-net_2.0.yaml --name Dexnet-150-60-seed$i
    python tools/train.py /home/deepstation/Downloads/scale_test/dexnet_150_070_seed$i --config_filename cfg/train_dex-net_2.0.yaml --name Dexnet-150-70-seed$i
    python tools/train.py /home/deepstation/Downloads/scale_test/dexnet_150_080_seed$i --config_filename cfg/train_dex-net_2.0.yaml --name Dexnet-150-80-seed$i
    python tools/train.py /home/deepstation/Downloads/scale_test/dexnet_150_090_seed$i --config_filename cfg/train_dex-net_2.0.yaml --name Dexnet-150-90-seed$i


    python tools/train.py /home/deepstation/Downloads/scale_test/egad_150_010_seed$i --config_filename cfg/train_dex-net_2.0.yaml --name EGAD-150-10-seed$i
    python tools/train.py /home/deepstation/Downloads/scale_test/egad_150_020_seed$i --config_filename cfg/train_dex-net_2.0.yaml --name EGAD-150-20-seed$i
    python tools/train.py /home/deepstation/Downloads/scale_test/egad_150_030_seed$i --config_filename cfg/train_dex-net_2.0.yaml --name EGAD-150-30-seed$i
    python tools/train.py /home/deepstation/Downloads/scale_test/egad_150_040_seed$i --config_filename cfg/train_dex-net_2.0.yaml --name EGAD-150-40-seed$i
    python tools/train.py /home/deepstation/Downloads/scale_test/egad_150_050_seed$i --config_filename cfg/train_dex-net_2.0.yaml --name EGAD-150-50-seed$i
    python tools/train.py /home/deepstation/Downloads/scale_test/egad_150_060_seed$i --config_filename cfg/train_dex-net_2.0.yaml --name EGAD-150-60-seed$i
    python tools/train.py /home/deepstation/Downloads/scale_test/egad_150_070_seed$i --config_filename cfg/train_dex-net_2.0.yaml --name EGAD-150-70-seed$i
    python tools/train.py /home/deepstation/Downloads/scale_test/egad_150_080_seed$i --config_filename cfg/train_dex-net_2.0.yaml --name EGAD-150-80-seed$i
    python tools/train.py /home/deepstation/Downloads/scale_test/egad_150_090_seed$i --config_filename cfg/train_dex-net_2.0.yaml --name EGAD-150-90-seed$i

    python tools/train.py /home/deepstation/Downloads/scale_test/gfdb_006_150_v2_010_seed$i --config_filename cfg/train_dex-net_2.0.yaml --name GFDB-006-150-10-seed$i
    python tools/train.py /home/deepstation/Downloads/scale_test/gfdb_006_150_v2_020_seed$i --config_filename cfg/train_dex-net_2.0.yaml --name GFDB-006-150-20-seed$i
    python tools/train.py /home/deepstation/Downloads/scale_test/gfdb_006_150_v2_030_seed$i --config_filename cfg/train_dex-net_2.0.yaml --name GFDB-006-150-30-seed$i
    python tools/train.py /home/deepstation/Downloads/scale_test/gfdb_006_150_v2_040_seed$i --config_filename cfg/train_dex-net_2.0.yaml --name GFDB-006-150-40-seed$i
    python tools/train.py /home/deepstation/Downloads/scale_test/gfdb_006_150_v2_050_seed$i --config_filename cfg/train_dex-net_2.0.yaml --name GFDB-006-150-50-seed$i
    python tools/train.py /home/deepstation/Downloads/scale_test/gfdb_006_150_v2_060_seed$i --config_filename cfg/train_dex-net_2.0.yaml --name GFDB-006-150-60-seed$i
    python tools/train.py /home/deepstation/Downloads/scale_test/gfdb_006_150_v2_070_seed$i --config_filename cfg/train_dex-net_2.0.yaml --name GFDB-006-150-70-seed$i
    python tools/train.py /home/deepstation/Downloads/scale_test/gfdb_006_150_v2_080_seed$i --config_filename cfg/train_dex-net_2.0.yaml --name GFDB-006-150-80-seed$i
    python tools/train.py /home/deepstation/Downloads/scale_test/gfdb_006_150_v2_090_seed$i --config_filename cfg/train_dex-net_2.0.yaml --name GFDB-006-150-90-seed$i

    python tools/train.py /home/deepstation/Downloads/scale_test/gfdb_030_150_v2_010_seed$i --config_filename cfg/train_dex-net_2.0.yaml --name GFDB-030-150-10-seed$i
    python tools/train.py /home/deepstation/Downloads/scale_test/gfdb_030_150_v2_020_seed$i --config_filename cfg/train_dex-net_2.0.yaml --name GFDB-030-150-20-seed$i
    python tools/train.py /home/deepstation/Downloads/scale_test/gfdb_030_150_v2_030_seed$i --config_filename cfg/train_dex-net_2.0.yaml --name GFDB-030-150-30-seed$i
    python tools/train.py /home/deepstation/Downloads/scale_test/gfdb_030_150_v2_040_seed$i --config_filename cfg/train_dex-net_2.0.yaml --name GFDB-030-150-40-seed$i
    python tools/train.py /home/deepstation/Downloads/scale_test/gfdb_030_150_v2_050_seed$i --config_filename cfg/train_dex-net_2.0.yaml --name GFDB-030-150-50-seed$i
    python tools/train.py /home/deepstation/Downloads/scale_test/gfdb_030_150_v2_060_seed$i --config_filename cfg/train_dex-net_2.0.yaml --name GFDB-030-150-60-seed$i
    python tools/train.py /home/deepstation/Downloads/scale_test/gfdb_030_150_v2_070_seed$i --config_filename cfg/train_dex-net_2.0.yaml --name GFDB-030-150-70-seed$i
    python tools/train.py /home/deepstation/Downloads/scale_test/gfdb_030_150_v2_080_seed$i --config_filename cfg/train_dex-net_2.0.yaml --name GFDB-030-150-80-seed$i
    python tools/train.py /home/deepstation/Downloads/scale_test/gfdb_030_150_v2_090_seed$i --config_filename cfg/train_dex-net_2.0.yaml --name GFDB-030-150-90-seed$i

    python tools/train.py /home/deepstation/Downloads/scale_test/gfdb_070_150_v2_010_seed$i --config_filename cfg/train_dex-net_2.0.yaml --name GFDB-070-150-10-seed$i
    python tools/train.py /home/deepstation/Downloads/scale_test/gfdb_070_150_v2_020_seed$i --config_filename cfg/train_dex-net_2.0.yaml --name GFDB-070-150-20-seed$i
    python tools/train.py /home/deepstation/Downloads/scale_test/gfdb_070_150_v2_030_seed$i --config_filename cfg/train_dex-net_2.0.yaml --name GFDB-070-150-30-seed$i
    python tools/train.py /home/deepstation/Downloads/scale_test/gfdb_070_150_v2_040_seed$i --config_filename cfg/train_dex-net_2.0.yaml --name GFDB-070-150-40-seed$i
    python tools/train.py /home/deepstation/Downloads/scale_test/gfdb_070_150_v2_050_seed$i --config_filename cfg/train_dex-net_2.0.yaml --name GFDB-070-150-50-seed$i
    python tools/train.py /home/deepstation/Downloads/scale_test/gfdb_070_150_v2_060_seed$i --config_filename cfg/train_dex-net_2.0.yaml --name GFDB-070-150-60-seed$i
    python tools/train.py /home/deepstation/Downloads/scale_test/gfdb_070_150_v2_070_seed$i --config_filename cfg/train_dex-net_2.0.yaml --name GFDB-070-150-70-seed$i
    python tools/train.py /home/deepstation/Downloads/scale_test/gfdb_070_150_v2_080_seed$i --config_filename cfg/train_dex-net_2.0.yaml --name GFDB-070-150-80-seed$i
    python tools/train.py /home/deepstation/Downloads/scale_test/gfdb_070_150_v2_090_seed$i --config_filename cfg/train_dex-net_2.0.yaml --name GFDB-070-150-90-seed$i

    python tools/train.py /home/deepstation/Downloads/scale_test/gfdb_006_030_070_150_010_seed$i --config_filename cfg/train_dex-net_2.0.yaml --name GFDB-006-030-070-150-10-seed$i
    python tools/train.py /home/deepstation/Downloads/scale_test/gfdb_006_030_070_150_020_seed$i --config_filename cfg/train_dex-net_2.0.yaml --name GFDB-006-030-070-150-20-seed$i
    python tools/train.py /home/deepstation/Downloads/scale_test/gfdb_006_030_070_150_030_seed$i --config_filename cfg/train_dex-net_2.0.yaml --name GFDB-006-030-070-150-30-seed$i
    python tools/train.py /home/deepstation/Downloads/scale_test/gfdb_006_030_070_150_040_seed$i --config_filename cfg/train_dex-net_2.0.yaml --name GFDB-006-030-070-150-40-seed$i
    python tools/train.py /home/deepstation/Downloads/scale_test/gfdb_006_030_070_150_050_seed$i --config_filename cfg/train_dex-net_2.0.yaml --name GFDB-006-030-070-150-50-seed$i
    python tools/train.py /home/deepstation/Downloads/scale_test/gfdb_006_030_070_150_060_seed$i --config_filename cfg/train_dex-net_2.0.yaml --name GFDB-006-030-070-150-60-seed$i
    python tools/train.py /home/deepstation/Downloads/scale_test/gfdb_006_030_070_150_070_seed$i --config_filename cfg/train_dex-net_2.0.yaml --name GFDB-006-030-070-150-70-seed$i
    python tools/train.py /home/deepstation/Downloads/scale_test/gfdb_006_030_070_150_080_seed$i --config_filename cfg/train_dex-net_2.0.yaml --name GFDB-006-030-070-150-80-seed$i
    python tools/train.py /home/deepstation/Downloads/scale_test/gfdb_006_030_070_150_090_seed$i --config_filename cfg/train_dex-net_2.0.yaml --name GFDB-006-030-070-150-90-seed$i

    python tools/train.py /home/deepstation/grasp-fractal/gqcnn-sim/3rdparty/dexnet/primitive_010_seed$i --config_filename cfg/train_dex-net_2.0.yaml --name Primitive-150-10-seed$i
    python tools/train.py /home/deepstation/grasp-fractal/gqcnn-sim/3rdparty/dexnet/primitive_020_seed$i --config_filename cfg/train_dex-net_2.0.yaml --name Primitive-150-20-seed$i
    python tools/train.py /home/deepstation/grasp-fractal/gqcnn-sim/3rdparty/dexnet/primitive_030_seed$i --config_filename cfg/train_dex-net_2.0.yaml --name Primitive-150-30-seed$i
    python tools/train.py /home/deepstation/grasp-fractal/gqcnn-sim/3rdparty/dexnet/primitive_040_seed$i --config_filename cfg/train_dex-net_2.0.yaml --name Primitive-150-40-seed$i
    python tools/train.py /home/deepstation/grasp-fractal/gqcnn-sim/3rdparty/dexnet/primitive_050_seed$i --config_filename cfg/train_dex-net_2.0.yaml --name Primitive-150-50-seed$i
    python tools/train.py /home/deepstation/grasp-fractal/gqcnn-sim/3rdparty/dexnet/primitive_060_seed$i --config_filename cfg/train_dex-net_2.0.yaml --name Primitive-150-60-seed$i
    python tools/train.py /home/deepstation/grasp-fractal/gqcnn-sim/3rdparty/dexnet/primitive_070_seed$i --config_filename cfg/train_dex-net_2.0.yaml --name Primitive-150-70-seed$i
    python tools/train.py /home/deepstation/grasp-fractal/gqcnn-sim/3rdparty/dexnet/primitive_080_seed$i --config_filename cfg/train_dex-net_2.0.yaml --name Primitive-150-80-seed$i
    python tools/train.py /home/deepstation/grasp-fractal/gqcnn-sim/3rdparty/dexnet/primitive_090_seed$i --config_filename cfg/train_dex-net_2.0.yaml --name Primitive-150-90-seed$i

done