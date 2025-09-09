#!/bin/bash

python tools/train.py /home/deepstation/Downloads/scale_test/dexnet_150_100_seed123_pos010 --config_filename cfg/train_dex-net_2.0.yaml --name Dexnet-150-pos010
python tools/train.py /home/deepstation/Downloads/scale_test/dexnet_150_100_seed123_pos020 --config_filename cfg/train_dex-net_2.0.yaml --name Dexnet-150-pos020
python tools/train.py /home/deepstation/Downloads/scale_test/dexnet_150_100_seed123_pos030 --config_filename cfg/train_dex-net_2.0.yaml --name Dexnet-150-pos030
python tools/train.py /home/deepstation/Downloads/scale_test/dexnet_150_100_seed123_pos040 --config_filename cfg/train_dex-net_2.0.yaml --name Dexnet-150-pos040
python tools/train.py /home/deepstation/Downloads/scale_test/dexnet_150_100_seed123_pos050 --config_filename cfg/train_dex-net_2.0.yaml --name Dexnet-150-pos050
python tools/train.py /home/deepstation/Downloads/scale_test/dexnet_150_100_seed123_pos060 --config_filename cfg/train_dex-net_2.0.yaml --name Dexnet-150-pos060
python tools/train.py /home/deepstation/Downloads/scale_test/dexnet_150_100_seed123_pos070 --config_filename cfg/train_dex-net_2.0.yaml --name Dexnet-150-pos070
python tools/train.py /home/deepstation/Downloads/scale_test/dexnet_150_100_seed123_pos080 --config_filename cfg/train_dex-net_2.0.yaml --name Dexnet-150-pos080
python tools/train.py /home/deepstation/Downloads/scale_test/dexnet_150_100_seed123_pos090 --config_filename cfg/train_dex-net_2.0.yaml --name Dexnet-150-pos090

