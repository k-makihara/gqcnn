#!/bin/bash

python tools/train.py /home/deepstation/Downloads/scale_test/dexnet_10 --config_filename cfg/train_dex-net_2.0.yaml --name Dexnet-10

python tools/train.py /home/deepstation/Downloads/scale_test/dexnet_50 --config_filename cfg/train_dex-net_2.0.yaml --name Dexnet-50

python tools/train.py /home/deepstation/Downloads/scale_test/dexnet_150 --config_filename cfg/train_dex-net_2.0.yaml --name Dexnet-150

python tools/train.py /home/deepstation/Downloads/scale_test/egad_10 --config_filename cfg/train_dex-net_2.0.yaml --name EGAD-10

python tools/train.py /home/deepstation/Downloads/scale_test/egad_50 --config_filename cfg/train_dex-net_2.0.yaml --name EGAD-50

python tools/train.py /home/deepstation/Downloads/scale_test/egad_150 --config_filename cfg/train_dex-net_2.0.yaml --name EGAD-150

python tools/train.py /home/deepstation/Downloads/scale_test/gfdb_006_10 --config_filename cfg/train_dex-net_2.0.yaml --name GFDB-006-10

python tools/train.py /home/deepstation/Downloads/scale_test/gfdb_006_50 --config_filename cfg/train_dex-net_2.0.yaml --name GFDB-006-50