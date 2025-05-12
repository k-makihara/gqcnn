#!/bin/bash

python tools/finetune.py /home/deepstation/Downloads/realtrainset_6 Dexnet-10 --config_filename cfg/train_dex-net_2.0.yaml --name pt-Dexnet-10-ft-realtrainset6

python tools/finetune.py /home/deepstation/Downloads/realtrainset_6 Dexnet-50 --config_filename cfg/train_dex-net_2.0.yaml --name pt-Dexnet-50-ft-realtrainset6

python tools/finetune.py /home/deepstation/Downloads/realtrainset_6 EGAD-10 --config_filename cfg/train_dex-net_2.0.yaml --name pt-EGAD-10-ft-realtrainset6

python tools/finetune.py /home/deepstation/Downloads/realtrainset_6 EGAD-50 --config_filename cfg/train_dex-net_2.0.yaml --name pt-EGAD-50-ft-realtrainset6

python tools/finetune.py /home/deepstation/Downloads/realtrainset_6 GFDB-006-10 --config_filename cfg/train_dex-net_2.0.yaml --name pt-GFDB-006-10-ft-realtrainset6

python tools/finetune.py /home/deepstation/Downloads/realtrainset_6 GFDB-006-50 --config_filename cfg/train_dex-net_2.0.yaml --name pt-GFDB-006-50-ft-realtrainset6

python tools/finetune.py /home/deepstation/Downloads/realtrainset Dexnet-10 --config_filename cfg/train_dex-net_2.0.yaml --name pt-Dexnet-10-ft-realtrainset

python tools/finetune.py /home/deepstation/Downloads/realtrainset Dexnet-50 --config_filename cfg/train_dex-net_2.0.yaml --name pt-Dexnet-50-ft-realtrainset

python tools/finetune.py /home/deepstation/Downloads/realtrainset EGAD-10 --config_filename cfg/train_dex-net_2.0.yaml --name pt-EGAD-10-ft-realtrainset

python tools/finetune.py /home/deepstation/Downloads/realtrainset EGAD-50 --config_filename cfg/train_dex-net_2.0.yaml --name pt-EGAD-50-ft-realtrainset

python tools/finetune.py /home/deepstation/Downloads/realtrainset GFDB-006-10 --config_filename cfg/train_dex-net_2.0.yaml --name pt-GFDB-006-10-ft-realtrainset

python tools/finetune.py /home/deepstation/Downloads/realtrainset GFDB-006-50 --config_filename cfg/train_dex-net_2.0.yaml --name pt-GFDB-006-50-ft-realtrainset