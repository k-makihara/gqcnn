#!/bin/bash

python tools/finetune.py /home/deepstation/Downloads/realtrainset_6 Dexnet-150 --config_filename cfg/train_dex-net_2.0.yaml --name pt-Dexnet-150-ft-realtrainset6

python tools/finetune.py /home/deepstation/Downloads/realtrainset_6 EGAD-150 --config_filename cfg/train_dex-net_2.0.yaml --name pt-EGAD-150-ft-realtrainset6

python tools/finetune.py /home/deepstation/Downloads/realtrainset_6 Grasp-FractalMeshDB-006-150-v2 --config_filename cfg/train_dex-net_2.0.yaml --name pt-GFDB-006-150-ft-realtrainset6

python tools/finetune.py /home/deepstation/Downloads/realtrainset_6 Grasp-FractalMeshDB-030-150-v2 --config_filename cfg/train_dex-net_2.0.yaml --name pt-GFDB-030-150-ft-realtrainset6

python tools/finetune.py /home/deepstation/Downloads/realtrainset_6 Grasp-FractalMeshDB-070-150-v2 --config_filename cfg/train_dex-net_2.0.yaml --name pt-GFDB-070-150-ft-realtrainset6

python tools/finetune.py /home/deepstation/Downloads/realtrainset_6 Grasp-FractalMeshDB-006-030-070-150 --config_filename cfg/train_dex-net_2.0.yaml --name pt-GFDB-006-030-070-150-ft-realtrainset6

python tools/finetune.py /home/deepstation/Downloads/realtrainset_6 Primitive-150 --config_filename cfg/train_dex-net_2.0.yaml --name pt-Primitive-150-ft-realtrainset6