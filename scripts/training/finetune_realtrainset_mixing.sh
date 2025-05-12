#!/bin/bash

python tools/finetune.py /home/deepstation/Downloads/realtrainset Dexnet-EGAD-150 --config_filename cfg/train_dex-net_2.0.yaml --name pt-Dexnet-EGAD-150-ft-realtrainset

python tools/finetune.py /home/deepstation/Downloads/realtrainset Dexnet-GFDB006-150 --config_filename cfg/train_dex-net_2.0.yaml --name pt-Dexnet-GFDB006-150-ft-realtrainset

python tools/finetune.py /home/deepstation/Downloads/realtrainset Dexnet-GFDB030-150 --config_filename cfg/train_dex-net_2.0.yaml --name pt-Dexnet-GFDB030-150-ft-realtrainset

python tools/finetune.py /home/deepstation/Downloads/realtrainset Dexnet-GFDB070-150 --config_filename cfg/train_dex-net_2.0.yaml --name pt-Dexnet-GFDB070-150-ft-realtrainset

python tools/finetune.py /home/deepstation/Downloads/realtrainset Dexnet-GFDB006030070-150 --config_filename cfg/train_dex-net_2.0.yaml --name pt-Dexnet-GFDB006030070-150-ft-realtrainset

python tools/finetune.py /home/deepstation/Downloads/realtrainset Dexnet-GFDB006030070full-150 --config_filename cfg/train_dex-net_2.0.yaml --name pt-Dexnet-GFDB006030070full-150-ft-realtrainset

python tools/finetune.py /home/deepstation/Downloads/realtrainset Dexnet-Primitive-150 --config_filename cfg/train_dex-net_2.0.yaml --name pt-Primitive-150-ft-realtrainset