#!/bin/bash

python tools/finetune.py /home/deepstation/Downloads/realtrainset Dexnet-best --config_filename cfg/train_dex-net_2.0.yaml --name pt-Dexnet-best-ft-realtrainset

python tools/finetune.py /home/deepstation/Downloads/realtrainset EGAD-best --config_filename cfg/train_dex-net_2.0.yaml --name pt-EGAD-best-ft-realtrainset

python tools/finetune.py /home/deepstation/Downloads/realtrainset GFDB-006-best --config_filename cfg/train_dex-net_2.0.yaml --name pt-GFDB-006-best-ft-realtrainset

python tools/finetune.py /home/deepstation/Downloads/realtrainset GFDB-030-best --config_filename cfg/train_dex-net_2.0.yaml --name pt-GFDB-030-best-ft-realtrainset

python tools/finetune.py /home/deepstation/Downloads/realtrainset GFDB-070-best --config_filename cfg/train_dex-net_2.0.yaml --name pt-GFDB-070-best-ft-realtrainset

python tools/finetune.py /home/deepstation/Downloads/realtrainset Primitive-best --config_filename cfg/train_dex-net_2.0.yaml --name pt-Primitive-best-ft-realtrainset