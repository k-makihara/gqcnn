#!/bin/bash

python tools/finetune.py /home/deepstation/Downloads/realtrainset_6 Dexnet-best --config_filename cfg/train_dex-net_2.0.yaml --name pt-Dexnet-best-ft-realtrainset6

python tools/finetune.py /home/deepstation/Downloads/realtrainset_6 EGAD-best --config_filename cfg/train_dex-net_2.0.yaml --name pt-EGAD-best-ft-realtrainset6

python tools/finetune.py /home/deepstation/Downloads/realtrainset_6 GFDB-006-best --config_filename cfg/train_dex-net_2.0.yaml --name pt-GFDB-006-best-ft-realtrainset6

python tools/finetune.py /home/deepstation/Downloads/realtrainset_6 GFDB-030-best --config_filename cfg/train_dex-net_2.0.yaml --name pt-GFDB-030-best-ft-realtrainset6

python tools/finetune.py /home/deepstation/Downloads/realtrainset_6 GFDB-070-best --config_filename cfg/train_dex-net_2.0.yaml --name pt-GFDB-070-best-ft-realtrainset6

python tools/finetune.py /home/deepstation/Downloads/realtrainset_6 Primitive-best --config_filename cfg/train_dex-net_2.0.yaml --name pt-Primitive-best-ft-realtrainset6