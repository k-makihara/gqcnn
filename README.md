## Note: Python 2.x support has officially been dropped.

# Berkeley AUTOLAB's GQCNN Package
<p>
   <a href="https://travis-ci.org/BerkeleyAutomation/gqcnn/">
       <img alt="Build Status" src="https://travis-ci.org/BerkeleyAutomation/gqcnn.svg?branch=master">
   </a>
   <a href="https://github.com/BerkeleyAutomation/gqcnn/releases/latest">
       <img alt="Release" src="https://img.shields.io/github/release/BerkeleyAutomation/gqcnn.svg?style=flat">
   </a>
   <a href="https://github.com/BerkeleyAutomation/gqcnn/blob/master/LICENSE">
       <img alt="Software License" src="https://img.shields.io/badge/license-REGENTS-brightgreen.svg">
   </a>
   <a>
       <img alt="Python 3 Versions" src="https://img.shields.io/badge/python-3.5%20%7C%203.6%20%7C%203.7%20%7C%203.10-yellow.svg">
   </a>
   <a>
       <img alt="Ubnutu Versions" src="https://img.shields.io/badge/ubuntu-16.04%20%7C%2018.04%20%7C%2020.04-green.svg">
   </a>
</p>

## Package Overview
The gqcnn Python package is for training and analysis of Grasp Quality Convolutional Neural Networks (GQ-CNNs). It is part of the ongoing [Dexterity-Network (Dex-Net)](https://berkeleyautomation.github.io/dex-net/) project created and maintained by the [AUTOLAB](https://autolab.berkeley.edu) at UC Berkeley.

## System requirements
Any system may work if it can be installed tensorflow 2. CUDA and CuDNN versions depend on your specific GPUs.

### Successfully worked versions info
```
OS: Ubuntu 20.04
GPU: NVIDIA RTX A6000 48GB
CUDA: 11.2
cuDNN: ??
Nvidia driver: 470.129.06
python: 3.8.10
```

```
OS: Rocky Linux (ABCI 2.0 V-node)
GPU: NVIDIA V100 for NVLink 16GiB HBM2 x4
gcc: 12.2 (gcc/12.2.0)
python: 3.10.10 (python/3.10/3.10.10)
CUDA: 11.8 (cuda/11.8/11.8.0)
cuDNN: 8.6.0 (cudnn/8.6/8.6.0)
```

## Versions (the original script is written in tensoflow 1.15, but it uses tensorflow 2.12 and tensorflow.compat.v1)
```
Package                      Version   Editable project location \
---------------------------- --------- ------------------------------------
absl-py                      1.4.0
astunparse                   1.6.3
attrs                        23.1.0
autolab-core                 1.1.1
autolab-perception           1.0.0
cachetools                   5.3.0
certifi                      2023.5.7
chardet                      5.1.0
charset-normalizer           3.1.0
colorlog                     6.7.0
contourpy                    1.0.7
cycler                       0.11.0
dill                         0.3.6
ffmpeg-python                0.2.0
flatbuffers                  23.5.9
fonttools                    4.39.4
freetype-py                  2.4.0
future                       0.18.3
gast                         0.4.0
google-auth                  2.18.0
google-auth-oauthlib         1.0.0
google-pasta                 0.2.0
GPUtil                       1.4.0
gqcnn                        1.3.0    
grpcio                       1.54.2
h5py                         3.8.0
idna                         3.4
imageio                      2.28.1
jax                          0.4.10
joblib                       1.2.0
jsonschema                   4.18.0a6
jsonschema-specifications    2023.3.6
keras                        2.12.0
kiwisolver                   1.4.4
lazy_loader                  0.2
libclang                     16.0.0
lxml                         4.9.2
mapbox-earcut                1.0.1
Markdown                     3.4.3
MarkupSafe                   2.1.2
matplotlib                   3.7.1
ml-dtypes                    0.1.0
mpmath                       1.3.0
multiprocess                 0.70.14
networkx                     3.1
numpy                        1.23.5
oauthlib                     3.2.2
opencv-python                4.7.0.72
opt-einsum                   3.3.0
packaging                    23.1
Pillow                       9.5.0
pip                          22.3.1
protobuf                     4.23.0
psutil                       5.9.5
pyasn1                       0.5.0
pyasn1-modules               0.3.0
pycollada                    0.7.2
pyglet                       2.0.7
PyOpenGL                     3.1.0
pyparsing                    3.1.0b1
pyrender                     0.1.45
pyserial                     3.5
python-dateutil              2.8.2
PyWavelets                   1.4.1
referencing                  0.28.1
requests                     2.30.0
requests-oauthlib            1.3.1
rpds-py                      0.7.1
rsa                          4.9
Rtree                        1.0.1
ruamel.std.argparse          0.8.3
ruamel.yaml                  0.2
ruamel.yaml.clib             0.2.7
scikit-image                 0.21.0rc0
scikit-learn                 1.2.2
scipy                        1.10.1
setproctitle                 1.3.2
setuptools                   51.1.2
shapely                      2.0.1
six                          1.16.0
svg.path                     6.3
sympy                        1.12
tensorboard                  2.12.3
tensorboard-data-server      0.7.0
tensorflow                   2.12.0
tensorflow-estimator         2.12.0
tensorflow-io-gcs-filesystem 0.32.0
termcolor                    2.3.0
threadpoolctl                3.1.0
tifffile                     2023.4.12
trimesh                      3.21.6
typing_extensions            4.5.0
urllib3                      1.26.15
visualization                1.0.0
Werkzeug                     2.3.4
wheel                        0.40.0
wrapt                        1.14.1
xxhash                       3.2.0
```
## Installation and Usage
Please see the [docs](https://berkeleyautomation.github.io/gqcnn/) for installation and usage instructions.

In pip installation, something problem may happen in depended packages. Berkeley-AUTOLAB specific packages should be installed.

- autolab-core
- autolab-perception
- visualization

The installation is following to Dexnet install [docs](https://berkeleyautomation.github.io/dex-net/code.html)

```bash
# go to the root directory that you cloned the package

cd /path/to/gqcnn
mkdir deps
cd deps

git clone https://github.com/BerkeleyAutomation/autolab_core.git
git clone https://github.com/BerkeleyAutomation/perception.git
git clone https://github.com/k-makihara/visualization.git

# install autolab_core
cd autolab_core
python setup.py develop
cd ..

# install perception
cd perception
python setup.py develop
cd ..

# install visualization
cd visualization
python setup.py develop
cd ..

```

and, please do again in the [docs](https://berkeleyautomation.github.io/gqcnn/)

```bash
cd /path/to/gqcnn
pip install .
```

In these installations, if you may meet additional errors, please install this package from apt and pip.　Not all packages are necessary, so it's better to select and install only the required packages.

```bash
# install apt deps
sudo apt-get install cmake libvtk5-dev python-vtk python-sip python-qt4 libosmesa6-dev meshlab libhdf5-dev

# install pip deps
pip install numpy scipy scikit-learn scikit-image opencv-python pyassimp tensorflow h5py mayavi matplotlib catkin_pkg multiprocess dill cvxopt ipython pillow pyhull setproctitle trimesh

```


## Citation
If you use any part of this code in a publication, please cite [the appropriate Dex-Net publication](https://berkeleyautomation.github.io/gqcnn/index.html#academic-use).



