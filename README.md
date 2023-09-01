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

### Installation

If you only use in python, please do the followings

```bash
cd /path/to/gqcnn
pip install .
```

If you use in ROS, see the [docs](https://berkeleyautomation.github.io/gqcnn/). But in this repository, the ROS implementation is not confirmed in python3.

### Trouble shooting in Installation

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

### Virtual environments

Singularity is confirmed, use this Dockerfile and compile, but local installation is better to use because it may something problem still remained

```
FROM nvcr.io/nvidia/tensorflow:20.01-tf1-py2

ENV DEBIAN_FRONTEND=noninteractive

ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

ENV DISPLAY host.docker.internal:0.0

RUN apt-get -y update && apt-get -y upgrade && \
   apt-get install -y software-properties-common  && \
   add-apt-repository -y ppa:openscad/releases && \
   apt-get -y update && \
   apt-get -y install \
   x11-apps \
   build-essential \
   cmake \
   wget \
   git \
   nano \
   curl \
   vim \
   apt-utils \
   python-dev \
   python-pip \
   python3-dev \
   python3-pip \
   libglib2.0-0  \
   xvfb \
   libgl1-mesa-dev \
   libxkbcommon-x11-0 \
   libavcodec-dev \
   libavformat-dev \
   libswscale-dev \
   qt5-default \
   libqt5widgets5 \
   unity-gtk-module-common \
   unity-gtk3-module \
   xorg \
   libgl1-mesa-glx \
   mesa-utils \
   language-pack-en-base \
   libglew-dev \
   libcanberra-gtk-module \
   libcanberra-gtk3-module \
   bash \
   gcc \
   gfortran \
   g++ \
   make \
   file \
   unzip \
   python-sip \
   python-qt4 \
   libosmesa6-dev \
   meshlab \
   libhdf5-dev \
   freeglut3 \
   freeglut3-dev \
   openscad \
   blender \
   python-tk \
   ffmpeg \
   default-jdk


RUN mkdir -p /home/co


# Install libspatialindex
RUN cd /home/co && \
   wget https://download.osgeo.org/libspatialindex/spatialindex-src-1.7.0.tar.gz && \
   tar -xzf spatialindex-src-1.7.0.tar.gz && \
   cd spatialindex-src-1.7.0 && \
   ./configure && \
   make && \
   make install && \
   ldconfig


# Install newest pip versions
#RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
#    python get-pip.py && \
#    python3 get-pip.py && \
#    pip2 install --upgrade setuptools && \
#    pip3 install --upgrade setuptools


# Install python2 dependencies (largely very specific to get dexnet to actually run).
RUN pip2 install numpy==1.16.5 scipy==1.2.2 scikit-learn==0.20.4 scikit-image==0.14.5 opencv-python==4.1.2.30 pyassimp tensorflow==2.0.0 h5py networkx==2.2 tornado==3.2
RUN pip2 install imageio==2.6.1
RUN pip2 install vtk
RUN pip2 install traitsui==6.1.3 pyface==6.1.2 mayavi==4.5.0
RUN pip2 install matplotlib==2.2.4 multiprocess dill ipython==5.8.0 Pillow==6.2.1 setproctitle trimesh
RUN pip2 install pyhull
RUN pip2 install ffmpeg-python
RUN pip2 install pathlib2 pathlib
RUN pip2 install joblib==0.13.1 matplotlib==2.2.0 ruamel.yaml==0.16.0
RUN pip2 install cvxopt==1.1.9
RUN pip2 install colorlog==5.0.1
RUN pip2 install shapely


# Install Dex-Net and dependencies.
RUN mkdir -p /home/co/dexnet && \
   cd /home/co/dexnet && \
   wget https://github.com/dougsm/egad/releases/download/v0.1/dexnet_deployment.zip --no-check-certificate && \
   unzip dexnet_deployment.zip && \
   cd deps && \
   # install SDFGen
   cd SDFGen && \
   sh install.sh && \
   cd .. && \
   # install Boost.NumPy
   cd Boost.NumPy && \
   sh install.sh && \
   cd .. && \
   # return to dex-net directory
   # install meshpy
   cd meshpy && \
   python setup.py develop && \
   cd .. && \
   # install all Berkeley AUTOLAB modules
   # autolab_core
   cd autolab_core && \
   python setup.py develop && \
   cd .. && \
   # perception
   cd perception && \
   python setup.py develop && \
   cd .. && \
   # visualization
   cd visualization && \
   python setup.py develop && \
   cd .. && \
   # gqcnn
   cd gqcnn && \
   pip install . && \
   cd .. && \
   # dexnet
   cd .. && \
   python setup.py develop

RUN cd /home/co && \
   git clone https://github.com/k-makihara/openrave-installation.git && \
   cd openrave-installation && \
   ./install-dependencies.sh && \
   ./install-osg.sh && \
   ./install-fcl.sh && \
   ./install-openrave.sh


```



## Citation
If you use any part of this code in a publication, please cite [the appropriate Dex-Net publication](https://berkeleyautomation.github.io/gqcnn/index.html#academic-use).



