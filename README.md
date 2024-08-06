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
OS: macOS 13.5.2
python: 3.9.9
```

### Installation

If you only use in python, please do the followings

```bash
cd /path/to/gqcnn
pip install .
```

If you use in ROS, see the [docs](https://berkeleyautomation.github.io/gqcnn/). But in this repository, the ROS implementation is not confirmed in python3.

### Usage

First, you need to try GQ-CNN for example data. 
The pre-trained model is Dexnet-4.0 for parallel jaw gripper

```bash
./scripts/policies/run_all_dex-net_4.0_pj_examples.sh
```

If the script run sccessully, you can use your own data

```bash
python python examples/policy.py GQCNN-4.0-PJ --depth_image /path/to/depth_image --segmask /path/to/segmentation_mask --config_filename cfg/examples/replication/dex-net_4.0_pj.yaml --camera_intr /path/to/camera_intrinsics
```



If there are something error in examples/policy.py

```bash
pip install ruamel.yaml==0.2
pip install numpy==1.23.5
```



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



