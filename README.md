<div align="center">
  <h3 align="center">Pointrix</h3>
  <p align="center">
      <picture>
      <img alt="Pointrix" src="docs/images/logo_2.png" width="80%">
      </picture>
  </p>
  <p align="center">
    A differentiable point-based rendering library.
    <br />
    <a href="https://countermaker.github.io/pointrix.io/">
    <strong>Document | </strong></a>
    <a href="https://countermaker.github.io/pointrix.io/">
    <strong>Paper (Comming soon)| </strong></a>
    <a href="https://countermaker.github.io/pointrix.io/">
    <strong>Wechat group (Comming soon)</strong></a>
    <br />
    <br />
    <!-- <a href="https://github.com/othneildrew/Best-README-Template">View Demo</a>
    ·
    <a href="https://github.com/othneildrew/Best-README-Template/issues">Report Bug</a>
    ·
    <a href="https://github.com/othneildrew/Best-README-Template/issues">Request Feature</a> -->
  </p>
</div>

Pointrix is a differentiable point-based rendering library which has following properties:
- **Rich Feature**:
  - Pointrix supports the implementation of various types of tasks such as:
    - Static Scene Reconstruction: 
      - **3D Gaussian Splatting for Real-Time Radiance Field Rendering (2023 Siggraph Best Paper)**
    - Dynamic Scene Reconstruction: 
      - **Deformable 3D Gaussians for High-Fidelity Monocular Dynamic Scene Reconstruction (2024 CVPR)**
    - Text to 3D generation: 
      - MVDream: Multi-view Diffusion for 3D Generation (2023 Arxiv)
      
- **Highly Extensible**:
  - Pointrix adopts a modular design, with clear structure and easy extensibility. 
- **Powerful Backend**:
  - DPTR which offer foundational functionalities for point rendering serves as the backend of Pointrix.

<div style="display:flex; justify-content:center;">
  <video width="320" height="240" controls>
    <source src="docs/images/result1.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>
  <video width="320" height="240" controls>
    <source src="docs/images/result2.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>
  <video width="320" height="240" controls>
    <source src="docs/images/result3.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>
</div>


## Prerequisites

### Installations
1. Install the following package:

First,create a new conda environment and activate it:

```bash
conda create -n pointrix python=3.9
conda activate pointrix
```

Then, you need to install pytorch
```bash
conda install pytorch==2.1.1 torchvision==0.16.1 pytorch-cuda=12.1 -c pytorch -c nvidia
```
Other dependencies:

```
pip install -r requirements.txt
```

Finally, install our DPTR rendering kernel:

```bash
# Install official diff-gaussian-rasterization
# clone the code from github
git clone https://github.com/NJU-3DV/DPTR.git --recursive
cd DPTR
# install dptr
pip install .
```
```bash
# Install simple knn
git clone https://gitlab.inria.fr/bkerbl/simple-knn.git
cd simple-knn
python setup.py install
pip install .
```

Note: we support both gaussian original kernel and DPTR kernel.

## Running our example

1. Download the data and put it in your folder:

http://storage.googleapis.com/gresearch/refraw360/360_v2.zip

2. Run the following command to train the model:

**you need to modify the data path in the config file to the path of the data you downloaded.**

```bash
cd Pointrix
pip install -e .
cd projects/gaussian_splatting
python launch.py --config ./configs/colmap_dptr.yaml
```

## Try other methods

### 1. Dynamic Gaussian
1. Download the iphone dataset and put it in your folder:
https://drive.google.com/drive/folders/1cBw3CUKu2sWQfc_1LbFZGbpdQyTFzDEX

2. Run the following command to train the model:

**you need to modify the data path in the config file to the path of the data you downloaded.**

```bash
cd Pointrix
pip install -e .
cd projects/deformable_gaussian
python launch.py --config deform.yaml
```

### 2. Generation (under construction, please refer to generate branch)




# TODO
Welcome to submit PR if you have any idea or methods:

- [ ] Support opencv camera
- [ ] Add gaussian flow (CVPR2024) methods (in one week)
- [ ] Add GUI for visualization (in one week)
- [ ] support camera optimization.