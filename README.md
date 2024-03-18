<div align="center">
  <p align="center">
      <picture>
      <img alt="Pointrix" src="https://github.com/pointrix-project/pointrix/assets/32637882/e0bd7ce3-fbf3-40f3-889c-7c882f3eed20" width="80%">
      </picture>
  </p>
  <p align="center">
    A differentiable point-based rendering library.
    <br />
    <a href="https://countermaker.github.io/pointrix.io/">
    <strong>Document | </strong></a>
    <a href="https://countermaker.github.io/pointrix.io/">
    <strong>Paper (Comming soon) | </strong></a>
    <a href="https://github.com/pointrix-project/dptr">
    <strong>DPRT Render Kernel</strong></a>
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
- **Powerful Backend**:
  - Support **"Render Anything"**(depth, normal, optical flow, anything you want)  and **"Backward Anything"** (Intrinsics & Extrinsics).
  - Modular design and easy to modify, support open-gl and opencv camera.
- **Rich Feature**:
  - Static Scene Reconstruction: 
    - **3D Gaussian Splatting for Real-Time Radiance Field Rendering (2023 Siggraph Best Paper)**
  - Dynamic Scene Reconstruction: 
    - **Deformable 3D Gaussians for High-Fidelity Monocular Dynamic Scene Reconstruction (2024 CVPR)**
    - **Gaussian-Flow: 4D Reconstruction with Dynamic 3D Gaussian Particle (2024 CVPR)**
  - Text to 3D generation: 
    - MVDream: Multi-view Diffusion for 3D Generation (2023 Arxiv)
      
- **Highly Extensible and Designed for Research**:
  - Pointrix adopts a modular design, with clear structure and easy extensibility. 
  - Only few codes need to be modified if you want to add a new method. 


<div style="display:flex;">
  <img src="https://github.com/pointrix-project/pointrix/assets/32637882/61795e5a-f91a-4a2a-b6ce-9a341a16145e" width="30%" />
  <img src="https://github.com/pointrix-project/pointrix/assets/32637882/616b7af8-3a8a-455a-ac1e-a62e9dc146d2" width="30%" />
  <img src="https://github.com/pointrix-project/pointrix/assets/32637882/41920617-86aa-4500-982f-5145b90e3336" width="30%" />
</div>

# WIP
- [ ] Introduction of Pointrix by video
- [ ] **Add GUI for visualization (in one week)**
- [ ] Add gaussian flow (CVPR2024) methods (in one week)
- [ ] Add Relightable Gaussian (CVPR2024) methods
- [ ] **support camera optimization  (in one week).**

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
git clone https://github.com/pointrix-project/dptr.git --recursive
cd dptr
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

# you can also run this if you have install gaussian original kernel
python launch.py --config ./configs/colmap.yaml
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


Welcome to submit PR if you have any idea or methods.