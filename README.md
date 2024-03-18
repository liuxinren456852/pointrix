<div align="center">
  <p align="center">
      <picture>
      <img alt="Pointrix" src="https://github.com/pointrix-project/pointrix/assets/32637882/e0bd7ce3-fbf3-40f3-889c-7c882f3eed20" width="80%">
      </picture>
  </p>
  <p align="center">
    A differentiable point-based rendering library.
    <br />
    <a href="https://pointrix-project.github.io/pointrix/">
    <strong>Document🏠 | </strong></a>
    <a href="https://countermaker.github.io/pointrix.io/">
    <strong>Paper📄 (Comming soon) | </strong></a>
    <a href="https://github.com/pointrix-project/dptr">
    <strong>DPRT Backend🌐 </strong></a>
    <br />
    <br />
    <!-- <a href="https://github.com/othneildrew/Best-README-Template">View Demo</a>
    ·
    <a href="https://github.com/othneildrew/Best-README-Template/issues">Report Bug</a>
    ·
    <a href="https://github.com/othneildrew/Best-README-Template/issues">Request Feature</a> -->
  </p>
</div>

[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fpointrix-project%2Fpointrix&count_bg=%2396114C&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=visitors&edge_flat=false)](https://hits.seeyoufarm.com)
![Hits](https://img.shields.io/github/stars/pointrix-project/pointrix)
![Static Badge](https://img.shields.io/badge/Pointrix_document-Pointrix_document?color=hsl&link=https%3A%2F%2Fpointrix-project.github.io%2Fpointrix)
![GitHub Pipenv locked dependency version](https://img.shields.io/github/pipenv/locked/dependency-version/pointrix-project/dptr/dptr)




Pointrix is a differentiable point-based rendering library which has following properties:
- **Powerful Backend**:
  - Support **"Render Anything"**(depth, normal, optical flow, anything you want)  and **"Backward Anything"** (Intrinsics & Extrinsics).
  - Modular design and easy to modify, support open-gl and opencv camera.
- **Rich Feature**:
  - Static Scene Reconstruction: 
    - **[Vanilla 3DGS](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/3d_gaussian_splatting_high.pdf) (2023 Siggraph Best Paper)**
  - Dynamic Scene Reconstruction: 
    - **[Deformable 3DGS](https://arxiv.org/abs/2309.13101) (2024 CVPR)**
    - **[Gaussian-Flow](https://arxiv.org/abs/2312.03431) (2024 CVPR)**
  - Text to 3D generation: 
    - [MVDream](https://arxiv.org/abs/2308.16512) (2023 Arxiv)
      
- **Highly Extensible and Designed for Research**:
  - Pointrix adopts a modular design, with clear structure and easy extensibility. 
  - Only few codes need to be modified if you want to add a new method. 


<div style="display:flex;">
  <img src="https://github.com/pointrix-project/pointrix/assets/32637882/61795e5a-f91a-4a2a-b6ce-9a341a16145e" width="30%" />
  <img src="https://github.com/pointrix-project/pointrix/assets/32637882/616b7af8-3a8a-455a-ac1e-a62e9dc146d2" width="30%" />
  <img src="https://github.com/pointrix-project/pointrix/assets/32637882/928a142e-38cb-48e6-847b-1c6d4b95f7a3" width="30%" />
</div>

## contributors
<a href="https://github.com/pointrix-project/pointrix/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=pointrix-project/pointrix" />
</a>

Made with [contrib.rocks](https://contrib.rocks).

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
### 1. Lego
1. Download the lego data and put it in your folder:

```bash
wget http://cseweb.ucsd.edu/\~viscomp/projects/LF/papers/ECCV20/nerf/nerf_example_data.zip
```

2. Run the following command to train the model (...data path in the config file...):

```bash
cd Pointrix
pip install -e .
cd projects/gaussian_splatting
python launch.py --config ./configs/nerf_dptr.yaml

# you can also run this if you have installed gaussian original kernel
python launch.py --config ./configs/nerf.yaml
```

### 2. Mip-nerf 360 or other colmap dataset
1. Download the data and put it in your folder:

http://storage.googleapis.com/gresearch/refraw360/360_v2.zip

2. Run the following command to train the model (...data path in the config file...):

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

### 2. Generation (WIP)


# WIP
- [ ] Introduction video
- [ ] **Add GUI for visualization (this week).**
- [ ] **Implementataion of Gaussian-Flow (CVPR 2024) (this week).**
- [ ] Implementataion of MVDream (this week).
- [ ] Implementataion of Relightable Gaussian (arXiv 2023).
- [ ] **Support camera optimization  (this week).**

Welcome to join us or submit PR if you have any idea or methods.




