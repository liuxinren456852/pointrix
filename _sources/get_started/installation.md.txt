# Installation

Get started with our package with these steps:

## 1. Install package

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
```bash
pip install -r requirements.txt
```

Finally, install the following packages:

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