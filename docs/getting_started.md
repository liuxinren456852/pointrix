# Getting started

Get started with our package with these steps:

### 1. Install package

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
git clone https://github.com/graphdeco-inria/diff-gaussian-rasterization.git
cd diff-gaussian-rasterization
python setup.py install
pip install .
```
```bash
# Install simple knn
git clone https://gitlab.inria.fr/bkerbl/simple-knn.git
cd simple-knn
python setup.py install
pip install .
```

### 2. Download the data

<http://storage.googleapis.com/gresearch/refraw360/360_v2.zip>

Put the data into your folder.

### 3. Run the code

you need to modify the data path in the config file to the path of the data you downloaded.

```bash
cd projects/gaussian_splatting
python launch.py --config ./configs/colmap.yaml
```