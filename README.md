<br />
<div align="center">
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>裁剪图片</title>
<style>
  /* 设置容器的大小，超出部分将被裁剪 */
  .image-container {
    width: 300px;
    height: 100px;
    overflow: hidden; /* 超出部分隐藏 */
    background-image: url('docs/images/logo_transparent.png');
    background-size: cover; /* 背景图像尺寸调整为完全覆盖容器 */
    background-position: center; /* 背景图像位置居中 */
  }
</style>
</head>
<body>

<!-- 图片容器 -->
<div class="image-container">
  <!-- 图片链接 -->
  <a href="https://github.com/othneildrew/Best-README-Template"></a>
</div>


  <p align="center">
    A differentiable point-based rendering library
    <br />
    <!-- <a href="https://github.com/othneildrew/Best-README-Template"> -->
    <strong>Explore the docs »</strong></a>
    <br />
    <br />
    <!-- <a href="https://github.com/othneildrew/Best-README-Template">View Demo</a>
    ·
    <a href="https://github.com/othneildrew/Best-README-Template/issues">Report Bug</a>
    ·
    <a href="https://github.com/othneildrew/Best-README-Template/issues">Request Feature</a> -->
  </p>
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

Other dependencies:
```bash
pip install -r requirements.txt
```

Finally, install the following packages:
```bash
# Install official diff-gaussian-rasterization
git clone https://github.com/graphdeco-inria/diff-gaussian-rasterization.git  --recursive
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

2. Download the data and put it in your folder:

http://storage.googleapis.com/gresearch/refraw360/360_v2.zip

3. Run the following command to train the model:

you need to modify the data path in the config file to the path of the data you downloaded.

```bash
cd Pointrix
pip install -e .
cd projects/gaussian_splatting
python launch.py --config ./configs/colmap.yaml
```
