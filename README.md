<br />
<div align="center">
  <!-- <a href="https://github.com/othneildrew/Best-README-Template">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a> -->

  <h3 align="center">Pointrix</h3>

  <p align="center">
    A differentiable point-based rendering libraries
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

```bash
# Install official diff-gaussian-rasterization
python -m pip install git+https://github.com/graphdeco-inria/diff-gaussian-rasterization.git

# Install pointrix
git clone https://github.com/NJU-3DV/Pointrix.git
cd Pointrix
python -m pip install .

# use -e if you want to modify the code
python -m pip install -e .
```

## Getting Started

```bash
cd projects/gaussian_splatting

python launch.py --config configs/nerf.yaml \
        trainer.output_path='output/lego' \
        dataset.data_path='nerf_synthetic/lego'
```
