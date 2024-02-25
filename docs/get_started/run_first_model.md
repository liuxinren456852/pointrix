# Run your first model

### 1. Download the data

<http://storage.googleapis.com/gresearch/refraw360/360_v2.zip>

Put the data into your folder.

### 2. Run the code

you need to modify the data path in the config file to the path of the data you downloaded.

```bash
cd projects/gaussian_splatting
python launch.py --config ./configs/colmap.yaml
```