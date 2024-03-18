import os
import json
import numpy as np
from PIL import Image
from pathlib import Path

from pointrix.dataset.base_data import DATA_FORMAT_REGISTRY, BaseReFormatData, SimplePointCloud
from pointrix.camera.camera import Camera

C0 = 0.28209479177387814
def SH2RGB(sh):
    return sh * C0 + 0.5

def camera_nerfies_from_JSON(path, scale):
    """Loads a JSON camera into memory."""
    with open(path, 'r') as fp:
        camera_json = json.load(fp)

    # Fix old camera JSON.
    if 'tangential' in camera_json:
        camera_json['tangential_distortion'] = camera_json['tangential']

    return dict(
        orientation=np.array(camera_json['orientation']),
        position=np.array(camera_json['position']),
        focal_length=camera_json['focal_length'] * scale,
        principal_point=np.array(camera_json['principal_point']) * scale,
        skew=camera_json['skew'],
        pixel_aspect_ratio=camera_json['pixel_aspect_ratio'],
        radial_distortion=np.array(camera_json['radial_distortion']),
        tangential_distortion=np.array(camera_json['tangential_distortion']),
        image_size=np.array((int(round(camera_json['image_size'][0] * scale)),
                             int(round(camera_json['image_size'][1] * scale)))),
    )

@DATA_FORMAT_REGISTRY.register()
class NerfiesReFormat(BaseReFormatData):
    def __init__(self,
                 data_root: Path,
                 split: str = 'train',
                 cached_image: bool = True,
                 scale: float = 1.0):
        super().__init__(data_root, split, cached_image, scale)
    
    def load_camera(self, split: str):
        with open(f'{self.data_root}/scene.json', 'r') as f:
            scene_json = json.load(f)
        with open(f'{self.data_root}/metadata.json', 'r') as f:
            meta_json = json.load(f)
        with open(f'{self.data_root}/dataset.json', 'r') as f:
            dataset_json = json.load(f)
        
        coord_scale = scene_json['scale']
        scene_center = scene_json['center']

        self.scene_center = scene_center
        self.coord_scale = coord_scale

        name = self.data_root.split('/')[-2]
        if name.startswith('vrig'):
            train_img = dataset_json['train_ids']
            val_img = dataset_json['val_ids']
            all_img = train_img + val_img
            ratio = 0.25
        elif name.startswith('interp'):
            all_id = dataset_json['ids']
            train_img = all_id[::4]
            val_img = all_id[2::4]
            all_img = train_img + val_img
            ratio = 0.5
        else:  # for hypernerf
            train_img = dataset_json['ids']
            all_img = train_img
            ratio = 1.0

        train_num = len(train_img)

        all_cam = [meta_json[i]['camera_id'] for i in all_img]
        all_time = [meta_json[i]['appearance_id'] for i in all_img]
        max_time = max(all_time)
        all_time = [meta_json[i]['appearance_id'] / max_time for i in all_img]
        selected_time = set(all_time)

        # all poses
        all_cam_params = []
        for im in all_img:
            camera = camera_nerfies_from_JSON(f'{self.data_root}/camera/{im}.json', ratio)
            camera['position'] = camera['position'] - scene_center
            camera['position'] = camera['position'] * coord_scale
            all_cam_params.append(camera)

        all_img = [f'{self.data_root}/rgb/{int(1 / ratio)}x/{i}.png' for i in all_img]

        cameras = []
        for idx in range(len(all_img)):
            image_path = all_img[idx]
            image = np.array(Image.open(image_path))
            image = Image.fromarray((image).astype(np.uint8))
            image_name = Path(image_path).stem

            orientation = all_cam_params[idx]['orientation'].T
            position = -all_cam_params[idx]['position'] @ orientation
            focal = all_cam_params[idx]['focal_length']
            fid = all_time[idx]
            T = position
            R = orientation

            camera = Camera(idx=idx, fid=fid, R=R, T=T, width=image.size[0], height=image.size[1], rgb_file_name=image_name,
                            rgb_file_path=image_path, fx=focal, fy=focal, cx=image.size[0]/2, cy=image.size[1]/2, bg=0.0)
            cameras.append(camera)
        
        if split == 'train':
            cameras_results = cameras[0::2]
        else:
            cameras_results = cameras[1::2]
        
        return cameras_results
    
    def load_image_filenames(self, cameras, split):
        """
        The function for loading the image files names typically requires user customization.

        Parameters
        ----------
        split: The split of the data.
        """
        image_filenames = []
        for camera in cameras:
            image_filenames.append(camera.rgb_file_path)
        return image_filenames
    
    def load_pointcloud(self):
        xyz = np.load(os.path.join(self.data_root, "points.npy"))
        xyz = (xyz - self.scene_center) * self.coord_scale
        num_pts = xyz.shape[0]
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = SimplePointCloud(positions=xyz, colors=SH2RGB(
            shs), normals=np.zeros((num_pts, 3)))
        return pcd