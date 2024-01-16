import sys
sys.path.append('../../')
from pointrix.camera.camera import Camera, TrainableCamera
import torch

def test_camera():
    idx = 0
    width = 100
    height = 100
    fovX = 35
    fovY = 35
    bg = 0.0
    R = torch.eye(3, 3)
    T = torch.ones(1, 3)

    camera = Camera(idx=idx, width=width, height=height,
                    fovX=fovX, fovY=fovY, bg=bg, R=R, T=T)

    print(camera.world_view_transform)
    print(camera.projection_matrix)
    print(camera.full_proj_transform)
    print(camera.intrinsics_matrix)
    print(camera.camera_center)
    print(camera.image_height)

    camera.load2device('cuda')

    print(camera.world_view_transform)
    print(camera.projection_matrix)
    print(camera.full_proj_transform)
    print(camera.intrinsics_matrix)
    print(camera.camera_center)
    print(camera.image_height)


def test_trainable_camera():
    idx = 0
    width = 100
    height = 100
    fovX = 35
    fovY = 35
    bg = 0.0
    R = torch.eye(3, 3)
    T = torch.ones(1, 3)

    camera = TrainableCamera(
        idx=idx, width=width, height=height, fovX=fovX, fovY=fovY, bg=bg, R=R, T=T)

    print(camera.param_groups)


if __name__ == "__main__":
    test_camera()
    test_trainable_camera()
