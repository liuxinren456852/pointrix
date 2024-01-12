import torch
import numpy as np

def fov2focal(fov, pixels):
    return pixels / (2 * torch.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2 * torch.arctan(pixels / (2 * focal))

class SimpleCamera:
    def __init__(self, cam_idx, R, T, width=800, height=800, fovY=35, bg=0.0):
        
        self.idx = cam_idx
        self.R = R
        self.T = T
        self.width = width
        self.height = height

        # input Fov is in degree, and turned into radian here
        self.fovY = torch.deg2rad(torch.tensor(fovY,dtype=torch.float, device="cuda"))
        self.fovX = focal2fov(fov2focal(self.fovY, height), width)
        
        self.bg = torch.Tensor([bg, bg, bg]).cuda()

        self.world_view_transform = torch.tensor(self.getWorld2View(R, T)).transpose(0, 1).cuda()
        self.projection_matrix = self.getProjectionMatrix(self.fovX, self.fovY).transpose(0, 1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    @staticmethod
    def getWorld2View(R, t):
        Rt = np.zeros((4, 4))
        Rt[:3, :3] = R.transpose()
        Rt[:3, 3] = t
        Rt[3, 3] = 1.0
        return np.float32(Rt)
    
    def get_intrinsics(self):
        focal_x = self.width / (2 * np.tan(self.fovX * 0.5))
        focal_y = self.height / (2 * np.tan(self.fovY * 0.5))

        return torch.tensor([[focal_x, 0, self.width / 2],
                             [0, focal_y, self.height / 2],
                             [0, 0, 1]], device='cuda', dtype=torch.float32)
    
    @staticmethod
    def getProjectionMatrix(fovX, fovY, znear=0.1, zfar=100):
        tanHalfFovY = torch.tan((fovY / 2))
        tanHalfFovX = torch.tan((fovX / 2))

        top = tanHalfFovY * znear
        bottom = -top
        right = tanHalfFovX * znear
        left = -right

        P = torch.zeros(4, 4)

        z_sign = 1.0

        P[0, 0] = 2.0 * znear / (right - left)
        P[1, 1] = 2.0 * znear / (top - bottom)
        P[0, 2] = (right + left) / (right - left)
        P[1, 2] = (top + bottom) / (top - bottom)
        P[3, 2] = z_sign
        P[2, 2] = z_sign * zfar / (zfar - znear)
        P[2, 3] = -(zfar * znear) / (zfar - znear)

        return P