import math
import torch
from torch import Tensor
from torch import nn
import numpy as np
from jaxtyping import Float
from numpy.typing import NDArray
from typing import Union, List
from dataclasses import dataclass, field
# from pointrix.camera.camera_utils import se3_exp_map
from pointrix.utils.pose import se3_exp_map

@dataclass()
class Camera:
    """
    Camera class used in Pointrix

    Parameters
    ----------
    idx: int
        The index of the camera.
    width: int
        The width of the image.
    height: int
        The height of the image.
    R: Float[Tensor, "3 3"]
        The rotation matrix of the camera.
    T: Float[Tensor, "3 1"]
        The translation vector of the camera.
    fx: float
        The focal length of the camera in x direction.
    fy: float
        The focal length of the camera in y direction.
    cx: float
        The center of the image in x direction.
    cy: float
        The center of the image in y direction.
    fovX: float
        The field of view of the camera in x direction.
    fovY: float
        The field of view of the camera in y direction.
    bg: float
        The background color of the camera.
    rgb_file_name: str
        The path of the image.
    radius: float
        The radius of the camera.
    scene_scale: float
        The scale of the scene.

    Notes
    -----
    fx, fy, cx, cy and fovX, fovY are mutually exclusive. 
    If fx, fy, cx, cy are provided, fovX, fovY will be calculated from them. 
    If fovX, fovY are provided, fx, fy, cx, cy will be calculated from them.

    Examples
    --------
    >>> idx = 1
    >>> width = 800
    >>> height = 600
    >>> R = np.eye(3)
    >>> T = np.zeros(3)
    >>> focal_length_x = 800
    >>> focal_length_y = 800
    >>> camera = Camera(idx=idx, R=R, T=T, width=width, height=height, rgb_file_name='1_rgb.png',
                        fx=focal_length_x, fy=focal_length_y, cx=width/2, cy=height/2, bg=0.0, scene_scale=1.0)
    """
    idx: int
    width: int
    height: int
    R: Union[Float[Tensor, "3 3"], NDArray]
    T: Union[Float[Tensor, "3 1"], NDArray]
    fx: Union[float, None] = None
    fy: Union[float, None] = None
    cx: Union[float, None] = None
    cy: Union[float, None] = None
    fovX: Union[float, None] = None
    fovY: Union[float, None] = None
    bg: float = 0.0
    rgb_file_name: str = None
    radius: float = 0.0
    scene_scale: float = 1.0
    _world_view_transform: Float[Tensor, "4 4"] = field(init=False)
    _projection_matrix: Float[Tensor, "4 4"] = field(init=False)
    _intrinsics_matrix: Float[Tensor, "3 3"] = field(init=False)
    _full_proj_transform: Float[Tensor, "4 4"] = field(init=False)
    _camera_center: Float[Tensor, "3"] = field(init=False)

    def __post_init__(self):
        assert (self.fx is not None and self.fy is not None and self.cx is not None and self.cy is not None
                or self.fovX is not None and self.fovY is not None), "Either fx, fy, cx, cy or fovX, fovY must be provided"
        if self.fx is None:
            self.fx = self.width / (2 * np.tan(self.fovX * 0.5))
            self.fy = self.height / (2 * np.tan(self.fovY * 0.5))
            self.cx = self.width / 2
            self.cy = self.height / 2
        elif self.fovX is None:
            # TODO: check if this is correct
            self.fovX = 2 * math.atan(self.width / (2 * self.fx))
            self.fovY = 2 * math.atan(self.height / (2 * self.fy))

        if not isinstance(self.R, Tensor):
            self.R = torch.tensor(self.R)
        if not isinstance(self.T, Tensor):
            self.T = torch.tensor(self.T)
        self._world_view_transform = self.getWorld2View(
            self.R, self.T, scale=self.scene_scale
        ).transpose(0, 1)
        self._projection_matrix = self.getProjectionMatrix(
            self.fovX, self.fovY).transpose(0, 1)
        self._full_proj_transform = (self.world_view_transform.unsqueeze(
            0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self._intrinsics_matrix = self._get_intrinsics(
            self.fx, self.fy, self.cx, self.cy)
        self._camera_center = self.world_view_transform.inverse()[3, :3]

    def load2device(self, device:str):
        """
        Load all the parameters of the camera to the device.
        
        Parameters
        ----------
        device: str
            The device to load the parameters to.

        Notes
        -----
            This function will load part of the parameters of the camera to the device.
        """
        self._world_view_transform = self._world_view_transform.to(device)
        self._projection_matrix = self._projection_matrix.to(device)
        self._full_proj_transform = self._full_proj_transform.to(device)
        self._intrinsics_matrix = self._intrinsics_matrix.to(device)
        self._camera_center = self._camera_center.to(device)

    @staticmethod
    def getWorld2View(
        R: Float[Tensor, "3 3"], 
        t: Float[Tensor, "3 1"], 
        scale: float=1.0,
        translate: Float[Tensor, "3"] = torch.tensor([0., 0., 0.])
    ) -> Float[Tensor, "4 4"]:
        """
        Get the world to view transform.
        
        Parameters
        ----------
        R: Float[Tensor, "3 3"]
            The rotation matrix of the camera.
        t: Float[Tensor, "3 1"]
            The translation vector of the camera.
        scale: float
            The scale of the scene.
        translate: Float[Tensor, "3"]
            The translation vector of the scene.
            
        Returns
        -------
        Rt: Float[Tensor, "4 4"]
            The world to view transform.
    
        Notes
        -----
        only used in the camera class

        """
        Rt = torch.zeros((4, 4))
        Rt[:3, :3] = R.transpose(0, 1)
        Rt[:3, 3] = t
        Rt[3, 3] = 1.0
    
        C2W = torch.linalg.inv(Rt)
        cam_center = C2W[:3, 3]
        cam_center = (cam_center + translate) * scale
        C2W[:3, 3] = cam_center
        Rt = torch.linalg.inv(C2W)
        return Rt.float()

    @staticmethod
    def _get_intrinsics(fx: float, fy: float, cx: float, cy: float) -> Float[Tensor, "3 3"]:
        """
        Get the intrinsics matrix.
        
        Parameters
        ----------
        fx: float
            The focal length of the camera in x direction.
        fy: float
            The focal length of the camera in y direction.
        cx: float
            The center of the image in x direction.
        cy: float
            The center of the image in y direction.
            
        Returns
        -------
        intrinsics: Float[Tensor, "3 3"]
            The intrinsics matrix.
        Notes
        -----
        only used in the camera class
        """
        return torch.tensor([[fx, 0, cx],
                             [0, fy, cy],
                             [0, 0, 1]], dtype=torch.float32)

    @staticmethod
    def getProjectionMatrix(fovX: float, fovY: float, znear: float = 0.01, zfar: float = 100) -> Float[Tensor, "4 4"]:
        """
        Get the projection matrix.
            
        Parameters
        ----------
        fovX: float
            The field of view of the camera in x direction.
        fovY: float
            The field of view of the camera in y direction.
        znear: float
            The near plane of the camera.
        zfar: float
            The far plane of the camera.
        
        Returns
        -------
        P: Float[Tensor, "4 4"]
            The projection matrix.
        Notes
        -----
        only used in the camera class

        """
        tanHalfFovY = np.tan((fovY / 2))
        tanHalfFovX = np.tan((fovX / 2))

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

    @property
    def world_view_transform(self) -> Float[Tensor, "4 4"]:
        """
        Get the world to view transform from the camera.
              
        Returns
        -------
        _world_view_transform: Float[Tensor, "4 4"]
            The world to view transform.
       
        Notes
        -----
        property of the camera class

        """
        return self._world_view_transform

    @property
    def projection_matrix(self) -> Float[Tensor, "4 4"]:
        """
        Get the projection matrix from the camera.
              
        Returns
        -------
        _projection_matrix: Float[Tensor, "4 4"]
       
        Notes
        -----
        property of the camera class

        """
        return self._projection_matrix

    @property
    def full_proj_transform(self) -> Float[Tensor, "4 4"]:
        """
        Get the full projection matrix from the camera.
              
        Returns
        -------
        _full_proj_transform: Float[Tensor, "4 4"]
       
        Notes
        -----
        property of the camera class

        """
        return self._full_proj_transform

    @property
    def intrinsics_matrix(self) -> Float[Tensor, "3 3"]:
        """
        Get the intrinsics matrix from the camera.
              
        Returns
        -------
        _intrinsics_matrix: Float[Tensor, "3 3"]
       
        Notes
        -----
        property of the camera class

        """
        return self._intrinsics_matrix

    @property
    def camera_center(self) -> Float[Tensor, "1 3"]:
        """
        Get the camera center from the camera.
              
        Returns
        -------
        _camera_center: Float[Tensor, "1 3"]
       
        Notes
        -----
        property of the camera class

        """
        return self._camera_center

    @property
    def image_height(self) -> int:
        """
        Get the image height from the camera.
              
        Returns
        -------
        height: int
            The image height.
       
        Notes
        -----
        property of the camera class

        """
        return self.height

    @property
    def image_width(self) -> int:
        """
        Get the image width from the camera.
              
        Returns
        -------
        width: int
            The image width.
       
        Notes
        -----
        property of the camera class

        """
        return self.width


@dataclass()
class TrainableCamera(Camera):
    """
    Trainable Camera class used in Pointrix

    Parameters
    ----------
    idx: int
        The index of the camera.
    width: int
        The width of the image.
    height: int
        The height of the image.
    R: Float[Tensor, "3 3"]
        The rotation matrix of the camera.
    T: Float[Tensor, "3 1"]
        The translation vector of the camera.
    fx: float
        The focal length of the camera in x direction.
    fy: float
        The focal length of the camera in y direction.
    cx: float
        The center of the image in x direction.
    cy: float
        The center of the image in y direction.
    fovX: float
        The field of view of the camera in x direction.
    fovY: float
        The field of view of the camera in y direction.
    bg: float
        The background color of the camera.
    rgb_file_name: str
        The path of the image.
    radius: float
        The radius of the camera.
    scene_scale: float
        The scale of the scene.

    Notes
    -----
    fx, fy, cx, cy and fovX, fovY are mutually exclusive. 
    If fx, fy, cx, cy are provided, fovX, fovY will be calculated from them. 
    If fovX, fovY are provided, fx, fy, cx, cy will be calculated from them.

    Examples
    --------
    >>> idx = 1
    >>> width = 800
    >>> height = 600
    >>> R = np.eye(3)
    >>> T = np.zeros(3)
    >>> focal_length_x = 800
    >>> focal_length_y = 800
    >>> camera = TrainableCamera(idx=idx, R=R, T=T, width=width, height=height, rgb_file_name='1_rgb.png',
                        fx=focal_length_x, fy=focal_length_y, cx=width/2, cy=height/2, bg=0.0, scene_scale=1.0)
    """
    def __post_init__(self):
        super().__post_init__()
        self._omega = nn.Parameter(torch.zeros(6).requires_grad_(True))

    @property
    def param_groups(self)->List:
        """
        Get the parameter groups of the camera.
              
        Returns
        -------
        param_groups: List
       
        Notes
        -----
        property of the camera class

        """
        return [self._omega]

    @property
    def _exp_factor(self) -> Float[Tensor, "4 4"]:
        """
        Get the exponential fix factor of the camera.
              
        Returns
        -------
        exp_factor: Float[Tensor, "4 4"]
       
        Notes
        -----
        property of the camera class

        """
        return se3_exp_map(self._omega.view(1, 6)).view(4, 4)

    @property
    def world_view_transform(self) -> Float[Tensor, "4 4"]:
        """
        Get the fixed world to view transform from the camera.
              
        Returns
        -------
        _world_view_transform: Float[Tensor, "4 4"]
            The world to view transform.
       
        Notes
        -----
        property of the camera class

        """
        return self._world_view_transform @ self._exp_factor

    @property
    def full_proj_transform(self) -> Float[Tensor, "4 4"]:
        """
        Get the full projection matrix from the camera.
              
        Returns
        -------
        _full_proj_transform: Float[Tensor, "4 4"]
       
        Notes
        -----
        property of the camera class

        """
        return (self.world_view_transform.unsqueeze(0).bmm(self._projection_matrix.unsqueeze(0))).squeeze(0)

    def load2device(self, device):
        """
        Load all the parameters of the camera to the device.
        
        Parameters
        ----------
        device: str
            The device to load the parameters to.

        Notes
        -----
            This function will load part of the parameters of the camera to the device.
        """
        self._world_view_transform = self._world_view_transform.to(device)
        self._projection_matrix = self._projection_matrix.to(device)
        self._full_proj_transform = self._full_proj_transform.to(device)
        self._intrinsics_matrix = self._intrinsics_matrix.to(device)
        self._camera_center = self._camera_center.to(device)
        self._omega = self._omega.to(device)
