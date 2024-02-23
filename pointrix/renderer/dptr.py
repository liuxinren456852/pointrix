import torch
import numpy as np
import dptr.gs as gs
from .base_splatting import RENDERER_REGISTRY, GaussianSplattingRender


@RENDERER_REGISTRY.register()
class DPTRRender(GaussianSplattingRender):
    """
    A class for rendering point clouds using DPTR.

    Parameters
    ----------
    cfg : dict
        The configuration dictionary.
    white_bg : bool
        Whether the background is white or not.
    device : str
        The device to use.
    update_sh_iter : int, optional
        The iteration to update the spherical harmonics degree, by default 1000.
    """

    def __init__(self, cfg, white_bg, device, update_sh_iter=1000, **kwargs):
        super(DPTRRender, self).__init__(cfg, white_bg, device, update_sh_iter, **kwargs)

    def render_iter(self,
                    FovX,
                    FovY,
                    height,
                    width,
                    world_view_transform,
                    full_proj_transform,
                    camera_center,
                    position,
                    opacity,
                    scaling,
                    rotation,
                    shs,
                    scaling_modifier=1.0,
                    render_xyz=False,
                    **kwargs) -> dict:
        """
        Render the point cloud for one iteration

        Parameters
        ----------
        FovX : float
            The field of view in the x-axis.
        FovY : float
            The field of view in the y-axis.
        height : float
            The height of the image.
        width : float
            The width of the image.
        world_view_transform : torch.Tensor
            The world view transformation matrix.
        full_proj_transform : torch.Tensor
            The full projection transformation matrix.
        camera_center : torch.Tensor
            The camera center.
        position : torch.Tensor
            The position of the point cloud.
        opacity : torch.Tensor
            The opacity of the point cloud.
        scaling : torch.Tensor
            The scaling of the point cloud.
        rotation : torch.Tensor
            The rotation of the point cloud.
        shs : torch.Tensor
            The spherical harmonics of the point cloud.
        scaling_modifier : float, optional
            The scaling modifier, by default 1.0
        render_xyz : bool, optional
            Whether to render the xyz or not, by default False
        
        Returns
        -------
        dict
            The rendered image, the viewspace points, 
            the visibility filter, the radii, the xyz, 
            the color, the rotation, the scales, and the xy.
        """
        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        # import pdb; pdb.set_trace()
        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.

        direction = (position.cuda() - camera_center.repeat(position.shape[0], 1).cuda())
        direction = direction / direction.norm(dim=1, keepdim=True)
        rgb = gs.compute_sh(shs, 3, direction)

        camparams = torch.Tensor([
            width / (2 * np.tan(FovX * 0.5)),
            height / (2 * np.tan(FovY * 0.5)),
            float(width) / 2, 
            float(height) / 2]).cuda().float()
        
        (uv, depth) = gs.project_point(
            position, 
            world_view_transform.cuda(), 
            full_proj_transform.cuda(), 
            camparams, width, height)
        
        visible = depth != 0

        # compute cov3d
        cov3d = gs.compute_cov3d(scaling, rotation, visible)

        # ewa project
        (conic, radius, tiles_touched) = gs.ewa_project(
            position, 
            cov3d, 
            world_view_transform.cuda(), 
            camparams, 
            uv, 
            width, 
            height, 
            visible
        )

        # sort
        (gaussian_ids_sorted, tile_range) = gs.sort_gaussian(
            uv, depth, width, height, radius, tiles_touched
        )

        # alpha blending
        ndc = torch.zeros_like(uv, requires_grad=True)
        try:
            ndc.retain_grad()
        except:
            pass

        render_feature = gs.alpha_blending(
            uv, conic, opacity, rgb, 
            gaussian_ids_sorted, tile_range, 1.0, width, height, ndc
        )

        return {"render": render_feature,
                "viewspace_points": ndc,
                "visibility_filter": radius > 0,
                "radii": radius
                }