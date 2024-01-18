#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

def splatting_render(
    FovX,
    FovY,
    height,
    width,
    world_view_transform, 
    full_proj_transform,
    camera_center,
    active_sh_degree,
    position,
    opacity,
    scaling,
    rotation, 
    shs,
    bg_color : torch.Tensor, 
    scaling_modifier = 1.0, 
    render_xyz=False,
    **kwargs,
):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(
        position, 
        dtype=position.dtype, 
        requires_grad=True, 
        device="cuda"
    ) + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(FovX * 0.5)
    tanfovy = math.tan(FovY * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(height),
        image_width=int(width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=world_view_transform.cuda(),
        projmatrix=full_proj_transform.cuda(),
        sh_degree=active_sh_degree,
        campos=camera_center.cuda(),
        prefiltered=False,
        # computer_xyz=render_xyz,
        debug=False,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = position
    means2D = screenspace_points
    cov3D_precomp = None
        
    # trbfscale = pc._time_scale_params
    # trbfdistanceoffset = pc.timestamp_final
    # trbfdistance =  trbfdistanceoffset / torch.exp(trbfscale) 
    # trbfoutput = torch.exp(-1*trbfdistance.pow(2))
    
    # opacity = opacity * trbfoutput 

    colors_precomp = None

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    (
        # num_rendered, 
        rendered_image, 
        # opacity, 
        # depth, 
        # render_xyz, 
        radii,
    ) = rasterizer(
        means3D = means3D.contiguous(),
        means2D = means2D.contiguous(),
        shs = shs.contiguous(),
        colors_precomp = colors_precomp,
        opacities = opacity.contiguous(),
        scales = scaling.contiguous(),
        rotations = rotation.contiguous(),
        cov3D_precomp = cov3D_precomp
    )


    # import pdb; pdb.set_trace()
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            # "opacity": opacity,
            # "render_xyz": render_xyz,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "xyz": means3D,
            "color": shs,
            "rot": rotation,
            "scales": scaling,
            "xy": means2D,}
