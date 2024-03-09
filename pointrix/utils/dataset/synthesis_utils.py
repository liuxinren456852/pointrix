import functools
import logging
import math
import numpy as np
from collections.abc import Mapping
from torch.utils.data import Dataset
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union
from .dataset_utils import focal2fov, fov2focal
import random

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config
from point_e.diffusion.sampler import PointCloudSampler
from point_e.models.download import load_checkpoint
from point_e.models.configs import MODEL_CONFIGS, model_from_config
from point_e.util.plotting import plot_point_cloud
from pointrix.camera.camera import Camera


def safe_normalize(x, eps=1e-20):
    return x / torch.sqrt(torch.clamp(torch.sum(x * x, -1, keepdim=True), min=eps))


# TODO base_name ,device，prompt都可以放到cfg
def init_by_point_e(base_name, prompt,num_pts):
    print('creating base model...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_model = model_from_config(MODEL_CONFIGS[base_name], device)
    base_model.eval()
    base_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[base_name])

    print('creating upsample model...')
    upsampler_model = model_from_config(MODEL_CONFIGS['upsample'], device)
    upsampler_model.eval()
    upsampler_diffusion = diffusion_from_config(DIFFUSION_CONFIGS['upsample'])

    print('downloading base checkpoint...')
    base_model.load_state_dict(load_checkpoint(base_name, device))

    print('downloading upsampler checkpoint...')
    upsampler_model.load_state_dict(load_checkpoint('upsample', device))

    sampler = PointCloudSampler(
        device=device,
        models=[base_model, upsampler_model],
        diffusions=[base_diffusion, upsampler_diffusion],
        num_points=[1024, num_pts - 1024],
        guidance_scale=[3.0, 0.0],
        aux_channels=['R', 'G', 'B'],
        # Do not condition the upsampler at all
        model_kwargs_key_filter=('texts', ''),
    )

    samples = None

    for x in tqdm(sampler.sample_batch_progressive(batch_size=1, model_kwargs=dict(texts=[prompt]))):
        samples = x
    pc = sampler.output_to_point_clouds(samples)[0]
    xyz = pc.coords
    rgb = np.zeros_like(xyz)
    rgb[:, 0], rgb[:, 1], rgb[:,
                              2] = pc.channels["R"], pc.channels["G"], pc.channels["B"]
    del base_model, upsampler_model, sampler
    torch.cuda.empty_cache()
    return xyz, rgb


def random_pos(size, param_range, gamma=1):
    lower, higher = param_range[0], param_range[1]

    mid = lower + (higher - lower) * 0.5
    radius = (higher - lower) * 0.5

    rand_ = torch.rand(size)  # 0, 1
    sign = torch.where(torch.rand(size) > 0.5,
                       torch.ones(size) * -1., torch.ones(size))
    rand_ = sign * (rand_ ** gamma)

    return (rand_ * radius) + mid


def rand_poses(size,
               cfg,
               radius_range: List[float] = [1, 1.5],
               theta_range: List[float] = [0, 120],
               phi_range: List[float] = [0, 360],
               angle_overhead: int = 30,
               angle_front: int = 60,
               uniform_sphere_rate: float = 0.5,
               rand_cam_gamma: float = 1):
    ''' generate random poses from an orbit camera
    Args:
        size: batch size of generated poses.
        device: where to allocate the output.
        radius: camera radius
        theta_range: [min, max], should be in [0, pi]
        phi_range: [min, max], should be in [0, 2 * pi]
    Return:
        poses: [size, 4, 4]
    '''

    theta_range = np.array(theta_range) / 180 * np.pi
    phi_range = np.array(phi_range) / 180 * np.pi
    angle_overhead = angle_overhead / 180 * np.pi
    angle_front = angle_front / 180 * np.pi

    radius = random_pos(size, radius_range)

    if random.random() < uniform_sphere_rate:
        unit_centers = F.normalize(
            torch.stack([
                torch.randn(size),
                torch.abs(torch.randn(size)),
                torch.randn(size),
            ], dim=-1), p=2, dim=1
        )
        thetas = torch.acos(unit_centers[:, 1])
        phis = torch.atan2(unit_centers[:, 0], unit_centers[:, 2])
        phis[phis < 0] += 2 * np.pi
        centers = unit_centers * radius.unsqueeze(-1)
    else:

        thetas = random_pos(size, theta_range, rand_cam_gamma)
        phis = random_pos(size, phi_range, rand_cam_gamma)
        phis[phis < 0] += 2 * np.pi

        centers = torch.stack([
            radius * torch.sin(thetas) * torch.sin(phis),
            radius * torch.sin(thetas) * torch.cos(phis),
            radius * torch.cos(thetas),
        ], dim=-1)  # [B, 3]

    targets = 0

    # jitters
    if cfg.jitter_pose:
        jit_center = cfg.jitter_center  # 0.015  # was 0.2
        jit_target = cfg.jitter_target
        centers += torch.rand_like(centers) * jit_center - jit_center/2.0
        targets += torch.randn_like(centers) * jit_target

    # lookat
    forward_vector = safe_normalize(centers - targets)
    up_vector = torch.FloatTensor([0, 0, 1]).unsqueeze(0).repeat(size,1)
    right_vector = safe_normalize(
        torch.cross(forward_vector, up_vector, dim=-1))

    if cfg.jitter_pose:
        up_noise = torch.randn_like(up_vector) * cfg.jitter_up
    else:
        up_noise = 0

    up_vector = safe_normalize(torch.cross(
        right_vector, forward_vector, dim=-1) + up_noise)  # forward_vector

    poses = torch.eye(4, dtype=torch.float).unsqueeze(0).repeat(size, 1, 1)
    poses[:, :3, :3] = torch.stack(
        (-right_vector, up_vector, forward_vector), dim=-1)  # up_vector
    poses[:, :3, 3] = centers

    # back to degree
    thetas = thetas / np.pi * 180
    phis = phis / np.pi * 180

    return poses.numpy(), thetas.numpy(), phis.numpy(), radius.numpy()


def circle_poses(radius=torch.tensor([3.2]), theta=torch.tensor([60]), phi=torch.tensor([0]), angle_overhead=30, angle_front=60):

    theta = theta / 180 * np.pi
    phi = phi / 180 * np.pi
    angle_overhead = angle_overhead / 180 * np.pi
    angle_front = angle_front / 180 * np.pi

    centers = torch.stack([
        radius * torch.sin(theta) * torch.sin(phi),
        radius * torch.sin(theta) * torch.cos(phi),
        radius * torch.cos(theta),
    ], dim=-1)  # [B, 3]

    # lookat
    forward_vector = safe_normalize(centers)
    up_vector = torch.FloatTensor(
        [0, 0, 1]).unsqueeze(0).repeat(len(centers), 1)
    right_vector = safe_normalize(
        torch.cross(forward_vector, up_vector, dim=-1))
    up_vector = safe_normalize(torch.cross(
        right_vector, forward_vector, dim=-1))

    poses = torch.eye(4, dtype=torch.float).unsqueeze(
        0).repeat(len(centers), 1, 1)
    poses[:, :3, :3] = torch.stack(
        (-right_vector, up_vector, forward_vector), dim=-1)
    poses[:, :3, 3] = centers

    return poses.numpy()


def generate_circle_cameras(cfg, size=8, render45=False):
    # random focal
    fov = cfg.default_fovy
    cam_infos = []
    polars_list=[]
    azimuths_list=[]
    radius_list=[]
    # generate specific data structure
    focal=fov2focal(fov, cfg.image_h)
    
    look_pos=2.7*cfg.radius_now_scale
    for idx in range(size):
        thetas = torch.FloatTensor([cfg.default_polar])
        phis = torch.FloatTensor([(idx / size) * 360])
        radius = torch.FloatTensor([look_pos])
        # random pose on the fly
        poses = circle_poses(radius=radius, theta=thetas, phi=phis,
                             angle_overhead=cfg.angle_overhead, angle_front=cfg.angle_front)
        matrix = np.linalg.inv(poses[0])
        R = -np.transpose(matrix[:3, :3])
        R[:, 0] = -R[:, 0]
        T = -matrix[:3, 3]
        

        # delta polar/azimuth/radius to default view
        delta_polar = thetas - cfg.default_polar
        delta_azimuth = phis - cfg.default_azimuth
        delta_azimuth[delta_azimuth > 180] -= 360  # range in [-180, 180]
        delta_radius = radius - cfg.default_radius
        
        polars_list.append(delta_polar)
        azimuths_list.append(delta_azimuth)
        radius_list.append(delta_radius)
        
        camera = Camera(idx=idx, R=R, T=T, width=cfg.image_w, height=cfg.image_h, rgb_file_name=f"{idx:03d}_rgb.png",
                        fx=focal, fy=focal, cx=cfg.image_w/2, cy=cfg.image_h/2, bg=0.0, scene_scale=1.0)
        cam_infos.append(camera)
    if render45:
        for idx in range(size):
            thetas = torch.FloatTensor([cfg.default_polar*2//3])
            phis = torch.FloatTensor([(idx / size) * 360])
            radius = torch.FloatTensor([cfg.default_radius])
            # random pose on the fly
            poses = circle_poses(radius=radius, theta=thetas, phi=phis,
                                 angle_overhead=cfg.angle_overhead, angle_front=cfg.angle_front)
            matrix = np.linalg.inv(poses[0])
            R = -np.transpose(matrix[:3, :3])
            R[:, 0] = -R[:, 0]
            T = -matrix[:3, 3]
            focal=fov2focal(fov, cfg.image_h)

            # delta polar/azimuth/radius to default view
            delta_polar = thetas - cfg.default_polar
            delta_azimuth = phis - cfg.default_azimuth
            delta_azimuth[delta_azimuth > 180] -= 360  # range in [-180, 180]
            delta_radius = radius - cfg.default_radius

            polars_list.append(delta_polar)
            azimuths_list.append(delta_azimuth)
            radius_list.append(delta_radius)

            camera = Camera(idx=idx, R=R, T=T, width=cfg.image_w, height=cfg.image_h, rgb_file_name=f"{idx:03d}_rgb.png",
                        fx=focal, fy=focal, cx=cfg.image_w/2, cy=cfg.image_h/2, bg=0.0, scene_scale=1.0)
            cam_infos.append(camera)

    spherical_coordinate_infos={"polar":polars_list,"azimuth":azimuths_list,"radius":radius_list}
    return cam_infos, spherical_coordinate_infos


def generate_random_cameras(size,cfg, SSAA=True):

    radius_range=[]
    radius_range.append(cfg.radius_range[0])
    radius_range.append(cfg.radius_range[1])
    radius_range[0]*=cfg.radius_now_scale
    radius_range[1]*=cfg.radius_now_scale
    # random pose on the fly
    poses, thetas, phis, radius = rand_poses(size, cfg, radius_range=radius_range, theta_range=cfg.theta_range, phi_range=cfg.phi_range,
                                             angle_overhead=cfg.angle_overhead, angle_front=cfg.angle_front, uniform_sphere_rate=cfg.uniform_sphere_rate,
                                             rand_cam_gamma=cfg.rand_cam_gamma)
    # delta polar/azimuth/radius to default view

    delta_polar = thetas - cfg.default_polar
    delta_azimuth = phis - cfg.default_azimuth
    delta_azimuth[delta_azimuth > 180] -= 360  # range in [-180, 180]
    delta_radius = radius - cfg.default_radius
    # random focal
    fov = random.random() * \
        (cfg.fovy_range[1] - cfg.fovy_range[0]) + cfg.fovy_range[0]

    cam_infos = []

    if SSAA:
        ssaa = cfg.SSAA
    else:
        ssaa = 1

    image_h = cfg.image_h * ssaa
    image_w = cfg.image_w * ssaa
    focal=fov2focal(fov, image_h)

    # generate specific data structure
    for idx in range(size):
        matrix = np.linalg.inv(poses[idx])
        R = -np.transpose(matrix[:3, :3])
        R[:, 0] = -R[:, 0]
        T = -matrix[:3, 3]
        # fovy = focal2fov(fov2focal(fov, image_h), image_w)
        # FovY = fovy
        # FovX = fov
        # focal=fov2focal(cfg.fov, image_h)

        camera = Camera(idx=idx, R=R, T=T, width=cfg.image_w, height=cfg.image_h, rgb_file_name=f"{idx:03d}_rgb.png",
                        fx=focal, fy=focal, cx=cfg.image_w/2, cy=cfg.image_h/2, bg=0.0, scene_scale=1.0)
        cam_infos.append(camera)
    
    spherical_coordinate_infos={"polar":delta_polar,"azimuth":delta_azimuth,"radius":delta_radius}
    return cam_infos, spherical_coordinate_infos
