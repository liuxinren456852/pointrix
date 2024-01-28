import os
import time
import torch
import numpy as np
from tqdm import tqdm
from datetime import datetime
from pytorch_msssim import ms_ssim
from pointrix.utils.losses import l1_loss

import imageio
from collections import defaultdict

from simple_knn._C import distCUDA2

# FIXME: this is a hack to build lpips loss and lpips metric
from lpips import LPIPS
lpips_net = LPIPS(net="vgg").to("cuda")
def lpips_norm_fn(x): return x[None, ...] * 2 - 1
def lpips_norm_b_fn(x): return x * 2 - 1
def lpips_fn(x, y): return lpips_net(lpips_norm_fn(x), lpips_norm_fn(y)).mean()


def lpips_b_fn(x, y): return lpips_net(
    lpips_norm_b_fn(x), lpips_norm_b_fn(y)).mean()


def inverse_sigmoid(x):
    return torch.log(x/(1-x))


def build_rotation(r):
    norm = torch.sqrt(r[:, 0]*r[:, 0] + r[:, 1]*r[:, 1] +
                      r[:, 2]*r[:, 2] + r[:, 3]*r[:, 3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R


def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):

    s = scaling_modifier * scaling

    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float)

    R = build_rotation(rotation)

    L[:, 0, 0] = s[:, 0]
    L[:, 1, 1] = s[:, 1]
    L[:, 2, 2] = s[:, 2]

    L = R @ L
    # L = build_scaling_rotation(scaling_modifier * scaling, rotation)
    L = L @ L.transpose(1, 2)

    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float)

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]

    return uncertainty


def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

# TODO: rewite this function, it is ugly


def validation_process(render_func, datapipeline, global_step=0, logger=None):
    l1_test = 0.0
    psnr_test = 0.0
    ssims_test = 0.0
    lpips_test = 0.0
    val_dataset = datapipeline.validation_dataset
    val_dataset_size = len(val_dataset)
    progress_bar = tqdm(
        range(0, val_dataset_size),
        desc="Validation progress",
        leave=False,
    )
    for i in range(0, val_dataset_size):
        b_i = val_dataset[i]
        render_results = render_func(b_i)
        image = torch.clamp(render_results["render"], 0.0, 1.0)
        gt_image = torch.clamp(b_i['image'].to("cuda").float(), 0.0, 1.0)
        # opacity = render_results["opacity"]
        # depth = render_results["depth"]
        # depth_normal = (depth - depth.min()) / (depth.max() - depth.min())

        if logger:
            image_name = os.path.basename(b_i['camera'].rgb_file_name)
            iteration = global_step
            logger.add_images(
                "test" + f"_view_{image_name}/render", image[None], global_step=iteration)
            logger.add_images(
                "test" + f"_view_{image_name}/ground_truth", gt_image[None], global_step=iteration)
            # logger.add_images("test" + f"_view_{image_name}/opacity", opacity[None], global_step=iteration)
            # logger.add_images("test" + f"_view_{image_name}/depth", depth_normal[None], global_step=iteration)

        l1_test += l1_loss(image, gt_image).mean().double()
        psnr_test += psnr(image, gt_image).mean().double()
        ssims_test += ms_ssim(
            image[None], gt_image[None], data_range=1, size_average=True
        )
        lpips_test += lpips_fn(image, gt_image).item()
        progress_bar.update(1)
    progress_bar.close()
    l1_test /= val_dataset_size
    psnr_test /= val_dataset_size
    ssims_test /= val_dataset_size
    lpips_test /= val_dataset_size
    print(f"\n[ITER {iteration}] Evaluating test: L1 {l1_test:.5f} PSNR {psnr_test:.5f} SSIMS {ssims_test:.5f} LPIPS {lpips_test:.5f}")
    if logger:
        iteration = global_step
        logger.add_scalar(
            "test" + '/loss_viewpoint - l1_loss',
            l1_test,
            iteration
        )
        logger.add_scalar(
            "test" + '/loss_viewpoint - psnr',
            psnr_test,
            iteration
        )
        logger.add_scalar(
            "test" + '/loss_viewpoint - ssims',
            ssims_test,
            iteration
        )
        logger.add_scalar(
            "test" + '/loss_viewpoint - lpips',
            lpips_test,
            iteration
        )

# TODO: rewite this function, it is ugly


def render_batch(render_func, batch):
    renders = []
    viewspace_points = []
    visibilitys = []
    radiis = []
    for b_i in batch:
        render_results = render_func(b_i)
        renders.append(render_results["render"])
        viewspace_points.append(render_results["viewspace_points"])
        visibilitys.append(render_results["visibility_filter"].unsqueeze(0))
        radiis.append(render_results["radii"].unsqueeze(0))

    radii = torch.cat(radiis, 0).max(dim=0).values
    visibility = torch.cat(visibilitys).any(dim=0)
    images = torch.stack(renders)

    return images, radii, visibility, viewspace_points


def gaussian_point_init(position, max_sh_degree):
    num_points = len(position)
    avg_dist = torch.clamp_min(
        distCUDA2(position.cuda()), 
        0.0000001
    )[..., None].cpu()
    # position_np = position.detach().cpu().numpy()
    # Build the nearest neighbors model
    # from sklearn.neighbors import NearestNeighbors

    # k = 3
    # nn_model = NearestNeighbors(
    #     n_neighbors=k + 1, 
    #     algorithm="auto", 
    #     metric="euclidean"
    # ).fit(position_np)
    
    # distances, indices = nn_model.kneighbors(position_np)
    # distances = distances[:, 1:].astype(np.float32)
    # distances = torch.from_numpy(distances)
    # avg_dist = distances.mean(dim=-1, keepdim=True)
    
    # avg_dist = torch.clamp_min(avg_dist, 0.0000001)
    scales = torch.log(torch.sqrt(avg_dist)).repeat(1, 3)
    rots = torch.zeros((num_points, 4))
    rots[:, 0] = 1

    init_one = torch.ones(
        (num_points, 1),
        dtype=torch.float32
    )
    opacities = inverse_sigmoid(0.1 * init_one)
    features_rest = torch.zeros(
        (num_points, (max_sh_degree+1) ** 2 - 1, 3),
        dtype=torch.float32
    )

    return scales, rots, opacities, features_rest


def video_process(render_func, datapipeline, render_path, save_npz=False, pcd=None):
    to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)
    dataset = datapipeline.video_dataset
    dataset_size = len(dataset)
    progress_bar = tqdm(
        range(0, dataset_size),
        desc="Video progress",
        leave=False,
    )
    render_images = []
    if save_npz:
        gaussian_collect = defaultdict(list)
    for i in range(0, dataset_size):
        b_i = dataset[i]
        render_results = render_func(b_i)
        image = torch.clamp(render_results["render"], 0.0, 1.0)
        render_images.append(to8b(image).transpose(1,2,0))
        if save_npz:
            gaussian_collect['means3D'].append(
                pcd.get_position_flow.detach().cpu().numpy()
            )
            gaussian_collect['unnorm_rotations'].append(
                pcd.get_rotation_flow.detach().cpu().numpy()
            )
            gaussian_collect['shs'].append(
                pcd.get_shs.detach().cpu().numpy()
            )
            gaussian_collect['logit_opacities'] = (
                pcd.opacity.detach().cpu().numpy()
            )
            gaussian_collect['log_scales'] = (
                pcd.scaling.detach().cpu().numpy()
            )
        
        progress_bar.update(1)
    progress_bar.close()
    
    timestamp = time.time()
    formatted_timestamp = datetime.fromtimestamp(timestamp).strftime('%Y%m%d-%H%M%S')
    
    imageio.mimwrite(
        os.path.join(
            render_path, 
            f"video@{formatted_timestamp}", 
            'video_rgb.mp4'
        ), 
        render_images, fps=25, quality=8
    )
    if save_npz:
        np.savez(        
            os.path.join(
                render_path, 
                f"video@{formatted_timestamp}", 
                'params.npz',
            ),
            **gaussian_collect,
        )
