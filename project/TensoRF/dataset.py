"""Create model."""
# coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright 2023 Dell(18588220928@163.com), All Rights Reserved.
# ***
# ***    File Author: Dell, 2023年 01月 12日 星期四 23:31:37 CST
# ***
# ************************************************************************************/
#
import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

# import numpy as np
from PIL import Image
import json

from tqdm import tqdm
import pdb


def create_meshgrid(height: int, width: int):
    """Generates a coordinate grid for an image.
    Return:
        torch.Tensor: returns a grid tensor with shape :math:`(1, H, W, 2)`.
    """
    xs: torch.Tensor = torch.linspace(0, width - 1, width, dtype=torch.float32)
    ys: torch.Tensor = torch.linspace(0, height - 1, height, dtype=torch.float32)

    # generate grid by stacking coordinates
    base_grid: torch.Tensor = torch.stack(torch.meshgrid([xs, ys])).transpose(1, 2)  # 2xHxW
    return torch.unsqueeze(base_grid, dim=0).permute(0, 2, 3, 1)  # 1xHxWx2


def get_rays(directions, c2w):
    """
    Inputs:
        directions: (H, W, 3) precomputed ray directions in camera coordinate
        c2w: (3, 4) transformation matrix from camera coordinate to world coordinate
    Outputs:
        rays_o: (H * W, 3), the origin of the rays in world coordinate
        rays_d: (H * W, 3), the normalized direction of the rays in world coordinate
    """
    # Rotate ray directions from camera coordinate to the world coordinate
    rays_d = directions @ c2w[:3, :3].T  # (H, W, 3)
    # rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    # The origin of all rays is the camera origin in world coordinate
    rays_o = c2w[:3, 3].expand(rays_d.shape)  # (H, W, 3)

    rays_d = rays_d.view(-1, 3)
    rays_o = rays_o.view(-1, 3)
    # (Pdb) c2w
    # tensor([[    -0.9999,     -0.0042,      0.0133,     -0.0538],
    #         [    -0.0140,      0.2997,     -0.9539,      3.8455],
    #         [    -0.0000,     -0.9540,     -0.2997,      1.2081],
    #         [     0.0000,      0.0000,      0.0000,      1.0000]])
    return rays_o, rays_d


def get_ray_directions(H, W, focal):
    """
    Inputs:
        H, W, focal: image height, width and focal length
    Outputs:
        directions: (H, W, 3), the direction of the rays in camera coordinate
    """
    # H = 800
    # W = 800
    # focal = [1111.1110, 1111.1110]

    grid = create_meshgrid(H, W)[0] + 0.5
    # grid.size() -- [800, 800, 2]
    # (Pdb) grid[0][0] -- [0.5000, 0.5000]
    # (Pdb) grid[799][799] -- [799.5000, 799.5000]

    i, j = grid.unbind(-1)
    # the direction here is without +0.5 pixel centering as calibration is not so accurate
    # see https://github.com/bmild/nerf/issues/24
    center = [W / 2, H / 2]
    directions = torch.stack(
        [(i - center[0]) / focal[0], (j - center[1]) / focal[1], torch.ones_like(i)], -1
    )  # (H, W, 3)
    # (Pdb) pp directions[0,0] -- [-0.3595, -0.3595,  1.0000]
    # (Pdb) pp directions[799,799] -- [0.3595, 0.3595, 1.0000]
    # (Pdb) pp directions[400,400] -- [0.0005, 0.0005, 1.0000]
    return directions / torch.norm(directions, dim=-1, keepdim=True)  # normal


class BlenderDataset(Dataset):
    def __init__(self, datadir, split="train", downsample=1.0, batch_size=2048):
        # datadir = 'data/lego'
        self.near_far = [2.0, 6.0]
        self.scene_bbox = [[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]]
        self.root_dir = datadir
        self.split = split
        self.downsample = downsample
        self.batch_size = batch_size
        self.read_meta()

    def get_c2w_rays_rgbs(self, i):
        blender2opencv = torch.tensor([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], dtype=torch.float32)

        frame = self.meta["frames"][i]
        c2w = torch.tensor(frame["transform_matrix"], dtype=torch.float32) @ blender2opencv
        rays_o, rays_d = get_rays(self.directions, c2w)  # both (H*W, 3)
        # rays_o = rays_o[mask] # Stupid idea
        # rays_d = rays_d[mask] # Stupid idea

        # rays_o.size() -- [640000, 3], rays_o[i] -- [2.1031, -1.0323,  3.2804]
        # rays_d.size() -- [640000, 3]
        rays = torch.cat([rays_o, rays_d], 1)

        image_path = os.path.join(self.root_dir, f"{frame['file_path']}.png")
        self.image_paths += [image_path]
        image = Image.open(image_path)
        image = image.resize((self.W, self.H))

        image = T.ToTensor()(image)  # image.size() -- [4, 800, 800]
        # mask = (image[3, :, :] > 0.0).flatten() # Stupid idea

        image = image.view(4, -1).permute(1, 0)  # image.size() -- [640000, 4]
        image = image[:, :3] * image[:, -1:] + (1 - image[:, -1:])  # blend A to RGB
        # image = image[mask] # Stupid idea
        return c2w, rays, image  # rgbs

    def read_meta(self):
        # lego/transforms_train.json
        with open(os.path.join(self.root_dir, f"transforms_{self.split}.json"), "r") as f:
            self.meta = json.load(f)

        # read first image for W, H
        frame = self.meta["frames"][0]
        image_path = os.path.join(self.root_dir, f"{frame['file_path']}.png")
        image = Image.open(image_path)
        self.H = int(image.height / self.downsample)
        self.W = int(image.width / self.downsample)

        # self.meta['camera_angle_x'] -- 0.6911112070083618
        camera_fx = torch.tensor(self.meta["camera_angle_x"])
        self.focal = 0.5 * self.W / torch.tan(0.5 * camera_fx)  # original focal length
        self.focal *= self.downsample  # modify focal length to match downsample

        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = get_ray_directions(self.H, self.W, [self.focal, self.focal])  # (H, W, 3)
        self.intrinsics = torch.tensor(
            [[self.focal, 0.0, self.W / 2.0], [0.0, self.focal, self.H / 2.0], [0.0, 0.0, 1.0]]
        )
        # (Pdb) self.intrinsics
        # tensor([[ 1111.1111,     0.0000,   400.0000],
        #         [    0.0000,  1111.1111,   400.0000],
        #         [    0.0000,     0.0000,     1.0000]])
        self.image_paths = []
        self.poses = []
        self.all_rays = []
        self.all_rgbs = []

        frame_lists = list(range(0, len(self.meta["frames"])))
        if self.split == "test" or self.split == "val":
            frame_lists = frame_lists[:50]  # speed upp test

        blender2opencv = torch.tensor([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], dtype=torch.float32)

        for i in tqdm(frame_lists, desc=f"Loading {self.split} data"):  # img_list:#
            c2w, rays, rgbs = self.get_c2w_rays_rgbs(i)
            self.poses += [c2w]  # camert to world
            self.all_rgbs += [rgbs]
            self.all_rays += [rays]

        self.poses = torch.stack(self.poses)
        self.all_rays = torch.cat(self.all_rays, 0)  # self.all_rays.size() -- [100x800x800, 6]
        self.all_rgbs = torch.cat(self.all_rgbs, 0)  # self.all_rgbs.size() -- [100x800x800, 3], RGB

        # filter rays
        self.all_rays, self.all_rgbs = self.filter_rays(self.all_rays, self.all_rgbs)
        # shuffle
        shuffle_index = torch.randperm(self.all_rays.size(0))
        self.all_rays = self.all_rays[shuffle_index]
        self.all_rgbs = self.all_rgbs[shuffle_index]

    def filter_rays(self, all_rays, all_rgbs, chunk=10240 * 5):
        N = torch.tensor(all_rays.shape[:-1]).prod()  # 64000000

        mask = []
        index_chunks = torch.split(torch.arange(N), chunk)
        for index in index_chunks:
            rays_chunk = all_rays[index]
            rays_o, rays_d = rays_chunk[..., :3], rays_chunk[..., 3:6]
            vec = torch.where(rays_d == 0, torch.full_like(rays_d, 1e-6), rays_d)
            rate_a = (torch.tensor(self.scene_bbox[1])/self.downsample - rays_o) / vec
            rate_b = (torch.tensor(self.scene_bbox[0])/self.downsample - rays_o) / vec
            t_min = torch.minimum(rate_a, rate_b).amax(-1)
            t_max = torch.maximum(rate_a, rate_b).amin(-1)
            mask_inbbox = t_max > t_min
            mask.append(mask_inbbox)

        mask = torch.cat(mask).view(all_rgbs.shape[:-1])

        print(f"Ray filtering ratio: {torch.sum(mask) / N:.4f}")
        return all_rays[mask], all_rgbs[mask]

    def __len__(self):
        N = len(self.all_rgbs)
        return N // self.batch_size + int(N % self.batch_size > 0)  # len(self.all_rgbs)

    def __getitem__(self, idx):
        # return {"rays": self.all_rays[idx], "rgbs": self.all_rgbs[idx]}
        start = idx * self.batch_size
        end = (idx + 1) * self.batch_size
        return {"rays": self.all_rays[start:end, :], "rgbs": self.all_rgbs[start:end, :]}

    def __repr__(self):
        return f"{self.split} sets: {self.root_dir}, {len(self.image_paths)} Images({self.W}x{self.H}), {len(self.all_rgbs)/1000000:.2f}M Rays"


if __name__ == "__main__":
    ds = TensoRF.BlenderDataset("data/lego", split="test", downsample=1.0)
    print(ds)
    pdb.set_trace()
