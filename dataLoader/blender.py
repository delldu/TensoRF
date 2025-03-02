import torch
from torch.utils.data import Dataset
import json
from tqdm import tqdm
import os
from PIL import Image
from torchvision import transforms as T
import pdb

from .ray_utils import *


class BlenderDataset(Dataset):
    def __init__(self, datadir, split="train", downsample=1.0):
        # datadir = './data/nerf_synthetic/lego'
        self.near_far = [2.0, 6.0]
        self.scene_bbox = torch.tensor([[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]])
        self.blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

        self.root_dir = datadir
        self.split = split
        self.img_wh = (int(800 / downsample), int(800 / downsample))
        self.transform = T.ToTensor()

        self.read_meta()
        # self.define_proj_mat()

        self.center = torch.mean(self.scene_bbox, axis=0).float().view(1, 1, 3)
        self.radius = (self.scene_bbox[1] - self.center).float().view(1, 1, 3)
        self.downsample = downsample

        # self.center -- [[[0., 0., 0.]]]
        # self.radius -- [[[1.5000, 1.5000, 1.5000]]]
        # (Pdb) self.scene_bbox
        # [[-1.5000, -1.5000, -1.5000],
        # [ 1.5000,  1.5000,  1.5000]]

    def read_meta(self):
        # /data/nerf_synthetic/lego/transforms_train.json
        with open(os.path.join(self.root_dir, f"transforms_{self.split}.json"), "r") as f:
            self.meta = json.load(f)

        w, h = self.img_wh
        # self.meta['camera_angle_x'] -- 0.6911112070083618
        self.focal = 0.5 * 800 / np.tan(0.5 * self.meta["camera_angle_x"])  # original focal length
        self.focal *= self.img_wh[0] / 800  # modify focal length to match size self.img_wh

        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = get_ray_directions(h, w, [self.focal, self.focal])  # (h, w, 3)
        self.directions = self.directions / torch.norm(self.directions, dim=-1, keepdim=True)
        self.intrinsics = torch.tensor([[self.focal, 0, w / 2], [0, self.focal, h / 2], [0, 0, 1]]).float()
        # (Pdb) self.intrinsics
        # tensor([[ 1111.1111,     0.0000,   400.0000],
        #         [    0.0000,  1111.1111,   400.0000],
        #         [    0.0000,     0.0000,     1.0000]])
        self.image_paths = []
        self.poses = []
        self.all_rays = []
        self.all_rgbs = []
        self.all_masks = []
        self.downsample = 1.0

        idxs = list(range(0, len(self.meta["frames"])))
        if self.split == "test":
            idxs = idxs[:50]  # speed upp test

        for i in tqdm(idxs, desc=f"Loading data {self.split} ({len(idxs)})"):  # img_list:#
            frame = self.meta["frames"][i]
            # (Pdb) self.blender2opencv
            # array([[ 1,  0,  0,  0],
            #        [ 0, -1,  0,  0],
            #        [ 0,  0, -1,  0],
            #        [ 0,  0,  0,  1]])
            pose = np.array(frame["transform_matrix"]) @ self.blender2opencv
            c2w = torch.FloatTensor(pose)
            self.poses += [c2w]  # camert to world

            image_path = os.path.join(self.root_dir, f"{frame['file_path']}.png")
            self.image_paths += [image_path]
            img = Image.open(image_path)

            if self.downsample != 1.0:  # False
                img = img.resize(self.img_wh, Image.LANCZOS)

            img = self.transform(img)  # (4, h, w)
            img = img.view(4, -1).permute(1, 0)  # (h*w, 4) RGBA
            img = img[:, :3] * img[:, -1:] + (1 - img[:, -1:])  # blend A to RGB
            self.all_rgbs += [img]

            rays_o, rays_d = get_rays(self.directions, c2w)  # both (h*w, 3)
            # rays_o.size() -- [640000, 3], rays_o[i] -- [2.1031, -1.0323,  3.2804]
            # rays_d.size() -- [640000, 3]
            self.all_rays += [torch.cat([rays_o, rays_d], 1)]  # (h*w, 6)

        self.poses = torch.stack(self.poses)
        self.all_rays = torch.cat(self.all_rays, 0)  # (len(self.meta['frames])*h*w, 3)
        self.all_rgbs = torch.cat(self.all_rgbs, 0)  # (len(self.meta['frames])*h*w, 3)
        # self.all_rays.size() -- [100x800x800, 6]
        # self.all_rgbs.size() -- [100x800x800, 3], RGB

    def __len__(self):
        return len(self.all_rgbs)

    def __getitem__(self, idx):
        return {"rays": self.all_rays[idx], "rgbs": self.all_rgbs[idx]}
