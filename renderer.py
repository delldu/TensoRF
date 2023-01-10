import torch

from tqdm.auto import tqdm
from utils import *
import pdb


def OctreeRender_trilinear_fast(rays, tensorf, chunk=4096, is_train=False, device="cuda"):
    rgbs = []
    depths = []

    N_rays_all = rays.shape[0]
    N = N_rays_all // chunk + int(N_rays_all % chunk > 0)
    for i in range(N):  # tqdm(range(N)): # range(N)
        rays_chunk = rays[i * chunk : (i + 1) * chunk].to(device)
        rgb_map, depth_map = tensorf(rays_chunk, is_train=is_train)
        rgbs.append(rgb_map)
        depths.append(depth_map)

    return torch.cat(rgbs), torch.cat(depths)
