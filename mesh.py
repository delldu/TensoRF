import os

# from tqdm.auto import tqdm
from opt import config_parser
from renderer import *

from utils import *
from models.tensoRF import TensorVMSplit
import pdb
import torch
from dataLoader.blender import BlenderDataset
from dataLoader.ray_utils import *


torch.set_printoptions(sci_mode=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
renderer = OctreeRender_trilinear_fast  # tensorf batch forward


def save_points(xyzs, rgbs, filename):
    """Save to point cloud file"""
    print(f"Saving {xyzs.size(0)} points to {filename} ...")

    float_formatter = lambda x: "%.4f" % x
    file = open(filename, "w")
    file.write(
        """ply
    format ascii 1.0
    element vertex %d
    property float x
    property float y
    property float z
    property uchar red
    property uchar green
    property uchar blue
    property uchar alpha
    end_header
    """
        % (xyzs.size(0))
    )
    for i in torch.cat((xyzs, rgbs), dim=1):
        file.write(
            "{} {} {} {} {} {} 0\n".format(
                float_formatter(i[0]),
                float_formatter(i[1]),
                float_formatter(i[2]),
                int(i[3] * 255.0),
                int(i[4] * 255.0),
                int(i[5] * 255.0),
            )
        )
    file.close()


def save_input_points(args, tensorf):
    # def save_points(args, filename):
    filename = f"{args.ckpt[:-3]}_input.ply"
    train_dataset = BlenderDataset(args.datadir, split="train", downsample=args.downsample_train)
    allrays, allrgbs = train_dataset.all_rays, train_dataset.all_rgbs
    allrays, allrgbs = tensorf.filtering_rays(allrays, allrgbs)
    size = allrays.size(0)

    select_index = torch.randperm(size)
    size = size // 100
    select_index = select_index[0:size]
    allrays = allrays[select_index]
    allrgbs = allrgbs[select_index]

    rays_o, rays_d = allrays[:, 0:3], allrays[:, 3:6]

    with torch.no_grad():
        rgb_map, depth_map = renderer(
            allrays,
            tensorf,
            chunk=args.batch_size,
            device=device,
            is_train=False,
        )
    rgb_map = rgb_map.cpu()
    depth_map = depth_map.cpu()

    aabb = tensorf.aabb.cpu()
    pc_xyz = rays_o + rays_d * depth_map.view(-1, 1)
    mask = torch.all(torch.cat([pc_xyz > aabb[0], pc_xyz < aabb[1]], dim=-1), dim=-1)
    pc_xyz = pc_xyz[mask]
    rgb_map = rgb_map[mask]

    save_points(pc_xyz, rgb_map, filename)


@torch.no_grad()
def export_mesh(args):
    ckpt = torch.load(args.ckpt, map_location=device)
    aabb = torch.tensor([[-1.5000, -1.5000, -1.5000],
            [1.5000,  1.5000,  1.5000]], device=device)
    grid_size = [128, 128, 128]
    tensorf = TensorVMSplit(
        aabb,
        grid_size,
        device,
    )

    tensorf.load_state_dict(ckpt)

    alpha, dense_xyz = tensorf.get_dense_alpha()
    # alpha.size() -- [115, 205, 133]
    convert_sdf_samples_to_ply(alpha.cpu(), f"{args.ckpt[:-3]}.ply", bbox=tensorf.aabb.cpu(), level=0.005)

    save_input_points(args, tensorf)


if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(20211202)

    args = config_parser()
    print(args)
    export_mesh(args)
