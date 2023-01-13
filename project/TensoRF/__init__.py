"""TensoRF Package."""  # coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2023(18588220928@163.com) All Rights Reserved.
# ***
# ***    File Author: Dell, 2023年 01月 12日 星期四 23:31:37 CST
# ***
# ************************************************************************************/
#

__version__ = "1.0.0"

import os
import math
import torch
import todos
from . import tensorf, dataset
from tqdm import tqdm
import pdb

DATA_ROOT_DIR = "data/lego"


def train_data(bs=2048):
    """Get data loader for training, bs means batch_size."""
    ds = dataset.BlenderDataset(DATA_ROOT_DIR, split="train", downsample=1.0, batch_size=bs)
    return torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=4)


def test_data(bs=2048):
    """Get data loader for test, bs means batch_size."""
    ds = dataset.BlenderDataset(DATA_ROOT_DIR, split="test", downsample=1.0, batch_size=bs)
    return torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=4)


def valid_data(bs=2048):
    """Get data loader for test, bs means batch_size."""
    ds = dataset.BlenderDataset(DATA_ROOT_DIR, split="val", downsample=1.0, batch_size=bs)
    return torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=4)


def get_model(model_path, aabb=None):
    """Create model."""
    if aabb is not None:
        model = tensorf.TensorVMSplit(aabb=aabb)
    else:
        model = tensorf.TensorVMSplit()

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))

    device = todos.model.get_device()
    model = model.to(device)
    # model.eval()

    # print(f"Running model on {device} ...")
    # model = torch.jit.script(model)
    # todos.data.mkdir("output")
    # if not os.path.exists("output/lego_tensorf.torch"):
    #     model.save("output/lego_tensorf.torch")

    return model, device


class Counter(object):
    """Class Counter."""

    def __init__(self):
        """Init average."""

        self.reset()

    def reset(self):
        """Reset average."""

        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """Update average."""

        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_epoch(loader, model, optimizer, device, tag="train"):
    """Trainning model ..."""

    model.train()
    psnr_loss = Counter()

    #
    # /************************************************************************************
    # ***
    # ***    MS: Define Loss Function
    # ***
    # ************************************************************************************/
    #
    loss_function = torch.nn.MSELoss(reduction="mean")

    with tqdm(loader) as t:
        t.set_description(tag)

        count = 0
        for data in t:
            rays, rgbs = data["rays"], data["rgbs"]
            count = count + 1

            # Transform data to device
            rays = rays.squeeze(0).to(device)
            rgbs = rgbs.squeeze(0).to(device)

            output_rgbs, _ = model(rays, is_train=True)

            # Loss
            loss = loss_function(output_rgbs, rgbs)  # torch.mean((output_rgbs - rgbs) ** 2) #
            psnr_loss.update(loss.detach().item(), 1)

            loss += 8e-6 * model.dense_L1()  # optimize planes

            # Optimizer
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if count % 10 == 0:
                avg_loss = psnr_loss.avg
                t.set_postfix(
                    loss="{:.4f}".format(avg_loss), PSNR="{:.2f}".format(-10.0 * math.log(avg_loss) / math.log(10.0))
                )
                t.update()

                psnr_loss.reset()
                for param_group in optimizer.param_groups:
                    param_group["lr"] = param_group["lr"] * 0.9992327661102197  # xxxx8888

        return psnr_loss.avg


def valid_epoch(loader, model, device, tag="valid"):
    """Validating model  ..."""

    psnr_loss = Counter()

    model.eval()

    #
    # /************************************************************************************
    # ***
    # ***    MS: Define Loss Function
    # ***
    # ************************************************************************************/
    #
    loss_function = torch.nn.MSELoss(reduction="mean")

    with tqdm(loader) as t:
        t.set_description(tag)

        for data in t:
            rays, rgbs = data["rays"], data["rgbs"]

            # Transform data to device
            rays = rays.squeeze(0).to(device)
            rgbs = rgbs.squeeze(0).to(device)

            with torch.no_grad():
                output_rgbs, _ = model(rays, is_train=False)

            # Loss
            loss = loss_function(output_rgbs, rgbs)
            psnr_loss.update(loss.item(), 1)

            t.set_postfix(
                loss="{:.4f}".format(psnr_loss.avg),
                PSNR="{:.2f}".format(-10.0 * math.log(psnr_loss.avg) / math.log(10.0)),
            )
            t.update(1)


def point_predict(model, rays, device, batch_size=2048):
    rgbs = []
    depths = []

    N_rays_all = rays.shape[0]
    N = N_rays_all // batch_size + int(N_rays_all % batch_size > 0)
    for i in tqdm(range(N)):
        rays_chunk = rays[i * batch_size : (i + 1) * batch_size].to(device)
        with torch.no_grad():
            rgb_map, depth_map = model(rays_chunk, is_train=False)
        rgbs.append(rgb_map.cpu())
        depths.append(depth_map.cpu())
    rgb_map, depth_map = torch.cat(rgbs), torch.cat(depths)

    # Filter
    aabb = model.aabb.cpu()
    rays_o, rays_d = rays[:, 0:3], rays[:, 3:6]
    xyz_map = rays_o + rays_d * depth_map.view(-1, 1)
    mask = torch.all(torch.cat([xyz_map > aabb[0], xyz_map < aabb[1]], dim=-1), dim=-1)
    xyz_map = xyz_map[mask]
    rgb_map = rgb_map[mask]

    return xyz_map, rgb_map
