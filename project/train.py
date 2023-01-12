# coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2023(18588220928@163.com), All Rights Reserved.
# ***
# ***    File Author: Dell, 2023年 01月 12日 星期四 23:31:37 CST
# ***
# ************************************************************************************/
#
import todos
import TensoRF

import os
import pdb
import torch
import numpy as np

from tqdm.auto import tqdm


torch.set_printoptions(sci_mode=False)


class SimpleSampler:
    def __init__(self, total, batch):
        self.total = total
        self.batch = batch
        self.curr = total
        self.ids = None

    def nextids(self):
        self.curr += self.batch
        if self.curr + self.batch > self.total:
            self.ids = torch.LongTensor(np.random.permutation(self.total))
            self.curr = 0
        return self.ids[self.curr : self.curr + self.batch]


if __name__ == "__main__":

    todos.data.mkdir("output")

    model, device = TensoRF.get_model()
    train_dataset = TensoRF.get_dataset()
    grad_vars = model.get_optparam_groups()
    batch_size = 2048
    epochs = 3000
    lr_decay_ratio = 0.1
    progress_refresh_rate = 10

    lr_factor = lr_decay_ratio ** (1 / epochs)  # lr_decay_ratio -- 0.1
    # lr_factor -- 0.9992327661102197
    print("lr decay", lr_decay_ratio)

    optimizer = torch.optim.Adam(grad_vars, betas=(0.9, 0.99))

    PSNRs = []

    allrays, allrgbs = train_dataset.all_rays, train_dataset.all_rgbs
    trainingSampler = SimpleSampler(allrays.shape[0], batch_size)

    L1_reg_weight = 8e-5
    print("initial L1_reg_weight", L1_reg_weight)

    pbar = tqdm(range(epochs), miniters=progress_refresh_rate)
    for iteration in pbar:
        ray_indexs = trainingSampler.nextids()
        rays_train, rgb_train = allrays[ray_indexs].to(device), allrgbs[ray_indexs].to(device)

        # rgb_map, alphas_map, depth_map
        # rays_train = rays_train.to(device)
        rgb_map, depth_map = model(rays_train, is_train=True)

        loss = torch.mean((rgb_map - rgb_train) ** 2)

        # loss
        total_loss = loss
        loss_reg_L1 = model.dense_L1()
        total_loss += L1_reg_weight * loss_reg_L1

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        loss = loss.detach().item()

        PSNRs.append(-10.0 * np.log(loss) / np.log(10.0))
        # for param_group in optimizer.param_groups:
        #     param_group["lr"] = param_group["lr"] * lr_factor  # xxxx8888

        # Print the current values of the losses.
        if iteration % progress_refresh_rate == 0:
            pbar.set_description(
                f"Iteration {iteration:05d}:" + f" train_psnr = {float(np.mean(PSNRs)):.2f}" + f" mse = {loss:.6f}"
            )
            PSNRs = [] # reset PSNR
            for param_group in optimizer.param_groups:
                param_group["lr"] = param_group["lr"] * lr_factor  # xxxx8888

    torch.save(model.state_dict(), f"output/lego_tensorf.th")
