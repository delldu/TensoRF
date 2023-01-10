import os
import sys
import pdb

import torch

from tqdm.auto import tqdm

from renderer import *
from utils import *

from opt import config_parser
from dataLoader import dataset_dict
from models.tensoRF import TensorVMSplit
from dataLoader.blender import BlenderDataset

torch.set_printoptions(sci_mode=False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
renderer = OctreeRender_trilinear_fast  # tensorf batch forward


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


def do_train(args):
    # init dataset
    train_dataset = BlenderDataset(args.datadir, split="train", downsample=args.downsample_train)
    test_dataset = BlenderDataset(args.datadir, split="test", downsample=args.downsample_train)

    # init resolution
    logfolder = f"{args.basedir}/{args.expname}"

    # init log file
    os.makedirs(logfolder, exist_ok=True)

    # init parameters
    aabb = train_dataset.scene_bbox.to(device)
    grid_size = N_to_reso(args.n_voxel_init, aabb)  # [128, 128, 128]
    near_far = train_dataset.near_far

    tensorf = TensorVMSplit(
        aabb,
        grid_size,
        device,
        near_far=near_far,
    )

    # (Pdb) tensorf
    # TensorVMSplit(
    #   (dense_plane): ParameterList(
    #       (0): Parameter containing: [torch.cuda.FloatTensor of size 1x16x128x128 (GPU 0)]
    #       (1): Parameter containing: [torch.cuda.FloatTensor of size 1x16x128x128 (GPU 0)]
    #       (2): Parameter containing: [torch.cuda.FloatTensor of size 1x16x128x128 (GPU 0)]
    #   )
    #   (dense_line): ParameterList(
    #       (0): Parameter containing: [torch.cuda.FloatTensor of size 1x16x128x1 (GPU 0)]
    #       (1): Parameter containing: [torch.cuda.FloatTensor of size 1x16x128x1 (GPU 0)]
    #       (2): Parameter containing: [torch.cuda.FloatTensor of size 1x16x128x1 (GPU 0)]
    #   )
    #   (color_plane): ParameterList(
    #       (0): Parameter containing: [torch.cuda.FloatTensor of size 1x48x128x128 (GPU 0)]
    #       (1): Parameter containing: [torch.cuda.FloatTensor of size 1x48x128x128 (GPU 0)]
    #       (2): Parameter containing: [torch.cuda.FloatTensor of size 1x48x128x128 (GPU 0)]
    #   )
    #   (color_line): ParameterList(
    #       (0): Parameter containing: [torch.cuda.FloatTensor of size 1x48x128x1 (GPU 0)]
    #       (1): Parameter containing: [torch.cuda.FloatTensor of size 1x48x128x1 (GPU 0)]
    #       (2): Parameter containing: [torch.cuda.FloatTensor of size 1x48x128x1 (GPU 0)]
    #   )
    #   (basis_mat): Linear(in_features=144, out_features=27, bias=False)
    #   (renderModule): MLPRender_Fea(
    #     (mlp): Sequential(
    #       (0): Linear(in_features=150, out_features=128, bias=True)
    #       (1): ReLU(inplace=True)
    #       (2): Linear(in_features=128, out_features=128, bias=True)
    #       (3): ReLU(inplace=True)
    #       (4): Linear(in_features=128, out_features=3, bias=True)
    #     )
    #   )
    # )
    grad_vars = tensorf.get_optparam_groups(args.lr_init, args.lr_basis)
    # (Pdb) for g in grad_vars: print(g)
    # {'params': ParameterList(
    #     (0): Parameter containing: [torch.cuda.FloatTensor of size 1x16x128x1 (GPU 0)]
    #     (1): Parameter containing: [torch.cuda.FloatTensor of size 1x16x128x1 (GPU 0)]
    #     (2): Parameter containing: [torch.cuda.FloatTensor of size 1x16x128x1 (GPU 0)]
    # ), 'lr': 0.02}
    # {'params': ParameterList(
    #     (0): Parameter containing: [torch.cuda.FloatTensor of size 1x16x128x128 (GPU 0)]
    #     (1): Parameter containing: [torch.cuda.FloatTensor of size 1x16x128x128 (GPU 0)]
    #     (2): Parameter containing: [torch.cuda.FloatTensor of size 1x16x128x128 (GPU 0)]
    # ), 'lr': 0.02}
    # {'params': ParameterList(
    #     (0): Parameter containing: [torch.cuda.FloatTensor of size 1x16x128x1 (GPU 0)]
    #     (1): Parameter containing: [torch.cuda.FloatTensor of size 1x16x128x1 (GPU 0)]
    #     (2): Parameter containing: [torch.cuda.FloatTensor of size 1x16x128x1 (GPU 0)]
    # ), 'lr': 0.02}
    # {'params': ParameterList(
    #     (0): Parameter containing: [torch.cuda.FloatTensor of size 1x16x128x128 (GPU 0)]
    #     (1): Parameter containing: [torch.cuda.FloatTensor of size 1x16x128x128 (GPU 0)]
    #     (2): Parameter containing: [torch.cuda.FloatTensor of size 1x16x128x128 (GPU 0)]
    # ), 'lr': 0.02}
    # {'params': <generator object Module.parameters at 0x7f075c487040>, 'lr': 0.001}
    # {'params': <generator object Module.parameters at 0x7f075c4870b0>, 'lr': 0.001}
    lr_factor = args.lr_decay_ratio ** (1 / args.n_iters)  # lr_decay_ratio -- 0.1
    # lr_factor -- 0.9992327661102197
    print("lr decay", args.lr_decay_ratio)

    optimizer = torch.optim.Adam(grad_vars, betas=(0.9, 0.99))

    torch.cuda.empty_cache()
    PSNRs = []

    allrays, allrgbs = train_dataset.all_rays, train_dataset.all_rgbs
    allrays, allrgbs = tensorf.filtering_rays(allrays, allrgbs)
    trainingSampler = SimpleSampler(allrays.shape[0], args.batch_size)

    L1_reg_weight = args.L1_weight_inital
    print("initial L1_reg_weight", L1_reg_weight)

    pbar = tqdm(range(args.n_iters), miniters=args.progress_refresh_rate, file=sys.stdout)
    for iteration in pbar:
        ray_idx = trainingSampler.nextids()
        rays_train, rgb_train = allrays[ray_idx], allrgbs[ray_idx].to(device)

        # rgb_map, alphas_map, depth_map
        rgb_map, depth_map = renderer(
            rays_train,
            tensorf,
            chunk=args.batch_size,
            device=device,
            is_train=True,
        )

        loss = torch.mean((rgb_map - rgb_train) ** 2)

        # loss
        total_loss = loss
        loss_reg_L1 = tensorf.dense_L1()
        total_loss += L1_reg_weight * loss_reg_L1

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        loss = loss.detach().item()

        PSNRs.append(-10.0 * np.log(loss) / np.log(10.0))
        for param_group in optimizer.param_groups:
            param_group["lr"] = param_group["lr"] * lr_factor

        # Print the current values of the losses.
        if iteration % args.progress_refresh_rate == 0:
            pbar.set_description(
                f"Iteration {iteration:05d}:" + f" train_psnr = {float(np.mean(PSNRs)):.2f}" + f" mse = {loss:.6f}"
            )
            PSNRs = [] # reset PSNR

    torch.save(tensorf.state_dict(), f"{logfolder}/{args.expname}.th")


if __name__ == "__main__":

    torch.set_default_dtype(torch.float32)
    torch.manual_seed(20211202)
    np.random.seed(20211202)

    args = config_parser()
    print(args)
    # (Pdb) pp args
    # Namespace(config='configs/lego.txt',
    #     expname='tensorf_lego_VM', basedir='./log',
    #     datadir='./data/nerf_synthetic/lego', progress_refresh_rate=10,
    #     downsample_train=1.0, downsample_test=1.0,
    #     model_name='TensorVMSplit', batch_size=4096, n_iters=3000,
    #     dataset_name='blender', lr_init=0.02, lr_basis=0.001,
    #     lr_decay_ratio=0.1,
    #     lr_upsample_reset=1, L1_weight_inital=8e-05,
    #     L1_weight_rest=4e-05,
    #     ckpt=None, render_only=0, render_test=1,
    #     render_train=0, render_path=0, export_mesh=0,
    #     accumulate_decay=0.998,
    #     step_ratio=0.5,
    #     idx_view=0, vis_every=10000)

    do_train(args)
