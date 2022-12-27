
import os
from tqdm.auto import tqdm
from opt import config_parser

# import json, random
from renderer import *
from utils import *
# from torch.utils.tensorboard import SummaryWriter
import datetime

from dataLoader import dataset_dict
import sys
import pdb

import torch
torch.set_printoptions(sci_mode=False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
renderer = OctreeRender_trilinear_fast # tensorf batch forward

class SimpleSampler:
    def __init__(self, total, batch):
        self.total = total
        self.batch = batch
        self.curr = total
        self.ids = None

    def nextids(self):
        self.curr+=self.batch
        if self.curr + self.batch > self.total:
            self.ids = torch.LongTensor(np.random.permutation(self.total))
            self.curr = 0
        return self.ids[self.curr:self.curr+self.batch]


@torch.no_grad()
def export_mesh(args):

    ckpt = torch.load(args.ckpt, map_location=device)
    kwargs = ckpt['kwargs']
    kwargs.update({'device': device})
    tensorf = eval(args.model_name)(**kwargs)
    tensorf.load(ckpt)

    alpha,_ = tensorf.getDenseAlpha()

    # alpha.size() -- [115, 205, 133]
    convert_sdf_samples_to_ply(alpha.cpu(), f'{args.ckpt[:-3]}.ply',bbox=tensorf.aabb.cpu(), level=0.005)


@torch.no_grad()
def render_test(args):
    return

    # init dataset
    dataset = dataset_dict[args.dataset_name]
    test_dataset = dataset(args.datadir, split='test', downsample=args.downsample_train)
    white_bg = test_dataset.white_bg
    # args.downsample_train -- 1.0
    if not os.path.exists(args.ckpt):
        print('the ckpt path does not exists!!')
        return

    ckpt = torch.load(args.ckpt, map_location=device)
    kwargs = ckpt['kwargs']
    kwargs.update({'device': device})
    tensorf = eval(args.model_name)(**kwargs) # args.model_name -- 'TensorVMSplit'
    tensorf.load(ckpt)

    logfolder = os.path.dirname(args.ckpt)

    if args.render_train:
        os.makedirs(f'{logfolder}/imgs_train_all', exist_ok=True)
        train_dataset = dataset(args.datadir, split='train', downsample=args.downsample_train, is_stack=True)
        PSNRs_test = evaluation(train_dataset,tensorf, args, renderer, f'{logfolder}/imgs_train_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, device=device)
        print(f'======> {args.expname} train all psnr: {np.mean(PSNRs_test)} <========================')

    if args.render_test:
        os.makedirs(f'{logfolder}/{args.expname}/imgs_test_all', exist_ok=True)
        evaluation(test_dataset,tensorf, args, renderer, f'{logfolder}/{args.expname}/imgs_test_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, device=device)

    # xxxx8888
    # if args.render_path:
    #     c2ws = test_dataset.render_path
    #     os.makedirs(f'{logfolder}/{args.expname}/imgs_path_all', exist_ok=True)
    #     evaluation_path(test_dataset,tensorf, c2ws, renderer, f'{logfolder}/{args.expname}/imgs_path_all/',
    #                             N_vis=-1, N_samples=-1, white_bg = white_bg, device=device)

def reconstruction(args):

    # init dataset
    dataset = dataset_dict[args.dataset_name]
    train_dataset = dataset(args.datadir, split='train', downsample=args.downsample_train, is_stack=False)
    test_dataset = dataset(args.datadir, split='test', downsample=args.downsample_train, is_stack=True)
    white_bg = train_dataset.white_bg
    near_far = train_dataset.near_far

    # init resolution
    n_lamb_sigma = args.n_lamb_sigma
    n_lamb_sh = args.n_lamb_sh
    logfolder = f'{args.basedir}/{args.expname}'
    
    # init log file
    os.makedirs(logfolder, exist_ok=True)
    # os.makedirs(f'{logfolder}/imgs_vis', exist_ok=True)
    # os.makedirs(f'{logfolder}/imgs_rgba', exist_ok=True)
    # os.makedirs(f'{logfolder}/rgba', exist_ok=True)

    # init parameters
    aabb = train_dataset.scene_bbox.to(device)
    reso_cur = N_to_reso(args.N_voxel_init, aabb) # [128, 128, 128]
    nSamples = min(args.nSamples, cal_n_samples(reso_cur, args.step_ratio)) # 100x100x100 --> 443

    if args.ckpt is not None:
        ckpt = torch.load(args.ckpt, map_location=device)
        kwargs = ckpt['kwargs']
        kwargs.update({'device':device})
        tensorf = eval(args.model_name)(**kwargs)
        tensorf.load(ckpt)
    else:
        tensorf = eval(args.model_name)(aabb, reso_cur, device,
                    density_n_comp=n_lamb_sigma, appearance_n_comp=n_lamb_sh, app_dim=args.data_dim_color, near_far=near_far,
                    alphaMask_thres=args.alpha_mask_thre, distance_scale=args.distance_scale,
                    view_pe=args.view_pe, feat_pe=args.feat_pe, featureC=args.featureC, step_ratio=args.step_ratio, feat2denseAct=args.feat2denseAct)

    # (Pdb) tensorf
    # TensorVMSplit(
    #   (density_plane): ParameterList(
    #       (0): Parameter containing: [torch.cuda.FloatTensor of size 1x16x128x128 (GPU 0)]
    #       (1): Parameter containing: [torch.cuda.FloatTensor of size 1x16x128x128 (GPU 0)]
    #       (2): Parameter containing: [torch.cuda.FloatTensor of size 1x16x128x128 (GPU 0)]
    #   )
    #   (density_line): ParameterList(
    #       (0): Parameter containing: [torch.cuda.FloatTensor of size 1x16x128x1 (GPU 0)]
    #       (1): Parameter containing: [torch.cuda.FloatTensor of size 1x16x128x1 (GPU 0)]
    #       (2): Parameter containing: [torch.cuda.FloatTensor of size 1x16x128x1 (GPU 0)]
    #   )
    #   (app_plane): ParameterList(
    #       (0): Parameter containing: [torch.cuda.FloatTensor of size 1x48x128x128 (GPU 0)]
    #       (1): Parameter containing: [torch.cuda.FloatTensor of size 1x48x128x128 (GPU 0)]
    #       (2): Parameter containing: [torch.cuda.FloatTensor of size 1x48x128x128 (GPU 0)]
    #   )
    #   (app_line): ParameterList(
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
    lr_factor = args.lr_decay_ratio**(1/args.n_iters) # lr_decay_ratio -- 0.1
    # lr_factor -- 0.9992327661102197
    print("lr decay", args.lr_decay_ratio)
    
    optimizer = torch.optim.Adam(grad_vars, betas=(0.9,0.99))

    #linear in logrithmic space
    # args.upsample_list=[2000, 3000, 4000, 5500, 7000]
    N_voxel_list = (torch.round(torch.exp(torch.linspace(np.log(args.N_voxel_init), \
        np.log(args.N_voxel_final), len(args.upsample_list)+1))).long()).tolist()[1:]


    torch.cuda.empty_cache()
    PSNRs,PSNRs_test = [],[0]

    allrays, allrgbs = train_dataset.all_rays, train_dataset.all_rgbs
    allrays, allrgbs = tensorf.filtering_rays(allrays, allrgbs, bbox_only=True)
    trainingSampler = SimpleSampler(allrays.shape[0], args.batch_size)
    
    L1_reg_weight = args.L1_weight_inital
    print("initial L1_reg_weight", L1_reg_weight)

    pbar = tqdm(range(args.n_iters), miniters=args.progress_refresh_rate, file=sys.stdout)
    for iteration in pbar:
        ray_idx = trainingSampler.nextids()
        rays_train, rgb_train = allrays[ray_idx], allrgbs[ray_idx].to(device)

        #rgb_map, alphas_map, depth_map, weights, uncertainty
        rgb_map, depth_map = renderer(rays_train, tensorf, chunk=args.batch_size,
                                N_samples=nSamples, white_bg = white_bg, device=device, is_train=True)

        loss = torch.mean((rgb_map - rgb_train) ** 2)

        # loss
        total_loss = loss
        loss_reg_L1 = tensorf.density_L1()
        total_loss += L1_reg_weight*loss_reg_L1

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        loss = loss.detach().item()
        
        PSNRs.append(-10.0 * np.log(loss) / np.log(10.0))

        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * lr_factor

        # Print the current values of the losses.
        if iteration % args.progress_refresh_rate == 0:
            pbar.set_description(
                f'Iteration {iteration:05d}:'
                + f' train_psnr = {float(np.mean(PSNRs)):.2f}'
                + f' test_psnr = {float(np.mean(PSNRs_test)):.2f}'
                + f' mse = {loss:.6f}'
            )
            PSNRs = []

        if iteration % args.vis_every == args.vis_every - 1 and args.N_vis!=0:
            PSNRs_test = evaluation(test_dataset,tensorf, args, renderer, f'{logfolder}/imgs_vis/', N_vis=args.N_vis,
                            prtx=f'{iteration:06d}_', N_samples=nSamples, white_bg = white_bg)

        if iteration in args.update_alphamask_list:
            if reso_cur[0] * reso_cur[1] * reso_cur[2]<256**3:# update volume resolution
                reso_mask = reso_cur
            new_aabb = tensorf.updateAlphaMask(tuple(reso_mask))

            if iteration == args.update_alphamask_list[0]:
                tensorf.shrink(new_aabb)
                L1_reg_weight = args.L1_weight_rest
                print("continuing L1_reg_weight", L1_reg_weight)

            if iteration == args.update_alphamask_list[1]:
                # filter rays outside the bbox
                allrays, allrgbs = tensorf.filtering_rays(allrays, allrgbs)
                trainingSampler = SimpleSampler(allrgbs.shape[0], args.batch_size)

        if iteration in args.upsample_list:
            n_voxels = N_voxel_list.pop(0)
            reso_cur = N_to_reso(n_voxels, tensorf.aabb)
            nSamples = min(args.nSamples, cal_n_samples(reso_cur,args.step_ratio))
            tensorf.upsample_volume_grid(reso_cur)

            if args.lr_upsample_reset: # True
                print("reset lr to initial")
                lr_scale = 1.0 #0.1 ** (iteration / args.n_iters)
            else:
                lr_scale = args.lr_decay_ratio ** (iteration / args.n_iters)
            grad_vars = tensorf.get_optparam_groups(args.lr_init*lr_scale, args.lr_basis*lr_scale)
            optimizer = torch.optim.Adam(grad_vars, betas=(0.9, 0.99))
        

    tensorf.save(f'{logfolder}/{args.expname}.th')


    if args.render_train:
        os.makedirs(f'{logfolder}/imgs_train_all', exist_ok=True)
        train_dataset = dataset(args.datadir, split='train', downsample=args.downsample_train)
        PSNRs_test = evaluation(train_dataset,tensorf, args, renderer, f'{logfolder}/imgs_train_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, device=device)
        print(f'======> {args.expname} test all psnr: {np.mean(PSNRs_test)} <========================')

    if args.render_test:
        os.makedirs(f'{logfolder}/imgs_test_all', exist_ok=True)
        PSNRs_test = evaluation(test_dataset,tensorf, args, renderer, f'{logfolder}/imgs_test_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, device=device)
        print(f'======> {args.expname} test all psnr: {np.mean(PSNRs_test)} <========================')


if __name__ == '__main__':

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
    #     n_lamb_sigma=[16, 16, 16], n_lamb_sh=[48, 48, 48], 
    #     data_dim_color=27, 
    #     alpha_mask_thre=0.0001, distance_scale=25, 
    #     view_pe=2, feat_pe=2, featureC=128, 
    #     ckpt=None, render_only=0, render_test=1, 
    #     render_train=0, render_path=0, export_mesh=0, 
    #     lindisp=False, perturb=1.0, accumulate_decay=0.998, 
    #     feat2denseAct='softplus', nSamples=1000000.0, 
    #     step_ratio=0.5, white_bkgd=False, N_voxel_init=2097156, 
    #     N_voxel_final=16777216, upsample_list=[2000, 3000, 4000, 5500, 7000], 
    #     update_alphamask_list=[2000, 4000], idx_view=0, N_vis=5, vis_every=10000)

    if  args.export_mesh:
        export_mesh(args)

    if args.render_only and (args.render_test or args.render_path):
        render_test(args)
    else:
        reconstruction(args)

