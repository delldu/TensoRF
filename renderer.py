import torch,os,imageio

from tqdm.auto import tqdm
# from dataLoader.ray_utils import get_rays
# from models.tensoRF import TensorVM, TensorCP, raw2alpha, TensorVMSplit, AlphaGridMask
from models.tensoRF import raw2alpha, TensorVMSplit, AlphaGridMask
from utils import *
# from dataLoader.ray_utils import ndc_rays_blender
import pdb

def OctreeRender_trilinear_fast(rays, tensorf, chunk=4096, N_samples=-1, white_bg=True, is_train=False, device='cuda'):
    rgbs, depth_maps = [], []
    N_rays_all = rays.shape[0]
    for chunk_idx in range(N_rays_all // chunk + int(N_rays_all % chunk > 0)):
        rays_chunk = rays[chunk_idx * chunk:(chunk_idx + 1) * chunk].to(device)
        rgb_map, depth_map = tensorf(rays_chunk, is_train=is_train, white_bg=white_bg, N_samples=N_samples)
        rgbs.append(rgb_map)
        depth_maps.append(depth_map)
    
    return torch.cat(rgbs) # , torch.cat(depth_maps)

@torch.no_grad()
def evaluation(test_dataset,tensorf, args, renderer, savePath=None, N_vis=5, prtx='', N_samples=-1,
               white_bg=False, device='cuda'):
    PSNRs, rgb_maps, depth_maps = [], [], []
    ssims,l_alex,l_vgg=[],[],[]
    os.makedirs(savePath, exist_ok=True)
    os.makedirs(savePath+"/rgbd", exist_ok=True)

    try:
        tqdm._instances.clear()
    except Exception:
        pass

    near_far = test_dataset.near_far
    # test_dataset.all_rays.size() -- [200, 640000, 6]
    # N_vis --  -1
    img_eval_interval = 1 if N_vis < 0 else max(test_dataset.all_rays.shape[0] // N_vis,1)
    idxs = list(range(0, test_dataset.all_rays.shape[0], img_eval_interval))
    # idxs -- [0, 1, ..., 199]

    # img_eval_interval -- 1
    # test_dataset.all_rays[0::img_eval_interval].size() -- [200, 640000, 6]
    # for idx, samples in tqdm(enumerate(test_dataset.all_rays[0::img_eval_interval]), file=sys.stdout):

    W, H = test_dataset.img_wh # (800, 800)
    pbar = tqdm(total=test_dataset.all_rays.size(0), desc=f'Evaluation')
    for idx, samples in enumerate(test_dataset.all_rays[0::img_eval_interval]):
        pbar.update(1)

        rays = samples.view(-1,samples.shape[-1])

        # rgb_map, depth_map = renderer(rays, tensorf, chunk=4096, N_samples=N_samples,
        #                                 white_bg = white_bg, device=device)
        rgb_map = renderer(rays, tensorf, chunk=4096, N_samples=N_samples, 
            white_bg = white_bg, device=device)

        rgb_map = rgb_map.clamp(0.0, 1.0)
        # rgb_map, depth_map = rgb_map.reshape(H, W, 3).cpu(), depth_map.reshape(H, W).cpu()
        rgb_map = rgb_map.reshape(H, W, 3).cpu()

        # depth_map, _ = visualize_depth_numpy(depth_map.numpy(),near_far)
        # test_dataset.all_rgbs.size() -- [200, 800, 800, 3]

        if len(test_dataset.all_rgbs):
            gt_rgb = test_dataset.all_rgbs[idxs[idx]].view(H, W, 3)
            loss = torch.mean((rgb_map - gt_rgb) ** 2)
            PSNRs.append(-10.0 * np.log(loss.item()) / np.log(10.0))

        rgb_map = (rgb_map.numpy() * 255).astype('uint8')

        # xxxx8888
        # rgb_maps.append(rgb_map)
        # depth_maps.append(depth_map)
        # if savePath is not None:
        #     imageio.imwrite(f'{savePath}/{prtx}{idx:03d}.png', rgb_map)
        #     rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
        #     imageio.imwrite(f'{savePath}/rgbd/{prtx}{idx:03d}.png', rgb_map)

    # xxxx8888
    # imageio.mimwrite(f'{savePath}/{prtx}video.mp4', np.stack(rgb_maps), fps=30, quality=10)
    # imageio.mimwrite(f'{savePath}/{prtx}depthvideo.mp4', np.stack(depth_maps), fps=30, quality=10)

    if PSNRs:
        psnr = np.mean(np.asarray(PSNRs))
        np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([psnr]))

    return PSNRs
