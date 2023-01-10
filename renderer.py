import torch, os, imageio

from tqdm.auto import tqdm
from models.tensoRF import raw2alpha, TensorVMSplit, AlphaGridMask
from utils import *
import pdb


def OctreeRender_trilinear_fast(rays, tensorf, chunk=4096, N_samples=-1, white_bg=True, is_train=False, device="cuda"):
    rgbs = []
    depths = []

    N_rays_all = rays.shape[0]
    N = N_rays_all // chunk + int(N_rays_all % chunk > 0)
    for i in range(N):  # tqdm(range(N)): # range(N)
        rays_chunk = rays[i * chunk : (i + 1) * chunk].to(device)
        rgb_map, depth_map = tensorf(rays_chunk, is_train=is_train, white_bg=white_bg, N_samples=N_samples)
        rgbs.append(rgb_map)
        depths.append(depth_map)

    return torch.cat(rgbs), torch.cat(depths)


@torch.no_grad()
def evaluation(
    test_dataset, tensorf, renderer, savePath=None, N_vis=5, prtx="", N_samples=-1, white_bg=False, device="cuda"
):
    PSNRs = []
    ssims, l_alex, l_vgg = [], [], []
    os.makedirs(savePath, exist_ok=True)

    try:
        tqdm._instances.clear()
    except Exception:
        pass

    # near_far = test_dataset.near_far
    # test_dataset.all_rays.size() -- [200, 640000, 6]
    # N_vis --  -1
    img_eval_interval = 1 if N_vis < 0 else max(test_dataset.all_rays.shape[0] // N_vis, 1)
    idxs = list(range(0, test_dataset.all_rays.shape[0], img_eval_interval))
    # idxs -- [0, 1, ..., 199]

    # img_eval_interval -- 1
    # test_dataset.all_rays[0::img_eval_interval].size() -- [200, 640000, 6]

    W, H = test_dataset.img_wh  # (800, 800)
    pbar = tqdm(total=test_dataset.all_rays.size(0), desc=f"Evaluation")
    for idx, samples in enumerate(test_dataset.all_rays[0::img_eval_interval]):
        pbar.update(1)

        rays = samples.view(-1, samples.shape[-1])
        rgb_map, rgb_depth = renderer(rays, tensorf, chunk=4096, N_samples=N_samples, white_bg=white_bg, device=device)
        rgb_map = rgb_map.clamp(0.0, 1.0).reshape(H, W, 3).cpu()

        if len(test_dataset.all_rgbs):
            gt_rgb = test_dataset.all_rgbs[idxs[idx]].view(H, W, 3)
            loss = torch.mean((rgb_map - gt_rgb) ** 2)
            PSNRs.append(-10.0 * np.log(loss.item()) / np.log(10.0))

    if PSNRs:
        psnr = np.mean(np.asarray(PSNRs))
        np.savetxt(f"{savePath}/{prtx}mean.txt", np.asarray([psnr]))
        print("Evaluation: ", np.asarray([psnr]))

    return PSNRs
