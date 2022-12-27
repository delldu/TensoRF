import torch
import torch.nn
import torch.nn.functional as F
from .sh import eval_sh_bases
import numpy as np
import time
import pdb

def positional_encoding(positions, freqs):
    # positions.size() -- [9, 27]
    # freqs -- 2

    freq_bands = (2**torch.arange(freqs).float()).to(positions.device) # freq_bands.size() -- [2]
    # positions[..., None].size() -- [9, 27, 1]
    # positions.shape[:-1] + (freqs * positions.shape[-1], ) -- torch.Size([9, 54])
    pts = (positions[..., None] * freq_bands).reshape(
        positions.shape[:-1] + (freqs * positions.shape[-1], ))
    pts = torch.cat([torch.sin(pts), torch.cos(pts)], dim=-1)

    return pts # pts.size() -- [9, 108]

def raw2alpha(sigma, dist):
    # sigma, dist  [N_rays, N_samples]
    # sigma.size() -- [4096, 443]
    # dist.size() -- [4096, 443]
    alpha = 1. - torch.exp(-sigma*dist) # alpha.size() -- [4096, 443]
    ones = torch.ones(alpha.shape[0], 1).to(alpha.device) # ones.size() -- [4096, 1]
    T = torch.cumprod(torch.cat([ones, 1. - alpha + 1e-10], -1), -1) # T.size() -- [4096, 444]
    weights = alpha * T[:, :-1]  # T[:, :-1].size() -- [4096, 443]

    return alpha, weights # T[:,-1:] -- bg_weight


class AlphaGridMask(torch.nn.Module):
    def __init__(self, device, aabb, alpha_volume):
        super(AlphaGridMask, self).__init__()
        # (Pdb) aabb
        # tensor([[-1.5000, -1.5000, -1.5000],
        #         [ 1.5000,  1.5000,  1.5000]], device='cuda:0')
        # alpha_volume.size() -- [128, 128, 128]
        self.device = device
        self.aabb=aabb.to(self.device)
        self.aabbSize = self.aabb[1] - self.aabb[0]
        self.invgridSize = 1.0/self.aabbSize * 2
        self.alpha_volume = alpha_volume.view(1,1,*alpha_volume.shape[-3:])
        self.gridSize = torch.LongTensor([alpha_volume.shape[-1],alpha_volume.shape[-2],alpha_volume.shape[-3]]).to(self.device)
        # self.gridSize -- tensor([128, 128, 128], device='cuda:0')

    def sample_alpha(self, xyz_sampled):
        # xyz_sampled.size() -- [482223, 3]
        xyz_sampled = self.normalize_coord(xyz_sampled)
        alpha_vals = F.grid_sample(self.alpha_volume, xyz_sampled.view(1,-1,1,1,3), align_corners=True).view(-1)
        # alpha_vals.size() -- [482223]
        return alpha_vals

    def normalize_coord(self, xyz_sampled):
        # self.invgridSize -- tensor([0.6667, 0.6667, 0.6667], device='cuda:0')
        return (xyz_sampled - self.aabb[0]) * self.invgridSize - 1


class MLPRender_Fea(torch.nn.Module):
    def __init__(self,inChanel, view_pe=6, feat_pe=6, featureC=128):
        super(MLPRender_Fea, self).__init__()
        # self = MLPRender_Fea(
        #   (mlp): Sequential(
        #     (0): Linear(in_features=150, out_features=128, bias=True)
        #     (1): ReLU(inplace=True)
        #     (2): Linear(in_features=128, out_features=128, bias=True)
        #     (3): ReLU(inplace=True)
        #     (4): Linear(in_features=128, out_features=3, bias=True)
        #   )
        # )
        # inChanel = 27
        # view_pe = 2
        # feat_pe = 2
        # featureC = 128

        self.in_mlpC = (2*view_pe*3 + 2*feat_pe*inChanel) + 3 + inChanel # 150
        self.view_pe = view_pe
        self.feat_pe = feat_pe
        layer1 = torch.nn.Linear(self.in_mlpC, featureC)
        layer2 = torch.nn.Linear(featureC, featureC)
        layer3 = torch.nn.Linear(featureC,3)

        self.mlp = torch.nn.Sequential(
            layer1, torch.nn.ReLU(inplace=True),
            layer2, torch.nn.ReLU(inplace=True),
            layer3)

        torch.nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(self, pts, viewdirs, features): # pts -- useless ??????
        indata = [features, viewdirs]
        if self.feat_pe > 0: # True
            indata += [positional_encoding(features, self.feat_pe)]
        if self.view_pe > 0: # True
            indata += [positional_encoding(viewdirs, self.view_pe)]
        mlp_in = torch.cat(indata, dim=-1)
        rgb = self.mlp(mlp_in)
        rgb = torch.sigmoid(rgb)

        return rgb

class TensorBase(torch.nn.Module):
    def __init__(self, aabb, gridSize, device, 
                    density_n_comp = [16, 16, 16], 
                    appearance_n_comp = [16, 16, 16], app_dim = 27,
                    alphaMask = None, near_far=[2.0,6.0],
                    alphaMask_thres=0.0001, distance_scale=25, rayMarch_weight_thres=0.0001,
                    view_pe = 2, feat_pe = 2, featureC=128, step_ratio=0.5,
                    feat2denseAct = 'softplus'):
        super(TensorBase, self).__init__()
        # self = TensorVMSplit()
        # aabb = tensor([[-1.5000, -1.5000, -1.5000],
        #         [ 1.5000,  1.5000,  1.5000]], device='cuda:0')
        # gridSize = [128, 128, 128]
        # device = device(type='cuda')

        self.density_n_comp = density_n_comp
        self.app_n_comp = appearance_n_comp
        self.app_dim = app_dim
        self.aabb = aabb
        self.alphaMask = alphaMask
        self.device=device

        self.density_shift = -10.0 # shift density in softplus; making density = 0  when feature == 0
        self.alphaMask_thres = alphaMask_thres # 0.0001
        self.distance_scale = distance_scale
        self.rayMarch_weight_thres = rayMarch_weight_thres
        self.feat2denseAct = feat2denseAct

        self.near_far = near_far
        self.step_ratio = step_ratio

        self.update_stepSize(gridSize)

        self.matMode = [[0,1], [0,2], [1,2]]
        self.vecMode =  [2, 1, 0]
        # self.comp_w = [1,1,1]

        self.init_svd_volume(gridSize[0], device)

        self.view_pe, self.feat_pe, self.featureC = view_pe, feat_pe, featureC
        self.renderModule = MLPRender_Fea(self.app_dim, view_pe, feat_pe, featureC).to(device)

        print("view_pe", view_pe, "feat_pe", feat_pe)
        print(self.renderModule)

    def update_stepSize(self, gridSize):
        print("aabb", self.aabb.view(-1))
        print("grid size", gridSize)
        self.aabbSize = self.aabb[1] - self.aabb[0]
        self.invaabbSize = 2.0/self.aabbSize
        self.gridSize= torch.LongTensor(gridSize).to(self.device)
        self.units = self.aabbSize / (self.gridSize-1)
        self.stepSize=torch.mean(self.units)*self.step_ratio

        aabbDiag = torch.sqrt(torch.sum(torch.square(self.aabbSize)))
        self.nSamples = int((aabbDiag / self.stepSize).item()) + 1

        print("sampling step size: ", self.stepSize)
        print("sampling number: ", self.nSamples)

    def normalize_coord(self, xyz_sampled):
        return (xyz_sampled-self.aabb[0]) * self.invaabbSize - 1

    def get_kwargs(self):
        return {
            'aabb': self.aabb,
            'gridSize':self.gridSize.tolist(),
            'density_n_comp': self.density_n_comp,
            'appearance_n_comp': self.app_n_comp,
            'app_dim': self.app_dim,

            'alphaMask_thres': self.alphaMask_thres,
            'distance_scale': self.distance_scale,
            'rayMarch_weight_thres': self.rayMarch_weight_thres,
            'feat2denseAct': self.feat2denseAct,

            'near_far': self.near_far,
            'step_ratio': self.step_ratio,

            'view_pe': self.view_pe,
            'feat_pe': self.feat_pe,
            'featureC': self.featureC
        }

    def save(self, path):
        kwargs = self.get_kwargs()
        ckpt = {'kwargs': kwargs, 'state_dict': self.state_dict()}
        if self.alphaMask is not None:
            alpha_volume = self.alphaMask.alpha_volume.bool().cpu().numpy()
            ckpt.update({'alphaMask.shape':alpha_volume.shape})
            ckpt.update({'alphaMask.mask':np.packbits(alpha_volume.reshape(-1))})
            ckpt.update({'alphaMask.aabb': self.alphaMask.aabb.cpu()})
        torch.save(ckpt, path)

    def load(self, ckpt):
        if 'alphaMask.aabb' in ckpt.keys(): # True
            length = np.prod(ckpt['alphaMask.shape'])
            alpha_volume = torch.from_numpy(np.unpackbits(ckpt['alphaMask.mask'])[:length].reshape(ckpt['alphaMask.shape']))
            self.alphaMask = AlphaGridMask(self.device, ckpt['alphaMask.aabb'].to(self.device), alpha_volume.float().to(self.device))
        self.load_state_dict(ckpt['state_dict'])

    # def sample_ray_ndc(self, rays_o, rays_d, is_train=True, N_samples=-1):
    #     N_samples = N_samples if N_samples > 0 else self.nSamples
    #     near, far = self.near_far
    #     interpx = torch.linspace(near, far, N_samples).unsqueeze(0).to(rays_o)
    #     if is_train:
    #         interpx += torch.rand_like(interpx).to(rays_o) * ((far - near) / N_samples)

    #     rays_pts = rays_o[..., None, :] + rays_d[..., None, :] * interpx[..., None]
    #     mask_outbbox = ((self.aabb[0] > rays_pts) | (rays_pts > self.aabb[1])).any(dim=-1)
    #     return rays_pts, interpx, ~mask_outbbox

    def sample_ray(self, rays_o, rays_d, is_train=True, N_samples=-1):
        # is_train = True
        # N_samples = 443
        # rays_o.size() -- [4096, 3]
        # rays_d.size() -- [4096, 3]
        N_samples = N_samples if N_samples>0 else self.nSamples
        near, far = self.near_far
        vec = torch.where(rays_d==0, torch.full_like(rays_d, 1e-6), rays_d)
        rate_a = (self.aabb[1] - rays_o) / vec
        rate_b = (self.aabb[0] - rays_o) / vec
        t_min = torch.minimum(rate_a, rate_b).amax(-1).clamp(min=near, max=far)

        rng = torch.arange(N_samples)[None].float()
        if is_train:
            rng = rng.repeat(rays_d.shape[-2],1)
            rng += torch.rand_like(rng[:,[0]])
        step = self.stepSize * rng.to(rays_o.device)
        interpx = (t_min[...,None] + step)

        rays_pts = rays_o[...,None,:] + rays_d[...,None,:] * interpx[...,None]
        mask_outbbox = ((self.aabb[0] > rays_pts) | (rays_pts>self.aabb[1])).any(dim=-1)

        # rays_pts.size() -- [4096, 443, 3]
        # interpx.size() -- [4096, 443]
        # mask_outbbox.size() -- [4096, 443]
        return rays_pts, interpx, ~mask_outbbox

    # def shrink(self, new_aabb, voxel_size):
    #     pass

    @torch.no_grad()
    def getDenseAlpha(self,gridSize=None):
        # gridSize = (128, 128, 128)
        gridSize = self.gridSize if gridSize is None else gridSize

        samples = torch.stack(torch.meshgrid(
            torch.linspace(0, 1, gridSize[0]),
            torch.linspace(0, 1, gridSize[1]),
            torch.linspace(0, 1, gridSize[2]),
        ), -1).to(self.device)
        dense_xyz = self.aabb[0] * (1-samples) + self.aabb[1] * samples

        alpha = torch.zeros_like(dense_xyz[...,0])
        for i in range(gridSize[0]):
            alpha[i] = self.compute_alpha(dense_xyz[i].view(-1,3), self.stepSize).view((gridSize[1], gridSize[2]))

        # alpha.size() -- [128, 128, 128]
        # dense_xyz.size() -- [128, 128, 128, 3]
        return alpha, dense_xyz

    @torch.no_grad()
    def updateAlphaMask(self, gridSize=(200,200,200)):
        # gridSize = (128, 128, 128)
        alpha, dense_xyz = self.getDenseAlpha(gridSize)
        dense_xyz = dense_xyz.transpose(0,2).contiguous()
        alpha = alpha.clamp(0,1).transpose(0,2).contiguous()[None,None]
        total_voxels = gridSize[0] * gridSize[1] * gridSize[2]

        ks = 3
        alpha = F.max_pool3d(alpha, kernel_size=ks, padding=ks // 2, stride=1).view(gridSize[::-1])
        alpha[alpha>=self.alphaMask_thres] = 1
        alpha[alpha<self.alphaMask_thres] = 0

        self.alphaMask = AlphaGridMask(self.device, self.aabb, alpha)

        valid_xyz = dense_xyz[alpha>0.5]

        xyz_min = valid_xyz.amin(0)
        xyz_max = valid_xyz.amax(0)

        new_aabb = torch.stack((xyz_min, xyz_max))

        total = torch.sum(alpha)
        print(f"bbox: {xyz_min, xyz_max} alpha rest %%%f"%(total/total_voxels*100))

        # ==> pdb.set_trace()
        return new_aabb

    @torch.no_grad()
    def filtering_rays(self, all_rays, all_rgbs, N_samples=256, chunk=10240*5, bbox_only=False):
        # N_samples = 256
        # chunk = 51200
        # bbox_only = True
        print('========> filtering rays ...')
        tt = time.time()
        # all_rays.shape[:-1] -- [64000000]
        N = torch.tensor(all_rays.shape[:-1]).prod() # 64000000

        mask_filtered = []
        idx_chunks = torch.split(torch.arange(N), chunk)
        for idx_chunk in idx_chunks:
            rays_chunk = all_rays[idx_chunk].to(self.device)

            rays_o, rays_d = rays_chunk[..., :3], rays_chunk[..., 3:6]
            if bbox_only:
                vec = torch.where(rays_d == 0, torch.full_like(rays_d, 1e-6), rays_d)
                rate_a = (self.aabb[1] - rays_o) / vec
                rate_b = (self.aabb[0] - rays_o) / vec
                t_min = torch.minimum(rate_a, rate_b).amax(-1) #.clamp(min=near, max=far)
                t_max = torch.maximum(rate_a, rate_b).amin(-1) #.clamp(min=near, max=far)
                mask_inbbox = t_max > t_min
            else:
                xyz_sampled, _,_ = self.sample_ray(rays_o, rays_d, N_samples=N_samples, is_train=False)
                mask_inbbox= (self.alphaMask.sample_alpha(xyz_sampled).view(xyz_sampled.shape[:-1]) > 0).any(-1)

            mask_filtered.append(mask_inbbox.cpu())

        mask_filtered = torch.cat(mask_filtered).view(all_rgbs.shape[:-1])

        print(f'Ray filtering done! takes {time.time()-tt} s. ray mask ratio: {torch.sum(mask_filtered) / N}')
        return all_rays[mask_filtered], all_rgbs[mask_filtered]


    def feature2density(self, density_features):
        # density_features.size() -- [985197]
        if self.feat2denseAct == "softplus": # True
            return F.softplus(density_features + self.density_shift) # self.density_shift == -10
        elif self.feat2denseAct == "relu":
            return F.relu(density_features)

    def compute_alpha(self, xyz_locs, length=1):
        # xyz_locs.size() -- [16384, 3]
        # length = tensor(0.0118, device='cuda:0')

        if self.alphaMask is not None: # False
            alphas = self.alphaMask.sample_alpha(xyz_locs)
            alpha_mask = alphas > 0
        else:
            alpha_mask = torch.ones_like(xyz_locs[:,0], dtype=bool)
            
        sigma = torch.zeros(xyz_locs.shape[:-1], device=xyz_locs.device)

        if alpha_mask.any(): # True
            xyz_sampled = self.normalize_coord(xyz_locs[alpha_mask])
            sigma_feature = self.compute_densityfeature(xyz_sampled)
            valid_sigma = self.feature2density(sigma_feature)
            sigma[alpha_mask] = valid_sigma
        
        alpha = 1 - torch.exp(-sigma*length).view(xyz_locs.shape[:-1])
        return alpha # alpha.size() -- [16384]


    def forward(self, rays_chunk, white_bg=True, is_train=False, N_samples=-1):
        # rays_chunk.size() -- [4096, 6]
        # white_bg = True
        # is_train = True
        # N_samples = 443

        # sample points
        viewdirs = rays_chunk[:, 3:6]
        xyz_sampled, z_vals, ray_valid = self.sample_ray(rays_chunk[:, :3], viewdirs, 
            is_train=is_train,N_samples=N_samples)
        dists = torch.cat((z_vals[:, 1:] - z_vals[:, :-1], torch.zeros_like(z_vals[:, :1])), dim=-1)
        viewdirs = viewdirs.view(-1, 1, 3).expand(xyz_sampled.shape)

        sigma = torch.zeros(xyz_sampled.shape[:-1], device=xyz_sampled.device)
        rgb = torch.zeros((*xyz_sampled.shape[:2], 3), device=xyz_sampled.device)
        
        if self.alphaMask is not None: # False
            alphas = self.alphaMask.sample_alpha(xyz_sampled[ray_valid])
            alpha_mask = alphas > 0
            ray_invalid = ~ray_valid
            ray_invalid[ray_valid] |= (~alpha_mask)
            ray_valid = ~ray_invalid

        if ray_valid.any(): # True ?
            xyz_sampled = self.normalize_coord(xyz_sampled)
            sigma_feature = self.compute_densityfeature(xyz_sampled[ray_valid])
            valid_sigma = self.feature2density(sigma_feature)
            sigma[ray_valid] = valid_sigma

        # dists.size() -- [4096, 443]
        alpha, weight = raw2alpha(sigma, dists * self.distance_scale)

        app_mask = weight > self.rayMarch_weight_thres # self.rayMarch_weight_thres -- 0.0001
        if app_mask.any(): # False
            app_features = self.compute_appfeature(xyz_sampled[app_mask])
            valid_rgbs = self.renderModule(xyz_sampled[app_mask], viewdirs[app_mask], app_features)
            rgb[app_mask] = valid_rgbs

        acc_map = torch.sum(weight, -1)
        rgb_map = torch.sum(weight[..., None] * rgb, -2)

        if white_bg or (is_train and torch.rand((1,))<0.5):
            rgb_map = rgb_map + (1. - acc_map[..., None])
        
        rgb_map = rgb_map.clamp(0,1)
        with torch.no_grad():
            depth_map = torch.sum(weight * z_vals, -1)
            depth_map = depth_map + (1. - acc_map) * rays_chunk[..., -1]

        # rgb_map.size() -- [4096, 3]
        # depth_map.size() -- [4096]
        return rgb_map, depth_map # rgb, sigma, alpha, weight, bg_weight

