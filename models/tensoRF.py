import torch
import torch.nn as nn
import torch.nn.functional as F
# from .sh import eval_sh_bases
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
        self.grid_size = torch.LongTensor([alpha_volume.shape[-1],alpha_volume.shape[-2],alpha_volume.shape[-3]]).to(self.device)
        # self.grid_size -- tensor([128, 128, 128], device='cuda:0')

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
    def __init__(self,inChanel, view_pe=6, feat_pe=6, feature_dim=128):
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
        # feature_dim = 128

        self.in_mlpC = (2*view_pe*3 + 2*feat_pe*inChanel) + 3 + inChanel # 150
        self.view_pe = view_pe
        self.feat_pe = feat_pe
        layer1 = torch.nn.Linear(self.in_mlpC, feature_dim)
        layer2 = torch.nn.Linear(feature_dim, feature_dim)
        layer3 = torch.nn.Linear(feature_dim,3)

        self.mlp = torch.nn.Sequential(
            layer1, torch.nn.ReLU(inplace=True),
            layer2, torch.nn.ReLU(inplace=True),
            layer3)

        torch.nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(self, viewdirs, features):
        indata = [features, viewdirs]
        if self.feat_pe > 0: # True
            indata += [positional_encoding(features, self.feat_pe)]
        if self.view_pe > 0: # True
            indata += [positional_encoding(viewdirs, self.view_pe)]
        mlp_in = torch.cat(indata, dim=-1)
        rgb = self.mlp(mlp_in)
        rgb = torch.sigmoid(rgb)

        return rgb

class TensorVMSplit(nn.Module):
    def __init__(self, aabb, grid_size, device, 
                    dense_n_comp = [16, 16, 16], 
                    color_n_comp = [16, 16, 16], 
                    color_data_dim = 27,
                    alpha_mask = None, near_far=[2.0, 6.0],
                    view_pe = 2, feat_pe = 2, 
                    feature_dim=128, step_ratio=0.5):
        super(TensorVMSplit, self).__init__()
        # self = TensorVMSplit()
        # aabb = tensor([[-1.5000, -1.5000, -1.5000],
        #         [ 1.5000,  1.5000,  1.5000]], device='cuda:0')
        # grid_size = [128, 128, 128]
        # device = device(type='cuda')

        self.dense_n_comp = dense_n_comp
        self.color_n_comp = color_n_comp
        self.color_data_dim = color_data_dim
        self.aabb = aabb
        self.alpha_mask = alpha_mask
        self.device=device

        self.dense_shift = -10.0 # shift density in softplus; making density = 0  when feature == 0
        self.alpha_mask_threshold = 0.0001
        self.distance_scale = 25 # 5x5 sigma 
        self.march_weight_threshold = 0.0001

        self.near_far = near_far
        self.step_ratio = step_ratio

        self.update_step_size(grid_size)

        self.matMode = [[0,1], [0,2], [1,2]]
        self.vecMode =  [2, 1, 0]

        self.init_svd_volume(device)

        self.view_pe, self.feat_pe, self.feature_dim = view_pe, feat_pe, feature_dim
        self.render_model = MLPRender_Fea(self.color_data_dim, view_pe, feat_pe, feature_dim).to(device)

        print("view_pe", view_pe, "feat_pe", feat_pe)
        print(self.render_model)


    def init_svd_volume(self, device):
        # self.dense_n_comp -- [16, 16, 16]
        # self.app_n_comp -- [16, 16, 16]
        # self.color_data_dim -- 27
        self.dense_plane, self.dense_line = self.init_one_svd(self.dense_n_comp, self.grid_size, 0.1, device)
        self.color_plane, self.color_line = self.init_one_svd(self.color_n_comp, self.grid_size, 0.1, device)
        self.basis_mat = torch.nn.Linear(sum(self.color_n_comp), self.color_data_dim, bias=False).to(device)

    def init_one_svd(self, n_component, grid_size, scale, device):
        plane_coef, line_coef = [], []
        for i in range(len(self.vecMode)): # self.vecMode -- [2, 1, 0]
            vec_id = self.vecMode[i]
            mat_id_0, mat_id_1 = self.matMode[i]
            # self.matMode -- [[0, 1], [0, 2], [1, 2]]
            plane_coef.append(torch.nn.Parameter(
                scale * torch.randn((1, n_component[i], grid_size[mat_id_1], grid_size[mat_id_0]))))  #
            line_coef.append(
                torch.nn.Parameter(scale * torch.randn((1, n_component[i], grid_size[vec_id], 1))))

        # n_component = [16, 16, 16]
        # grid_size = tensor([128, 128, 128], device='cuda:0')
        # scale = 0.1
        # ==> plane_coef[0|1|2].size() -- [1, 16, 128, 128]
        #     line_coef[0|1|2].size() -- [1, 16, 128, 1]
        return torch.nn.ParameterList(plane_coef).to(device), torch.nn.ParameterList(line_coef).to(device)

    def update_step_size(self, grid_size):
        print("aabb", self.aabb.view(-1))
        print("grid size", grid_size)
        self.aabbSize = self.aabb[1] - self.aabb[0]
        self.invaabbSize = 2.0/self.aabbSize
        self.grid_size= torch.LongTensor(grid_size).to(self.device)
        self.units = self.aabbSize / (self.grid_size - 1)
        self.step_size=torch.mean(self.units) * self.step_ratio

        diag = torch.sqrt(torch.sum(torch.square(self.aabbSize)))
        self.n_samples = int((diag / self.step_size).item()) + 1

        print("sampling step size: ", self.step_size)
        print("sampling number: ", self.n_samples)

    def normalize_coord(self, xyz_sampled):
        return (xyz_sampled - self.aabb[0]) * self.invaabbSize - 1

    def get_kwargs(self):
        return {
            'aabb': self.aabb,
            'grid_size':self.grid_size.tolist(),
            'dense_n_comp': self.dense_n_comp,
            'color_n_comp': self.color_n_comp,
            'color_data_dim': self.color_data_dim,

            'alpha_mask_threshold': self.alpha_mask_threshold,
            'distance_scale': self.distance_scale,
            'march_weight_threshold': self.march_weight_threshold,

            'near_far': self.near_far,
            'step_ratio': self.step_ratio,

            'view_pe': self.view_pe,
            'feat_pe': self.feat_pe,
            'feature_dim': self.feature_dim
        }

    def save(self, path):
        kwargs = self.get_kwargs()
        ckpt = {'kwargs': kwargs, 'state_dict': self.state_dict()}
        if self.alpha_mask is not None:
            alpha_volume = self.alpha_mask.alpha_volume.bool().cpu().numpy()
            ckpt.update({'alpha_mask.shape':alpha_volume.shape})
            ckpt.update({'alpha_mask.mask':np.packbits(alpha_volume.reshape(-1))})
            ckpt.update({'alpha_mask.aabb': self.alpha_mask.aabb.cpu()})
        torch.save(ckpt, path)

    def load(self, ckpt):
        if 'alpha_mask.aabb' in ckpt.keys(): # True
            length = np.prod(ckpt['alpha_mask.shape'])
            alpha_volume = torch.from_numpy(np.unpackbits(ckpt['alpha_mask.mask'])[:length].reshape(ckpt['alpha_mask.shape']))
            self.alpha_mask = AlphaGridMask(self.device, ckpt['alpha_mask.aabb'].to(self.device), alpha_volume.float().to(self.device))
        self.load_state_dict(ckpt['state_dict'])


    def sample_ray(self, rays_o, rays_d, is_train=True, N_samples=-1):
        # is_train = True
        # N_samples = 443
        # rays_o.size() -- [4096, 3]
        # rays_d.size() -- [4096, 3]
        N_samples = N_samples if N_samples > 0 else self.n_samples
        near, far = self.near_far
        vec = torch.where(rays_d==0, torch.full_like(rays_d, 1e-6), rays_d)
        rate_a = (self.aabb[1] - rays_o) / vec
        rate_b = (self.aabb[0] - rays_o) / vec
        t_min = torch.minimum(rate_a, rate_b).amax(-1).clamp(min=near, max=far)

        rng = torch.arange(N_samples)[None].float()
        if is_train:
            rng = rng.repeat(rays_d.shape[-2], 1) # rays_d.shape[-2] -- 4096
            rng += torch.rand_like(rng[:, [0]]) # torch.rand_like(rng[:, [0]]) -- [[0.0327]]
        step = self.step_size * rng.to(rays_o.device)
        interpx = (t_min[..., None] + step)

        rays_points = rays_o[...,None, :] + rays_d[...,None,:] * interpx[..., None]
        mask_outbbox = ((self.aabb[0] > rays_points) | (rays_points>self.aabb[1])).any(dim=-1)

        # rays_points.size() -- [4096, 443, 3]
        # interpx.size() -- [4096, 443]
        # mask_outbbox.size() -- [4096, 443]
        return rays_points, interpx, ~mask_outbbox

    @torch.no_grad()
    def get_dense_alpha(self, grid_size=None):
        # grid_size = (128, 128, 128)
        grid_size = self.grid_size if grid_size is None else grid_size
        X, Y, Z = grid_size

        samples = torch.stack(torch.meshgrid(
            torch.linspace(0, 1, X),
            torch.linspace(0, 1, Y),
            torch.linspace(0, 1, Z),
        ), -1).to(self.device)
        dense_xyz = self.aabb[0] * (1.0 - samples) + self.aabb[1] * samples

        alpha = torch.zeros_like(dense_xyz[..., 0])
        for i in range(X):
            alpha[i] = self.compute_alpha(dense_xyz[i].view(-1,3), self.step_size).view((Y, Z))

        # alpha.size() -- [128, 128, 128]
        # dense_xyz.size() -- [128, 128, 128, 3]
        return alpha, dense_xyz

    @torch.no_grad()
    def update_alpha_mask(self, grid_size):
        # grid_size = (128, 128, 128)
        X, Y, Z = grid_size
        alpha, dense_xyz = self.get_dense_alpha(grid_size)
        dense_xyz = dense_xyz.transpose(0,2).contiguous()
        alpha = alpha.clamp(0,1).transpose(0,2).contiguous()[None,None]
        total_voxels = X * Y * Z

        ks = 3
        alpha = F.max_pool3d(alpha, kernel_size=ks, padding=ks // 2, stride=1).view(Z, Y, X)
        alpha[alpha >= self.alpha_mask_threshold] = 1.0
        alpha[alpha < self.alpha_mask_threshold] = 0.0

        self.alpha_mask = AlphaGridMask(self.device, self.aabb, alpha)

        valid_xyz = dense_xyz[alpha > 0.5]
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
                mask_inbbox= (self.alpha_mask.sample_alpha(xyz_sampled).view(xyz_sampled.shape[:-1]) > 0).any(-1)

            mask_filtered.append(mask_inbbox.cpu())

        mask_filtered = torch.cat(mask_filtered).view(all_rgbs.shape[:-1])

        print(f'Ray filtering done! takes {time.time()-tt} s. ray mask ratio: {torch.sum(mask_filtered) / N}')
        return all_rays[mask_filtered], all_rgbs[mask_filtered]


    def feature2dense(self, density_features):
        # density_features.size() -- [985197]
        return F.softplus(density_features + self.dense_shift) # self.dense_shift == -10 for softplus

    def compute_alpha(self, xyz_locs, length=1):
        # xyz_locs.size() -- [16384, 3]
        # length = tensor(0.0118, device='cuda:0')

        if self.alpha_mask is not None: # False
            alphas = self.alpha_mask.sample_alpha(xyz_locs)
            alpha_mask = alphas > 0
        else:
            alpha_mask = torch.ones_like(xyz_locs[:,0], dtype=bool)
            
        sigma = torch.zeros(xyz_locs.shape[:-1], device=xyz_locs.device)

        if alpha_mask.any(): # True
            xyz_sampled = self.normalize_coord(xyz_locs[alpha_mask])
            sigma_feature = self.dense2feature(xyz_sampled)
            valid_sigma = self.feature2dense(sigma_feature)
            sigma[alpha_mask] = valid_sigma
        
        alpha = 1.0 - torch.exp(-sigma*length).view(xyz_locs.shape[:-1])
        return alpha # alpha.size() -- [16384]


    def forward(self, rays_chunk, white_bg=True, is_train=False, N_samples=-1):
        # rays_chunk.size() -- [4096, 6]
        # white_bg = True
        # is_train = True
        # N_samples = 443

        # sample points
        view_o = rays_chunk[:, 0:3]
        view_d = rays_chunk[:, 3:6]
        xyz_sampled, z_vals, ray_valid = self.sample_ray(view_o, view_d, 
            is_train=is_train, N_samples=N_samples)
        # z_vals.size() -- [4096, 443]
        dists = torch.cat((z_vals[:, 1:] - z_vals[:, :-1], torch.zeros_like(z_vals[:, :1])), dim=-1)
        # dists.size() -- [4096, 443]
        view_d = view_d.view(-1, 1, 3).expand(xyz_sampled.shape)

        # xyz_sampled.shape[:-1] -- [4096, 443]
        sigma = torch.zeros(xyz_sampled.shape[:-1], device=xyz_sampled.device)
        rgb = torch.zeros((*xyz_sampled.shape[:2], 3), device=xyz_sampled.device)
        # xyz_sampled.shape[:2] -- [4096, 443]

        if self.alpha_mask is not None: # False
            alphas = self.alpha_mask.sample_alpha(xyz_sampled[ray_valid])
            alpha_mask = alphas > 0
            ray_invalid = ~ray_valid
            ray_invalid[ray_valid] |= (~alpha_mask)
            ray_valid = ~ray_invalid

        if ray_valid.any(): # True ?
            xyz_sampled = self.normalize_coord(xyz_sampled)
            sigma_feature = self.dense2feature(xyz_sampled[ray_valid])
            valid_sigma = self.feature2dense(sigma_feature)
            sigma[ray_valid] = valid_sigma

        # dists.size() -- [4096, 443], delta -- gradient ?
        alpha, weight = raw2alpha(sigma, dists * self.distance_scale)

        color_mask = weight > self.march_weight_threshold # self.march_weight_threshold -- 0.0001
        if color_mask.any(): # False
            color_features = self.color2feature(xyz_sampled[color_mask])
            valid_rgbs = self.render_model(view_d[color_mask], color_features)
            rgb[color_mask] = valid_rgbs

        acc_map = torch.sum(weight, -1)
        rgb_map = torch.sum(weight[..., None] * rgb, -2)

        if white_bg or (is_train and torch.rand((1,)) < 0.5):
            rgb_map = rgb_map + (1. - acc_map[..., None])
        
        return rgb_map.clamp(0,1)

    def get_optparam_groups(self, lr_init_spatialxyz = 0.02, lr_init_network = 0.001):
        # what's app ?
        grad_vars = [{'params': self.dense_line, 'lr': lr_init_spatialxyz}, 
                     {'params': self.dense_plane, 'lr': lr_init_spatialxyz},
                     {'params': self.color_line, 'lr': lr_init_spatialxyz}, 
                     {'params': self.color_plane, 'lr': lr_init_spatialxyz},
                     {'params': self.basis_mat.parameters(), 'lr':lr_init_network}]
        if isinstance(self.render_model, torch.nn.Module):
            grad_vars += [{'params':self.render_model.parameters(), 'lr':lr_init_network}]
        return grad_vars


    def dense_L1(self):
        total = 0
        for idx in range(len(self.dense_plane)):
            total = total + torch.mean(torch.abs(self.dense_plane[idx])) + torch.mean(torch.abs(self.dense_line[idx]))# + torch.mean(torch.abs(self.color_plane[idx])) + torch.mean(torch.abs(self.dense_plane[idx]))
        # ==> pdb.set_trace()
        return total
    
    def dense2feature(self, xyz_sampled):
        # xyz_sampled.size() -- [985197, 3]
        # self.matMode -- [[0, 1], [0, 2], [1, 2]]
        # self.vecMode -- [2, 1, 0]

        # plane + line basis
        coordinate_plane = torch.stack((xyz_sampled[..., self.matMode[0]], xyz_sampled[..., self.matMode[1]], xyz_sampled[..., self.matMode[2]])).detach().view(3, -1, 1, 2)
        coordinate_line = torch.stack((xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2)
        # coordinate_plane.size() -- [3, 985197, 1, 2]
        # coordinate_line.size() -- [3, 985197, 1, 2]

        sigma_feature = torch.zeros((xyz_sampled.shape[0],), device=xyz_sampled.device)
        # self.dense_plane
        # ParameterList(
        #     (0): Parameter containing: [torch.cuda.FloatTensor of size 1x16x128x128 (GPU 0)]
        #     (1): Parameter containing: [torch.cuda.FloatTensor of size 1x16x128x128 (GPU 0)]
        #     (2): Parameter containing: [torch.cuda.FloatTensor of size 1x16x128x128 (GPU 0)]
        # )
        for idx_plane in range(len(self.dense_plane)):
            plane_coef_point = F.grid_sample(self.dense_plane[idx_plane], coordinate_plane[[idx_plane]],
                                                align_corners=True).view(-1, *xyz_sampled.shape[:1])
            line_coef_point = F.grid_sample(self.dense_line[idx_plane], coordinate_line[[idx_plane]],
                                            align_corners=True).view(-1, *xyz_sampled.shape[:1])
            sigma_feature = sigma_feature + torch.sum(plane_coef_point * line_coef_point, dim=0)

        return sigma_feature # sigma_feature.size() -- [985197]


    def color2feature(self, xyz_sampled):
        # xyz_sampled.size() -- [9, 3]

        # plane + line basis
        coordinate_plane = torch.stack((xyz_sampled[..., self.matMode[0]], xyz_sampled[..., self.matMode[1]], xyz_sampled[..., self.matMode[2]])).detach().view(3, -1, 1, 2)
        coordinate_line = torch.stack((xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2)

        plane_coef_point,line_coef_point = [],[]
        # (Pdb) self.color_plane
        # ParameterList(
        #     (0): Parameter containing: [torch.cuda.FloatTensor of size 1x16x128x128 (GPU 0)]
        #     (1): Parameter containing: [torch.cuda.FloatTensor of size 1x16x128x128 (GPU 0)]
        #     (2): Parameter containing: [torch.cuda.FloatTensor of size 1x16x128x128 (GPU 0)]
        # )
        for idx_plane in range(len(self.color_plane)):
            plane_coef_point.append(F.grid_sample(self.color_plane[idx_plane], coordinate_plane[[idx_plane]],
                                                align_corners=True).view(-1, *xyz_sampled.shape[:1]))
            line_coef_point.append(F.grid_sample(self.color_line[idx_plane], coordinate_line[[idx_plane]],
                                            align_corners=True).view(-1, *xyz_sampled.shape[:1]))
        plane_coef_point, line_coef_point = torch.cat(plane_coef_point), torch.cat(line_coef_point)

        # plane_coef_point.size() -- [48, 9]
        # line_coef_point.size() -- [48, 9]
        return self.basis_mat((plane_coef_point * line_coef_point).T)

    @torch.no_grad()
    def up_sampling_vm(self, plane_coef, line_coef, res_target):
        # res_target = [115, 205, 133]
        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            mat_id_0, mat_id_1 = self.matMode[i]
            plane_coef[i] = torch.nn.Parameter(
                F.interpolate(plane_coef[i].data, size=(res_target[mat_id_1], res_target[mat_id_0]), mode='bilinear',
                              align_corners=True))
            line_coef[i] = torch.nn.Parameter(
                F.interpolate(line_coef[i].data, size=(res_target[vec_id], 1), mode='bilinear', align_corners=True))
        return plane_coef, line_coef

    @torch.no_grad()
    def upsample_volume_grid(self, res_target):
        self.color_plane, self.color_line = self.up_sampling_vm(self.color_plane, self.color_line, res_target)
        self.dense_plane, self.dense_line = self.up_sampling_vm(self.dense_plane, self.dense_line, res_target)

        self.update_step_size(res_target)
        print(f'upsamping to {res_target}')
        # ==> pdb.set_trace()

    @torch.no_grad()
    def shrink(self, new_aabb):
        # (Pdb) new_aabb
        # tensor([[-0.6732, -1.1929, -0.5079],
        #         [ 0.6732,  1.1929,  1.0512]], device='cuda:0')

        print("====> shrinking ...")
        xyz_min, xyz_max = new_aabb
        t_l, b_r = (xyz_min - self.aabb[0]) / self.units, (xyz_max - self.aabb[0]) / self.units
        # print(new_aabb, self.aabb)
        # print(t_l, b_r,self.alpha_mask.alpha_volume.shape)
        t_l, b_r = torch.round(torch.round(t_l)).long(), torch.round(b_r).long() + 1
        b_r = torch.stack([b_r, self.grid_size]).amin(0)

        for i in range(len(self.vecMode)):
            mode0 = self.vecMode[i]
            self.dense_line[i] = torch.nn.Parameter(
                self.dense_line[i].data[...,t_l[mode0]:b_r[mode0],:]
            )
            self.color_line[i] = torch.nn.Parameter(
                self.color_line[i].data[...,t_l[mode0]:b_r[mode0],:]
            )
            mode0, mode1 = self.matMode[i]
            self.dense_plane[i] = torch.nn.Parameter(
                self.dense_plane[i].data[...,t_l[mode1]:b_r[mode1],t_l[mode0]:b_r[mode0]]
            )
            self.color_plane[i] = torch.nn.Parameter(
                self.color_plane[i].data[...,t_l[mode1]:b_r[mode1],t_l[mode0]:b_r[mode0]]
            )


        if not torch.all(self.alpha_mask.grid_size == self.grid_size):
            t_l_r, b_r_r = t_l / (self.grid_size-1), (b_r-1) / (self.grid_size-1)
            correct_aabb = torch.zeros_like(new_aabb)
            correct_aabb[0] = (1-t_l_r)*self.aabb[0] + t_l_r*self.aabb[1]
            correct_aabb[1] = (1-b_r_r)*self.aabb[0] + b_r_r*self.aabb[1]
            print("aabb", new_aabb, "\ncorrect aabb", correct_aabb)
            new_aabb = correct_aabb

        newSize = b_r - t_l
        # newSize -- tensor([ 58, 102,  67], device='cuda:0')
        self.aabb = new_aabb
        self.update_step_size((newSize[0], newSize[1], newSize[2]))
        # ==> pdb.set_trace()
