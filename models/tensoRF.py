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
    freq_bands = (2 ** torch.arange(freqs).float()).to(positions.device)  # freq_bands.size() -- [2]
    # positions[..., None].size() -- [9, 27, 1]
    pts = (positions[..., None] * freq_bands).reshape(
        positions.shape[:-1] + (freqs * positions.shape[-1],)) # size() -- [9, 54]
    pts = torch.cat([torch.sin(pts), torch.cos(pts)], dim=-1)

    return pts  # pts.size() -- [9, 108]


def raw2alpha(sigma, dist):
    # sigma.size() -- [4096, 443]
    # dist.size() -- [4096, 443]
    alpha = 1.0 - torch.exp(-sigma * dist)  # alpha.size() -- [4096, 443]
    ones = torch.ones(alpha.shape[0], 1).to(alpha.device)  # ones.size() -- [4096, 1]
    T = torch.cumprod(torch.cat([ones, 1.0 - alpha + 1e-10], -1), -1)  # T.size() -- [4096, 444]
    weights = alpha * T[:, :-1]  # T[:, :-1].size() -- [4096, 443]

    return alpha, weights  # T[:,-1:] -- bg_weight


class MLPRender_Fea(nn.Module):
    def __init__(self, inChanel, view_pe=2, feat_pe=2, feature_dim=128):
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
        self.in_mlpC = (2 * view_pe * 3 + 2 * feat_pe * inChanel) + 3 + inChanel  # 150
        self.view_pe = view_pe
        self.feat_pe = feat_pe
        self.mlp = nn.Sequential(
            nn.Linear(self.in_mlpC, feature_dim), nn.ReLU(inplace=True), 
            nn.Linear(feature_dim, feature_dim), nn.ReLU(inplace=True),
            nn.Linear(feature_dim, 3))

        nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(self, rays_d, features):
        indata = [features, rays_d]
        indata += [positional_encoding(features, self.feat_pe)]
        indata += [positional_encoding(rays_d, self.view_pe)]
        # (Pdb) len(indata) -- 4
        # (Pdb) indata[0].size() -- [9, 27]
        # (Pdb) indata[1].size() -- [9, 3]
        # (Pdb) indata[2].size() -- [9, 108]
        # (Pdb) indata[3].size() -- [9, 12]
        mlp_in = torch.cat(indata, dim=1) # size() -- [9, 150]
        return torch.sigmoid(self.mlp(mlp_in)) # rgb, size() -- [9, 3]


class TensorVMSplit(nn.Module):
    def __init__(
        self,
        aabb,
        grid_size,
        device,
        dense_n_comp=[16, 16, 16],
        color_n_comp=[16, 16, 16],
        color_data_dim=27,
        near_far=[2.0, 6.0],
        view_pe=2,
        feat_pe=2,
        feature_dim=128,
        step_ratio=0.5,
    ):
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
        self.device = device

        self.dense_shift = -10.0  # shift density in softplus; making density = 0  when feature == 0
        self.alpha_mask_threshold = 0.0001
        self.distance_scale = 25  # 5x5 sigma
        self.march_weight_threshold = 0.0001

        self.near_far = near_far
        self.step_ratio = step_ratio

        self.update_step_size(grid_size)

        self.matMode = [[0, 1], [0, 2], [1, 2]]
        self.vecMode = [2, 1, 0]

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
        self.basis_mat = nn.Linear(sum(self.color_n_comp), self.color_data_dim, bias=False).to(device)

    def init_one_svd(self, n_component, grid_size, scale, device):
        plane_coef, line_coef = [], []
        for i in range(len(self.vecMode)):  # self.vecMode -- [2, 1, 0]
            vec_id = self.vecMode[i]
            mat_id_0, mat_id_1 = self.matMode[i]
            # self.matMode -- [[0, 1], [0, 2], [1, 2]]
            plane_coef.append(
                nn.Parameter(scale * torch.randn((1, n_component[i], grid_size[mat_id_1], grid_size[mat_id_0])))
            )  #
            line_coef.append(nn.Parameter(scale * torch.randn((1, n_component[i], grid_size[vec_id], 1))))

        # n_component = [16, 16, 16]
        # grid_size = tensor([128, 128, 128], device='cuda:0')
        # scale = 0.1
        # ==> plane_coef[0|1|2].size() -- [1, 16, 128, 128]
        #     line_coef[0|1|2].size() -- [1, 16, 128, 1]
        return nn.ParameterList(plane_coef).to(device), nn.ParameterList(line_coef).to(device)

    def update_step_size(self, grid_size):
        print("aabb", self.aabb.view(-1))
        print("grid size", grid_size)
        self.aabbSize = self.aabb[1] - self.aabb[0]
        self.invaabbSize = 2.0 / self.aabbSize
        self.grid_size = torch.LongTensor(grid_size).to(self.device)
        unit = self.aabbSize / (self.grid_size - 1)
        self.step_size = torch.mean(unit) * self.step_ratio

        diag = torch.sqrt(torch.sum(torch.square(self.aabbSize)))
        self.n_samples = int((diag / self.step_size).item()) + 1
        print("sampling step size: ", self.step_size)

    def normalize_coord(self, xyz_sampled):
        return (xyz_sampled - self.aabb[0]) * self.invaabbSize - 1


    def sample_ray(self, rays_o, rays_d, is_train=True):
        # is_train = True
        # rays_o.size() -- [4096, 3]
        # rays_d.size() -- [4096, 3]
        near, far = self.near_far
        vec = torch.where(rays_d == 0, torch.full_like(rays_d, 1e-6), rays_d)
        rate_a = (self.aabb[1] - rays_o) / vec
        rate_b = (self.aabb[0] - rays_o) / vec
        t_min = torch.minimum(rate_a, rate_b).amax(-1).clamp(min=near, max=far)  # size() -- [4096]

        rng = torch.arange(self.n_samples)[None].float() # size() -- [1, 440]
        if is_train:
            rng = rng.repeat(rays_d.shape[-2], 1)  # rays_d.shape[-2] -- 4096 ==> rng.size() -- [4096, 440]
            rng += torch.rand_like(rng[:, [0]])  # torch.rand_like(rng[:, [0]]) -- [[0.0327]]
        # self.step_size -- tensor(0.0118, device='cuda:0')
        # rng -- tensor([[    0.0327,     1.0327,     2.0327,  ...,   440.0327,   441.0327,
        #    442.0327],
        # [    0.7392,     1.7392,     2.7392,  ...,   440.7392,   441.7392,
        #    442.7392]])
        step = self.step_size * rng.to(rays_o.device)
        interpx = t_min[..., None] + step  # interpx.size() -- [4096, 443]
        # (Pdb) rays_o.size() -- [4096, 3]
        # (Pdb) rays_o[...,None, :].size() -- [4096, 1, 3]
        rays_points = rays_o[..., None, :] + rays_d[..., None, :] * interpx[..., None]
        mask_outbbox = ((self.aabb[0] > rays_points) | (rays_points > self.aabb[1])).any(dim=-1)

        # rays_points.size() -- [4096, 443, 3]
        # interpx.size() -- [4096, 443]
        # mask_outbbox.size() -- [4096, 443]

        return rays_points, interpx, ~mask_outbbox

    @torch.no_grad()
    def get_dense_alpha(self):
        X, Y, Z = self.grid_size # # grid_size = (128, 128, 128)
        samples = torch.stack(
            torch.meshgrid(
                torch.linspace(0, 1, X),
                torch.linspace(0, 1, Y),
                torch.linspace(0, 1, Z),
            ),
            -1,
        ).to(self.device)
        dense_xyz = self.aabb[0] * (1.0 - samples) + self.aabb[1] * samples

        alpha = torch.zeros_like(dense_xyz[..., 0])
        for i in range(X):
            alpha[i] = self.compute_alpha(dense_xyz[i].view(-1, 3)).view((Y, Z))

        # alpha.size() -- [128, 128, 128]
        # dense_xyz.size() -- [128, 128, 128, 3]
        return alpha, dense_xyz

    @torch.no_grad()
    def filtering_rays(self, all_rays, all_rgbs, chunk=10240 * 5):
        # chunk = 51200
        # bbox_only = True
        print("========> filtering rays ...")
        tt = time.time()
        # all_rays.shape[:-1] -- [64000000]
        N = torch.tensor(all_rays.shape[:-1]).prod()  # 64000000

        mask_filtered = []
        idx_chunks = torch.split(torch.arange(N), chunk)
        for idx_chunk in idx_chunks:
            rays_chunk = all_rays[idx_chunk].to(self.device)

            rays_o, rays_d = rays_chunk[..., :3], rays_chunk[..., 3:6]
            vec = torch.where(rays_d == 0, torch.full_like(rays_d, 1e-6), rays_d)
            rate_a = (self.aabb[1] - rays_o) / vec
            rate_b = (self.aabb[0] - rays_o) / vec
            t_min = torch.minimum(rate_a, rate_b).amax(-1)  # .clamp(min=near, max=far)
            t_max = torch.maximum(rate_a, rate_b).amin(-1)  # .clamp(min=near, max=far)
            mask_inbbox = t_max > t_min
            mask_filtered.append(mask_inbbox.cpu())

        mask_filtered = torch.cat(mask_filtered).view(all_rgbs.shape[:-1])

        print(f"Ray filtering done! takes {time.time()-tt} s. ray mask ratio: {torch.sum(mask_filtered) / N}")
        return all_rays[mask_filtered], all_rgbs[mask_filtered]

    def feature2dense(self, density_features):
        # density_features.size() -- [985197]
        return F.softplus(density_features + self.dense_shift)  # self.dense_shift == -10 for softplus

    def compute_alpha(self, xyz):
        # xyz.size() -- [16384, 3]
        # length = tensor(0.0118, device='cuda:0')
        # alpha_mask = torch.ones_like(xyz[:, 0], dtype=bool)
        # sigma = torch.zeros(xyz.shape[:-1], device=xyz.device)

        # if alpha_mask.any():  # True
        #     xyz_sampled = self.normalize_coord(xyz[alpha_mask])
        #     sigma_feature = self.dense2feature(xyz_sampled)
        #     valid_sigma = self.feature2dense(sigma_feature)
        #     sigma[alpha_mask] = valid_sigma

        xyz_sampled = self.normalize_coord(xyz)
        sigma_feature = self.dense2feature(xyz_sampled)
        sigma = self.feature2dense(sigma_feature)
        # self.step_size -- tensor(0.0118, device='cuda:0')
        alpha = 1.0 - torch.exp(-sigma * self.step_size).view(xyz.shape[:-1])
        return alpha  # alpha.size() -- [16384]

    def forward(self, rays_chunk, is_train=False):
        rays_o = rays_chunk[:, 0:3]
        rays_d = rays_chunk[:, 3:6]
        xyz_sampled, z_vals, ray_valid = self.sample_ray(rays_o, rays_d, is_train=is_train)
        # z_vals.size() -- [4096, 440]
        # ray_valid.size() -- [4096, 440]
        N, S, C = xyz_sampled.size() # [4096, 440, 3]

        dists = torch.cat((z_vals[:, 1:] - z_vals[:, :-1], torch.zeros_like(z_vals[:, :1])), dim=-1)
        rays_d = rays_d.view(-1, 1, 3).expand(xyz_sampled.shape)

        sigma = torch.zeros(N, S).to(xyz_sampled.device)
        if ray_valid.any():  # True ?
            xyz_sampled = self.normalize_coord(xyz_sampled)
            sigma_feature = self.dense2feature(xyz_sampled[ray_valid])
            valid_sigma = self.feature2dense(sigma_feature)
            sigma[ray_valid] = valid_sigma
        # ==> sigma

        rgb = torch.zeros(N, S, 3).to(xyz_sampled.device)
        alpha, weight = raw2alpha(sigma, dists * self.distance_scale) # weight.size() -- [4096, 440]
        color_mask = weight > self.march_weight_threshold
        if color_mask.any():
            color_features = self.color2feature(xyz_sampled[color_mask])
            valid_rgbs = self.render_model(rays_d[color_mask], color_features)
            rgb[color_mask] = valid_rgbs
        # ==> rgb

        acc_map = torch.sum(weight, dim=1) # acc_map.size() -- [4096]
        rgb_map = torch.sum(weight[..., None] * rgb, -2) # [4096, 3] -- zero !!!
        rgb_map = rgb_map + (1.0 - acc_map[..., None]) # rgb_map.size() -- [4096, 3]
        rgb_map = rgb_map.clamp(0.0, 1.0)
        # ==> rgb_map

        with torch.no_grad():
            depth_map = torch.sum(weight * z_vals, -1) # size() -- [4096]
            depth_map = depth_map + (1.0 - acc_map) * rays_chunk[..., -1]
        # ==> depth_map 

        # rgb_map.size() -- [4096, 3]
        # depth_map.size() -- [4096]
        return rgb_map, depth_map  # rgb, sigma, alpha, weight, bg_weight

    def get_optparam_groups(self, lr_init_spatialxyz=0.02, lr_init_network=0.001):
        # what's app ?
        grad_vars = [
            {"params": self.dense_line, "lr": lr_init_spatialxyz},
            {"params": self.dense_plane, "lr": lr_init_spatialxyz},
            {"params": self.color_line, "lr": lr_init_spatialxyz},
            {"params": self.color_plane, "lr": lr_init_spatialxyz},
            {"params": self.basis_mat.parameters(), "lr": lr_init_network},
        ]
        grad_vars += [{"params": self.render_model.parameters(), "lr": lr_init_network}]
        return grad_vars

    def dense_L1(self):
        total = 0
        for i in range(len(self.dense_plane)):
            total += torch.mean(torch.abs(self.dense_plane[i])) + torch.mean(torch.abs(self.dense_line[i]))
        # ==> pdb.set_trace()
        return total

    def dense2feature(self, xyz_sampled):
        N, C = xyz_sampled.size() # [985197, 3]

        # plane + line basis
        # self.matMode -- [[0, 1], [0, 2], [1, 2]]
        coord_plane = (
            torch.stack(
                (
                    xyz_sampled[..., self.matMode[0]], # [985197, 2]
                    xyz_sampled[..., self.matMode[1]], # [985197, 2]
                    xyz_sampled[..., self.matMode[2]], # [985197, 2]
                )
            )
            .detach()
            .view(3, -1, 1, 2)
        )
        # coord_plane.size() -- [3, 985197, 1, 2]

        # self.vecMode -- [2, 1, 0]
        coord_line = torch.stack(
            (xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]])
        )
        coord_line = (
            torch.stack((torch.zeros_like(coord_line), coord_line), dim=-1).detach().view(3, -1, 1, 2)
        )
        # coord_line.size() -- [3, 985197, 1, 2]

        sigma_feature = torch.zeros((N,), device=xyz_sampled.device) # [985197]
        # self.dense_plane
        # ParameterList(
        #     (0): Parameter containing: [torch.cuda.FloatTensor of size 1x16x128x128 (GPU 0)]
        #     (1): Parameter containing: [torch.cuda.FloatTensor of size 1x16x128x128 (GPU 0)]
        #     (2): Parameter containing: [torch.cuda.FloatTensor of size 1x16x128x128 (GPU 0)]
        # )
        for i in range(len(self.dense_plane)): # len() -- 3
            plane_coef_point = F.grid_sample(self.dense_plane[i], coord_plane[[i]], 
                align_corners=True).view(-1, N)
            line_coef_point = F.grid_sample(self.dense_line[i], coord_line[[i]],
                align_corners=True).view(-1, N)

            sigma_feature = sigma_feature + torch.sum(plane_coef_point * line_coef_point, dim=0)

        return sigma_feature  # sigma_feature.size() -- [985197]

    def color2feature(self, xyz_sampled):
        # xyz_sampled.size() -- [9, 3]

        # plane + line basis
        coord_plane = (
            torch.stack(
                (
                    xyz_sampled[..., self.matMode[0]],
                    xyz_sampled[..., self.matMode[1]],
                    xyz_sampled[..., self.matMode[2]],
                )
            )
            .detach()
            .view(3, -1, 1, 2)
        )
        coord_line = torch.stack(
            (xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]])
        )
        coord_line = (
            torch.stack((torch.zeros_like(coord_line), coord_line), dim=-1).detach().view(3, -1, 1, 2)
        )

        plane_coef_point, line_coef_point = [], []
        # (Pdb) self.color_plane
        # ParameterList(
        #     (0): Parameter containing: [torch.cuda.FloatTensor of size 1x16x128x128 (GPU 0)]
        #     (1): Parameter containing: [torch.cuda.FloatTensor of size 1x16x128x128 (GPU 0)]
        #     (2): Parameter containing: [torch.cuda.FloatTensor of size 1x16x128x128 (GPU 0)]
        # )
        for i in range(len(self.color_plane)):
            plane_coef_point.append(
                F.grid_sample(self.color_plane[i], coord_plane[[i]], align_corners=True).view(
                    -1, *xyz_sampled.shape[:1]
                )
            )
            line_coef_point.append(
                F.grid_sample(self.color_line[i], coord_line[[i]], align_corners=True).view(
                    -1, *xyz_sampled.shape[:1]
                )
            )
        plane_coef_point, line_coef_point = torch.cat(plane_coef_point), torch.cat(line_coef_point)

        # plane_coef_point.size() -- [48, 9]
        # line_coef_point.size() -- [48, 9]
        return self.basis_mat((plane_coef_point * line_coef_point).T)

