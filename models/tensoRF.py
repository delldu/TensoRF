from .tensorBase import *
import pdb

class TensorVMSplit(TensorBase):
    def __init__(self, aabb, gridSize, device, **kargs):
        super(TensorVMSplit, self).__init__(aabb, gridSize, device, **kargs)
        # gridSize = [128, 128, 128]

    def init_svd_volume(self, res, device):
        # self.density_n_comp -- [16, 16, 16]
        # self.app_n_comp -- [16, 16, 16]
        # self.app_dim -- 27
        self.density_plane, self.density_line = self.init_one_svd(self.density_n_comp, self.gridSize, 0.1, device)
        self.app_plane, self.app_line = self.init_one_svd(self.app_n_comp, self.gridSize, 0.1, device)
        self.basis_mat = torch.nn.Linear(sum(self.app_n_comp), self.app_dim, bias=False).to(device)

    def init_one_svd(self, n_component, gridSize, scale, device):
        plane_coef, line_coef = [], []
        for i in range(len(self.vecMode)): # self.vecMode -- [2, 1, 0]
            vec_id = self.vecMode[i]
            mat_id_0, mat_id_1 = self.matMode[i]
            # self.matMode -- [[0, 1], [0, 2], [1, 2]]
            plane_coef.append(torch.nn.Parameter(
                scale * torch.randn((1, n_component[i], gridSize[mat_id_1], gridSize[mat_id_0]))))  #
            line_coef.append(
                torch.nn.Parameter(scale * torch.randn((1, n_component[i], gridSize[vec_id], 1))))

        # n_component = [16, 16, 16]
        # gridSize = tensor([128, 128, 128], device='cuda:0')
        # scale = 0.1
        # ==> plane_coef[0|1|2].size() -- [1, 16, 128, 128]
        #     line_coef[0|1|2].size() -- [1, 16, 128, 1]
        return torch.nn.ParameterList(plane_coef).to(device), torch.nn.ParameterList(line_coef).to(device)
    
    def get_optparam_groups(self, lr_init_spatialxyz = 0.02, lr_init_network = 0.001):
        # what's app ?
        grad_vars = [{'params': self.density_line, 'lr': lr_init_spatialxyz}, 
                     {'params': self.density_plane, 'lr': lr_init_spatialxyz},
                     {'params': self.app_line, 'lr': lr_init_spatialxyz}, 
                     {'params': self.app_plane, 'lr': lr_init_spatialxyz},
                     {'params': self.basis_mat.parameters(), 'lr':lr_init_network}]
        if isinstance(self.renderModule, torch.nn.Module):
            grad_vars += [{'params':self.renderModule.parameters(), 'lr':lr_init_network}]
        return grad_vars


    def density_L1(self):
        total = 0
        for idx in range(len(self.density_plane)):
            total = total + torch.mean(torch.abs(self.density_plane[idx])) + torch.mean(torch.abs(self.density_line[idx]))# + torch.mean(torch.abs(self.app_plane[idx])) + torch.mean(torch.abs(self.density_plane[idx]))
        # ==> pdb.set_trace()
        return total
    
    def compute_densityfeature(self, xyz_sampled):
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
        # self.density_plane
        # ParameterList(
        #     (0): Parameter containing: [torch.cuda.FloatTensor of size 1x16x128x128 (GPU 0)]
        #     (1): Parameter containing: [torch.cuda.FloatTensor of size 1x16x128x128 (GPU 0)]
        #     (2): Parameter containing: [torch.cuda.FloatTensor of size 1x16x128x128 (GPU 0)]
        # )
        for idx_plane in range(len(self.density_plane)):
            plane_coef_point = F.grid_sample(self.density_plane[idx_plane], coordinate_plane[[idx_plane]],
                                                align_corners=True).view(-1, *xyz_sampled.shape[:1])
            line_coef_point = F.grid_sample(self.density_line[idx_plane], coordinate_line[[idx_plane]],
                                            align_corners=True).view(-1, *xyz_sampled.shape[:1])
            sigma_feature = sigma_feature + torch.sum(plane_coef_point * line_coef_point, dim=0)

        return sigma_feature # sigma_feature.size() -- [985197]


    def compute_appfeature(self, xyz_sampled):
        # xyz_sampled.size() -- [9, 3]

        # plane + line basis
        coordinate_plane = torch.stack((xyz_sampled[..., self.matMode[0]], xyz_sampled[..., self.matMode[1]], xyz_sampled[..., self.matMode[2]])).detach().view(3, -1, 1, 2)
        coordinate_line = torch.stack((xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2)

        plane_coef_point,line_coef_point = [],[]
        # (Pdb) self.app_plane
        # ParameterList(
        #     (0): Parameter containing: [torch.cuda.FloatTensor of size 1x16x128x128 (GPU 0)]
        #     (1): Parameter containing: [torch.cuda.FloatTensor of size 1x16x128x128 (GPU 0)]
        #     (2): Parameter containing: [torch.cuda.FloatTensor of size 1x16x128x128 (GPU 0)]
        # )
        for idx_plane in range(len(self.app_plane)):
            plane_coef_point.append(F.grid_sample(self.app_plane[idx_plane], coordinate_plane[[idx_plane]],
                                                align_corners=True).view(-1, *xyz_sampled.shape[:1]))
            line_coef_point.append(F.grid_sample(self.app_line[idx_plane], coordinate_line[[idx_plane]],
                                            align_corners=True).view(-1, *xyz_sampled.shape[:1]))
        plane_coef_point, line_coef_point = torch.cat(plane_coef_point), torch.cat(line_coef_point)

        # plane_coef_point.size() -- [48, 9]
        # line_coef_point.size() -- [48, 9]
        return self.basis_mat((plane_coef_point * line_coef_point).T)

    @torch.no_grad()
    def up_sampling_VM(self, plane_coef, line_coef, res_target):
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
        self.app_plane, self.app_line = self.up_sampling_VM(self.app_plane, self.app_line, res_target)
        self.density_plane, self.density_line = self.up_sampling_VM(self.density_plane, self.density_line, res_target)

        self.update_stepSize(res_target)
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
        # print(t_l, b_r,self.alphaMask.alpha_volume.shape)
        t_l, b_r = torch.round(torch.round(t_l)).long(), torch.round(b_r).long() + 1
        b_r = torch.stack([b_r, self.gridSize]).amin(0)

        for i in range(len(self.vecMode)):
            mode0 = self.vecMode[i]
            self.density_line[i] = torch.nn.Parameter(
                self.density_line[i].data[...,t_l[mode0]:b_r[mode0],:]
            )
            self.app_line[i] = torch.nn.Parameter(
                self.app_line[i].data[...,t_l[mode0]:b_r[mode0],:]
            )
            mode0, mode1 = self.matMode[i]
            self.density_plane[i] = torch.nn.Parameter(
                self.density_plane[i].data[...,t_l[mode1]:b_r[mode1],t_l[mode0]:b_r[mode0]]
            )
            self.app_plane[i] = torch.nn.Parameter(
                self.app_plane[i].data[...,t_l[mode1]:b_r[mode1],t_l[mode0]:b_r[mode0]]
            )


        if not torch.all(self.alphaMask.gridSize == self.gridSize):
            t_l_r, b_r_r = t_l / (self.gridSize-1), (b_r-1) / (self.gridSize-1)
            correct_aabb = torch.zeros_like(new_aabb)
            correct_aabb[0] = (1-t_l_r)*self.aabb[0] + t_l_r*self.aabb[1]
            correct_aabb[1] = (1-b_r_r)*self.aabb[0] + b_r_r*self.aabb[1]
            print("aabb", new_aabb, "\ncorrect aabb", correct_aabb)
            new_aabb = correct_aabb

        newSize = b_r - t_l
        # newSize -- tensor([ 58, 102,  67], device='cuda:0')
        self.aabb = new_aabb
        self.update_stepSize((newSize[0], newSize[1], newSize[2]))
        # ==> pdb.set_trace()
