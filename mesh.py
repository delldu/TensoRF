
import os
# from tqdm.auto import tqdm
from opt import config_parser

from utils import *
from models.tensoRF import TensorVMSplit
import pdb
import torch
torch.set_printoptions(sci_mode=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def export_mesh(args):

    ckpt = torch.load(args.ckpt, map_location=device)
    kwargs = ckpt['kwargs']
    kwargs.update({'device': device})

    del kwargs['alpha_mask_threshold']
    del kwargs['distance_scale']
    del kwargs['march_weight_threshold']
    # del kwargs['']

    tensorf = TensorVMSplit(**kwargs)
    tensorf.load(ckpt)

    alpha,_ = tensorf.get_dense_alpha()

    # alpha.size() -- [115, 205, 133]
    convert_sdf_samples_to_ply(alpha.cpu(), f'{args.ckpt[:-3]}.ply',bbox=tensorf.aabb.cpu(), level=0.005)


if __name__ == '__main__':
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(20211202)

    args = config_parser()
    print(args)
    export_mesh(args)
