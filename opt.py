import configargparse

def config_parser(cmd=None):
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, help='config file path')
    parser.add_argument("--expname", type=str, help='experiment name')
    parser.add_argument("--basedir", type=str, default='./log',
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern',
                        help='input data directory')
    parser.add_argument("--progress_refresh_rate", type=int, default=10,
                        help='how many iterations to show psnrs or iters')

    parser.add_argument('--downsample_train', type=float, default=1.0)
    parser.add_argument('--downsample_test', type=float, default=1.0)

    # loader options
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--n_iters", type=int, default=30000)

    # training options
    # learning rate
    parser.add_argument("--lr_init", type=float, default=0.02, help='learning rate')    
    parser.add_argument("--lr_basis", type=float, default=1e-3, help='learning rate')
    parser.add_argument("--lr_decay_ratio", type=float, default=0.1,
                        help='the target decay ratio; after decay_iters inital lr decays to lr*ratio')
    parser.add_argument("--lr_upsample_reset", type=int, default=1,
                        help='reset lr to inital after upsampling')

    # loss
    parser.add_argument("--L1_weight_inital", type=float, default=0.0, help='loss weight')
    parser.add_argument("--L1_weight_rest", type=float, default=0.0, help='loss weight')
    
    # model
    # volume options
    parser.add_argument("--dense_n_comp", type=int, action="append")
    parser.add_argument("--color_n_comp", type=int, action="append")
    parser.add_argument("--color_data_dim", type=int, default=27)

    # network decoder
    parser.add_argument("--view_pe", type=int, default=2, help='number of pe for view')
    parser.add_argument("--feat_pe", type=int, default=2, help='number of pe for features')
    parser.add_argument("--feature_dim", type=int, default=128, help='hidden feature channel in MLP')


    parser.add_argument("--ckpt", type=str, default=None,
                        help='specific weights npy file to reload for coarse network')
    # parser.add_argument("--render_only", type=int, default=0)
    parser.add_argument("--render_test", type=int, default=0)

    # rendering options
    parser.add_argument("--accumulate_decay", type=float, default=0.998)
    parser.add_argument('--n_samples', type=int, default=1e6,
                        help='sample point each ray, pass 1e6 if automatic adjust')
    parser.add_argument('--step_ratio',type=float,default=0.5)


    ## blender flags
    parser.add_argument('--n_voxel_init', type=int, default=100**3)
    parser.add_argument('--n_voxel_final', type=int, default=300**3)
    parser.add_argument("--upsample_list", type=int, action="append")
    parser.add_argument("--update_alphamask_list", type=int, action="append")

    # logging/saving options
    parser.add_argument("--N_vis", type=int, default=5, help='N images to vis')
    parser.add_argument("--vis_every", type=int, default=10000, help='frequency of visualize the image')

    return parser.parse_args()