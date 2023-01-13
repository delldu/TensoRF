"""Model test."""
# coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2023, All Rights Reserved.
# ***
# ***    File Author: Dell, 2023年 01月 12日 星期四 23:31:37 CST
# ***
# ************************************************************************************/
#
import os
import argparse
import torch
import TensoRF
import pdb  # For debug


def save_points(xyzs, rgbs, filename):
    """Save to point cloud file"""
    print(f"Saving {xyzs.size(0)} points to {filename} ...")

    float_formatter = lambda x: "%.4f" % x
    file = open(filename, "w")
    file.write(
        """ply
    format ascii 1.0
    element vertex %d
    property float x
    property float y
    property float z
    property uchar red
    property uchar green
    property uchar blue
    property uchar alpha
    end_header
    """
        % (xyzs.size(0))
    )
    for i in torch.cat((xyzs, rgbs), dim=1):
        file.write(
            "{} {} {} {} {} {} 0\n".format(
                float_formatter(i[0]),
                float_formatter(i[1]),
                float_formatter(i[2]),
                int(i[3] * 255.0),
                int(i[4] * 255.0),
                int(i[5] * 255.0),
            )
        )
    file.close()


if __name__ == "__main__":
    """Test model."""

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--outputdir", type=str, default="output", help="output directory")
    parser.add_argument("--checkpoint", type=str, default="output/model.pth", help="checkpoint file")
    parser.add_argument("--bs", type=int, default=2048, help="batch size")
    args = parser.parse_args()

    # Create directory to store result
    if not os.path.exists(args.outputdir):
        os.makedirs(args.outputdir)

    filename = f"{args.outputdir}/points.ply"

    # get dataset
    ds = TensoRF.train_data().dataset

    # get model
    net, device = TensoRF.get_model(args.checkpoint, aabb=ds.scene_bbox)
    net = net.eval()

    size = ds.all_rays.size(0) // 1000
    allrays = ds.all_rays[0:size, :]

    xyz_map, rgb_map = TensoRF.point_predict(net, allrays, device, args.bs)

    save_points(xyz_map, rgb_map, filename)
