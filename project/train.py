# coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2023(18588220928@163.com), All Rights Reserved.
# ***
# ***    File Author: Dell, 2023年 01月 12日 星期四 23:31:37 CST
# ***
# ************************************************************************************/
#
import argparse
import os
import pdb  # For debug
import torch
import torch.optim as optim

import TensoRF

if __name__ == "__main__":
    """Trainning model."""

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--outputdir", type=str, default="output", help="output directory")
    parser.add_argument("--checkpoint", type=str, default="output/model.pth", help="checkpoint file")
    parser.add_argument("--bs", type=int, default=4096, help="batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()

    # Create directory to store result
    if not os.path.exists(args.outputdir):
        os.makedirs(args.outputdir)

    # Step 1: get data loader
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #
    #     class name file MUST BE created for net
    #     please see load_class_names, save_class_names
    #
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    train_dl = TensoRF.train_data(bs=args.bs)
    valid_dl = TensoRF.valid_data(bs=args.bs)

    # Step 2: get net
    net, device = TensoRF.get_model(args.checkpoint, aabb=train_dl.dataset.scene_bbox)

    #
    # /************************************************************************************
    # ***
    # ***    MS: Construct Optimizer and Learning Rate Scheduler
    # ***
    # ************************************************************************************/
    #
    # params = [p for p in net.parameters() if p.requires_grad]
    # optimizer = optim.SGD(params, lr=args.lr, momentum=0.9)

    grad_vars = net.get_optparam_groups()
    optimizer = torch.optim.Adam(grad_vars, betas=(0.9, 0.99))
    opt_step_size = (args.epochs + 2) // 3  # from 0.01 -> 0.001, 0.0001, 0.00001
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=opt_step_size, gamma=0.1)

    for epoch in range(args.epochs):
        print("Epoch {}/{}, learning rate: {:.8f} ...".format(epoch + 1, args.epochs, lr_scheduler.get_last_lr()[0]))
        TensoRF.train_epoch(train_dl, net, optimizer, device, tag="train")
        TensoRF.valid_epoch(valid_dl, net, device, tag="valid")

        lr_scheduler.step()

        #
        # /************************************************************************************
        # ***
        # ***    MS: Define Save Model Strategy
        # ***
        # ************************************************************************************/
        #
        if epoch == (args.epochs // 2) or (epoch == args.epochs - 1):
            print("Saving model ...")
            torch.save(net.state_dict(), args.checkpoint)
