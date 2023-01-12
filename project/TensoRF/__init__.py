"""Image/Video Autops Package."""  # coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2023(18588220928@163.com) All Rights Reserved.
# ***
# ***    File Author: Dell, 2023年 01月 12日 星期四 23:31:37 CST
# ***
# ************************************************************************************/
#

__version__ = "1.0.0"

import os
import torch

import todos
from . import tensorf, dataset

import pdb


def get_model():
    """Create model."""
    model = tensorf.TensorVMSplit()
    if os.path.exists("models/lego_tensorf.pth"):
        model.load_weights(model_path="models/lego_tensorf.pth")

    device = todos.model.get_device()
    model = model.to(device)
    # model.eval()

    # print(f"Running model on {device} ...")
    # model = torch.jit.script(model)
    # todos.data.mkdir("output")
    # if not os.path.exists("output/lego_tensorf.torch"):
    #     model.save("output/lego_tensorf.torch")

    return model, device


def get_dataset():
    return dataset.BlenderDataset("data/lego", split="train")
