"""
Reference:
https://arxiv.org/pdf/2312.08426
"""


from __future__ import annotations

import math
from typing import Callable, Dict, List

import torch

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Add to the top of visualize/single_qubit_visualization.py
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from train.single_qubit.single_qubit_script import *


angle_vec_dict = {
    1/4 : [1.34820, 1.32669, 1.77042, 2.16800],
    1/3 : [1.41901, 1.35864, 1.77664, 2.13759],
    1/2 : [1.55280, 1.42267, 1.78586, 2.07559],
    2/3 : [1.67478, 1.47865, 1.78919, 2.02043],
    3/4 : [1.73053, 1.49972, 1.78853, 1.99939],
    1   : [1.87342, 1.52524, 1.78436, 1.97330]
}

unitaries = {
    "X(pi)" : [(1, 0)],
    "X(pi/2)" : [(1/2, 0)],
    "H" : [(1, 0), (1/2, 1/2)],
    "Z(pi/4)" : [(1/2, 0), (1/4, 1/2), (1/2, 0)]
}


def SCOREn_config(n, phi):
    angle_vec = angle_vec_dict[n]
    config = []
    Angle = np.pi * n

    for i, angle in enumerate(angle_vec):
        config.append({
            "phi": torch.tensor([phi + (i % 2) * np.pi], dtype=torch.complex64),
            "theta": torch.tensor([np.pi/2], dtype=torch.complex64),
            "angle": torch.tensor([angle * np.pi], dtype=torch.complex64)
        })
        Angle += (-1)**(len(angle_vec) - 1 - i) * 2 * angle * np.pi

    config.append({
        "phi": torch.tensor([phi], dtype=torch.complex64),
        "theta": torch.tensor([np.pi/2], dtype=torch.complex64),
        "angle": torch.tensor([Angle], dtype=torch.complex64)
    })

    for i, angle in reversed(list(enumerate(angle_vec))):
        config.append({
            "phi": torch.tensor([phi + (i % 2) * np.pi], dtype=torch.complex64),
            "theta": torch.tensor([np.pi/2], dtype=torch.complex64),
            "angle": torch.tensor([angle * np.pi], dtype=torch.complex64)
        })

    config_to_tensor = [
        [0, 1, x["phi"], torch.sin(x["theta"]) * x["angle"]]
        for x in config
    ]

    return torch.tensor(config_to_tensor)


def build_SCORE_pulses():
    SCORE_pulses = []

    for target in unitaries:
        pulses = []

        for n, phi in reversed(unitaries[target]):
            pulses.append(SCOREn_config(n, phi * np.pi))
        
        SCORE_pulses.append(torch.stack(pulses))

    return [x.reshape(-1, x.shape[-1]) for x in SCORE_pulses]

