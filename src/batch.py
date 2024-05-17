import json
import sys
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from tqdm import tqdm
import torch.utils.data as td

import src.config
from datasets.ffhq_lite import FFHQ
from datasets.menpo import Menpo
from datasets.menpo2d import Menpo2D
from datasets.palsy import Palsy
from datasets.wlp300 import WLP300
from landmarks import lmutils
from landmarks.lmutils import calc_landmark_nme
from src.useful_utils import *
import torch
from datasets.aflw20003d import AFLW20003D
from datasets.wflw import WFLW
from csl_common.utils.nn import Batch
import pandas as pd
import os
import glob
from landmarks.fabrec import load_net

from csl_common.utils import nn, cropping
from csl_common import utils
from torchvision import transforms as tf

INPUT_SIZE = 256

transforms = [utils.transforms.CenterCrop(INPUT_SIZE)]
transforms += [utils.transforms.ToTensor()]
transforms += [utils.transforms.Normalize([0.518, 0.418, 0.361], [1, 1, 1])]
crop_to_tensor = tf.Compose(transforms)
transforms = tf.Compose(
    [
        utils.transforms.CenterCrop(INPUT_SIZE),
        utils.transforms.Normalize([0.518, 0.418, 0.361], [1, 1, 1]),
        # utils.transforms.ToTensor(),
    ]
)


def main():
    pass


if __name__ == '__main__':
    main()
