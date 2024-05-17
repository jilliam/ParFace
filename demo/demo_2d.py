import os
import cv2
import glob
import numpy as np

import torch
import torch.utils.data as td
from torchvision import transforms as tf

# from utils import cropping as fp
from csl_common.utils.nn import Batch
from csl_common.utils import nn, cropping
from csl_common import utils
from csl_common.vis import vis
from datasets import aflw20003d, palsy

from src.face_vis import draw_results
import src.config as config

from landmarks import fabrec
from landmarks import lmvis
from networks.aae import vis_reconstruction, load_net

snapshot_dir = os.path.join('ims')

INPUT_SIZE = 256

transforms = [utils.transforms.CenterCrop(INPUT_SIZE)]
transforms += [utils.transforms.ToTensor()]
transforms += [utils.transforms.Normalize([0.518, 0.418, 0.361], [1, 1, 1])]
crop_to_tensor = tf.Compose(transforms)


def load_image(im_dir, fname=None):
    from skimage import io
    if fname is None:
        img_path = im_dir
    else:
        img_path = os.path.join(im_dir, fname)
    img = io.imread(img_path)
    if img is None:
        raise IOError("\tError: Could not load image {}!".format(img_path))
    if len(img.shape) == 2 or img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[2] == 4:
        print(img_path, "converting RGBA to RGB...")
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    assert img.shape[2] == 3, "{}, invalid format: {}".format(img_path, img.shape)
    return img


def detect_in_crop(net, crop):
    with torch.no_grad():
        X_recon, lms_in_crop, X_lm_hm = net.detect_landmarks(crop)
    lms_in_crop = utils.nn.to_numpy(lms_in_crop) #.reshape(1, -1, 2))
    return X_recon, lms_in_crop, X_lm_hm


def test_crop(net, input_image, gt_landmarks, bb_for_crop=None, lms_for_crop=None, align=False, scale=1.0):
    assert bb_for_crop is not None or lms_for_crop is not None

    print("Cropping image")
    cropper = cropping.FaceCrop(input_image, bbox=bb_for_crop, landmarks=lms_for_crop,
                                align_face_orientation=align, scale=scale,
                                output_size=(INPUT_SIZE, INPUT_SIZE))
    crop = cropper.apply_to_image()
    landmarks = cropper.apply_to_landmarks(gt_landmarks)[0]

    item = {'image': crop, 'landmarks': landmarks, 'pose': None}
    item = crop_to_tensor(item)
    images = nn.atleast4d(item['image']).cuda()

    X_recon, lms, X_lm_hm = detect_in_crop(net, images)

    lmvis.visualize_batch_CVPR(images, landmarks, X_recon, X_lm_hm, lms, wait=0,
                               horizontal=False, f=10.0, show_recon=True, radius=2, draw_wireframes=True,
                               session_name='demo', normalized=True)

    # # reconstruct images from path
    # print("Reconstructing image from path")
    # f = 1 if images.shape[-1] < 512 else 0.5
    # out = vis_reconstruction(net,
    #                          images,
    #                          landmarks=None,
    #                          ncols=10,
    #                          fx=f, fy=f, denorm=True)
    #
    # print("Saving output")
    # img_filepath = os.path.join(out_dir, 'reconstructions', out_file)
    # os.makedirs(os.path.join(out_dir, 'reconstructions'), exist_ok=True)
    # cv2.imwrite(img_filepath, cv2.cvtColor(out, cv2.COLOR_RGB2BGR))


if __name__ == '__main__':

    model = '../out/data/models/snapshots/demo_2d'
    net = fabrec.load_net(model, num_landmarks=68)
    net.eval()

    im_dir = './ims'
    ext = ['png', 'jpg']
    files = []
    [files.extend(glob.glob(os.path.join(im_dir, '*.' + e))) for e in ext]
    files.sort()

    with torch.no_grad():

        for img_name in files:
            img = load_image(img_name)

            scalef = 1.0
            bb0 = [0, 0] + list(img.shape[:2][::-1])
            bb = utils.geometry.scaleBB(bb0, scalef, scalef, typeBB=2)
            test_crop(net, img, gt_landmarks=None, bb_for_crop=bb)

