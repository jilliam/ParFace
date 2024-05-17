import os
import cv2
import glob

from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.utils.data as td

# from csl_common.utils import cropping as fp
from csl_common.utils import nn, cropping, geometry
from csl_common.utils.nn import Batch
from csl_common import utils
from csl_common.vis.vis import to_disp_image
import csl_common.utils.ds_utils as ds_utils
from datasets import palsy

from torchvision import transforms as tf
from landmarks import lmvis
from landmarks import fabrec

import src.config as config

snapshot_dir = os.path.join('.')

# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

INPUT_SIZE = 256

transforms = [utils.transforms.CenterCrop(INPUT_SIZE)]
transforms += [utils.transforms.ToTensor()]
# transforms += [utils.transforms.Normalize([0.518, 0.418, 0.361], [1, 1, 1])]
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
        print(fname, "converting RGBA to RGB...")
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    assert img.shape[2] == 3, "{}, invalid format: {}".format(img_path, img.shape)
    return img


def detect_in_crop(net, crop):
    with torch.no_grad():
        X_recon, lms_in_crop, X_lm_hm = net.detect_landmarks(crop)
    lms_in_crop = utils.nn.to_numpy(lms_in_crop.reshape(1, -1, 2))
    return X_recon, lms_in_crop, X_lm_hm


def detect_3d_in_crop(net, crop):
    with torch.no_grad():
        X_recon, lms_in_crop, X_lm_hm = net.detect_landmarks_3d(crop)
    lms_in_crop = utils.nn.to_numpy(lms_in_crop)
    return X_recon, lms_in_crop, X_lm_hm


def test_crop(net, input_image, gt_landmarks, bb_for_crop=None, lms_for_crop=None, align=False, scale=1.0):
    assert bb_for_crop is not None or lms_for_crop is not None

    cropper = cropping.FaceCrop(input_image, bbox=bb_for_crop, landmarks=lms_for_crop,
                                align_face_orientation=align, scale=scale,
                                output_size=(INPUT_SIZE, INPUT_SIZE))
    crop = cropper.apply_to_image()
    landmarks = cropper.apply_to_landmarks(gt_landmarks)[0]

    item = {'image': crop, 'landmarks': landmarks, 'pose': None}
    item = crop_to_tensor(item)

    images = nn.atleast4d(item['image']).cuda()
    X_recon, lms, X_lm_hm = detect_in_crop(net, images)

    # lmvis.visualize_batch(images, landmarks, X_recon, X_lm_hm, lms, wait=0, clean=True)
    lmvis.visualize_batch_CVPR(images, landmarks, X_recon, X_lm_hm, lms, wait=0,
                               horizontal=True, show_recon=True, radius=2, draw_wireframes=True,
                               session_name="demo_3d")


def plot_2d(im, pred, dim_pt=2):

    num_land = len(pred)
    # print(num_land)
    for i in range(num_land):
        # cv2.circle(im, (int(gt[i, 0]), int(gt[i, 1])), dim_pt, (255, 0, 0), -1)
        cv2.circle(im, (int(pred[i, 0]), int(pred[i, 1])), dim_pt, (0, 255, 0), -1)
        if num_land == 68:  # 300W
            if i < 16 or (16 < i < 21) or (21 < i < 26) or (26 < i < 30) or (30 < i < 35) or (35 < i < 41) or \
                    (41 < i < 47) or (47 < i < 59) or (59 < i < 67):
                x_next = pred[i + 1, 0]
                y_next = pred[i + 1, 1]
                cv2.line(im, (int(pred[i, 0]), int(pred[i, 1])), (int(x_next), int(y_next)), (0, 255, 0), 1)

            if (i == 36) or (i == 42) or (i == 48) or (i == 60):
                if (i == 36) or (i == 42):
                    offset = 5
                elif i == 48:
                    offset = 11
                else:
                    offset = 7
                x_next = pred[i + offset, 0]
                y_next = pred[i + offset, 1]
                cv2.line(im, (int(pred[i, 0]), int(pred[i, 1])), (int(x_next), int(y_next)), (0, 255, 0), 1)

        elif num_land == 98:  # WFLW
            if i < 32 or (32 < i < 41) or (41 < i < 50) or (50 < i < 54) or (54 < i < 59) or (59 < i < 67) or \
                    (67 < i < 75) or (75 < i < 87) or (87 < i < 95):
                x_next = pred[i + 1, 0]
                y_next = pred[i + 1, 1]
                cv2.line(im, (int(pred[i, 0]), int(pred[i, 1])), (int(x_next), int(y_next)), (0, 255, 0), 1)

            if (i == 33) or (i == 42) or (i == 60) or (i == 68) or (i == 76) or (i == 88):
                if (i == 33) or (i == 42):
                    offset = 8
                elif (i == 60) or (i == 68) or (i == 88):
                    offset = 7
                else:
                    offset = 11

                x_next = pred[i + offset, 0]
                y_next = pred[i + offset, 1]
                cv2.line(im, (int(pred[i, 0]), int(pred[i, 1])), (int(x_next), int(y_next)), (0, 255, 0), 1)

    cv2.imshow('1', im)
    cv2.waitKey(0)


def plot_3d(pred, view='frontal'):
    # Move point cloud to center
    width = (np.max(pred[:, 0]) - np.min(pred[:, 0]))
    height = (np.max(pred[:, 1]) - np.min(pred[:, 1]))
    pred_mean_2d = np.mean(pred[:, :2], axis=0)
    pred[:, :2] -= pred_mean_2d - np.array([width, height])

    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection='3d')

    # Data for a three-dimensional line
    ax.plot3D(pred[0:17, 0], pred[0:17, 1], pred[0:17, 2], 'green')
    ax.plot3D(pred[17:22, 0], pred[17:22, 1], pred[17:22, 2], 'green')
    ax.plot3D(pred[22:27, 0], pred[22:27, 1], pred[22:27, 2], 'green')
    ax.plot3D(pred[27:31, 0], pred[27:31, 1], pred[27:31, 2], 'green')
    ax.plot3D(pred[31:36, 0], pred[31:36, 1], pred[31:36, 2], 'green')

    eye_l = np.append(pred[36:42, :], [pred[36, :]], axis=0)
    ax.plot3D(eye_l[:, 0], eye_l[:, 1], eye_l[:, 2], 'green')

    eye_r = np.append(pred[42:48, :], [pred[42, :]], axis=0)
    ax.plot3D(eye_r[:, 0], eye_r[:, 1], eye_r[:, 2], 'green')

    mouth_o = np.append(pred[48:60, :], [pred[48, :]], axis=0)
    ax.plot3D(mouth_o[:, 0], mouth_o[:, 1], mouth_o[:, 2], 'green')
    mouth_i = np.append(pred[60:68, :], [pred[60, :]], axis=0)
    ax.plot3D(mouth_i[:, 0], mouth_i[:, 1], mouth_i[:, 2], 'green')

    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_zticklabels([])

    font_size=10
    ax.set_xlabel('x', fontsize=font_size, labelpad=-10)
    ax.set_ylabel('y', fontsize=font_size, labelpad=-15)
    ax.set_zlabel('z', fontsize=font_size, labelpad=-10)

    # ax.set_xlim([0, 250.])  # set axes limits
    # ax.set_ylim([0, 250.])
    # ax.set_zlim([0, 250.])
    if view == 'frontal':
        ax.view_init(180, 0, vertical_axis='y')     # Plot frontal view
    else:
        ax.view_init(170, 45, vertical_axis='y')  # Plot with side view
    plt.show()


def landmark_detector_2d(net2d, img_full, gt_full=None, bbox=None, session_name='demo_3d'):
    roi_size = geometry.get_diagonal(INPUT_SIZE)
    # cropper = cropping.FaceCrop(img_full, bbox=bbox, landmarks=None, #gt_annot[:, :2],
    cropper = cropping.FaceCrop(img_full, bbox=bbox, landmarks=gt_full,
                                align_face_orientation=False, scale=1.0,
                                output_size=roi_size)
    crop = cropper.apply_to_image()
    gt_crop = cropper.apply_to_landmarks(gt_full)[0]

    item = {'image': crop, 'landmarks': gt_crop, 'pose': None}
    item = crop_to_tensor(item)
    images = nn.atleast4d(item['image']).cuda()

    X_recon, lms, X_lm_hm = detect_in_crop(net2d, images)

    lands = lms[0]
    lands[:, 0] += int((roi_size - INPUT_SIZE) / 2)
    lands[:, 1] += int((roi_size - INPUT_SIZE) / 2)
    lands = cropper.apply_to_landmarks_inv(lands)
    return lands


def landmark_detector_3d(net3d, img_full, gt_full=None, bbox=None, session_name='demo_3d'):
    roi_size = geometry.get_diagonal(INPUT_SIZE)
    cropper = cropping.FaceCrop(img_full, bbox=bbox, landmarks=gt_full,
                                align_face_orientation=False, scale=1.0,
                                output_size=roi_size)
    crop = cropper.apply_to_image()
    gt_crop = cropper.apply_to_landmarks(gt_full)[0]

    item = {'image': crop, 'landmarks': gt_crop, 'pose': None}
    item = crop_to_tensor(item)
    images = nn.atleast4d(item['image']).cuda()

    X_recon, lms, X_lm_hm = detect_3d_in_crop(net3d, images)

    lands = lms[0]
    lands[:, 0] += int((roi_size - INPUT_SIZE) / 2)
    lands[:, 1] += int((roi_size - INPUT_SIZE) / 2)
    lands = cropper.apply_to_landmarks_inv(lands)

    return lands


if __name__ == '__main__':

    model3d = '../out/data/models/snapshots/demo_3d'
    net_3d = fabrec.load_net(model3d, num_landmarks=68)
    net_3d.zero_grad()
    net_3d.eval()

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

            # Plot 3DA-2D landmarks
            l2d = landmark_detector_2d(net_3d, img, gt_full=None, bbox=bb, session_name="demo_3d")
            plot_2d(img, l2d, dim_pt=2)

            # Plot 3D landmarks
            l3d = landmark_detector_3d(net_3d, img, gt_full=None, bbox=bb, session_name="demo_3d")
            plot_3d(l3d, view='frontal')
