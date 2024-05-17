import os
import sys
import json

import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import torch
from tqdm import tqdm

from csl_common.utils import geometry, ds_utils
from csl_common.utils.nn import Batch
from datasets import facedataset
import src.config as cfg
from src.useful_utils import *


def get_transform(center, scale, res, rot=0):
    """
    General image utils functions
    """
    # Generate transformation matrix
    h = 200 * scale
    t = np.zeros((3, 3))
    t[0, 0] = float(res[1]) / h
    t[1, 1] = float(res[0]) / h
    t[0, 2] = res[1] * (-float(center[0]) / h + .5)
    t[1, 2] = res[0] * (-float(center[1]) / h + .5)
    t[2, 2] = 1
    if rot != 0:
        rot = -rot  # To match direction of rotation from cropping
        rot_mat = np.zeros((3, 3))
        rot_rad = rot * np.pi / 180
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0, :2] = [cs, -sn]
        rot_mat[1, :2] = [sn, cs]
        rot_mat[2, 2] = 1
        # Need to rotate around center
        t_mat = np.eye(3)
        t_mat[0, 2] = -res[1] / 2
        t_mat[1, 2] = -res[0] / 2
        t_inv = t_mat.copy()
        t_inv[:2, 2] *= -1
        t = np.dot(t_inv, np.dot(rot_mat, np.dot(t_mat, t)))
    return t


def apply_transform3d(pt, center, scale, res, z_res, invert=0, rot=0):
    # Transform pixel location to different reference
    t = get_transform(center, scale, res, rot=rot)
    if invert:
        t = np.linalg.inv(t)
    new_pt = np.array([pt[0] - 1, pt[1] - 1, 1.]).T
    new_pt = np.dot(t, new_pt)

    h = 200 * scale
    if invert:
        new_pt_z = (h / float(z_res)) * (pt[2] - float(z_res) / 2)
    else:
        new_pt_z = (float(z_res) / h) * pt[2] + float(z_res) / 2

    new_pt[2] = new_pt_z

    return new_pt[:3]


def transform3D(lm3d, center, scale):
    # Flip z value
    lm3d[:, 2] = -lm3d[:, 2]

    # z-axis mean 0
    lm3d[:, 2] -= np.mean(lm3d[:, 2])

    # Transform 3D landmarks based on center, scale, and input size
    if center[0] != -1:
        scale *= 1.25

    tpts_inp = lm3d.copy()
    for i in range(tpts_inp.shape[0]):
        tpts_inp[i, 0:3] = apply_transform3d(tpts_inp[i, 0:3] + 1, center, scale, [256, 256], 256, 0)

    return tpts_inp


class AFLW20003D(facedataset.FaceDataset):
    """
    AFLW2000-3D Dataset.
    This dataset is used for evaluating 3D face alignment.
    The annotations include bbox, 2D FL (21), 3D FL (68) and head pose (rotation).
    http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm
    """
    CROP_SOURCES = ['bb_detector', 'bb_ground_truth', 'lm_ground_truth']

    NUM_LANDMARKS = 68
    LANDMARKS_ONLY_OUTLINE = list(range(17))
    LANDMARKS_NO_OUTLINE = list(range(17, NUM_LANDMARKS))
    ALL_LANDMARKS = list(range(NUM_LANDMARKS))

    def __init__(self, root, cache_root=None, train=False, test_split='full',
                 crop_source='bb_ground_truth', return_landmark_heatmaps=False,
                 return_modified_images=False, use_cache=False, **kwargs):

        self.root_dir = root
        self.test_split = test_split

        self.annotations_file = "csv/aflw2000_3D_anno_vd.json"
        if not os.path.isfile(self.annotations_file):
            self.annotations_file = "../csv/aflw2000_3D_anno_vd.json"

        super().__init__(root=root,
                         train=train,
                         test_split=test_split,
                         cache_root=cache_root,
                         fullsize_img_dir=root,
                         crop_source=crop_source,
                         use_cache=use_cache,
                         # return_landmark_heatmaps=False,
                         return_landmark_heatmaps=return_landmark_heatmaps,
                         return_modified_images=return_modified_images,
                         **kwargs)

        # Shuffle faces
        # self.annotations = shuffle(self.annotations)
        print(f"Number of images: {len(self.annotations)}")

        if self.crop_type == 'fullsize':
            self.transform = lambda x: x

    def __getitem__(self, idx):
        sample = self.annotations.iloc[idx]

        if "objpos" in sample and "scale" in sample:
            center = np.array(sample.objpos)
            scale = sample.scale_provided

        landmarks_3d = np.array(sample.landmarks).astype(np.float32)
        # landmarks_3d = transform3D(landmarks_3d, center, scale)

        landmarks_2d = landmarks_3d[:, :2]
        # landmarks_2d = np.array(sample.landmarks_2d).astype(np.float32)

        bb = sample.bbox
        bb = [bb[0], bb[1], bb[0] + bb[2], bb[1] + bb[3]]
        bb = geometry.extend_bbox(bb, dt=0.05, db=0.10)
        landmarks_for_crop = None
        if self.crop_source == 'lm_ground_truth':
            landmarks_for_crop = landmarks_2d

        sample = self.get_sample(
            # os.path.join(self.root_dir, sample.img_paths),
            sample.img_paths,
            bb,
            landmarks_for_crop,
            landmarks_to_return=landmarks_2d,
            landmarks_3d=landmarks_3d)

        return sample

    def __len__(self):
        return len(self.annotations)

    def _load_annotations(self, split=None):
        with open(self.annotations_file) as f:
            annotations = pd.DataFrame(json.load(f))

        annotations["img_paths"] = annotations.img_paths.str.replace('AFLW2000/', '')

        print("Split", split)
        if split !='train' and self.test_split != 'full':
            print("Entering here")
            idx = self._eval_based_on_yaw_angle()
            annotations = annotations[idx]

        # # Remove images with 23 2d landmarks - [694, 1392, 1493]
        # annotations.drop(annotations.index[[694, 1392, 1493]], inplace=True)

        # if train:
        #     return annotations.head(int(len(annotations) * 0.9))

        # return annotations.tail(int(len(annotations) * 0.1) - 1)
        return annotations

    def _eval_based_on_yaw_angle(self):
        yaw_annotation_file = "csv/aflw2000_3D_yaw.npy"
        if not os.path.isfile(yaw_annotation_file):
            yaw_annotation_file = "../csv/aflw2000_3D_yaw.npy"
        yaw_angles = np.load(yaw_annotation_file)
        yaw_angles_abs = np.abs(yaw_angles)
        list_idx = yaw_angles_abs > 0

        if self.test_split == '0_to_30':
            list_idx = yaw_angles_abs <= 30
        elif self.test_split == '30_to_60':
            print('Inside 30_to_60')
            list_idx = np.bitwise_and(yaw_angles_abs > 30, yaw_angles_abs <= 60)
        elif self.test_split == '60_to_90':
            list_idx = yaw_angles_abs > 60

        return list_idx



cfg.register_dataset(AFLW20003D)


if __name__ == '__main__':
    from csl_common.vis import vis
    import time

    # transform = ds_utils.build_transform(deterministic=False, daug=4)

    dirs = cfg.get_dataset_paths('aflw20003d')
    ds = AFLW20003D(
        root=dirs[0],
        cache_root=dirs[1],
        train=True, deterministic=True,
        crop_source='bb_ground_truth',
        use_cache=False,
        align_face_orientation=False,
        return_modified_images=False,
        # transform=transform,
        image_size=256)
    micro_batch_loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=True, num_workers=0)

    f = 1.0
    t = time.perf_counter()
    for iter, data in enumerate(tqdm(micro_batch_loader)):
        print('t load:', time.perf_counter() - t)
        t = time.perf_counter()
        batch = Batch(data)
        # batch = data
        print(batch.images.shape)
        print('t Batch:', time.perf_counter() - t)

        inputs = batch.images.clone()
        # lms_3d = batch.landmarks_3d
        imgs = vis.to_disp_images(inputs, denorm=False)
        imgs = vis.add_landmarks_to_images(imgs, batch.landmarks, radius=3, color=(0, 255, 0), draw_wireframe=True)
        # imgs = vis.add_landmarks_to_images(imgs, data['landmarks_of'].numpy(), color=(1,0,0))
        vis.vis_square(imgs, nCols=5, fx=1, fy=1, normalize=False)
