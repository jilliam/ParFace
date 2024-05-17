import os
import glob

import json
import numpy as np
import pandas as pd
import torch.utils.data as td

from csl_common.utils.nn import Batch
from csl_common.utils import geometry, ds_utils
from datasets import facedataset
import src.config as config
from src.useful_utils import *


def read_landmarks_2d(path, is_unsupervised):
    lms = []
    with open(path) as f:
        for line in f:
            try:
                x, y = [float(e) for e in line.split()]
                lms.append((x, y))
            except:
                pass

    num_missing_land = 68 - len(lms)
    lands_complete = False if num_missing_land > 0 else True
    # if not is_unsupervised:
    #     assert (len(lms) == 68)
    # else:
    if is_unsupervised:
        for i in range(num_missing_land):
            lms.append([lms[0][0], lms[0][1]])

    return np.vstack(lms), lands_complete


class Menpo2D(facedataset.FaceDataset):
    """
    Collection of Menpo and 300W challenges
    The class retrieves 2D landmarks.
    """
    CROP_SOURCES = ['bb_detector', 'bb_ground_truth', 'lm_ground_truth']

    NUM_LANDMARKS = 68
    LANDMARKS_ONLY_OUTLINE = list(range(17))
    LANDMARKS_NO_OUTLINE = list(range(17, NUM_LANDMARKS))
    ALL_LANDMARKS = list(range(NUM_LANDMARKS))

    def __init__(self, root, cache_root=None, train=True, test_split='full',
                 crop_source='bb_ground_truth', return_landmark_heatmaps=False,
                 return_modified_images=False, unsupervised_stage=False, use_cache=False,
                 load_in_memory=True, tar_path="menpo.tar", **kwargs):

        if train and unsupervised_stage:
            tar_path = "menpo3D84.tar"
            self.annotations_file = "csv/menpo3D84.ljson"
            if not os.path.isfile(self.annotations_file):
                self.annotations_file = "../csv/menpo3D84.ljson"
        elif train:
            self.annotations_file = "csv/menpo_train.ljson"
            if not os.path.isfile(self.annotations_file):
                self.annotations_file = "../csv/menpo_train.ljson"
        elif test_split == 'common':
            tar_path = "menpo_test.tar"
            self.annotations_file = "csv/menpo_common.ljson"
            if not os.path.isfile(self.annotations_file):
                self.annotations_file = "../csv/menpo_common.ljson"
        elif test_split == 'challenging':
            tar_path = "menpo_test.tar"
            self.annotations_file = "csv/menpo_challenging.ljson"
            if not os.path.isfile(self.annotations_file):
                self.annotations_file = "../csv/menpo_challenging.ljson"
        elif test_split == 'full':
            tar_path = "menpo_test.tar"
            self.annotations_file = "csv/menpo_full.ljson"
            if not os.path.isfile(self.annotations_file):
                self.annotations_file = "../csv/menpo_full.ljson"
        elif test_split == '300w':
            tar_path = "300W.tar"
            self.annotations_file = "csv/300W.ljson"
            if not os.path.isfile(self.annotations_file):
                self.annotations_file = "../csv/300W.ljson"
        else:
            print("Menpo configuration not found")

        self.is_unsupervised = unsupervised_stage

        super().__init__(root=root,
                         cache_root=cache_root,
                         fullsize_img_dir=root,
                         train=train,
                         test_split=test_split,
                         crop_source=crop_source,
                         use_cache=use_cache,
                         return_landmark_heatmaps=return_landmark_heatmaps,
                         return_modified_images=return_modified_images,
                         load_in_memory=load_in_memory,
                         tar_path=tar_path,
                         **kwargs)

        if self.crop_type == 'fullsize':
            self.transform = lambda x: x

    def _load_annotations(self, split=None):
        with open(self.annotations_file, "r") as f:
            data_annot = json.load(f)

        # Shuffle faces
        annot = pd.DataFrame(data_annot)
        # annot = shuffle(annot, random_state=None)

        return annot

    def _load_annotations2(self, split):
        if self.is_unsupervised:
            split_defs = {
                'train': [
                    ('images/menpo3D84', None),
                    ('images/HELEN_trainset', None),
                    ('images/LFPW_trainset', None),
                    ('images/AFW', None),
                ]
            }
        else:
            split_defs = {
                'train': [
                    # ('images/menpo3D84', None),
                    ('images/HELEN_trainset', None),
                    ('images/LFPW_trainset', None),
                    ('images/AFW', None),
                ],
                'common': [
                    ('images/HELEN_testset', None),
                    ('images/LFPW_testset', None)
                ],
                'challenging': [
                    ('images/IBUG', None)
                ],
                'full': [
                    ('images/HELEN_testset', None),
                    ('images/LFPW_testset', None),
                    ('images/IBUG', None)
                ],
                '300w': [
                    ('images/300W', None),
                ]
            }

        ann = []

        for subset in split_defs[split]:
            im_dir, bbox_file_suffix = subset
            lm_dir = os.path.join(im_dir.replace("images", "annotations"), "gt_original2D")
            lm_dir_abs = os.path.join(self.root, lm_dir)

            # get image file paths and read GT landmarks
            ext = "*.jpg"
            if 'LFPW' in im_dir or '300W' in im_dir:
                ext = "*.png"
            for img_file in sorted(glob.glob(os.path.join(self.fullsize_img_dir, im_dir, ext))):
                path_abs_noext = os.path.splitext(img_file)[0]
                filename_noext = os.path.split(path_abs_noext)[1]
                filename = os.path.split(img_file)[1]
                path_rel = os.path.join(im_dir, filename)

                # load landmarks from *.pts files
                lm_path = os.path.join(lm_dir_abs, filename_noext + ".pts")
                landmarks, lands_complete = read_landmarks_2d(lm_path, self.is_unsupervised)
                if self.is_unsupervised or (not self.is_unsupervised and lands_complete):
                    ann.append(
                        {
                            'imgName': str(filename),
                            'fname': path_rel,
                            'landmarks': landmarks,
                        }
                    )

        return pd.DataFrame(ann)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        sample = self.annotations.iloc[idx]

        # landmarks = sample.landmarks.astype(np.float32)               # Landmarks read from .pts
        # landmarks = np.array(sample.landmarks_is).astype(np.float32)  # Menpo Image space annot
        # temp = landmarks[:, 0].copy()
        # landmarks[:, 0] = landmarks[:, 1]
        # landmarks[:, 1] = temp

        landmarks = np.array(sample.landmarks_2d).astype(np.float32)
        landmarks_for_crop = landmarks if self.crop_source == 'lm_ground_truth' else None

        # bb = sample.bb_detector if self.crop_source == 'bb_detector' else sample.bb_ground_truth
        x1, x2 = np.min(landmarks[:, 0]), np.max(landmarks[:, 0])
        y1, y2 = np.min(landmarks[:, 1]), np.max(landmarks[:, 1])
        bb = [x1, y1, x2, y2]
        bb = geometry.extend_bbox(bb, dt=0.2, db=0.12)
        # bb = geometry.extend_bbox(bb, dt=0.3, db=0.1)
        # print(sample.fname, bb)

        if self.is_unsupervised and landmarks.shape[0] == 39:
            landmarks = np.pad(landmarks, ((0, 29), (0, 0)), mode='constant', constant_values=0)

        # return self.get_sample(sample.fname,
        # return self.get_sample(os.path.join(self.root, sample.fname),
        name_type, extension = os.path.splitext(sample.img_path)
        name_file, type_file = os.path.splitext(name_type)
        return self.get_sample(name_file,
                               bb,
                               landmarks_for_crop,  # =None,
                               landmarks_to_return=landmarks,
                               landmarks_3d=None)


config.register_dataset(Menpo2D)

if __name__ == '__main__':
    from csl_common.vis import vis
    import torch

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    transform = ds_utils.build_transform(deterministic=False, daug=4)

    dirs = config.get_dataset_paths('menpo2d')
    ds = Menpo2D(root=dirs[0],
                 cache_root=dirs[1],
                 train=True,
                 # max_samples=300,
                 deterministic=True,
                 use_cache=False,
                 start=None,
                 test_split='challenging',
                 # test_split='common',
                 daug=0,
                 align_face_orientation=False,
                 unsupervised_stage=True,
                 crop_source='bb_ground_truth',
                 # crop_source='lm_ground_truth',
                 return_landmark_heatmaps=False,
                 with_occlusions=False,
                 landmark_sigma=7,
                 transform=transform,
                 image_size=256)
    dl = td.DataLoader(ds, batch_size=10, shuffle=True, num_workers=0)

    for it, data in enumerate(dl):
        batch = Batch(data, gpu=False)
        inputs = batch.images.clone()
        # lms_3d = batch.landmarks_3d
        imgs = vis.to_disp_images(inputs, denorm=False)
        imgs = vis.add_landmarks_to_images(imgs, batch.landmarks, radius=3, color=(0, 255, 0), draw_wireframe=True)
        vis.vis_square(imgs, nCols=5, fx=1, fy=1, normalize=False)
