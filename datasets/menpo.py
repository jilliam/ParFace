import os
import glob

import numpy as np
import pandas as pd
import torch.utils.data as td

from csl_common.utils.nn import Batch
from csl_common.utils import geometry
from datasets import facedataset
import src.config as config
from src.useful_utils import *


def read_landmarks_2d(path):
    lms = []
    with open(path) as f:
        for line in f:
            try:
                x, y = [float(e) for e in line.split()]
                lms.append((x, y))
            except:
                pass
    assert (len(lms) == 68)
    return np.vstack(lms)


def read_landmarks(path):
    import json
    import matplotlib.pyplot as plt
    with open(path) as f:
        data = json.load(f)
    landmarks = np.array(data["landmarks"]["points"])
    if landmarks.shape[1] == 2:
        landmarks[:, 0], landmarks[:, 1] = landmarks[:, 1], landmarks[:, 0].copy()
    return landmarks


class Menpo(facedataset.FaceDataset):
    CROP_SOURCES = ['bb_detector', 'bb_ground_truth', 'lm_ground_truth']

    NUM_LANDMARKS = 84
    LANDMARKS_ONLY_OUTLINE = list(range(33))
    LANDMARKS_NO_OUTLINE = list(range(33, NUM_LANDMARKS))
    ALL_LANDMARKS = list(range(NUM_LANDMARKS))

    def __init__(self, root, cache_root=None, train=True, test_split='full',
                 crop_source='bb_detector', return_landmark_heatmaps=False,
                 return_modified_images=False, use_cache=False, **kwargs):

        super().__init__(root=root,
                         cache_root=cache_root,
                         # fullsize_img_dir=os.path.join(root, 'images'),
                         fullsize_img_dir=root,
                         train=train,
                         test_split=test_split,
                         crop_source=crop_source,
                         use_cache=use_cache,
                         return_landmark_heatmaps=return_landmark_heatmaps,
                         return_modified_images=return_modified_images,
                         **kwargs)

        if self.crop_type == 'fullsize':
            self.transform = lambda x: x

    def _load_annotations(self, split):
        split_defs = {
            'train': [
                ('images/menpo3D84', None),
                ('images/HELEN_trainset', None),
                ('images/LFPW_trainset', None),
                ('images/AFW', None),
            ],
            'common': [
                ('images/HELEN_testset', None),
                ('images/LFPW_trainset', None)
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

        bboxes = []
        for id, subset in enumerate(split_defs[split]):
            im_dir, bbox_file_suffix = subset
            lm_dir = os.path.join(im_dir.replace("images", "annotations"), "gt_projected_image_space")

            # Full 3D landmarks
            lm_3d_dir = os.path.join(im_dir.replace("images", "annotations"), "gt_model_space")

            lm_dir_abs = "/ds/images/face/landmarks/FaceAlignment/" + lm_dir
            # get image file paths and read GT landmarks
            ext = "*.jpg"
            if 'LFPW' in im_dir or '300W' in im_dir:
                ext = "*.png"
            for img_file in sorted(glob.glob(os.path.join(self.fullsize_img_dir, im_dir, ext))):
                path_abs_noext = os.path.splitext(img_file)[0]
                filename_noext = os.path.split(path_abs_noext)[1]
                filename = os.path.split(img_file)[1]
                path_rel = os.path.join(im_dir, filename)

                lm_path = os.path.join(lm_dir_abs, filename_noext + ".ljson")
                # lm_3d_path = os.path.join(lm_3d_dir_abs, filename_noext + ".ljson")
                # load landmarks from *.ljson files
                landmarks = read_landmarks(lm_path)
                # landmarks_3d = read_landmarks(lm_3d_path)
                ann.append(
                    {
                        'imgName': str(filename),
                        'fname': path_rel,
                        'landmarks': landmarks,
                        # 'landmarks_3d': landmarks_3d
                    }
                )

        annotations = pd.DataFrame(ann)

        return annotations

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        sample = self.annotations.iloc[idx]
        # bb = sample.bb_detector if self.crop_source == 'bb_detector' else sample.bb_ground_truth
        # bb = geometry.extend_bbox(bb, dt=0.2, db=0.12)
        bb = None
        landmarks = sample.landmarks.astype(np.float32)
        # landmarks_3d = sample.landmarks_3d.astype(np.float32)
        landmarks_for_crop = None
        if self.crop_source == 'lm_ground_truth':
            landmarks_for_crop = landmarks
        return self.get_sample(sample.fname,
                               bb,
                               landmarks_for_crop,
                               landmarks_to_return=landmarks,
                               # landmarks_3d=landmarks_3d
                               )


config.register_dataset(Menpo)

if __name__ == '__main__':
    from csl_common.vis import vis
    import torch

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    dirs = config.get_dataset_paths('menpo')
    # ds = Menpo(root=dirs[0], cache_root=dirs[1], train=True, deterministic=True, use_cache=False, image_size=256,
    #            test_split='common', daug=0, align_face_orientation=True, crop_source='lm_ground_truth',
    #            return_landmark_heatmaps=True)

    ds = Menpo(root=dirs[0],
               cache_root=dirs[1],
               train=True,
               max_samples=300,
               use_cache=True,
               start=None,
               test_split='common',
               align_face_orientation=False,
               # crop_source='bb_ground_truth',
               crop_source='lm_ground_truth',
               return_landmark_heatmaps=True,
               with_occlusions=False,
               landmark_sigma=7,
               # transform=transform,
               image_size=256)
    dl = td.DataLoader(ds, batch_size=10, shuffle=True, num_workers=0)

    for data in dl:
        batch = Batch(data, gpu=False)
        inputs = batch.images.clone()
        lms_3d = batch.landmarks_3d
        imgs = vis.to_disp_images(inputs, denorm=True)
        imgs = vis.add_landmarks_to_images(imgs, batch.landmarks, radius=3, color=(0, 255, 0), draw_wireframe=True)
        # imgs = vis.add_landmarks_to_images(imgs, data['landmarks_of'].numpy(), color=(1,0,0))
        vis.vis_square(imgs, nCols=5, fx=1, fy=1, normalize=False)
    print(lms_3d.shape)
