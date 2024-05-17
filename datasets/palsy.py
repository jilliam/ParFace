import os
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import torch.utils.data as td

from csl_common.utils.nn import Batch
from csl_common.utils import geometry
from datasets import facedataset
import src.config as config


def gt_loader(file_path):
    land_file = file_path + '.txt'
    lms = []
    with open(land_file) as f:
        for line in f:
            try:
                x, y, z = [float(e) for e in line.split()]
                lms.append((x, y, z))
            except:
                pass
    assert (len(lms) == 68)
    landmarks = np.vstack(lms)

    return landmarks


class Palsy(facedataset.FaceDataset):
    """
    Parface Dataset.
    Videos of patients with face paralysis, collected from YouTube.
    Some sequences are annotated with 3D facial landmarks (68 FL).
    There is one tar file per sequence.
    """
    CROP_SOURCES = ['bb_detector', 'bb_ground_truth', 'lm_ground_truth']
    
    NUM_LANDMARKS = 68
    LANDMARKS_ONLY_OUTLINE = list(range(17))
    LANDMARKS_NO_OUTLINE = list(range(17, NUM_LANDMARKS))
    ALL_LANDMARKS = list(range(NUM_LANDMARKS))

    def __init__(self, root, cache_root=None, train=True, test_split='full', train_split='full',
                 crop_source='lm_ground_truth', return_landmark_heatmaps=False,
                 return_modified_images=False, use_cache=False, **kwargs):

        self.data_type = 'annotated'

        annot_path = 'csv' if os.path.isdir('csv') else '../csv'
        if test_split == 'parface':
            self.annotations_file = os.path.join(annot_path, 'palsy_test.ljson')
        elif train_split == '1':
            self.annotations_file = os.path.join(annot_path, 'palsy_labeled_1.ljson')
        elif train_split == '2':
            self.annotations_file = os.path.join(annot_path, 'palsy_labeled_2.ljson')
        elif train_split == '3':
            self.annotations_file = os.path.join(annot_path, 'palsy_labeled_3.ljson')
        elif train_split == '4':
            self.annotations_file = os.path.join(annot_path, 'palsy_labeled_4.ljson')
        elif train_split == '5':
            self.annotations_file = os.path.join(annot_path, 'palsy_labeled_5.ljson')
        elif train_split == '6':
            self.annotations_file = os.path.join(annot_path, 'palsy_labeled_6.ljson')
        elif train_split == 'unlabeled':
            self.data_type = 'unlabeled'
            self.annotations_file = os.path.join(annot_path, 'palsy_unlabeled.ljson')
        else:
            self.annotations_file = os.path.join(annot_path, "palsy_annotations.ljson")

        super().__init__(root=root,
                         cache_root=cache_root,
                         fullsize_img_dir=root,  # os.path.join(root, 'images'),
                         train=train,
                         test_split=test_split,
                         crop_source=crop_source,
                         use_cache=use_cache,
                         return_landmark_heatmaps=return_landmark_heatmaps,
                         return_modified_images=return_modified_images,
                         **kwargs)

        if self.crop_type == 'fullsize':
            self.transform = lambda x: x

    def _load_annotations(self, split=None):
        with open(self.annotations_file, "r") as f:
            data = json.load(f)

        # Shuffle faces
        annot = pd.DataFrame(data)
        # annot = shuffle(annot, random_state=None)

        return annot

    @property
    def labels(self):
        return None

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        sample = self.annotations.iloc[idx]
        bb = None

        if self.data_type == 'annotated':
            landmarks_gt = np.array(sample.landmarks_MAL3D).astype(np.float32)
            landmarks_2d = landmarks_gt[:, :2]
            landmarks_3d = landmarks_gt
        else:  # if self.data_type == 'unlabeled':
            landmarks_gt = np.array(sample.landmarks_FAN).astype(np.float32)
            landmarks_2d = None
            landmarks_3d = None

        # self.crop_source = "lm_ground_truth"
        landmarks_for_crop = None
        if self.crop_source == 'lm_ground_truth':
            landmarks_for_crop = landmarks_gt[:, :2]

        # return self.get_sample(sample.img_path, bb, landmarks_for_crop, landmarks_to_return=landmarks_for_crop)

        x1, x2 = np.min(landmarks_gt[:, 0]), np.max(landmarks_gt[:, 0])
        y1, y2 = np.min(landmarks_gt[:, 1]), np.max(landmarks_gt[:, 1])
        bb = [x1, y1, x2, y2]
        bb = geometry.extend_bbox(bb, dt=0.05, db=0.10)
        # print(sample.img_path, x1, y1, x2, y2,landmarks_gt[:, :2])

        filename = str.join("/", sample.img_path.split("/")[-2:])
        # print(landmarks_for_crop)
        # sample = self.get_sample(os.path.join(self.root, sample.img_path),
        sample = self.get_sample(os.path.join(self.root, filename),
                                 bb,
                                 landmarks_for_crop,
                                 landmarks_to_return=landmarks_2d,
                                 landmarks_3d=landmarks_3d)

        return sample
        

config.register_dataset(Palsy)

if __name__ == '__main__':
    from csl_common.vis import vis
    import torch

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    dirs = config.get_dataset_paths('palsy')
    ds = Palsy(root=dirs[0],
               cache_root=dirs[1],
               train=False,
               deterministic=True,
               use_cache=False,
               image_size=256,
               train_split='6',
               # test_split='parface',
               daug=0,
               align_face_orientation=False,
               crop_source='lm_ground_truth')
    dl = td.DataLoader(ds, batch_size=10, shuffle=True, num_workers=0)

    for it, data in enumerate(dl):
        batch = Batch(data, gpu=False)
        # print(batch.images.shape)

        inputs = batch.images.clone()
        # lms_3d = batch.landmarks_3d
        imgs = vis.to_disp_images(inputs, denorm=False)
        imgs = vis.add_landmarks_to_images(imgs, batch.landmarks, radius=3, color=(0, 255, 0), draw_wireframe=True)
        # imgs = vis.add_landmarks_to_images(imgs, data['landmarks_of'].numpy(), color=(1,0,0))
        vis.vis_square(imgs, nCols=5, fx=1, fy=1, normalize=False)
