import json
import os

import numpy as np
import pandas as pd
import torch
from sklearn.utils import shuffle

import src.config as cfg

from csl_common.utils import geometry, ds_utils
from csl_common.utils.nn import Batch
from datasets import facedataset
from src.useful_utils import *


class WLP300(facedataset.FaceDataset):
    CROP_SOURCES = ['bb_detector', 'bb_ground_truth', 'lm_ground_truth']

    NUM_LANDMARKS = 68
    LANDMARKS_ONLY_OUTLINE = list(range(17))
    LANDMARKS_NO_OUTLINE = list(range(17, NUM_LANDMARKS))
    ALL_LANDMARKS = list(range(NUM_LANDMARKS))

    def __init__(self, root, cache_root=None, train=True, test_split='full',
                 crop_source='bb_ground_truth', return_landmark_heatmaps=False,
                 return_modified_images=False, use_cache=False, **kwargs):

        self.root_dir = root

        self.annotations_file = "csv/300wLP_anno_tr.json"
        if not os.path.isfile(self.annotations_file):
            self.annotations_file = "../csv/300wLP_anno_tr.json"
        self.annotations = self._load_annotations(train)

        # Shuffle faces
        self.annotations = shuffle(self.annotations)
        print(f"Number of images: {len(self.annotations)}")

        super().__init__(root=root,
                         cache_root=cache_root,
                         fullsize_img_dir=root,
                         crop_source=crop_source,
                         use_cache=use_cache,
                         # return_landmark_heatmaps=False,
                         return_landmark_heatmaps=return_landmark_heatmaps,
                         return_modified_images=return_modified_images,
                         **kwargs)

        if self.crop_type == 'fullsize':
            self.transform = lambda x: x

    def __getitem__(self, idx):
        sample = self.annotations.iloc[idx]

        landmarks_3d = np.array(sample.landmarks).astype(np.float32)
        landmarks_2d = landmarks_3d[:, :2]
        landmarks_gt = np.array(sample.landmarks_2d).astype(np.float32)

        bb = sample.bbox
        bb = [bb[0], bb[1], bb[0] + bb[2], bb[1] + bb[3]]
        bb = geometry.extend_bbox(bb, dt=0.05, db=0.10)

        landmarks_for_crop = None
        if self.crop_source == 'lm_ground_truth':
            landmarks_for_crop = landmarks_gt

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

    def _load_annotations(self, train):
        with open(self.annotations_file) as f:
            annotations = pd.DataFrame(json.load(f))

        if train:
            return annotations.head(int(len(annotations) * 0.9))

        return annotations.tail(int(len(annotations) * 0.1) - 1)


cfg.register_dataset(WLP300)

if __name__ == '__main__':
    from csl_common.vis import vis
    import time

    transform = ds_utils.build_transform(deterministic=False, daug=4)

    dirs = cfg.get_dataset_paths('wlp300')
    ds = WLP300(
        root=dirs[0],
        cache_root=dirs[1],
        train=True, deterministic=True,
        crop_source='lm_ground_truth',
        use_cache=False,
        align_face_orientation=False,
        return_modified_images=False,
        transform=transform,
        image_size=256)
    micro_batch_loader = torch.utils.data.DataLoader(ds, batch_size=10, shuffle=False, num_workers=0)

    f = 1.0
    t = time.perf_counter()
    for iter, data in enumerate(micro_batch_loader):
        print('t load:', time.perf_counter() - t)
        t = time.perf_counter()
        batch = Batch(data)
        # print(batch['image'].shape)
        # print('t Batch:', time.perf_counter() - t)

        inputs = batch.images.clone()
        imgs = vis.to_disp_images(inputs, denorm=False)
        imgs = vis.add_landmarks_to_images(imgs, batch.landmarks, radius=3, color=(0, 255, 0), draw_wireframe=True)
        vis.vis_square(imgs, nCols=5, fx=1, fy=1, normalize=False)
