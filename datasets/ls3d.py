import os
import json

import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import torch.utils.data as td

from csl_common.utils.nn import Batch
from csl_common.utils import geometry
from datasets import facedataset
import src.config as config


class LS3D(facedataset.FaceDataset):
    """
    LS3D dataset.
    Produced using the method described in:
    How far are we from solving the 2D & 3D Face Alignment problem?(and a dataset of 230,000 3D facial landmarks)
    from Adrian Bulat and Georgios Tzimiropoulos.
    It provides 68 3DA2D landmarks.
    """
    CROP_SOURCES = ['bb_detector', 'bb_ground_truth', 'lm_ground_truth']

    NUM_LANDMARKS = 68
    LANDMARKS_ONLY_OUTLINE = list(range(17))
    LANDMARKS_NO_OUTLINE = list(range(17, NUM_LANDMARKS))
    ALL_LANDMARKS = list(range(NUM_LANDMARKS))

    def __init__(self, root, cache_root=None, train=True, test_split='full',
                 crop_source='lm_ground_truth', return_landmark_heatmaps=False,
                 return_modified_images=False, use_cache=False,
                 load_in_memory=True, tar_path="LS3D.tar", **kwargs):

        self.annotations_file = "csv/ls3d.ljson"
        if not os.path.isfile(self.annotations_file):
            self.annotations_file = "../csv/ls3d.ljson"

        super().__init__(root=root,
                         cache_root=cache_root,
                         fullsize_img_dir=root,  # os.path.join(root, 'images'),
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

        landmarks_gt = np.array(sample.landmarks).astype(np.float32)
        landmarks_3d = None
        landmarks_for_crop = None
        if self.crop_source == 'lm_ground_truth':
            landmarks_for_crop = landmarks_gt[:, :2]

        x1, x2 = np.min(landmarks_gt[:, 0]), np.max(landmarks_gt[:, 0])
        y1, y2 = np.min(landmarks_gt[:, 1]), np.max(landmarks_gt[:, 1])
        bb = [x1, y1, x2, y2]
        bb = geometry.extend_bbox(bb, dt=0.2, db=0.12)
        # bb = geometry.extend_bbox(bb, dt=0.05, db=0.10)

        # filename = str.join("/", sample.img_path.split("/")[-2:])
        if self.dataset:    # If dataset has been loaded in memory
            name_file, extension = os.path.splitext(sample.img_path)
        else:               # If files are read individually while training
            name_file = os.path.join(self.root, sample.img_path)
        sample = self.get_sample(name_file,
                                 bb,
                                 landmarks_for_crop,
                                 landmarks_to_return=landmarks_gt,
                                 landmarks_3d=landmarks_3d)

        return sample


config.register_dataset(LS3D)

if __name__ == '__main__':
    from csl_common.vis import vis
    import torch

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    dirs = config.get_dataset_paths('ls3d')
    ds = LS3D(root=dirs[0],
              cache_root=dirs[1],
              train=False,
              deterministic=True,
              use_cache=False,
              image_size=256,
              daug=0,
              align_face_orientation=False,
              crop_source='bb_ground_truth',
              # crop_source='lm_ground_truth',
              load_in_memory=True)
    dl = td.DataLoader(ds, batch_size=10, shuffle=True, num_workers=0)

    for it, data in enumerate(dl):
        batch = Batch(data, gpu=False)
        print(batch.images.shape)

        inputs = batch.images.clone()
        # lms_3d = batch.landmarks_3d
        imgs = vis.to_disp_images(inputs, denorm=False)
        imgs = vis.add_landmarks_to_images(imgs, batch.landmarks, radius=3, color=(0, 255, 0), draw_wireframe=True)
        vis.vis_square(imgs, nCols=5, fx=1, fy=1, normalize=False)
