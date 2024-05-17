import os

import numpy as np
import pandas as pd
import torch
from sklearn.utils import shuffle

from csl_common.utils import geometry
from datasets import facedataset
import src.config as cfg


class FFHQ(facedataset.FaceDataset):
    def __init__(self, root, cache_root="cache_root/", train=True, use_cache=False,
                 load_in_memory=True, tar_path="FFHQ_lite.tar", **kwargs):
        self.annotations_file = "csv/FFHQ_annotations.csv"
        self.annotations = pd.read_csv(self.annotations_file)

        # Shuffle faces
        self.annotations = shuffle(self.annotations)
        print(f"Number of images: {len(self.annotations)}")

        super().__init__(root=root,
                         cache_root=cache_root,
                         fullsize_img_dir="",
                         crop_dir="",
                         train=train,
                         use_cache=use_cache,
                         return_landmark_heatmaps=False,
                         # return_modified_images=return_modified_images,
                         load_in_memory=load_in_memory,
                         tar_path=tar_path,
                         **kwargs)

    def _load_annotations(self, split):
        return pd.read_csv(self.annotations_file)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        sample = self.annotations.iloc[idx]

        bb = [sample.x1, sample.y1, sample.x2, sample.y2]
        bb = geometry.extend_bbox(bb, dt=0.05, db=0.10)
        # bb = [x * 4 for x in bb]

        if self.dataset:    # If dataset has been loaded in memory
            name_file, extension = os.path.splitext(sample.image_path)
        else:               # If files are read individually while training
            name_file = os.path.join(self.root, sample.image_path)
        sample = self.get_sample(name_file, bb,
                                 landmarks_for_crop=None, landmarks_to_return=None,
                                 landmarks_3d=None)
        return sample


cfg.register_dataset(FFHQ)


if __name__ == '__main__':
    from csl_common.vis import vis
    import time

    dirs = cfg.get_dataset_paths('ffhq')
    ds = FFHQ(root=dirs[0],
              cache_root=dirs[1],
              train=True,
              deterministic=True,
              use_cache=False,
              align_face_orientation=False,
              return_modified_images=False,
              image_size=256,
              load_in_memory=True)
    micro_batch_loader = torch.utils.data.DataLoader(ds, batch_size=10, shuffle=True, num_workers=0)

    f = 1.0
    t = time.perf_counter()
    for iter, data in enumerate(micro_batch_loader):
        print('t load:', time.perf_counter() - t)
        t = time.perf_counter()
        batch = data
        print(batch['image'].shape)
        print('t Batch:', time.perf_counter() - t)

        inputs = batch['image'].clone()
        imgs = vis.to_disp_images(inputs, denorm=False)
        vis.vis_square(imgs, nCols=5, fx=1, fy=1, normalize=False)
