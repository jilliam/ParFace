import os

import pandas as pd
from sklearn.utils import shuffle
import torch

from csl_common.utils import geometry
from datasets import facedataset
import src.config as cfg


class Celeba(facedataset.FaceDataset):
    def __init__(self, root, cache_root="cache_root/", train=True, use_cache=False, **kwargs):
        self.annotations_file = "csv/celeba_annotations.csv"
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
                         **kwargs)

    def __getitem__(self, idx):
        sample = self.annotations.iloc[idx]

        bb = [sample.x1, sample.y1, sample.x2, sample.y2]
        bb = geometry.extend_bbox(bb, dt=0.05, db=0.10)

        sample = self.get_sample(os.path.join(self.root, sample.image_path), bb,
                                 landmarks_for_crop=None, landmarks_to_return=None,
                                 landmarks_3d=None)
        return sample

    def __len__(self):
        return len(self.annotations)

    def _load_annotations(self, split):
        return pd.read_csv(self.annotations_file)


cfg.register_dataset(Celeba)


if __name__ == '__main__':
    import time

    dirs = cfg.get_dataset_paths('celeba')
    ds = Celeba(root=dirs[0], train=True, deterministic=True, use_cache=False,
                align_face_orientation=False,
                return_modified_images=False, image_size=256)
    micro_batch_loader = torch.utils.data.DataLoader(ds, batch_size=10, shuffle=True, num_workers=0)

    f = 1.0
    t = time.perf_counter()
    for iter, data in enumerate(micro_batch_loader):
        print('t load:', time.perf_counter() - t)
        t = time.perf_counter()
        batch = data
        print(batch['image'].shape)
        print('t Batch:', time.perf_counter() - t)
