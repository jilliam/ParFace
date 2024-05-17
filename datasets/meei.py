import os

import pandas as pd
from sklearn.utils import shuffle
import torch

from csl_common.utils import geometry
from datasets import facedataset
import src.config as cfg


class MEEI(facedataset.FaceDataset):
    """
    MEEI Facial Palsy Photo and Video Standard Set (no facial landmark annotations).
    Standardized set of facial photographs and videos representing the entire spectrum
    of flaccid and nonflaccid (aberrantly regenerated or synkinetic) facial palsy.
    The degree of facial palsy was quantified using eFACE, House-Brackmann, and Sunnybrook scales.
    There are 5 categories for flaccid and nonflaccid facial palsy.
    The photos are stored in flaccid.tar, normal.tar and synkinetic.tar.
    The video frames are stored in the other tar files, where each tar has a sequence per patient.
    """
    def __init__(self, root, cache_root="cache_root/", train=True, use_cache=False,
                 load_in_memory=True, tar_path="multi", **kwargs):
        self.annotations_file = "csv/MEEI_annotations.csv"
        self.annotations = self._load_annotations(train)

        # Shuffle faces
        self.annotations = shuffle(self.annotations)
        print(f"Number of images: {len(self.annotations)}")

        super().__init__(root=root,
                         cache_root=cache_root,
                         fullsize_img_dir="",
                         train=train,
                         crop_dir="",
                         use_cache=use_cache,
                         return_landmark_heatmaps=False,
                         # return_modified_images=return_modified_images,
                         load_in_memory=load_in_memory,
                         tar_path=tar_path,
                         **kwargs)

    def _load_annotations(self, split=None):
        return pd.read_csv(self.annotations_file)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        sample = self.annotations.iloc[idx]

        bb = [sample.x1, sample.y1, sample.x2, sample.y2]
        bb = geometry.extend_bbox(bb, dt=0.05, db=0.10)

        if self.dataset:    # If dataset has been loaded in memory
            name_file = sample.image_path.split(os.extsep)[:-2]
            name_file = '.'.join(name_file)
        else:               # If files are read individually while training
            name_file = os.path.join(self.root, sample.img_path)
        sample = self.get_sample(name_file, bb,
                                 landmarks_for_crop=None, landmarks_to_return=None,
                                 landmarks_3d=None)
        return sample


cfg.register_dataset(MEEI)


if __name__ == '__main__':
    from csl_common.vis import vis
    import time

    dirs = cfg.get_dataset_paths('meei')
    ds = MEEI(root=dirs[0], cache_root=dirs[1], train=True,
              deterministic=True,
              use_cache=False,
              align_face_orientation=False,
              return_modified_images=False,
              image_size=256)
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
        # lms_3d = batch.landmarks_3d
        imgs = vis.to_disp_images(inputs, denorm=False)
        vis.vis_square(imgs, nCols=5, fx=1, fy=1, normalize=False)
