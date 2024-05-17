import os

import glob
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import torch

from csl_common.utils import geometry
from datasets import facedataset
import src.config as cfg


def gt_loader(file_path):
    land_file = file_path + '.lmk2d.txt'
    bbox_file = file_path + '.bbox.txt'
    lms, bbox = [], []
    with open(land_file) as f:
        for line in f:
            try:
                x, y = [float(e) for e in line.split()]
                lms.append((x, y))
            except:
                pass
    assert (len(lms) == 68)
    landmarks = np.vstack(lms)

    with open(bbox_file) as f:
        for line in f:
            try:
                x1, y1, x2, y2 = [float(e) for e in line.split()]
                bbox.append((x1, y1, x2, y2))
            except:
                pass
    assert (len(bbox) == 1)
    bboxes = np.vstack(bbox)

    return landmarks, bboxes


class NeuroFace(facedataset.FaceDataset):
    """
    Toronto NeuroFace Dataset.
    Videos of oro-facial gestures of patients with amyotrophic lateral sclerosis (ALS) and stroke.
    Videos and selected frames (~3300) with annotated bbox and 2D facial landmarks (68 FL).
    There is one tar file per category, with the selected frames and corresponding annotations:
    ALS.tar, Stroke.tar and HealthyControls.tar.
    All the images and annotations are collected also in a single file, all.tar.
    This dataloader reads the dataset from /all.
    https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9177259
    """
    CROP_SOURCES = ['bb_detector', 'bb_ground_truth', 'lm_ground_truth']

    NUM_LANDMARKS = 68
    LANDMARKS_ONLY_OUTLINE = list(range(17))
    LANDMARKS_NO_OUTLINE = list(range(17, NUM_LANDMARKS))
    ALL_LANDMARKS = list(range(NUM_LANDMARKS))

    def __init__(self, root, cache_root=None, train=True,
                 crop_source='bb_ground_truth', return_landmark_heatmaps=False,
                 return_modified_images=False, use_cache=False,
                 load_in_memory=True, tar_path="all.tar", **kwargs):

        if not load_in_memory:
            self.annotations = self._load_annotations(train)

            # Shuffle faces
            self.annotations = shuffle(self.annotations)
            print(f"Number of images: {len(self.annotations)}")

        super().__init__(root=root,
                         cache_root=cache_root,
                         fullsize_img_dir=root,
                         train=train,
                         crop_source=crop_source,
                         use_cache=use_cache,
                         return_landmark_heatmaps=return_landmark_heatmaps,
                         return_modified_images=return_modified_images,
                         load_in_memory=load_in_memory,
                         tar_path=tar_path,
                         **kwargs)

    def _load_annotations(self, split):
        ext = ['png', 'jpg', 'jpeg']
        files = []
        [files.extend(glob.glob(self.root + '*.' + e)) for e in ext]

        annot = []
        for file in files:
            img_name = os.path.basename(file)
            filename = str.join(".", file.split(".")[:-2])
            landmarks, bbox = gt_loader(filename)

            annot.append({'imgName': str(img_name), 'fname': file,
                          'landmarks': landmarks, 'bb_ground_truth': bbox})

        annotations = pd.DataFrame(annot)
        return annotations

    def __len__(self):
        if self.dataset:
            return len(self.dataset)
        else:
            return len(self.annotations)

    def __getitem__(self, idx):

        if self.dataset:
            keys_list = list(self.dataset)
            filename = keys_list[idx]
            bb = self.dataset[filename]['bbox']
            landmarks = self.dataset[filename]['lmk2d']
        else:
            sample = self.annotations.iloc[idx]
            filename = sample.fname
            bb = sample.bb_ground_truth[0]
            landmarks = sample.landmarks.astype(np.float32)

        bb = geometry.extend_bbox(bb, dt=0.05, db=0.10)
        # bb = geometry.extend_bbox(bb, dt=0.2, db=0.12)

        landmarks_for_crop = None
        if self.crop_source == 'lm_ground_truth':
            landmarks_for_crop = landmarks

        sample = self.get_sample(filename,
                                 bb,
                                 landmarks_for_crop,
                                 landmarks_to_return=landmarks,
                                 landmarks_3d=None)

        return sample


cfg.register_dataset(NeuroFace)


if __name__ == '__main__':
    from csl_common.vis import vis
    import time

    dirs = cfg.get_dataset_paths('neuroface')
    ds = NeuroFace(root=dirs[0],
                   cache_root=dirs[1],
                   train=False,
                   deterministic=True,
                   use_cache=False,
                   image_size=256,
                   align_face_orientation=False,
                   return_modified_images=False,
                   crop_source='bb_ground_truth',
                   # crop_source='lm_ground_truth',
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
        # lms_3d = batch.landmarks_3d
        imgs = vis.to_disp_images(inputs, denorm=False)
        imgs = vis.add_landmarks_to_images(imgs, batch['landmarks'], radius=3, color=(0, 255, 0), draw_wireframe=True)
        vis.vis_square(imgs, nCols=5, fx=1, fy=1, normalize=False)
