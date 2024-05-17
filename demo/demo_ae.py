import os
import cv2
import glob
import numpy as np

import torch
import torch.utils.data as td
from torchvision import transforms as tf

# from utils import cropping as fp
from csl_common.utils.nn import Batch
from csl_common.utils import nn, cropping
from csl_common import utils
from datasets import aflw20003d, palsy

from src.face_vis import draw_results
import src.config as config

from landmarks import fabrec
from networks.aae import vis_reconstruction, load_net

snapshot_dir = os.path.join('.')

INPUT_SIZE = 256

transforms = [utils.transforms.CenterCrop(INPUT_SIZE)]
transforms += [utils.transforms.ToTensor()]
transforms += [utils.transforms.Normalize([0.518, 0.418, 0.361], [1, 1, 1])]
crop_to_tensor = tf.Compose(transforms)


def load_image(im_dir, fname=None):
    from skimage import io
    if fname is None:
        img_path = im_dir
    else:
        img_path = os.path.join(im_dir, fname)
    img = io.imread(img_path)
    if img is None:
        raise IOError("\tError: Could not load image {}!".format(img_path))
    if len(img.shape) == 2 or img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[2] == 4:
        print(img_path, "converting RGBA to RGB...")
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    assert img.shape[2] == 3, "{}, invalid format: {}".format(img_path, img.shape)
    return img


def detect_in_crop(net, crop):
    with torch.no_grad():
        X_recon, lms_in_crop, X_lm_hm = net.detect_landmarks(crop)
    lms_in_crop = utils.nn.to_numpy(lms_in_crop.reshape(1, -1, 2))
    return X_recon, lms_in_crop, X_lm_hm


def test_crop(net, input_image, out_dir, gt_landmarks, bb_for_crop=None, lms_for_crop=None, align=False, scale=1.0):
    assert bb_for_crop is not None or lms_for_crop is not None

    cropper = cropping.FaceCrop(input_image, bbox=bb_for_crop, landmarks=lms_for_crop,
                                align_face_orientation=align, scale=scale,
                                output_size=(INPUT_SIZE, INPUT_SIZE))
    crop = cropper.apply_to_image()
    landmarks = cropper.apply_to_landmarks(gt_landmarks)[0]

    item = {'image': crop, 'landmarks': landmarks, 'pose': None}
    item = crop_to_tensor(item)
    images = nn.atleast4d(item['image']).cuda()

    # reconstruct images from path
    f = 1 if images.shape[-1] < 512 else 0.5
    out = vis_reconstruction(net,
                             images,
                             landmarks=None,
                             ncols=10,
                             fx=f, fy=f, denorm=True)
    filename = 'reconst.jpg'
    img_filepath = os.path.join(out_dir, 'reconstructions', filename)
    os.makedirs(img_filepath, exist_ok=True)
    # cv2.normalize(img, img, 0, 255, norm_type=cv2.NORM_MINMAX)
    cv2.imwrite(img_filepath, cv2.cvtColor(out, cv2.COLOR_RGB2BGR))


if __name__ == '__main__':

    model = 'ae'
    out_path = '../out/ae/'

    net = load_net(model)
    net.eval()

    with torch.no_grad():

        # for img in files:
        #     img = load_image(img)
        #
        #     scalef = 1.0
        #     bb0 = [0,0] + list(img.shape[:2][::-1])
        #     bb = utils.geometry.scaleBB(bb0, scalef, scalef, typeBB=2)
        #     test_crop(net, img, out_path, gt_landmarks=None, bb_for_crop=bb)

        # dataset = 'aflw20003d'
        dataset = 'palsy'
        root, cache_root = config.get_dataset_paths(dataset)
        dataset_cls = config.get_dataset_class(dataset)
        ds = dataset_cls(root=root, cache_root=cache_root, train=False, deterministic=True, use_cache=False,
                         # test_split='full', align_face_orientation=False, crop_source='lm_ground_truth', normalize=True,
                         test_split='parface', align_face_orientation=False, crop_source='lm_ground_truth', normalize=True,
                         return_landmark_heatmaps=False, daug=0, landmark_sigma=7, image_size=256) #, transform=crop_to_tensor)
        dl = td.DataLoader(ds, batch_size=10, shuffle=False, num_workers=0)
        # dat = next(iter(dl))
        # dl = Batch(data, n=10)

        with torch.no_grad():
            for it, data in enumerate(dl):
                batch = Batch(data, eval=eval, gpu=False)

                input_images = batch.target_images if batch.target_images is not None else batch.images

                # Reconstruct images
                item = {'image': input_images, 'landmarks': None, 'pose': None}
                images = nn.atleast4d(item['image']).cuda()

                f = 1 if images.shape[-1] < 512 else 0.5
                out = vis_reconstruction(net,
                                         images,
                                         landmarks=None,
                                         ncols=10,
                                         fx=f, fy=f, denorm=True)
                filename = 'reconst' + str(it) + '.jpg'
                img_filepath = os.path.join(out_path, 'reconstructions', filename)
                os.makedirs(os.path.join(out_path, 'reconstructions'), exist_ok=True)
                cv2.imwrite(img_filepath, cv2.cvtColor(out, cv2.COLOR_RGB2BGR))


