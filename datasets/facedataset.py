from abc import ABC
import io
import glob
import os
import sys
import tarfile
from typing import Any, Dict, Union, List

import numpy as np
from PIL import Image

from datasets.imagedataset import ImageDataset
from landmarks import lmutils


def read_dataset(archive_path: str, append_dir=False) -> Dict[str, Dict[str, bytes]]:
    # Optional counter for how many bytes of data were loaded.
    byte_count = 0
    if append_dir:
        base = os.path.basename(archive_path)
        tar_name, extension = os.path.splitext(base)
    else:
        tar_name = ""

    # A dictionary mapping sample names to samples. A sample is a dictionary mapping names of sample elements
    # to the content of the respective files in the tar archive.
    dataset = dict()
    with tarfile.open(archive_path, mode="r") as dataset_file:
        # Create a dictionary holding all members.
        for tar_member in dataset_file.getmembers():

            # Assumes that filenames are structured like "sample_name.sample_element_name.file_extension"
            name_tar_member = tar_member.name.replace("/", ".")
            name_tar_member = name_tar_member if not append_dir else tar_name + "/" + tar_member.name

            if len(name_tar_member.split(os.extsep)) > 2:
                # sample_name, element_name = name_tar_member.split(os.extsep)[:2]
                sample_name = name_tar_member.split(os.extsep)[:-2]
                sample_name = '.'.join(sample_name)
                element_name = name_tar_member.split(os.extsep)[-2]
                # print(sample_name, element_name)

                # Get sample from dataset dictionary. If the sample does not
                # have any elements yet, insert and return an empty dict.
                dataset_entry = dataset.setdefault(sample_name, dict())

                # Extract the file from the tar archive.
                file_in_tar = dataset_file.extractfile(tar_member)

                # Read the file content into main memory.
                file_content = file_in_tar.read()

                # Store it in the dataset dictionary.
                dataset_entry[element_name] = file_content

                # Record the size of the file
                byte_count += sys.getsizeof(file_content)
    print('Loaded {} bytes of data into main memory'.format(byte_count))
    return dataset


def decode_dataset(dataset: Dict[str, Dict[str, Any]], num_lands=68):
    """ Loop over the dataset and decode based on file type or name.
        Only do this if you have a very small dataset.
    """
    for sample_name, sample in dataset.items():
        decoded_sample = decode_sample(sample, num_lands)
        dataset[sample_name] = decoded_sample


def decode_sample(sample: Dict[str, Any], num_lands=68):
    decoded_sample = dict()
    for element_name, file_content in sample.items():
        decoded_sample[element_name] = decode_file(element_name, file_content, num_lands)
    return decoded_sample


def decode_file(element_name: str, file_content: bytes, num_lands=68):
    """ Decode a sample based on file type or name.

        This can also be done on the fly when accessing a specific element.
        This way you can avoid keeping all decoded objects in memory all the time.
    """
    # Choose appropriate decoder based on the name of the sample element.
    if element_name == 'rgb' or element_name == 'face_image':
        return decode_as_pillow(file_content, 'RGB')  # return decode_as_pillow(file_content, 'L')
    elif element_name == 'bbox':
        return decode_bbox_file(file_content)
    elif element_name == 'lmk2d':
        return decode_2d_landmark_file(file_content, num_lands)
    else:
        # The exception will let you know that you missed decoding an element.
        raise NotImplementedError('Decoding not implemented for sample element', element_name)


def decode_as_pillow(data: bytes, mode: str) -> Image.Image:
    """ Handles decoding images represented as bytes-like objects to
        Pillow images.

    :param data: The image data to convert as a bytes-like object.
    :param mode: The mode to convert the loaded pillow image to,
                 e.g. 'RGB', 'L', '1', ...
    :returns: The decoded Pillow image.
    """
    with io.BytesIO(data) as stream:
        img = Image.open(stream).convert(mode=mode)
    return img


def decode_2d_landmark_file(data: bytes, num_lands: int) -> np.array:
    stream = data.decode("utf-8")
    stream = stream.split()
    lands_2d = np.zeros((num_lands, 2))
    for i in range(num_lands):
        lands_2d[i, :] = np.array([float(stream[i*2]), float(stream[i*2+1])])
    # print(lands_2d)
    return lands_2d


def decode_bbox_file(data: bytes) -> np.array:
    elem = data.decode("utf-8")
    elem = elem.split()
    bbox = np.array([float(elem[0]), float(elem[1]), float(elem[2]), float(elem[3])])
    # print(bbox)
    return bbox


class FaceDataset(ImageDataset, ABC):
    NUM_LANDMARKS = 68
    LANDMARKS_ONLY_OUTLINE = list(range(17))
    LANDMARKS_NO_OUTLINE = list(range(17, NUM_LANDMARKS))
    ALL_LANDMARKS = LANDMARKS_ONLY_OUTLINE + LANDMARKS_NO_OUTLINE

    def __init__(self, return_landmark_heatmaps=False, landmark_sigma=9, align_face_orientation=False,
                 load_in_memory=False, tar_path="", train_split='full', **kwargs):
        super().__init__(**kwargs)
        self.return_landmark_heatmaps = return_landmark_heatmaps
        self.landmark_sigma = landmark_sigma
        self.empty_landmarks = np.zeros((self.NUM_LANDMARKS, 2), dtype=np.float32)
        self.empty_landmarks_3d = np.zeros((self.NUM_LANDMARKS, 3), dtype=np.float32)
        self.align_face_orientation = align_face_orientation
        self.train_split = train_split

        if load_in_memory and tar_path:
            if tar_path == 'multi':
                # Read several tar files from directory
                ext = ['tar']
                files = []
                [files.extend(glob.glob(self.root + '*.' + e)) for e in ext]

                self.dataset = dict()
                for file in files:
                    # Read the entire tar file into memory.
                    data_tar = read_dataset(file, True)
                    # Decode the entire dataset and keep the decoded samples in memory.
                    decode_dataset(data_tar, self.NUM_LANDMARKS)
                    self.dataset.update(data_tar)

            else:
                # Read the entire dataset into memory.
                tar_path = tar_path if os.path.isfile(tar_path) else os.path.join(self.root, tar_path)
                self.dataset = read_dataset(tar_path)
                # Decode the entire dataset and keep the decoded samples in memory.
                # This may increase the memory consumption significantly.
                decode_dataset(self.dataset, self.NUM_LANDMARKS)
        else:
            self.dataset = dict()

    @staticmethod
    def _get_expression(sample):
        return np.array([[0, 0, 0]], dtype=np.float32)

    @staticmethod
    def _get_identity(sample):
        return -1

    def _crop_landmarks(self, lms):
        return self.loader._cropper.apply_to_landmarks(lms)[0]

    def get_sample(self, filename, bb=None, landmarks_for_crop=None, id=None,
                   landmarks_to_return=None, landmarks_3d=None):
        try:
            crop_mode = 'landmarks' if landmarks_for_crop is not None else 'bounding_box'
            crop_params = {'landmarks': landmarks_for_crop,
                           'bb': bb,
                           'id': id,
                           'aligned': self.align_face_orientation,
                           'mode': crop_mode}
            if self.dataset:
                img = self.dataset[filename]['rgb']
                image = self.loader.load_crop_mem(img, filename, **crop_params)
            else:
                image = self.loader.load_crop(filename, **crop_params)
        except:
            print('Could not load image {}'.format(filename))
            raise

        relative_landmarks = self._crop_landmarks(landmarks_to_return) \
            if landmarks_to_return is not None else self.empty_landmarks

        # self.show_landmarks(image, landmarks)

        relative_landmarks_3d = self._crop_landmarks(landmarks_3d) \
            if landmarks_3d is not None else self.empty_landmarks_3d

        # if landmarks_3d is not None:
        #     sample['landmarks_3d'] = landmarks_3d

        sample = {'image': image,
                  'landmarks': relative_landmarks,
                  'pose': np.zeros(3, dtype=np.float32),
                  'landmarks_3d': relative_landmarks_3d}

        if self.transform is not None:
            sample = self.transform(sample)
        target = self.target_transform(sample) if self.target_transform else None

        if self.crop_type != 'fullsize':
            sample = self.crop_to_tensor(sample)
            if target is not None:
                target = self.crop_to_tensor(target)

        sample.update({
            'fnames': filename,
            'bb': bb if bb is not None else [0, 0, 0, 0]
            # 'landmarks_3d': relative_landmarks_3d
            # 'expression':self._get_expression(sample),
            # 'id': self._get_identity(sample),
        })

        if target is not None:
            sample['target'] = target

        if self.return_landmark_heatmaps and self.crop_type != 'fullsize':
            from landmarks import lmconfig as lmcfg
            heatmap_size = lmcfg.HEATMAP_SIZE
            scaled_landmarks = sample['landmarks'] * (heatmap_size / self.image_size)
            sample['lm_heatmaps'] = lmutils.create_landmark_heatmaps(scaled_landmarks, self.landmark_sigma,
                                                                     self.ALL_LANDMARKS, heatmap_size)

        return sample

    def show_landmarks(self, img, landmarks):
        import cv2
        for lm in landmarks:
            lm_x, lm_y = lm[0], lm[1]
            cv2.circle(img, (int(lm_x), int(lm_y)), 3, (0, 0, 255), -1)
        cv2.imshow('landmarks', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        cv2.waitKey(0)
