import os
import glob

import cv2
from facenet_pytorch import MTCNN
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


DATASET_INFO = {
    "FFHQ": {
        "path": "/ds/FFHQ/ffhq_lite/",
        "ext": "*.jpg"
    },
    "celeba": {
        "path": "/ds/images/celeba/img_align_celeba/",
        "ext": "*.jpg"
    },
    "MEEI": {
        "path": "/ds/MEEI/frames/",
        "ext": "*.png"
    },
    "affectnet": {
        "path": "/ds/AffectNet/",
        "ext": "*.jpg"
    }
}


def process_data(dataset="FFHQ", resize=False, out_path=''):
    dataset_path = DATASET_INFO[dataset]["path"]
    # images = sorted(glob.glob(dataset_path + DATASET_INFO[dataset]["ext"]))
    images = [y for x in os.walk(dataset_path) for y in sorted(glob.glob(os.path.join(x[0], DATASET_INFO[dataset]["ext"])))]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    mtcnn = MTCNN(device=device)
    df = pd.DataFrame(columns=["image_path", "x1", "y1", "x2", "y2", "scale"])

    for i, im_path in enumerate(tqdm(images)):
        # Read image
        im = plt.imread(im_path)

        # Resize image
        if resize:
            im = cv2.resize(im, (256, 256))

        # print(im_path)
        # Get bounding box coordinates from face detection model
        im_tensor = torch.tensor(im.copy(), dtype=torch.float, device=device)
        if dataset == "MEEI":  # or dataset == "FFHQ"  # If image type is .png?
            faces = mtcnn.detect(im_tensor * 255.)
        else:
            faces = mtcnn.detect(im_tensor)

        img_name = os.path.basename(im_path)
        dir_name = os.path.basename(os.path.dirname(im_path))
        file_name = os.path.join(dir_name, img_name)

        try:
            if faces[0] is not None:
                x1, y1, x2, y2 = faces[0][0]
                scale = faces[1][0]
                data = {"image_path": file_name, "x1": [x1], "y1": [y1], "x2": [x2], "y2": [y2], "scale": [scale]}
                df = pd.concat([df, pd.DataFrame(data)], ignore_index=True)

                if resize:
                    plt.imsave(os.path.join(out_path, f"{i + 1:06d}.jpg"), im)
            else:
                print(file_name)

        except Exception as e:
            print(f"Error occurred - {e}")
            continue

    # Store bounding box and scale as annotations
    csv_file = os.path.join(out_path, dataset + "_annotations.csv")
    df.to_csv(csv_file, index=False)
    print(f"Total images: {len(images)}. Bounding boxes: {df.shape[0]}")


def process_affectnet(dataset="affectnet", out_path=''):
    dataset_path = DATASET_INFO[dataset]["path"]
    images = [y for x in os.walk(dataset_path) for y in sorted(glob.glob(os.path.join(x[0], DATASET_INFO[dataset]["ext"])))]

    df = pd.DataFrame(columns=["image_path", "face_x", "face_y", "face_width", "face_height", "facial_landmarks",
                               "expression",  "valence", "arousal"])

    for i, im_path in enumerate(tqdm(images)):
        img_name = os.path.basename(im_path)
        dir_type = os.path.basename(os.path.dirname(im_path))
        dir_name = os.path.basename(os.path.dirname(os.path.dirname(im_path)))
        file_name = os.path.join(dir_name, dir_type, img_name)  # "train_set/images/filename"

        try:
            annot_file = os.path.splitext(img_name)[0] + "_lnd.npy"
            land_file = os.path.join(dataset_path, dir_name, "annotations", annot_file)
            lands = np.load(land_file)

            annot_file = os.path.splitext(img_name)[0] + "_exp.npy"
            exp_file = os.path.join(dataset_path, dir_name, "annotations", annot_file)
            exp = np.load(exp_file)

            annot_file = os.path.splitext(img_name)[0] + "_val.npy"
            val_file = os.path.join(dataset_path, dir_name, "annotations", annot_file)
            val = np.load(val_file)

            annot_file = os.path.splitext(img_name)[0] + "_aro.npy"
            aro_file = os.path.join(dataset_path, dir_name, "annotations", annot_file)
            aro = np.load(aro_file)

            x1, y1, x2, y2 = 0.0, 0.0, 223.0, 223.0
            data = {"image_path": file_name, "face_x": [x1], "face_y": [y1], "face_width": [x2], "face_height": [y2],
                    "facial_landmarks": [lands.tolist()], "expression": [exp.tolist()],
                    "valence": [val.tolist()], "arousal": [aro.tolist()]}
            df = pd.concat([df, pd.DataFrame(data)], ignore_index=True)

            print(file_name)

        except Exception as e:
            print(f"Error occurred - {e}")
            continue

    # Store bounding box and scale as annotations
    csv_file = os.path.join(out_path, dataset + "_annotations.csv")
    df.to_csv(csv_file, index=False)
    print(f"Total images: {len(images)}. Bounding boxes: {df.shape[0]}")


if __name__ == '__main__':
    # process_data("celeba")
    # process_data("MEEI", False, '/ds/MEEI/frames/')
    # process_data("FFHQ", False, '/ds/FFHQ/')
    # process_affectnet("affectnet", '/ds/AffectNet/')
    process_affectnet("affectnet", '/ds/AffectNet/new/')

