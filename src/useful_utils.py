import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.transform import resize
# from mpl_toolkits import mplot3d

norm = lambda x: (x - x.min()) / (x.max() - x.min())
to_tensor = lambda x: torch.tensor(x).permute(2, 0, 1).unsqueeze(0).cuda().float()
get_rec = lambda x: norm(x[0].detach().permute(1, 2, 0).cpu().numpy())
# process_img = lambda x: norm(cv2.resize(x, (256, 256)))
process_img = lambda x: norm(resize(x, (256, 256)))

skeletons = [[i, i + 1] for i in range(16)] + \
                 [[i, i + 1] for i in range(17, 21)] + \
                 [[i, i + 1] for i in range(22, 26)] + \
                 [[i, i + 1] for i in range(36, 41)] + [[41, 36]] + \
                 [[i, i + 1] for i in range(42, 47)] + [[47, 42]] + \
                 [[i, i + 1] for i in range(27, 30)] + \
                 [[i, i + 1] for i in range(31, 35)] + \
                 [[i, i + 1] for i in range(48, 59)] + [[59, 48]] + \
                 [[i, i + 1] for i in range(60, 67)] + [[67, 60]]

# def show_3d(lm):
#     print("3D lm")
#     assert lm.shape[1] == 3
#     fig = plt.figure()
#     ax = plt.axes(projection='3d')
#     ax.scatter3D(lm[:, 0], lm[:, 2], lm[:, 1])
#     # set_axes_equal(ax)
#     ax.view_init(-160, 30)
#     plt.show()


def imshow(img):
    npimg = np.array(img).astype(np.uint8)*255.
    plt.imshow(npimg)
    plt.axis('off')


def show_both(*args):
    _, axs = plt.subplots(1, 2, figsize=(12, 12))
    axs = axs.flatten()
    for im, ax in zip(args, axs):
        ax.imshow(im)
        ax.axis("off")
    plt.show()


def get_lm(pred):
    rec = norm(pred[0].detach().squeeze().permute(1, 2, 0).cpu().numpy())
    lm = pred[1].squeeze()
    return rec, lm


def show_joints(img, pts, show_idx=False, pairs=skeletons, ax=None):
    pairs = None
    if ax is None:
        ax = plt.subplot(111)

    if type(img) == torch.Tensor:
        img = img.cpu().detach().squeeze().permute(1, 2, 0).numpy()

    pts_np = pts.numpy() if type(pts) == torch.tensor else pts
    ax.imshow(img)

    for i in range(pts.shape[0]):
        if pts.shape[1] < 3 or pts[i, 2] > 0:
            # plt.plot(pts[i, 0], pts[i, 1], 'bo')
            ax.scatter(pts[i, 0], pts[i, 1], s=10, c='b', edgecolors='b', linewidths=0.3)
            if show_idx:
                plt.text(pts[i, 0], pts[i, 1], str(i))
            if pairs is not None:
                for p in pairs:
                    ax.plot(pts_np[p, 0], pts_np[p, 1], c='b', linewidth=0.3)
    # plt.axis('off')
    # plt.show()


def show_3D(predPts, pairs=skeletons, ax=None):
    pairs = None
    if ax is None:
        ax = plt.subplot(111, projection='3d')

    view_angle = (-160, 30)
    # view_angle = (90., 90.)
    if predPts.shape[1] > 2:
        ax.scatter(predPts[:, 2], predPts[:, 0], predPts[:, 1], s=10, c='c', marker='o', edgecolors='b', linewidths=0.5)
        # ax.scatter(predPts[:, 0], predPts[:, 1], predPts[:, 2], s=10, c='c', marker='o', edgecolors='b', linewidths=0.5)
        # ax_pred.scatter(predPts[0, 2], predPts[0, 0], predPts[0, 1], s=10, c='g', marker='*')
        if pairs is not None:
            for p in pairs:
                ax.plot(predPts[p, 2], predPts[p, 0], predPts[p, 1], c='b',  linewidth=0.5)
    else:
        ax.scatter([0] * predPts.shape[0], predPts[:, 0], predPts[:, 1], s=10, marker='*')
    # ax.set_xlabel('z', fontsize=10)
    # ax.set_ylabel('x', fontsize=10)
    # ax.set_zlabel('y', fontsize=10)
    # ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.view_init(*view_angle)
    # plt.show()


def show(im, lm=None, pairs=skeletons, c="b"):
    if type(im) == torch.Tensor:
        im = im.cpu().detach().squeeze().permute(1, 2, 0).numpy()
    plt.imshow(im)

    if lm is not None:
        if len(lm.shape) == 3 and type(lm) == torch.Tensor:
            lm = lm.cpu().detach().squeeze(0)
        if len(lm.shape) == 3:
            lm = lm.squeeze(0)
        plt.scatter(lm[:, 0], lm[:, 1], c=c, s=10)

        # if pairs is not None:
        #     for p in pairs:
        #         plt.plot(lm[p, 0], lm[p, 1], c='b', linewidth=0.3)
    plt.axis("off")
    plt.show()

