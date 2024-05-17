import os
import time

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity as compare_ssim

from csl_common.utils import nn as nn
from csl_common.utils.nn import to_numpy
from csl_common.vis import vis
from landmarks.lmutils import calc_landmark_nme, calc_landmark_ssim_score, to_single_channel_heatmap, \
    calc_landmark_nme_per_img, smooth_heatmaps
import src.config as config


def show_landmark_heatmaps(pred_heatmaps, gt_heatmaps, nimgs, f=1.0,
                           session_name="demo", epoch=0, curr_iter=0):
    vmax = 1.0
    rows_heatmaps = []
    if gt_heatmaps is not None:
        vmax = gt_heatmaps.max()
        if len(gt_heatmaps[0].shape) == 2:
            gt_heatmaps = [vis.color_map(hm, vmin=0, vmax=vmax, cmap=plt.cm.jet) for hm in gt_heatmaps]
        nCols = 1 if len(gt_heatmaps) == 1 else nimgs
        rows_heatmaps.append(cv2.resize(vis.make_grid(gt_heatmaps, nCols=nCols, padval=0), None, fx=f, fy=f))

    disp_pred_heatmaps = pred_heatmaps
    if len(pred_heatmaps[0].shape) == 2:
        disp_pred_heatmaps = [vis.color_map(hm, vmin=0, vmax=vmax, cmap=plt.cm.jet) for hm in pred_heatmaps]
    nCols = 1 if len(pred_heatmaps) == 1 else nimgs
    rows_heatmaps.append(cv2.resize(vis.make_grid(disp_pred_heatmaps, nCols=nCols, padval=0), None, fx=f, fy=f))

    # cv2.imshow('Landmark heatmaps', cv2.cvtColor(np.vstack(rows_heatmaps), cv2.COLOR_RGB2BGR))

    dirs = config.get_dataset_paths('outputs')[0]
    save_path = os.path.join(dirs, "landmarks/", session_name, "heatmaps")
    os.makedirs(save_path, exist_ok=True)
    name_file = "hm_epoch_" + str(epoch) + "_iter_" + str(curr_iter) + ".jpg"
    # plt.imsave(os.path.join(save_path, name_file), np.vstack(rows_heatmaps))
    time_str = time.strftime("%Y%m%d%H%M%S")
    plt.imsave(os.path.join(save_path, "hm_" + time_str + ".jpg"), np.vstack(rows_heatmaps))
    # cv2.imwrite(os.path.join(save_path, "hm_" + time_str + ".jpg"), np.vstack(rows_heatmaps))


def visualize_batch(images, landmarks, X_recon, X_lm_hm, lm_preds_max,
                    lm_heatmaps=None, target_images=None, lm_preds_cnn=None, ds=None, wait=0, ssim_maps=None,
                    landmarks_to_draw=None, ocular_norm='outer', horizontal=False, f=1.0,
                    overlay_heatmaps_input=False, overlay_heatmaps_recon=False, clean=False,
                    landmarks_only_outline=range(17), landmarks_no_outline=range(17, 68),
                    session_name="debug", epoch=0, curr_iter=0, normalized=False):

    gt_color = (0, 255, 0)
    pred_color = (0, 0, 255)
    image_size = images.shape[3]
    assert image_size in [128, 256]

    nimgs = min(10, len(images))
    images = nn.atleast4d(images)[:nimgs]
    num_landmarks = lm_preds_max.shape[1]

    if landmarks_to_draw is None:
        landmarks_to_draw = range(num_landmarks)

    nme_per_lm = None
    if landmarks is None:
        # print('num landmarks', lmcfg.NUM_LANDMARKS)
        lm_gt = np.zeros((nimgs, num_landmarks, 2))
    else:
        lm_gt = nn.atleast3d(to_numpy(landmarks))[:nimgs]
        nme_per_lm = calc_landmark_nme(lm_gt, lm_preds_max[:nimgs], ocular_norm=ocular_norm, image_size=image_size)
        lm_ssim_errs = 1 - calc_landmark_ssim_score(images, X_recon[:nimgs], lm_gt)

    lm_confs = None
    # show landmark heatmaps
    pred_heatmaps = None
    if X_lm_hm is not None:
        pred_heatmaps = to_single_channel_heatmap(to_numpy(X_lm_hm[:nimgs]))
        pred_heatmaps = [cv2.resize(im, None, fx=f, fy=f, interpolation=cv2.INTER_NEAREST) for im in pred_heatmaps]
        gt_heatmaps = None
        if lm_heatmaps is not None:
            gt_heatmaps = to_single_channel_heatmap(to_numpy(lm_heatmaps[:nimgs]))
            gt_heatmaps = np.array([cv2.resize(im, None, fx=f, fy=f, interpolation=cv2.INTER_NEAREST) for im in gt_heatmaps])
        show_landmark_heatmaps(pred_heatmaps, gt_heatmaps, nimgs, f=1, session_name=session_name, epoch=epoch, curr_iter=curr_iter)
        lm_confs = to_numpy(X_lm_hm).reshape(X_lm_hm.shape[0], X_lm_hm.shape[1], -1).max(axis=2)

    # resize images for display and scale landmarks accordingly
    lm_preds_max = lm_preds_max[:nimgs] * f
    if lm_preds_cnn is not None:
        lm_preds_cnn = lm_preds_cnn[:nimgs] * f
    lm_gt *= f

    input_images = vis.to_disp_images(images[:nimgs], denorm=normalized)
    if target_images is not None:
        disp_images = vis.to_disp_images(target_images[:nimgs], denorm=normalized)
    else:
        disp_images = vis.to_disp_images(images[:nimgs], denorm=normalized)
    disp_images = [cv2.resize(im, None, fx=f, fy=f, interpolation=cv2.INTER_NEAREST) for im in disp_images]

    recon_images = vis.to_disp_images(X_recon[:nimgs], denorm=normalized)
    disp_X_recon = [cv2.resize(im, None, fx=f, fy=f, interpolation=cv2.INTER_NEAREST) for im in recon_images.copy()]

    # overlay landmarks on input images
    if pred_heatmaps is not None and overlay_heatmaps_input:
        disp_images = [vis.overlay_heatmap(disp_images[i], pred_heatmaps[i]) for i in range(len(pred_heatmaps))]
    if pred_heatmaps is not None and overlay_heatmaps_recon:
        disp_X_recon = [vis.overlay_heatmap(disp_X_recon[i], pred_heatmaps[i]) for i in range(len(pred_heatmaps))]

    # Show input images
    #
    disp_images = vis.add_landmarks_to_images(disp_images, lm_gt[:nimgs], color=gt_color)
    disp_images = vis.add_landmarks_to_images(disp_images, lm_preds_max[:nimgs], lm_errs=nme_per_lm,
                                              color=pred_color, draw_wireframe=False, gt_landmarks=lm_gt,
                                              draw_gt_offsets=True)

    # disp_images = vis.add_landmarks_to_images(disp_images, lm_gt[:nimgs], color=(1,1,1), radius=1,
    #                                          draw_dots=True, draw_wireframe=True, landmarks_to_draw=landmarks_to_draw)
    # disp_images = vis.add_landmarks_to_images(disp_images, lm_preds_max[:nimgs], lm_errs=nme_per_lm,
    #                                           color=(1.0, 0.0, 0.0),
    #                                           draw_dots=True, draw_wireframe=True, radius=1,
    #                                           gt_landmarks=lm_gt, draw_gt_offsets=False,
    #                                           landmarks_to_draw=landmarks_to_draw)

    # Show reconstructions
    #
    X_recon_errs = 255.0 * torch.abs(images - X_recon[:nimgs]).reshape(len(images), -1).mean(dim=1)
    if not clean:
        disp_X_recon = vis.add_error_to_images(disp_X_recon[:nimgs], errors=X_recon_errs, format_string='{:>4.1f}')

    # modes of heatmaps
    # disp_X_recon = [overlay_heatmap(disp_X_recon[i], pred_heatmaps[i]) for i in range(len(pred_heatmaps))]
    if not clean:
        lm_errs_max = calc_landmark_nme_per_img(lm_gt, lm_preds_max, ocular_norm,
                                                landmarks_no_outline, image_size=image_size)
        lm_errs_max_outline = calc_landmark_nme_per_img(lm_gt, lm_preds_max, ocular_norm,
                                                        landmarks_only_outline, image_size=image_size)
        lm_errs_max_all = calc_landmark_nme_per_img(lm_gt, lm_preds_max, ocular_norm,
                                                    list(landmarks_only_outline)+list(landmarks_no_outline),
                                                    image_size=image_size)
        disp_X_recon = vis.add_error_to_images(disp_X_recon, lm_errs_max, loc='br-2', format_string='{:>5.2f}', vmax=15)
        disp_X_recon = vis.add_error_to_images(disp_X_recon, lm_errs_max_outline, loc='br-1', format_string='{:>5.2f}', vmax=15)
        disp_X_recon = vis.add_error_to_images(disp_X_recon, lm_errs_max_all, loc='br', format_string='{:>5.2f}', vmax=15)
    disp_X_recon = vis.add_landmarks_to_images(disp_X_recon, lm_gt, color=gt_color, draw_wireframe=True)

    # disp_X_recon = vis.add_landmarks_to_images(disp_X_recon, lm_preds_max[:nimgs],
    #                                            color=pred_color, draw_wireframe=False,
    #                                            lm_errs=nme_per_lm, lm_confs=lm_confs,
    #                                            lm_rec_errs=lm_ssim_errs, gt_landmarks=lm_gt,
    #                                            draw_gt_offsets=True, draw_dots=True)

    disp_X_recon = vis.add_landmarks_to_images(disp_X_recon, lm_preds_max[:nimgs],
                                               color=pred_color, draw_wireframe=True,
                                               gt_landmarks=lm_gt, draw_gt_offsets=True, lm_errs=nme_per_lm,
                                               draw_dots=True, radius=2)

    def add_confidences(disp_X_recon, lmids, loc):
        means = lm_confs[:,lmids].mean(axis=1)
        colors = vis.color_map(to_numpy(1-means), cmap=plt.cm.jet, vmin=0.0, vmax=0.4)
        return vis.add_error_to_images(disp_X_recon, means, loc=loc, format_string='{:>4.2f}', colors=colors)

    # disp_X_recon = add_confidences(disp_X_recon, lmcfg.LANDMARKS_NO_OUTLINE, 'bm-2')
    # disp_X_recon = add_confidences(disp_X_recon, lmcfg.LANDMARKS_ONLY_OUTLINE, 'bm-1')
    # disp_X_recon = add_confidences(disp_X_recon, lmcfg.ALL_LANDMARKS, 'bm')

    # print ssim errors
    ssim = np.zeros(nimgs)
    for i in range(nimgs):
        # ssim[i] = compare_ssim(input_images[i], recon_images[i], data_range=1.0, multichannel=True)
        ssim[i] = compare_ssim(input_images[i], recon_images[i], data_range=1.0, channel_axis=-1)
    if not clean:
        disp_X_recon = vis.add_error_to_images(disp_X_recon, 1 - ssim, loc='bl-1', format_string='{:>4.2f}',
                                               vmax=0.8, vmin=0.2)
    # print ssim torch errors
    if ssim_maps is not None and not clean:
        disp_X_recon = vis.add_error_to_images(disp_X_recon, ssim_maps.reshape(len(ssim_maps), -1).mean(axis=1),
                                               loc='bl-2', format_string='{:>4.2f}', vmin=0.0, vmax=0.4)

    rows = [vis.make_grid(disp_images, nCols=nimgs, normalize=False)]
    rows.append(vis.make_grid(disp_X_recon, nCols=nimgs))

    if ssim_maps is not None:
        disp_ssim_maps = to_numpy(nn.denormalized(ssim_maps)[:nimgs].transpose(0, 2, 3, 1))
        for i in range(len(disp_ssim_maps)):
            disp_ssim_maps[i] = vis.color_map(disp_ssim_maps[i].mean(axis=2), vmin=0.0, vmax=2.0)
        grid_ssim_maps = vis.make_grid(disp_ssim_maps, nCols=nimgs, fx=f, fy=f)
        # cv2.imshow('ssim errors', cv2.cvtColor(grid_ssim_maps, cv2.COLOR_RGB2BGR))

        timestr = time.strftime("%Y%m%d%H%M%S")
        dirs = config.get_dataset_paths('outputs')[0]
        save_path = os.path.join(dirs, "landmarks/", session_name, "ssim_errors")
        os.makedirs(save_path, exist_ok=True)
        name_file = "ssim_epoch_" + str(epoch) + "_iter_" + str(curr_iter) + ".jpg"
        # cv2.imwrite(os.path.join(save_path, "ssim_" + timestr + ".jpg"), grid_ssim_maps)
        plt.imsave(os.path.join(save_path, name_file), grid_ssim_maps)

    if horizontal:
        assert(nimgs == 1)
        disp_rows = vis.make_grid(rows, nCols=2)
    else:
        disp_rows = vis.make_grid(rows, nCols=1)
    wnd_title = 'Predicted Landmarks '
    if ds is not None:
        wnd_title += ds.__class__.__name__
    # cv2.imshow(wnd_title, cv2.cvtColor(disp_rows, cv2.COLOR_RGB2BGR))
    # cv2.waitKey(wait)

    timestr = time.strftime("%Y%m%d%H%M%S")
    dirs = config.get_dataset_paths('outputs')[0]
    save_path = os.path.join(dirs, "landmarks/", session_name, "ssim_errors")
    os.makedirs(save_path, exist_ok=True)
    name_file = "pred_lmk_epoch_" + str(epoch) + "_iter_" + str(curr_iter) + ".jpg"
    # name_file = "pred_lmk_" + timestr + ".jpg"
    cv2.normalize(disp_rows, disp_rows, 0, 255, norm_type=cv2.NORM_MINMAX)
    # cv2.imwrite(os.path.join(save_path, "pred_lmk_" + timestr + "cv2.jpg"), cv2.cvtColor(disp_rows,cv2.COLOR_RGB2BGR))
    plt.imsave(os.path.join(save_path, name_file), disp_rows)


def visualize_images(X, X_lm_hm, landmarks=None, show_recon=True, show_landmarks=True, show_heatmaps=False,
                     draw_wireframe=False, smoothing_level=2, heatmap_opacity=0.8, f=1, normalized=False):

    if show_recon:
        disp_X = vis.to_disp_images(X, denorm=normalized)
    else:
        disp_X = vis.to_disp_images(torch.zeros_like(X), denorm=normalized)
        heatmap_opacity = 1

    if X_lm_hm is not None:
        if smoothing_level > 0:
            X_lm_hm = smooth_heatmaps(X_lm_hm)
        if smoothing_level > 1:
            X_lm_hm = smooth_heatmaps(X_lm_hm)

    if show_heatmaps:
        pred_heatmaps = to_single_channel_heatmap(to_numpy(X_lm_hm))
        pred_heatmaps = [cv2.resize(im, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC) for im in pred_heatmaps]
        disp_X = [vis.overlay_heatmap(disp_X[i], pred_heatmaps[i], heatmap_opacity) for i in range(len(pred_heatmaps))]

    if show_landmarks and landmarks is not None:
        pred_color = (0,255,255)
        disp_X = vis.add_landmarks_to_images(disp_X, landmarks, color=pred_color, draw_wireframe=draw_wireframe)

    return disp_X


def generate_images(net, z_random, **kwargs):
    with torch.no_grad():
        X_gen_vis = net.P(z_random)[:, :3]
        X_lm_hm = net.LMH(net.P)
    return visualize_images(X_gen_vis, X_lm_hm, **kwargs)


def visualize_random_faces(net, nimgs=10, wait=10, f=1.0, session_name="debug", epoch=0, curr_iter=0):
    z_random = torch.randn(nimgs, net.z_dim).cuda()
    disp_X_gen = generate_images(net, z_random)
    grid_img = vis.make_grid(disp_X_gen, nCols=nimgs//2)
    # cv2.imshow("random faces", cv2.cvtColor(grid_img, cv2.COLOR_RGB2BGR))
    # cv2.waitKey(wait)

    timestr = time.strftime("%Y%m%d%H%M%S")
    dirs = config.get_dataset_paths('outputs')[0]
    save_path = os.path.join(dirs, "landmarks/", session_name, "random_faces")
    os.makedirs(save_path, exist_ok=True)
    name_file = "random_faces_epoch_" + str(epoch) + "_iter_" + str(curr_iter) + ".jpg"
    # cv2.imwrite(os.path.join(save_path, "random_faces_" + timestr + ".jpg"), grid_img)
    plt.imsave(os.path.join(save_path, name_file), grid_img)


def visualize_batch_CVPR(images, landmarks, X_recon, X_lm_hm, lm_preds, show_recon=True,
                         lm_heatmaps=None, ds=None, wait=0, horizontal=False, f=1.0, radius=2,
                         draw_wireframes=False, session_name=None, normalized=False):

    gt_color = (0, 255, 0)
    pred_color = (0, 255, 255)
    # pred_color = (255,0,0)

    nimgs = min(10, len(images))
    images = nn.atleast4d(images)[:nimgs]
    num_landmarks = lm_preds.shape[1]

    # if landmarks is None:
    #     print('num landmarks', num_landmarks)
    #     lm_gt = np.zeros((nimgs, num_landmarks, 2))
    # else:

    # show landmark heatmaps
    pred_heatmaps = None
    if X_lm_hm is not None:
        pred_heatmaps = to_single_channel_heatmap(to_numpy(X_lm_hm[:nimgs]))
        pred_heatmaps = [cv2.resize(im, None, fx=f, fy=f, interpolation=cv2.INTER_NEAREST) for im in pred_heatmaps]
        gt_heatmaps = None
        if lm_heatmaps is not None:
            gt_heatmaps = to_single_channel_heatmap(to_numpy(lm_heatmaps[:nimgs]))
            gt_heatmaps = np.array([cv2.resize(im, None, fx=f, fy=f, interpolation=cv2.INTER_NEAREST) for im in gt_heatmaps])
        show_landmark_heatmaps(pred_heatmaps, gt_heatmaps, nimgs, f=1, session_name=session_name)
        lm_confs = to_numpy(X_lm_hm).reshape(X_lm_hm.shape[0], X_lm_hm.shape[1], -1).max(axis=2)

    # resize images for display and scale landmarks accordingly
    lm_preds = lm_preds[:nimgs] * f

    rows = []

    disp_images = vis.to_disp_images(images[:nimgs], denorm=normalized)
    disp_images = [cv2.resize(im, None, fx=f, fy=f, interpolation=cv2.INTER_NEAREST) for im in disp_images]
    rows.append(vis.make_grid(disp_images, nCols=nimgs, normalize=False))

    heatmap_opacity = 1.0
    if show_recon:
        recon_images = vis.to_disp_images(X_recon[:nimgs], denorm=normalized)
    else:
        recon_images = vis.to_disp_images(torch.ones_like(X_recon[:nimgs]), denorm=normalized)
        heatmap_opacity = 1

    disp_X_recon = [cv2.resize(im, None, fx=f, fy=f, interpolation=cv2.INTER_NEAREST) for im in recon_images.copy()]
    rows.append(vis.make_grid(disp_X_recon, nCols=nimgs))

    # overlay landmarks on images
    disp_X_recon_hm = [cv2.resize(im, None, fx=f, fy=f, interpolation=cv2.INTER_NEAREST) for im in recon_images.copy()]
    disp_X_recon_hm = [vis.overlay_heatmap(disp_X_recon_hm[i], pred_heatmaps[i], heatmap_opacity) for i in range(len(pred_heatmaps))]
    rows.append(vis.make_grid(disp_X_recon_hm, nCols=nimgs))

    # reconstructions with prediction
    disp_X_recon_pred = [cv2.resize(im, None, fx=f, fy=f, interpolation=cv2.INTER_NEAREST) for im in recon_images.copy()]
    disp_X_recon_pred = vis.add_landmarks_to_images(disp_X_recon_pred, lm_preds, color=pred_color, radius=radius)
    rows.append(vis.make_grid(disp_X_recon_pred, nCols=nimgs))

    # reconstructions with ground truth (if gt available)
    if landmarks is not None:
        lm_gt = nn.atleast3d(to_numpy(landmarks))[:nimgs] * f
        disp_X_recon_gt = [cv2.resize(im, None, fx=f, fy=f, interpolation=cv2.INTER_NEAREST) for im in recon_images.copy()]
        disp_X_recon_gt = vis.add_landmarks_to_images(disp_X_recon_gt, lm_gt, color=gt_color, radius=radius)
        rows.append(vis.make_grid(disp_X_recon_gt, nCols=nimgs))

    # input images with prediction (and ground truth)
    disp_images_pred = vis.to_disp_images(images[:nimgs], denorm=normalized)
    disp_images_pred = [cv2.resize(im, None, fx=f, fy=f, interpolation=cv2.INTER_NEAREST) for im in disp_images_pred]
    # disp_images_pred = vis.add_landmarks_to_images(disp_images_pred, lm_gt, color=gt_color, radius=radius)
    disp_images_pred = vis.add_landmarks_to_images(disp_images_pred, lm_preds, color=pred_color, radius=radius,
                                                   draw_wireframe=draw_wireframes)
    rows.append(vis.make_grid(disp_images_pred, nCols=nimgs))

    if horizontal:
        assert(nimgs == 1)
        disp_rows = vis.make_grid(rows, nCols=len(rows))
    else:
        disp_rows = vis.make_grid(rows, nCols=1)
    wnd_title = 'recon errors '
    if ds is not None:
        wnd_title += ds.__class__.__name__
    # cv2.imshow(wnd_title, cv2.cvtColor(disp_rows, cv2.COLOR_RGB2BGR))
    # cv2.waitKey(wait)

    timestr = time.strftime("%Y%m%d%H%M%S")
    dirs = config.get_dataset_paths('outputs')[0]
    save_path = os.path.join(dirs, "landmarks/")
    if session_name:
        save_path = os.path.join(save_path, session_name)
    os.makedirs(save_path, exist_ok=True)
    # cv2.imwrite(os.path.join(save_path, "recon_errors" + timestr + ".jpg"), disp_rows)
    plt.imsave(os.path.join(save_path, "recon_errors" + timestr + ".jpg"), disp_rows)


def visualize_batch_CVPR2022(images, landmarks, X_recon, X_lm_hm, lm_preds, show_recon=True,
                             lm_heatmaps=None, ds=None, horizontal=False, f=1.0, radius=2,
                             draw_wireframes=False, file_name=None, normalized=False):

    gt_color = (0, 255, 0)
    pred_color = (0, 255, 255)

    nimgs = min(10, len(images))
    images = nn.atleast4d(images)[:nimgs]
    num_landmarks = lm_preds.shape[1]

    # show landmark heatmaps
    pred_heatmaps = None
    if X_lm_hm is not None:
        pred_heatmaps = to_single_channel_heatmap(to_numpy(X_lm_hm[:nimgs]))
        pred_heatmaps = [cv2.resize(im, None, fx=f, fy=f, interpolation=cv2.INTER_NEAREST) for im in pred_heatmaps]
        gt_heatmaps = None
        if lm_heatmaps is not None:
            gt_heatmaps = to_single_channel_heatmap(to_numpy(lm_heatmaps[:nimgs]))
            gt_heatmaps = np.array([cv2.resize(im, None, fx=f, fy=f, interpolation=cv2.INTER_NEAREST) for im in gt_heatmaps])
        show_landmark_heatmaps(pred_heatmaps, gt_heatmaps, nimgs, f=1)

    # resize images for display and scale landmarks accordingly
    lm_preds = lm_preds[:nimgs] * f

    rows = []
    disp_images = vis.to_disp_images(images[:nimgs], denorm=normalized)
    disp_images = [cv2.resize(im, None, fx=f, fy=f, interpolation=cv2.INTER_NEAREST) for im in disp_images]
    rows.append(vis.make_grid(disp_images, nCols=nimgs, normalize=False))

    heatmap_opacity = 1.0
    if show_recon:
        recon_images = vis.to_disp_images(X_recon[:nimgs], denorm=normalized)
    else:
        recon_images = vis.to_disp_images(torch.ones_like(X_recon[:nimgs]), denorm=normalized)
        heatmap_opacity = 1

    disp_X_recon = [cv2.resize(im, None, fx=f, fy=f, interpolation=cv2.INTER_NEAREST) for im in recon_images.copy()]
    rows.append(vis.make_grid(disp_X_recon, nCols=nimgs))

    # overlay landmarks on images
    disp_X_recon_hm = [cv2.resize(im, None, fx=f, fy=f, interpolation=cv2.INTER_NEAREST) for im in recon_images.copy()]
    disp_X_recon_hm = [vis.overlay_heatmap(disp_X_recon_hm[i], pred_heatmaps[i], heatmap_opacity) for i in range(len(pred_heatmaps))]
    rows.append(vis.make_grid(disp_X_recon_hm, nCols=nimgs))

    # reconstructions with prediction
    disp_X_recon_pred = [cv2.resize(im, None, fx=f, fy=f, interpolation=cv2.INTER_NEAREST) for im in recon_images.copy()]
    disp_X_recon_pred = vis.add_landmarks_to_images(disp_X_recon_pred, lm_preds, color=pred_color, radius=radius)
    rows.append(vis.make_grid(disp_X_recon_pred, nCols=nimgs))

    # reconstructions with ground truth (if gt available)
    if landmarks is not None:
        lm_gt = nn.atleast3d(to_numpy(landmarks))[:nimgs] * f
        disp_X_recon_gt = [cv2.resize(im, None, fx=f, fy=f, interpolation=cv2.INTER_NEAREST) for im in recon_images.copy()]
        disp_X_recon_gt = vis.add_landmarks_to_images(disp_X_recon_gt, lm_gt, color=gt_color, radius=radius)
        rows.append(vis.make_grid(disp_X_recon_gt, nCols=nimgs))

    # input images with prediction (and ground truth)
    disp_images_pred = vis.to_disp_images(images[:nimgs], denorm=normalized)
    disp_images_pred = [cv2.resize(im, None, fx=f, fy=f, interpolation=cv2.INTER_NEAREST) for im in disp_images_pred]
    disp_images_pred = vis.add_landmarks_to_images(disp_images_pred, lm_preds, color=pred_color, radius=radius,
                                                   draw_wireframe=draw_wireframes)
    rows.append(vis.make_grid(disp_images_pred, nCols=nimgs))

    if horizontal:
        assert(nimgs == 1)
        disp_rows = vis.make_grid(rows, nCols=len(rows))
    else:
        disp_rows = vis.make_grid(rows, nCols=1)
    wnd_title = 'recon errors '
    if ds is not None:
        wnd_title += ds.__class__.__name__

    save_path = os.path.dirname(file_name)
    os.makedirs(save_path, exist_ok=True)
    plt.imsave(file_name, disp_rows)


def visualize_batch_eval(images, landmarks, X_recon, lm_preds_max,
                         target_images=None, lm_preds_cnn=None, ds=None, wait=0, ssim_maps=None,
                         landmarks_to_draw=None, ocular_norm='outer', horizontal=False, f=1.0, clean=False,
                         landmarks_only_outline=range(17), landmarks_no_outline=range(17, 68),
                         session_name="debug", epoch=0, curr_iter=0, normalized=False):

    gt_color = (0, 255, 0)
    pred_color = (0, 0, 255)
    image_size = images.shape[3]
    assert image_size in [128, 256]

    nimgs = min(10, len(images))
    images = nn.atleast4d(images)[:nimgs]
    num_landmarks = lm_preds_max.shape[1]

    if landmarks_to_draw is None:
        landmarks_to_draw = range(num_landmarks)

    nme_per_lm = None
    if landmarks is None:
        # print('num landmarks', lmcfg.NUM_LANDMARKS)
        lm_gt = np.zeros((nimgs, num_landmarks, 2))
    else:
        lm_gt = nn.atleast3d(to_numpy(landmarks))[:nimgs]
        nme_per_lm = calc_landmark_nme(lm_gt, lm_preds_max[:nimgs], ocular_norm=ocular_norm, image_size=image_size)
        lm_ssim_errs = 1 - calc_landmark_ssim_score(images, X_recon[:nimgs], lm_gt)

    # resize images for display and scale landmarks accordingly
    lm_preds_max = lm_preds_max[:nimgs] * f
    if lm_preds_cnn is not None:
        lm_preds_cnn = lm_preds_cnn[:nimgs] * f
    lm_gt *= f

    input_images = vis.to_disp_images(images[:nimgs], denorm=normalized)
    if target_images is not None:
        disp_images = vis.to_disp_images(target_images[:nimgs], denorm=normalized)
    else:
        disp_images = vis.to_disp_images(images[:nimgs], denorm=normalized)
    disp_images = [cv2.resize(im, None, fx=f, fy=f, interpolation=cv2.INTER_NEAREST) for im in disp_images]

    recon_images = vis.to_disp_images(X_recon[:nimgs], denorm=normalized)
    disp_X_recon = [cv2.resize(im, None, fx=f, fy=f, interpolation=cv2.INTER_NEAREST) for im in recon_images.copy()]

    # Show input images
    disp_images = vis.add_landmarks_to_images(disp_images, lm_gt[:nimgs], color=gt_color)
    disp_images = vis.add_landmarks_to_images(disp_images, lm_preds_max[:nimgs], lm_errs=nme_per_lm,
                                              color=pred_color, draw_wireframe=False, gt_landmarks=lm_gt,
                                              draw_gt_offsets=True)

    # Show reconstructions
    X_recon_errs = 255.0 * torch.abs(images - X_recon[:nimgs]).reshape(len(images), -1).mean(dim=1)
    if not clean:
        disp_X_recon = vis.add_error_to_images(disp_X_recon[:nimgs], errors=X_recon_errs, format_string='{:>4.1f}')

    # modes of heatmaps
    if not clean:
        lm_errs_max = calc_landmark_nme_per_img(lm_gt, lm_preds_max, ocular_norm,
                                                landmarks_no_outline, image_size=image_size)
        lm_errs_max_outline = calc_landmark_nme_per_img(lm_gt, lm_preds_max, ocular_norm,
                                                        landmarks_only_outline, image_size=image_size)
        lm_errs_max_all = calc_landmark_nme_per_img(lm_gt, lm_preds_max, ocular_norm,
                                                    list(landmarks_only_outline)+list(landmarks_no_outline),
                                                    image_size=image_size)
        disp_X_recon = vis.add_error_to_images(disp_X_recon, lm_errs_max, loc='br-2', format_string='{:>5.2f}', vmax=15)
        disp_X_recon = vis.add_error_to_images(disp_X_recon, lm_errs_max_outline, loc='br-1', format_string='{:>5.2f}', vmax=15)
        disp_X_recon = vis.add_error_to_images(disp_X_recon, lm_errs_max_all, loc='br', format_string='{:>5.2f}', vmax=15)
    disp_X_recon = vis.add_landmarks_to_images(disp_X_recon, lm_gt, color=gt_color, draw_wireframe=True)

    disp_X_recon = vis.add_landmarks_to_images(disp_X_recon, lm_preds_max[:nimgs],
                                               color=pred_color, draw_wireframe=True,
                                               gt_landmarks=lm_gt, draw_gt_offsets=True, lm_errs=nme_per_lm,
                                               draw_dots=True, radius=2)

    # print ssim errors
    ssim = np.zeros(nimgs)
    for i in range(nimgs):
        # ssim[i] = compare_ssim(input_images[i], recon_images[i], data_range=1.0, multichannel=True)
        ssim[i] = compare_ssim(input_images[i], recon_images[i], data_range=1.0, channel_axis=-1)
    if not clean:
        disp_X_recon = vis.add_error_to_images(disp_X_recon, 1 - ssim, loc='bl-1', format_string='{:>4.2f}',
                                               vmax=0.8, vmin=0.2)
    # print ssim torch errors
    if ssim_maps is not None and not clean:
        disp_X_recon = vis.add_error_to_images(disp_X_recon, ssim_maps.reshape(len(ssim_maps), -1).mean(axis=1),
                                               loc='bl-2', format_string='{:>4.2f}', vmin=0.0, vmax=0.4)

    rows = [vis.make_grid(disp_images, nCols=nimgs, normalize=False), vis.make_grid(disp_X_recon, nCols=nimgs)]

    if horizontal:
        assert(nimgs == 1)
        disp_rows = vis.make_grid(rows, nCols=2)
    else:
        disp_rows = vis.make_grid(rows, nCols=1)
    wnd_title = 'Predicted Landmarks '
    if ds is not None:
        wnd_title += ds.__class__.__name__

    # cv2.normalize(disp_rows, disp_rows, 0, 255, norm_type=cv2.NORM_MINMAX)
    # cv2.imshow(wnd_title, cv2.cvtColor(disp_rows, cv2.COLOR_RGB2BGR))
    # cv2.waitKey(wait)

    dirs = config.get_dataset_paths('outputs')[0]
    save_path = os.path.join(dirs, "landmarks/", session_name, "ssim_errors")
    os.makedirs(save_path, exist_ok=True)
    name_file = "pred_lmk_epoch_" + str(epoch) + "_iter_" + str(curr_iter) + ".jpg"
    # name_file = "pred_lmk_" + timestr + ".jpg"
    cv2.normalize(disp_rows, disp_rows, 0, 255, norm_type=cv2.NORM_MINMAX)
    # cv2.imwrite(os.path.join(save_path, "pred_lmk_" + timestr + "cv2.jpg"), cv2.cvtColor(disp_rows,cv2.COLOR_RGB2BGR))
    plt.imsave(os.path.join(save_path, name_file), disp_rows)


def visualize_batch_WACV(images, landmarks, X_recon, X_lm_hm, lm_preds, show_recon=True,
                         lm_heatmaps=None, ds=None, wait=0, horizontal=False, f=1.0, radius=2,
                         draw_wireframes=False, session_name=None, normalized=False):

    gt_color = (0, 255, 0)
    pred_color = (0, 255, 255)
    # pred_color = (255,0,0)

    nimgs = min(10, len(images))
    images = nn.atleast4d(images)[:nimgs]
    num_landmarks = lm_preds.shape[1]

    # if landmarks is None:
    #     print('num landmarks', num_landmarks)
    #     lm_gt = np.zeros((nimgs, num_landmarks, 2))
    # else:

    # show landmark heatmaps
    pred_heatmaps = None
    if X_lm_hm is not None:
        pred_heatmaps = to_single_channel_heatmap(to_numpy(X_lm_hm[:nimgs]))
        pred_heatmaps = [cv2.resize(im, None, fx=f, fy=f, interpolation=cv2.INTER_NEAREST) for im in pred_heatmaps]
        gt_heatmaps = None
        if lm_heatmaps is not None:
            gt_heatmaps = to_single_channel_heatmap(to_numpy(lm_heatmaps[:nimgs]))
            gt_heatmaps = np.array([cv2.resize(im, None, fx=f, fy=f, interpolation=cv2.INTER_NEAREST) for im in gt_heatmaps])
        show_landmark_heatmaps(pred_heatmaps, gt_heatmaps, nimgs, f=1)
        lm_confs = to_numpy(X_lm_hm).reshape(X_lm_hm.shape[0], X_lm_hm.shape[1], -1).max(axis=2)

    # resize images for display and scale landmarks accordingly
    lm_preds = lm_preds[:nimgs] * f

    rows = []

    disp_images = vis.to_disp_images(images[:nimgs], denorm=normalized)
    disp_images = [cv2.resize(im, None, fx=f, fy=f, interpolation=cv2.INTER_NEAREST) for im in disp_images]
    rows.append(vis.make_grid(disp_images, nCols=nimgs, normalize=False))

    heatmap_opacity = 1.0
    if show_recon:
        recon_images = vis.to_disp_images(X_recon[:nimgs], denorm=normalized)
    else:
        recon_images = vis.to_disp_images(torch.ones_like(X_recon[:nimgs]), denorm=normalized)
        heatmap_opacity = 1

    disp_X_recon = [cv2.resize(im, None, fx=f, fy=f, interpolation=cv2.INTER_NEAREST) for im in recon_images.copy()]
    rows.append(vis.make_grid(disp_X_recon, nCols=nimgs))

    # overlay landmarks on images
    disp_X_recon_hm = [cv2.resize(im, None, fx=f, fy=f, interpolation=cv2.INTER_NEAREST) for im in recon_images.copy()]
    disp_X_recon_hm = [vis.overlay_heatmap(disp_X_recon_hm[i], pred_heatmaps[i], heatmap_opacity) for i in range(len(pred_heatmaps))]
    rows.append(vis.make_grid(disp_X_recon_hm, nCols=nimgs))

    # reconstructions with prediction
    disp_X_recon_pred = [cv2.resize(im, None, fx=f, fy=f, interpolation=cv2.INTER_NEAREST) for im in recon_images.copy()]
    disp_X_recon_pred = vis.add_landmarks_to_images(disp_X_recon_pred, lm_preds, color=pred_color, radius=radius)
    rows.append(vis.make_grid(disp_X_recon_pred, nCols=nimgs))

    # reconstructions with ground truth (if gt available)
    if landmarks is not None:
        lm_gt = nn.atleast3d(to_numpy(landmarks))[:nimgs] * f
        disp_X_recon_gt = [cv2.resize(im, None, fx=f, fy=f, interpolation=cv2.INTER_NEAREST) for im in recon_images.copy()]
        disp_X_recon_gt = vis.add_landmarks_to_images(disp_X_recon_gt, lm_gt, color=gt_color, radius=radius)
        rows.append(vis.make_grid(disp_X_recon_gt, nCols=nimgs))

    # input images with prediction (and ground truth)
    disp_images_pred = vis.to_disp_images(images[:nimgs], denorm=normalized)
    disp_images_pred = [cv2.resize(im, None, fx=f, fy=f, interpolation=cv2.INTER_NEAREST) for im in disp_images_pred]
    # disp_images_pred = vis.add_landmarks_to_images(disp_images_pred, lm_gt, color=gt_color, radius=radius)
    disp_images_pred = vis.add_landmarks_to_images(disp_images_pred, lm_preds, color=pred_color, radius=radius,
                                                   draw_wireframe=draw_wireframes)
    rows.append(vis.make_grid(disp_images_pred, nCols=nimgs))

    if horizontal:
        assert(nimgs == 1)
        disp_rows = vis.make_grid(rows, nCols=len(rows))
    else:
        disp_rows = vis.make_grid(rows, nCols=1)
    wnd_title = 'recon errors '
    if ds is not None:
        wnd_title += ds.__class__.__name__
    # cv2.imshow(wnd_title, cv2.cvtColor(disp_rows, cv2.COLOR_RGB2BGR))
    # cv2.waitKey(wait)

    timestr = time.strftime("%Y%m%d%H%M%S")
    dirs = config.get_dataset_paths('outputs')[0]
    save_path = os.path.join(dirs, "landmarks/")
    if session_name:
        save_path = os.path.join(save_path, session_name)
    os.makedirs(save_path, exist_ok=True)
    # cv2.imwrite(os.path.join(save_path, "recon_errors" + timestr + ".jpg"), disp_rows)
    plt.imsave(os.path.join(save_path, "recon_errors" + timestr + ".jpg"), disp_rows)


def plot_gt_vs_pred_3d(gt, pred):
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection='3d')

    # # Data for a three-dimensional line
    # zline = np.linspace(0, 15, 1000)
    # xline = np.sin(zline)
    # yline = np.cos(zline)
    ax.plot3D(gt[0:17, 0], gt[0:17, 1], gt[0:17, 2], 'red')
    ax.plot3D(pred[0:17, 0], pred[0:17, 1], pred[0:17, 2], 'green')
    ax.plot3D(gt[17:22, 0], gt[17:22, 1], gt[17:22, 2], 'red')
    ax.plot3D(pred[17:22, 0], pred[17:22, 1], pred[17:22, 2], 'green')
    ax.plot3D(gt[22:27, 0], gt[22:27, 1], gt[22:27, 2], 'red')
    ax.plot3D(pred[22:27, 0], pred[22:27, 1], pred[22:27, 2], 'green')
    ax.plot3D(gt[27:31, 0], gt[27:31, 1], gt[27:31, 2], 'red')
    ax.plot3D(pred[27:31, 0], pred[27:31, 1], pred[27:31, 2], 'green')
    ax.plot3D(gt[31:36, 0], gt[31:36, 1], gt[31:36, 2], 'red')
    ax.plot3D(pred[31:36, 0], pred[31:36, 1], pred[31:36, 2], 'green')

    eye_l_gt = np.append(gt[36:42, :], [gt[36, :]], axis=0)
    ax.plot3D(eye_l_gt[:, 0], eye_l_gt[:, 1], eye_l_gt[:, 2], 'red')
    eye_l = np.append(pred[36:42, :], [pred[36, :]], axis=0)
    ax.plot3D(eye_l[:, 0], eye_l[:, 1], eye_l[:, 2], 'green')

    eye_r_gt = np.append(gt[42:48, :], [gt[42, :]], axis=0)
    ax.plot3D(eye_r_gt[:, 0], eye_r_gt[:, 1], eye_r_gt[:, 2], 'red')
    eye_r = np.append(pred[42:48, :], [pred[42, :]], axis=0)
    ax.plot3D(eye_r[:, 0], eye_r[:, 1], eye_r[:, 2], 'green')

    mouth_o_gt = np.append(gt[48:60, :], [gt[48, :]], axis=0)
    ax.plot3D(mouth_o_gt[:, 0], mouth_o_gt[:, 1], mouth_o_gt[:, 2], 'red')
    mouth_o = np.append(pred[48:60, :], [pred[48, :]], axis=0)
    ax.plot3D(mouth_o[:, 0], mouth_o[:, 1], mouth_o[:, 2], 'green')
    mouth_i_gt = np.append(gt[60:68, :], [gt[60, :]], axis=0)
    ax.plot3D(mouth_i_gt[:, 0], mouth_i_gt[:, 1], mouth_i_gt[:, 2], 'red')
    mouth_i = np.append(pred[60:68, :], [pred[60, :]], axis=0)
    ax.plot3D(mouth_i[:, 0], mouth_i[:, 1], mouth_i[:, 2], 'green')

    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_zticklabels([])

    ax.view_init(180, 0, vertical_axis='y')
    # ax.view_init(170, 45, vertical_axis='y')
    # plt.savefig(out_name, dpi=200, bbox_inches='tight', pad_inches=0)
    fig.canvas.draw()

    # Save it to a numpy array.
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return data


def to_disp_figs(gt_l3d, pred_l3d):
    return [plot_gt_vs_pred_3d(gt_l3d[i], pred_l3d[i]) for i in range(len(gt_l3d))]


def visualize_batch_3d(images, landmarks, X_recon, lm_preds_max,
                       target_images=None, ds=None, wait=0, ssim_maps=None,
                       ocular_norm='outer', horizontal=False, f=1.0, clean=False,
                       landmarks_only_outline=range(17), landmarks_no_outline=range(17, 68),
                       session_name="debug", epoch=0, curr_iter=0, normalized=False):

    # gt_color = (0, 255, 0)
    gt_color = (255, 0, 0)
    # pred_color = (0, 0, 255)
    pred_color = (0, 255, 0)
    image_size = images.shape[3]
    assert image_size in [128, 256]

    nimgs = min(10, len(images))
    images = nn.atleast4d(images)[:nimgs]
    num_landmarks = lm_preds_max.shape[1]

    nme_per_lm = None
    if landmarks is None:
        lm_gt = np.zeros((nimgs, num_landmarks, 2))
    else:
        lm_gt = nn.atleast3d(to_numpy(landmarks))[:nimgs]
        nme_per_lm = calc_landmark_nme(lm_gt[:, :, :2], lm_preds_max[:, :, :2][:nimgs], ocular_norm=ocular_norm, image_size=image_size)

    # resize images for display and scale landmarks accordingly
    lm_preds_max = lm_preds_max[:nimgs] * f
    lm_gt *= f

    input_images = vis.to_disp_images(images[:nimgs], denorm=normalized)
    if target_images is not None:
        disp_images = vis.to_disp_images(target_images[:nimgs], denorm=normalized)
    else:
        disp_images = vis.to_disp_images(images[:nimgs], denorm=normalized)
    disp_images = [cv2.resize(im, None, fx=f, fy=f, interpolation=cv2.INTER_NEAREST) for im in disp_images]

    recon_images = vis.to_disp_images(X_recon[:nimgs], denorm=normalized)
    disp_X_recon = [cv2.resize(im, None, fx=f, fy=f, interpolation=cv2.INTER_NEAREST) for im in recon_images.copy()]

    # print("sizes", lm_gt.shape, lm_preds_max.shape)
    plots_3d = to_disp_figs(lm_gt[:nimgs], lm_preds_max[:nimgs])
    dim = (256, 256)
    disp_plots_3d = [cv2.resize(im, dim, fx=f, fy=f, interpolation=cv2.INTER_NEAREST) for im in plots_3d.copy()]

    # Show input images
    disp_images = vis.add_landmarks_to_images(disp_images, lm_gt[:, :, :2][:nimgs], color=gt_color)
    disp_images = vis.add_landmarks_to_images(disp_images, lm_preds_max[:, :, :2][:nimgs], lm_errs=nme_per_lm,
                                              color=pred_color, draw_wireframe=False, gt_landmarks=lm_gt[:, :, :2],
                                              draw_gt_offsets=True)

    # Show reconstructions
    X_recon_errs = 255.0 * torch.abs(images - X_recon[:nimgs]).reshape(len(images), -1).mean(dim=1)
    if not clean:
        disp_X_recon = vis.add_error_to_images(disp_X_recon[:nimgs], errors=X_recon_errs, format_string='{:>4.1f}')

    # modes of heatmaps
    if not clean:
        lm_errs_max = calc_landmark_nme_per_img(lm_gt, lm_preds_max, ocular_norm,
                                                landmarks_no_outline, image_size=image_size)
        lm_errs_max_outline = calc_landmark_nme_per_img(lm_gt, lm_preds_max, ocular_norm,
                                                        landmarks_only_outline, image_size=image_size)
        lm_errs_max_all = calc_landmark_nme_per_img(lm_gt, lm_preds_max, ocular_norm,
                                                    list(landmarks_only_outline)+list(landmarks_no_outline),
                                                    image_size=image_size)
        disp_X_recon = vis.add_error_to_images(disp_X_recon, lm_errs_max, loc='br-2', format_string='{:>5.2f}', vmax=15)
        disp_X_recon = vis.add_error_to_images(disp_X_recon, lm_errs_max_outline, loc='br-1', format_string='{:>5.2f}', vmax=15)
        disp_X_recon = vis.add_error_to_images(disp_X_recon, lm_errs_max_all, loc='br', format_string='{:>5.2f}', vmax=15)
    disp_X_recon = vis.add_landmarks_to_images(disp_X_recon, lm_gt[:, :, :2], color=gt_color, draw_wireframe=True)

    disp_X_recon = vis.add_landmarks_to_images(disp_X_recon, lm_preds_max[:, :, :2][:nimgs],
                                               color=pred_color, draw_wireframe=True,
                                               gt_landmarks=lm_gt[:, :, :2], draw_gt_offsets=True, lm_errs=nme_per_lm,
                                               draw_dots=True, radius=2)

    # print ssim errors
    ssim = np.zeros(nimgs)
    for i in range(nimgs):
        # ssim[i] = compare_ssim(input_images[i], recon_images[i], data_range=1.0, multichannel=True)
        ssim[i] = compare_ssim(input_images[i], recon_images[i], data_range=1.0, channel_axis=-1)
    if not clean:
        disp_X_recon = vis.add_error_to_images(disp_X_recon, 1 - ssim, loc='bl-1', format_string='{:>4.2f}',
                                               vmax=0.8, vmin=0.2)
    # print ssim torch errors
    if ssim_maps is not None and not clean:
        disp_X_recon = vis.add_error_to_images(disp_X_recon, ssim_maps.reshape(len(ssim_maps), -1).mean(axis=1),
                                               loc='bl-2', format_string='{:>4.2f}', vmin=0.0, vmax=0.4)

    # print("shapes", np.array(disp_images).shape, np.array(disp_X_recon).shape, np.array(disp_plots_3d).shape)
    rows = [vis.make_grid(disp_images, nCols=nimgs, normalize=False),
            # vis.make_grid(disp_X_recon, nCols=nimgs)] # ,
            vis.make_grid(disp_plots_3d, nCols=nimgs)]

    if horizontal:
        assert(nimgs == 1)
        disp_rows = vis.make_grid(rows, nCols=2)
    else:
        disp_rows = vis.make_grid(rows, nCols=1)
    wnd_title = 'Predicted Landmarks '
    if ds is not None:
        wnd_title += ds.__class__.__name__

    # cv2.normalize(disp_rows, disp_rows, 0, 255, norm_type=cv2.NORM_MINMAX)
    # cv2.imshow(wnd_title, cv2.cvtColor(disp_rows, cv2.COLOR_RGB2BGR))
    # cv2.waitKey(wait)

    dirs = config.get_dataset_paths('outputs')[0]
    save_path = os.path.join(dirs, "landmarks/", session_name, "ssim_errors")
    os.makedirs(save_path, exist_ok=True)
    name_file = "pred_lmk_epoch_" + str(epoch) + "_iter_" + str(curr_iter) + ".jpg"
    # name_file = "pred_lmk_" + timestr + ".jpg"
    cv2.normalize(disp_rows, disp_rows, 0, 255, norm_type=cv2.NORM_MINMAX)
    # cv2.imwrite(os.path.join(save_path, "pred_lmk_" + timestr + "cv2.jpg"), cv2.cvtColor(disp_rows,cv2.COLOR_RGB2BGR))
    plt.imsave(os.path.join(save_path, name_file), disp_rows)

