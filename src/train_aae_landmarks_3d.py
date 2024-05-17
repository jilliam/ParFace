import datetime
import os
import time
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn.modules.distance
import torch.optim as optim
import torch.utils.data as td
import torchvision

sys.path.append('.')
from src.aae_training import bool_str
from src.constants import TRAIN, VAL

import src.aae_training as aae_training
import src.config as cfg
import csl_common.utils.ds_utils as ds_utils
from datasets import wflw, menpo, menpo2d, wlp300, aflw20003d, palsy
from csl_common.utils import log
from csl_common.utils.nn import to_numpy, Batch
from src.train_aae_unsupervised import AAETraining
from landmarks import lmutils, lmvis, fabrec
import landmarks.lmconfig as lmcfg
from src.train_aae_landmarks import AAELandmarkTraining

minx, maxx = -256., 256.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def normalize_depth(z_coord):
    return (2 * (z_coord - minx) / (maxx - minx)) - 1


def denormalize_depth(z_coord_norm):
    return (z_coord_norm / 2) * (maxx - minx)


class AAELandmarkTraining3D(AAETraining):

    def __init__(self, datasets, args, session_name='debug', **kwargs):
        args.reset = False  # just to make sure we don't reset the discriminator by accident
        try:
            ds = datasets[TRAIN]
        except KeyError:
            ds = datasets[VAL]
        self.num_landmarks = ds.NUM_LANDMARKS
        self.all_landmarks = ds.ALL_LANDMARKS
        self.landmarks_no_outline = ds.LANDMARKS_NO_OUTLINE
        self.landmarks_only_outline = ds.LANDMARKS_ONLY_OUTLINE

        super().__init__(datasets, args, session_name, macro_batch_size=0, **kwargs)

        self.optimizer_lm_head = optim.Adam(self.saae.LMH.parameters(), lr=args.lr_heatmaps, betas=(0.9, 0.999),
                                            eps=1e-8)
        self.optimizer_lm_3d_head = optim.Adam(self.saae.LMH_3D.parameters(), lr=args.lr_heatmaps, betas=(0.9, 0.999),
                                               eps=1e-8)
        self.optimizer_E = optim.Adam(self.saae.Q.parameters(), lr=0.00002, betas=(0.9, 0.999))
        # self.optimizer_G = optim.Adam(self.saae.P.parameters(), lr=0.00002, betas=(0.9, 0.999))

    def _get_network(self, pretrained):
        return fabrec.Fabrec(self.num_landmarks, input_size=self.args.input_size, z_dim=self.args.embedding_dims)

    @staticmethod
    def print_eval_metrics(nmes, var_nmes, gtes, var_gtes, show=False):
        def ced_curve(_nmes):
            y = []
            x = np.linspace(0, 10, 50)
            for th in x:
                recall = 1.0 - lmutils.calc_landmark_failure_rate(_nmes, th)
                recall *= 1 / len(x)
                y.append(recall)
            return x, y
        def _plot_curves(bins, ced_values, legend_entries, title, x_limit=0.08,
                         colors=None, linewidth=3, fontsize=12, figure_size=None):
            # number of curves
            n_curves = len(ced_values)

            # if no colors are provided, sample them from the jet colormap
            if colors is None:
                cm = plt.get_cmap('jet')
                colors = [cm(1. * i / n_curves)[:3] for i in range(n_curves)]

            # plot all curves
            fig = plt.figure()
            ax = plt.gca()
            for i, y in enumerate(ced_values):
                plt.plot(bins, y, color=colors[i],
                         linestyle='-',
                         linewidth=linewidth,
                         label=legend_entries[i])

            # legend
            ax.legend(prop={'size': fontsize}, loc=0)

            # set axes limits
            ax.set_xlim([0., x_limit])
            ax.set_ylim([0., 1.])
            ax.set_yticks(np.arange(0., 1.1, 0.1))

            # grid
            plt.grid('on', linestyle='--', linewidth=0.5)

            # figure size
            if figure_size is not None:
                fig.set_size_inches(np.asarray(figure_size))

        def auc(recalls):
            return np.sum(recalls)

        # for err_scale in np.linspace(0.1, 1, 10):
        for err_scale in [1.0]:
            # print('\nerr_scale', err_scale)
            # print(np.clip(lm_errs_max_all, a_min=0, a_max=10).mean())

            std_nme = np.sqrt(var_nmes.mean())
            fr = lmutils.calc_landmark_failure_rate(nmes * err_scale)
            X, Y = ced_curve(nmes)
            std_gte = np.sqrt(var_gtes.mean())

            log.info('NME:   {:>6.3f}'.format(nmes.mean() * err_scale))
            log.info('STD_NME:   {:>6.3f}'.format(std_nme.mean() * err_scale))
            log.info('FR@10: {:>6.3f} ({})'.format(fr * 100, np.sum(nmes.mean(axis=1) > 10)))
            log.info('AUC:   {:>6.4f}'.format(auc(Y)))
            log.info('GTE:   {:>6.3f}'.format(gtes.mean() * err_scale))
            log.info('STD_GTE:   {:>6.3f}'.format(std_gte.mean() * err_scale))

            if show:
                import matplotlib.pyplot as plt
                fig, axes = plt.subplots(1, 4)
                axes[0].plot(X, Y)
                axes[1].hist(nmes.mean(axis=1), bins=20)

                count, bins_count = np.histogram(nmes.mean(axis=1), bins=10)
                pdf = count / sum(count)
                cdf = np.cumsum(pdf)
                axes[2].plot(bins_count[1:], cdf, label="CDF")

                count, bins_count = np.histogram(gtes.mean(axis=1), bins=10)
                pdf = count / sum(count)
                cdf = np.cumsum(pdf)
                axes[3].plot(bins_count[1:], cdf, label="CDF")


    def _print_iter_stats(self, stats):
        means = pd.DataFrame(stats).mean(numeric_only=True).to_dict()
        current = stats[-1]
        nmes = current.get('nmes', np.zeros(0))

        str_stats = ['[{ep}][{i}/{iters_per_epoch}] '
                     'l_rec={avg_loss_recon:.3f} '
                     'l_lms={avg_loss_lms:.4f} '                     
                     'loss_lms_3d={avg_loss_lms_3d:.4f} '  # 'loss_3d={avg_loss_lms_3d:.4f} '
                     'err_lms={avg_err_lms:.2f}/{avg_err_lms_outline:.2f}/{avg_err_lms_all:.2f} '
                     '{t_data:.2f}/{t_proc:.2f}/{t:.2f}s ({total_iter:06d} {total_time})'][0]
        log.info(str_stats.format(
            ep=current['epoch'] + 1, i=current['iter'] + 1, iters_per_epoch=self.iters_per_epoch,
            avg_loss=means.get('loss', -1),
            avg_loss_recon=means.get('loss_recon', -1),
            avg_ssim=1.0 - means.get('ssim', -1),
            avg_ssim_torch=means.get('ssim_torch', -1),
            avg_loss_activations=means.get('loss_activations', -1),
            avg_loss_lms=means.get('loss_lms', -1),
            avg_loss_lms_3d=means.get('loss_lms_3d', -1),
            # avg_loss_lms_3d=means.get('loss_3d', -1),
            avg_z_l1=means.get('z_l1', -1),
            avg_z_recon_mean=means.get('z_recon_mean', -1),
            t=means['iter_time'],
            t_data=means['time_data_loading'],
            t_proc=means['time_processing'],
            avg_err_lms=nmes[:, self.landmarks_no_outline].mean(),
            avg_err_lms_outline=nmes[:, self.landmarks_only_outline].mean(),
            avg_err_lms_all=nmes[:, self.all_landmarks].mean(),
            total_iter=self.total_iter + 1, total_time=str(datetime.timedelta(seconds=self._training_time()))
        ))

    def _print_epoch_summary(self, epoch_stats, epoch_start_time, eval=False):
        means = pd.DataFrame(epoch_stats).mean(numeric_only=True).to_dict()

        if 'nmes' in self.epoch_stats[0]:  # try:
            nmes = np.concatenate([s['nmes'] for s in self.epoch_stats if 'nmes' in s])
        else:  # except KeyError:
            nmes = np.zeros((1, 100))

        if 'var_nmes' in self.epoch_stats[0]:  # try:
            var_nmes = np.concatenate([s['var_nmes'] for s in self.epoch_stats if 'var_nmes' in s])
        else:  # except KeyError:
            var_nmes = np.zeros((1, 100))

        if 'gtes' in self.epoch_stats[0]:
            gtes = np.concatenate([s['gtes'] for s in self.epoch_stats if 'gtes' in s])
        else:
            gtes = np.zeros((1, 100))

        if 'var_gtes' in self.epoch_stats[0]:
            var_gtes = np.concatenate([s['var_gtes'] for s in self.epoch_stats if 'var_gtes' in s])
        else:
            var_gtes = np.zeros((1, 100))

        duration = int(time.time() - epoch_start_time)
        log.info("{}".format('-' * 100))
        str_stats = ['           '
                     'l_rec={avg_loss_recon:.3f} '
                     # 'ssim={avg_ssim:.3f} '
                     # 'ssim_torch={avg_ssim_torch:.3f} '
                     # 'z_mu={avg_z_recon_mean:.3f} '
                     'l_lms={avg_loss_lms:.4f} '
                     'l_lms_3d={avg_loss_lms_3d:.4f} '
                     'err_lms={avg_err_lms:.2f}/{avg_err_lms_outline:.2f}/{avg_err_lms_all:.2f} '
                     'err_lms_3d={avg_err_lms_3d:.2f}/{avg_err_lms_3d_outline:.2f}/{avg_err_lms_3d_all:.2f} '
                     '\tT: {time_epoch}'][0]
        log.info(str_stats.format(
            iters_per_epoch=self.iters_per_epoch,
            avg_loss=means.get('loss', -1),
            avg_loss_recon=means.get('loss_recon', -1),
            avg_ssim=1.0 - means.get('ssim', -1),
            avg_ssim_torch=means.get('ssim_torch', -1),
            avg_loss_lms=means.get('loss_lms', -1),
            avg_loss_lms_3d=means.get('loss_lms_3d', -1),
            # avg_loss_lms_3d=means.get('loss_3d', -1),
            avg_loss_lms_cnn=means.get('loss_lms_cnn', -1),
            avg_err_lms=nmes[:, self.landmarks_no_outline].mean(),
            avg_err_lms_outline=nmes[:, self.landmarks_only_outline].mean(),
            avg_err_lms_all=nmes[:, self.all_landmarks].mean(),
            avg_err_lms_3d=gtes[:, self.landmarks_no_outline].mean(),
            avg_err_lms_3d_outline=gtes[:, self.landmarks_only_outline].mean(),
            avg_err_lms_3d_all=gtes[:, self.all_landmarks].mean(),
            avg_z_recon_mean=means.get('z_recon_mean', -1),
            t=means['iter_time'],
            t_data=means['time_data_loading'],
            t_proc=means['time_processing'],
            total_iter=self.total_iter + 1, total_time=str(datetime.timedelta(seconds=self._training_time())),
            time_epoch=str(datetime.timedelta(seconds=duration))))
        try:
            recon_errors = np.concatenate([stats['l1_recon_errors'] for stats in self.epoch_stats])
            rmse = np.sqrt(np.mean(recon_errors ** 2))
            log.info("RMSE: {} ".format(rmse))
        except KeyError:
            # print("no l1_recon_error")
            pass

        if self.args.eval and nmes is not None:
            # benchmark_mode = hasattr(self.args, 'benchmark')
            # self.print_eval_metrics(nmes, show=benchmark_mode)
            self.print_eval_metrics(nmes, var_nmes, gtes, var_gtes, show=False)

    def eval_epoch(self):
        log.info("")
        log.info("Evaluating '{}'...".format(self.session_name))
        # log.info("")

        epoch_start_time = time.time()
        self.epoch_stats = []
        self.saae.eval()

        self._run_epoch(self.datasets[VAL], eval=True)
        # print average loss and accuracy over epoch
        self._print_epoch_summary(self.epoch_stats, epoch_start_time)
        return self.epoch_stats

    def train(self, num_epochs=None):

        log.info("")
        log.info("Starting training session '{}'...".format(self.session_name))
        # log.info("")

        while num_epochs is None or self.epoch < num_epochs:
            log.info('')
            log.info('Epoch {}/{}'.format(self.epoch + 1, num_epochs))
            log.info('=' * 10)

            self.epoch_stats = []
            epoch_start_time = time.time()

            self._run_epoch(self.datasets[TRAIN])

            # save model every few epochs
            if (self.epoch + 1) % self.snapshot_interval == 0:
                log.info("*** saving snapshot *** ")
                self._save_snapshot(is_best=False)

            # print average loss and accuracy over epoch
            self._print_epoch_summary(self.epoch_stats, epoch_start_time)

            if self._is_eval_epoch():
                self.eval_epoch()

            self.epoch += 1

        time_elapsed = time.time() - self.time_start_training
        log.info('Training completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    def _run_epoch(self, dataset, eval=False):
        batchsize = self.args.batchsize_eval if eval else self.args.batchsize
        self.iters_per_epoch = int(len(dataset) / batchsize)
        self.iter_start_time = time.time()
        self.iter_in_epoch = 0
        dataloader = td.DataLoader(dataset, batch_size=batchsize,
                                   shuffle=not eval,
                                   num_workers=self.workers, drop_last=not eval)
        for data in dataloader:
            self._run_batch(data, eval=eval)
            self.total_iter += 1
            self.saae.total_iter = self.total_iter
            self.iter_in_epoch += 1

    def _run_batch(self, data, eval=False, ds=None):
        time_data_loading = time.time() - self.iter_start_time
        time_proc_start = time.time()
        iter_stats = {'time_data_loading': time_data_loading}

        batch = Batch(data, eval=eval)

        self.saae.zero_grad()
        self.saae.eval()

        # Freeze LMH
        self.saae.LMH.zero_grad()
        self.saae.LMH.eval()

        # Freeze LMH 3D
        self.saae.LMH_3D.zero_grad()
        self.saae.LMH_3D.eval()

        input_images = batch.target_images if batch.target_images is not None else batch.images

        with torch.set_grad_enabled(self.args.train_encoder):
            z_sample = self.saae.Q(input_images)

        iter_stats.update({'z_recon_mean': z_sample.mean().item()})

        #######################
        # Reconstruction phase
        #######################
        with torch.set_grad_enabled(self.args.train_encoder and not eval):
            X_recon = self.saae.P(z_sample)

        # calculate reconstruction error for debugging and reporting
        with torch.no_grad():
            iter_stats['loss_recon'] = aae_training.loss_recon(batch.images, X_recon)

        #######################
        # Landmark predictions
        #######################
        train_lmhead = not eval
        lm_preds = None
        with torch.set_grad_enabled(train_lmhead):
            # self.saae.LMH.train(False)  # self.saae.LMH.train(train_lmhead)
            self.saae.LMH.train(self.args.finetune_2d)  # self.saae.LMH.train(train_lmhead)
            self.saae.LMH_3D.train(train_lmhead)
            X_lm_hm = self.saae.LMH(self.saae.P)
            if batch.lm_heatmaps is not None and batch.landmarks_3d is not None:

                if self.args.use_adaptive_wing_loss:
                    loss_lms = self.adaptive_wing_loss(batch.lm_heatmaps, X_lm_hm) * 100 * 3
                else:
                    loss_lms = F.mse_loss(batch.lm_heatmaps, X_lm_hm) * 100 * 3
                iter_stats.update({'loss_lms': loss_lms.item()})

                # Prepare z coordinate. Normalize between [-1, 1]
                depth_coordinate = normalize_depth(batch.landmarks_3d[..., 2])

                # Create input of 3D landmark detector: Heatmaps + Latent code z_sample
                if self.args.use_groundtruth:     # Concat z_sample with groundtruth heatmap
                    inp = torch.cat([batch.lm_heatmaps, z_sample.unsqueeze(2).unsqueeze(3).expand(-1, -1, 128, 128)], 1)
                elif self.args.use_predicted:     # concat z_sample with predicted heatmap from LHM
                    inp = torch.cat([X_lm_hm, z_sample.unsqueeze(2).unsqueeze(3).expand(-1, -1, 128, 128)], 1)
                # inp = torch.cat([torchvision.transforms.Resize((128, 128))(X_recon),
                #                  z_sample.unsqueeze(2).unsqueeze(3).expand(-1, -1, 128, 128)], 1)

                # Forward pass LMH_3D
                depth_pred = self.saae.LMH_3D(inp)

                loss_3d = F.mse_loss(depth_coordinate, depth_pred) * 100 * 3
                iter_stats.update({'loss_lms_3d': loss_3d.item()})  # iter_stats['loss_3d'] = loss_3d.item()

            if eval or self._is_printout_iter(eval):
                # expensive, so only calculate when every N iterations
                X_lm_hm = lmutils.smooth_heatmaps(X_lm_hm)
                lm_preds_max = self.saae.heatmaps_to_landmarks(X_lm_hm)

                if self.args.eval_3d or self._is_printout_iter(eval):
                    lm_gt = to_numpy(batch.landmarks_3d)
                    # lm_preds = to_numpy(torch.cat([lm_preds_max, denormalize_depth(depth_pred).unsqueeze(0)], 1))
                    lm_tensor = torch.from_numpy(lm_preds_max).to(device)
                    depth_tensor = denormalize_depth(depth_pred).unsqueeze(dim=2).to(device)
                    lm_preds = to_numpy(torch.cat([lm_tensor, depth_tensor], 2))
                else:
                    lm_gt = to_numpy(batch.landmarks)
                    lm_preds = to_numpy(lm_preds_max)

                # nmes = lmutils.calc_landmark_nme(lm_gt[:, :, :2], lm_preds[:, :, :2],
                nmes, var_nmes = lmutils.calc_landmark_nme_var(lm_gt[:, :, :2], lm_preds[:, :, :2],
                                                               ocular_norm=self.args.ocular_norm,
                                                               image_size=self.args.input_size)
                iter_stats.update({'nmes': nmes})
                iter_stats.update({'var_nmes': var_nmes})
                # print('var_nmes', var_nmes.mean())

                # gtes = lmutils.calc_landmark_nme(lm_gt, lm_preds, ocular_norm='outer',  # self.args.ocular_norm,
                gtes, var_gtes = lmutils.calc_landmark_nme_var(lm_gt, lm_preds, ocular_norm='outer', # self.args.ocular_norm,
                                                 image_size=self.args.input_size)
                iter_stats.update({'gtes': gtes})
                iter_stats.update({'var_gtes': var_gtes})

        if train_lmhead:
            loss_3d.backward()
            self.optimizer_lm_3d_head.step()

            if self.args.finetune_2d:
                loss_lms.backward()
                self.optimizer_lm_head.step()

            if self.args.train_encoder:
                self.optimizer_E.step()

        # statistics
        iter_stats.update({'epoch': self.epoch, 'timestamp': time.time(),
                           'iter_time': time.time() - self.iter_start_time,
                           'time_processing': time.time() - time_proc_start,
                           'iter': self.iter_in_epoch, 'total_iter': self.total_iter, 'batch_size': len(batch)})
        self.iter_start_time = time.time()

        self.epoch_stats.append(iter_stats)

        # print stats every N mini-batches
        if self._is_printout_iter(eval):
            self._print_iter_stats(self.epoch_stats[-self._print_interval(eval):])

            lmvis.visualize_batch_3d(batch.images, batch.landmarks_3d, X_recon, lm_preds,
                                     target_images=batch.target_images,
                                     ds=ds,
                                     ocular_norm=self.args.ocular_norm,
                                     clean=False,
                                     landmarks_only_outline=self.landmarks_only_outline,
                                     landmarks_no_outline=self.landmarks_no_outline,
                                     f=1.0,
                                     wait=self.wait,
                                     epoch=self.epoch,
                                     curr_iter=self.iter_in_epoch,
                                     session_name=self.session_name,
                                     normalized=self.args.normalize)


def run():
    from csl_common.utils.common import init_random

    if args.seed is not None:
        init_random(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    datasets = {}
    for phase, dsnames, num_samples in zip((TRAIN, VAL),
                                           (args.dataset_train, args.dataset_val),
                                           (args.train_count, args.val_count)):
        train = phase == TRAIN
        name = dsnames[0]
        transform = ds_utils.build_transform(deterministic=not train, daug=args.daug)
        root, cache_root = cfg.get_dataset_paths(name)
        dataset_cls = cfg.get_dataset_class(name)
        datasets[phase] = dataset_cls(root=root,
                                      cache_root=cache_root,
                                      train=train,
                                      max_samples=num_samples,
                                      # use_cache=args.use_cache,
                                      start=args.st,
                                      test_split=args.test_split,
                                      train_split=args.train_split,
                                      align_face_orientation=args.align,
                                      crop_source=args.crop_source,
                                      normalize=args.normalize,
                                      return_landmark_heatmaps=lmcfg.PREDICT_HEATMAP,
                                      with_occlusions=args.occ and train,
                                      landmark_sigma=args.sigma,
                                      transform=transform,
                                      image_size=args.input_size)

    fntr = AAELandmarkTraining3D(datasets, args, session_name=args.sessionname, snapshot_interval=args.save_freq,
                                 workers=args.workers, wait=args.wait)

    torch.backends.cudnn.benchmark = True
    if args.eval_3d:
        fntr.eval_epoch()
    else:
        fntr.train(num_epochs=args.epochs)


if __name__ == '__main__':

    import sys
    import configargparse

    np.set_printoptions(linewidth=np.inf)

    # Disable traceback on Ctrl+c
    import signal

    signal.signal(signal.SIGINT, lambda x, y: sys.exit(0))

    parser = configargparse.ArgParser()
    defaults = {
        'batchsize': 40,
        'train_encoder': False,
        'train_decoder': False
    }
    aae_training.add_arguments(parser, defaults)

    # Dataset
    parser.add_argument('--dataset', default=['menpo'], type=str, help='dataset for training and testing',
                        choices=['w300', 'aflw', 'wflw', 'menpo', 'menpo2d', 'wlp300', 'aflw20003d', 'palsy'], nargs='+')
    parser.add_argument('--test-split', default='common', type=str, help='test set split for 300W/AFLW/WFLW',
                        choices=['challenging', 'common', '300w', 'full', 'frontal', 'parface'] + wflw.SUBSETS)
    parser.add_argument('--train-split', default='6', type=str, help='train set split for ParFace training',
                        choices=['1', '2', '3', '4', '5', '6', 'full'])
    parser.add_argument('--normalize', action="store_true", help="normalize input image")

    # Landmarks
    parser.add_argument('--use-groundtruth', type=bool_str, default=True,
                        help='Use GT 2d landmarks to regress z coordinate')
    parser.add_argument('--use-predicted', type=bool_str, default=False,
                        help='Use predicted 2d landmark to regress z coordinate')
    parser.add_argument('--finetune-2d', type=bool_str, default=False, help='Fine-tune 2d landmark head')

    parser.add_argument('--use-adaptive-wing-loss', type=bool_str, default=False,
                        help='Use adaptive wing loss instead of MSE for landmarks training')
    parser.add_argument('--lr-heatmaps', default=0.00001, type=float, help='learning rate for landmark heatmap outputs')
    parser.add_argument('--sigma', default=7, type=float, help='size of landmarks in heatmap')
    parser.add_argument('-n', '--ocular-norm', default=lmcfg.LANDMARK_OCULAR_NORM, type=str,
                        help='how to normalize landmark errors', choices=['pupil', 'outer', 'none'])
    parser.add_argument('--eval-3d', type=bool_str, default=False)
    parser.add_argument('--gpu', type=str, default='1', help='gpu to use')

    args = parser.parse_args()

    args.dataset_train = args.dataset
    args.dataset_val = args.dataset

    if args.sessionname is None:
        if args.resume:
            modelname = os.path.split(args.resume)[0]
            args.sessionname = modelname
        else:
            args.sessionname = 'debug'

    run()
