import sys

sys.path.append('.')
from src.train_aae_landmarks import AAELandmarkTraining
from src.train_aae_landmarks_3d import AAELandmarkTraining3D
from src.aae_training import AAETraining
import src.config as cfg
from datasets import wflw, w300, aflw, palsy
import src.aae_training as aae_training
from src.constants import VAL
from csl_common.utils import log
from landmarks import lmconfig


class FabrecEval(AAELandmarkTraining):
    def __init__(self, datasets, args, session_name, **kwargs):
        args.reset = False  # just to make sure we don't reset the discriminator by accident
        ds = datasets[VAL]
        self.num_landmarks = ds.NUM_LANDMARKS
        self.all_landmarks = ds.ALL_LANDMARKS
        self.landmarks_no_outline = ds.LANDMARKS_NO_OUTLINE
        self.landmarks_only_outline = ds.LANDMARKS_ONLY_OUTLINE
        AAETraining.__init__(self, datasets, args, session_name, **kwargs)


class FabrecEval3D(AAELandmarkTraining3D):
    def __init__(self, datasets, args, session_name, **kwargs):
        args.reset = False  # just to make sure we don't reset the discriminator by accident
        ds = datasets[VAL]
        self.num_landmarks = ds.NUM_LANDMARKS
        self.all_landmarks = ds.ALL_LANDMARKS
        self.landmarks_no_outline = ds.LANDMARKS_NO_OUTLINE
        self.landmarks_only_outline = ds.LANDMARKS_ONLY_OUTLINE
        AAETraining.__init__(self, datasets, args, session_name, **kwargs)


def bool_str(x):
    return str(x).lower() in ['True', 'true', '1']


def run(args):
    if args.seed is not None:
        from csl_common.utils.common import init_random
        init_random(args.seed)
    # log.info(json.dumps(vars(args), indent=4))

    datasets = {}
    dsname = args.dataset_val[0]
    root, cache_root = cfg.get_dataset_paths(dsname)
    dataset_cls = cfg.get_dataset_class(dsname)
    datasets[VAL] = dataset_cls(root=root,
                                cache_root=cache_root,
                                train=False,
                                test_split=args.test_split,
                                max_samples=args.val_count,
                                start=args.st,
                                use_cache=args.use_cache,
                                align_face_orientation=args.align,
                                crop_source=args.crop_source,
                                normalize=args.normalize,
                                return_landmark_heatmaps=True,
                                landmark_sigma=args.sigma,
                                image_size=args.input_size)
    print(datasets[VAL])

    if not args.eval_3d:
        fntr = FabrecEval(datasets, args, args.sessionname, workers=args.workers, wait=args.wait)
    else:
        fntr = FabrecEval3D(datasets, args, args.sessionname, workers=args.workers, wait=args.wait)

    import torch
    torch.backends.cudnn.benchmark = True
    fntr.eval_epoch()


if __name__ == '__main__':
    # Disable traceback on Ctrl+c
    import sys
    import signal

    signal.signal(signal.SIGINT, lambda x, y: sys.exit(0))

    import configargparse
    import numpy as np

    np.set_printoptions(linewidth=np.inf)

    parser = configargparse.ArgParser()
    aae_training.add_arguments(parser)

    # Dataset
    parser.add_argument('--dataset', default=['w300'], type=str, choices=cfg.get_registered_dataset_names(),
                        nargs='+', help='dataset for training and testing')
    parser.add_argument('--test-split', default='full', type=str, help='test set split for 300W/AFLW/WFLW',
                        choices=['challenging', 'common', '300w', 'full', 'frontal', 'parface', '0_to_30', '30_to_60', '60_to_90'] + wflw.SUBSETS)
    parser.add_argument('--normalize', action="store_true", help="normalize input image")

    parser.add_argument('--use-adaptive-wing-loss', type=bool_str, default=False, help='Use adaptive wing loss 2D FA')
    parser.add_argument('--eval-3d', type=bool_str, default=False)
    parser.add_argument('--benchmark', default=False, action='store_true', help='evaluate performance on testset')

    # Landmarks
    parser.add_argument('--sigma', default=7, type=float, help='size of landmarks in heatmap')
    parser.add_argument('--ocular-norm', default=lmconfig.LANDMARK_OCULAR_NORM, type=str,
                        help='how to normalize landmark errors', choices=['pupil', 'outer', 'bb', 'none'])
    parser.add_argument('--use-groundtruth', type=bool_str, default=False,
                        help='Use GT 2d landmarks to regress z coordinate')
    parser.add_argument('--use-predicted', type=bool_str, default=True,
                        help='Use predicted 2d landmark to regress z coordinate')
    parser.add_argument('--finetune-2d', type=bool_str, default=False, help='Fine-tune 2d landmark head')

    args = parser.parse_args()

    if args.resume is None:
        raise ValueError("Please specify the model to be evaluated: '-r MODELNAME'")

    args.dataset_train = args.dataset
    args.dataset_val = args.dataset

    args.eval = True
    args.batchsize_eval = 10
    args.wait = 0
    args.workers = 0
    args.print_freq_eval = 1
    args.epochs = 1

    if args.benchmark:
        log.info('Switching to benchmark mode...')
        args.batchsize_eval = 10 #50
        args.wait = 10
        args.workers = 4
        args.print_freq_eval = 10
        args.epochs = 1
        args.val_count = None

    if args.sessionname is None:
        args.sessionname = args.resume

    run(args)
