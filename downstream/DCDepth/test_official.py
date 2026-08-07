import os
import os.path as osp
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2

from models.utils import MetricTool
from utils import compute_errors_pth, flip_lr, post_process_depth

try:
    from mmcv import Config
except ImportError:
    from mmengine import Config

from dataloaders import DATAMODULES
from models import MODELS
from tqdm import tqdm
from utils import inv_normalize


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        'config_name',
        type=str,
        help='The name of configuration file.'
    )
    parser.add_argument(
        'resume_from',
        type=str,
        help='Where to load checkpoint.'
    )
    parser.add_argument(
        '--vis',
        action='store_true'
    )
    parser.add_argument(
        '--do_kb_crop', 
        help='if set, crop input images as kitti benchmark images', 
        action='store_true'
    )

    return parser.parse_args()


def to_metric_depth(output: torch.Tensor, output_space: str):
    if output_space == 'log':
        return torch.exp(output)
    elif output_space == 'metric':
        return output
    else:
        raise NotImplementedError


@torch.no_grad()
def test(post_process: bool):
    # warnings.filterwarnings(action='ignore')
    device = torch.device('cuda')

    # configurations
    arg = parse_args()
    cfg = Config.fromfile(osp.join('configs/', f'{arg.config_name}.yaml'))

    # data module
    data = DATAMODULES.build(
        {
            'type': cfg.dataset.name,
            'cfg': cfg
        }
    )
    data.setup('test')
    loader = data.test_dataloader()
    dataset = cfg.dataset.name

    # read list file
    list_file = {
        'nyu': 'nyudepthv2_test_files_with_gt.txt',
        'kitti_eigen': 'eigen_test_files_with_gt.txt',
        'kitti_official': 'kitti_official_test.txt',
        'tofdc': 'TOFDC/TOFDC_test.txt'
    }[dataset]
    with open(osp.join('data_splits', list_file), 'r') as f:
        lines = f.readlines()

    num_test_samples = len(lines)
    print('now testing {} files with {}'.format(num_test_samples, arg.resume_from))

    # model
    model = MODELS.build({
        'type': cfg.model.type,
        'cfg': cfg
    })
    print(f'Testing with model {type(model).__name__}...')
    checkpoint = torch.load(arg.resume_from, map_location='cpu')
    if 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']
    model.load_state_dict(checkpoint, strict=True)
    print(f'Checkpoint is successfully loaded from {arg.resume_from}.')
    model = model.model
    model.to(device)
    model.eval()

    # define checkpoint configurations
    work_dir = osp.join(cfg.training.work_dir, arg.config_name)

    # create vis folder
    if arg.vis:
        vis_dir = osp.join(work_dir, 'vis')
        os.makedirs(vis_dir, exist_ok=True)

    pred_depths = []
    # begin test
    for batch_idx, batch in enumerate(tqdm(loader)):
        # fetch data
        image = batch['image'].to(device)

        depths = model(image)
        depth = to_metric_depth(depths[-1], cfg.model.output_space)
        if post_process:
            image_flipped = flip_lr(image)
            depth_flipped = to_metric_depth(model(image_flipped)[-1], cfg.model.output_space)
            pred_depth = post_process_depth(depth, depth_flipped)
        else:
            pred_depth = depth

        pred_depth = pred_depth.squeeze()

        # pred_depth[pred_depth < cfg.dataset.min_depth] = cfg.dataset.min_depth
        # pred_depth[pred_depth > cfg.dataset.max_depth] = cfg.dataset.max_depth
        # pred_depth[torch.isinf(pred_depth)] = cfg.dataset.max_depth
        # pred_depth[torch.isnan(pred_depth)] = cfg.dataset.min_depth

        pred_depth = pred_depth.cpu().numpy().squeeze()

        if arg.do_kb_crop:
            height, width = 352, 1216
            top_margin = int(height - 352)
            left_margin = int((width - 1216) / 2)
            pred_depth_uncropped = np.zeros((height, width), dtype=np.float32)
            pred_depth_uncropped[top_margin:top_margin + 352, left_margin:left_margin + 1216] = pred_depth
            pred_depth = pred_depth_uncropped

        pred_depths.append(pred_depth)

    save_name = osp.join(work_dir, 'result')
    
    print('Saving result pngs..')
    if not os.path.exists(save_name):
        try:
            os.mkdir(save_name)
            os.mkdir(save_name + '/raw')
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    for s in tqdm(range(num_test_samples)):
        filename_pred_png = save_name + '/raw/' + lines[s].split()[0].split('/')[-1].replace('.jpg', '.png')
        
        pred_depth = pred_depths[s]
        
        pred_depth_scaled = pred_depth * 256.0
        
        pred_depth_scaled = pred_depth_scaled.astype(np.uint16)
        cv2.imwrite(filename_pred_png, pred_depth_scaled, [cv2.IMWRITE_PNG_COMPRESSION, 0])



if __name__ == '__main__':
    test(True)
