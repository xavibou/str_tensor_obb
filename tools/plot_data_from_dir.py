# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import warnings

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.runner import load_checkpoint
from mmrotate.models import build_detector
import numpy as np
from mmdet.apis import inference_detector, init_detector, show_result_pyplot
import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
#from mmdet.apis import multi_gpu_test, single_gpu_test
from mmrotate.apis.test import multi_gpu_test, single_gpu_test
from mmdet.datasets import build_dataloader, replace_ImageToTensor

from mmrotate.datasets import build_dataset
from mmrotate.models import build_detector
from mmrotate.utils import (build_ddp, build_dp, compat_cfg, get_device,
                            setup_multi_processes)


def parse_args():
    """Parse parameters."""
    parser = argparse.ArgumentParser(description="MMRotate image inference script")
    parser.add_argument("config", help="Path to the config file")
    parser.add_argument("checkpoint", help="Path to the checkpoint file")
    parser.add_argument("image_dir", help="Path to the directory containing images")
    parser.add_argument("--out-dir", help="Directory to save inference results", default=None)
    parser.add_argument("--show", action="store_true", help="Show the results")
    parser.add_argument(
        "--show-score-thr",
        type=float,
        default=0.3,
        help="Score threshold for visualizing results (default: 0.3)",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if not os.path.isdir(args.image_dir):
        raise ValueError(f"Image directory {args.image_dir} does not exist")

    if args.out_dir is not None:
        os.makedirs(args.out_dir, exist_ok=True)

    # Load configuration and model
    cfg = Config.fromfile(args.config)
    cfg.model.pretrained = None  # Ensure the model is not retrained
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"

    model = init_detector(args.config, args.checkpoint, device='cuda:0')

    
    checkpoint = load_checkpoint(model, args.checkpoint, map_location="cpu")
    model.CLASSES = checkpoint["meta"]["CLASSES"] if "CLASSES" in checkpoint["meta"] else None

    # Wrap model for mixed-precision inference if necessary
    #fp16_cfg = cfg.get("fp16", None)
    #if fp16_cfg is not None:
    #    wrap_fp16_model(model)

    #model = model.to(cfg.device)
    #model.eval()

    # Process images in the directory
    for img_name in os.listdir(args.image_dir):
        img_path = os.path.join(args.image_dir, img_name)
        if not os.path.isfile(img_path):
            continue

        try:
            img = mmcv.imread(img_path)
            result = inference_detector(model, img_path)

            # Visualize or save results
            if args.show or args.out_dir:
                out_file = None
                if args.out_dir:
                    out_file = os.path.join(args.out_dir, img_name)
                model.show_result(
                    img,
                    result,
                    score_thr=args.show_score_thr,
                    show=args.show,
                    out_file=out_file,
                )

        except Exception as e:
            warnings.warn(f"Error processing {img_path}: {e}")


if __name__ == "__main__":
    main()
