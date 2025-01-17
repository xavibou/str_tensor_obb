from mmrotate.datasets import build_dataset
from mmrotate.core.visualization import imshow_det_rbboxes
import mmcv
import matplotlib.pyplot as plt
import os

# Define your config file for the dataset
config_file = '/home/boux/code/str_tensor_obb/configs/h2rbox_v2p/h2rbox_v2p_r50_fpn_6x_icdar_le90_str_tensor.py'

# Load the dataset
cfg = mmcv.Config.fromfile(config_file)
dataset = build_dataset(cfg.data.train)

# Choose an index for the image to visualize
index = 0  # Change index to visualize other images
data_info = dataset.load_annotations(dataset.ann_file)

# visualize all images and their ground truths
for i in range(len(data_info)):
    img = mmcv.imread(os.path.join(dataset.img_prefix, data_info[i]['filename']))
    out_file = '/home/boux/code/str_tensor_obb/data/icdar2015/visualizations/' + data_info[i]['filename']
    imshow_det_rbboxes(
        img,
        data_info[i]['ann']['bboxes'],
        data_info[i]['ann']['labels'],
        class_names=dataset.CLASSES,
        show=False,
        out_file=out_file)
    plt.show()