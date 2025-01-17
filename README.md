# Structure Tensor Representation for Robust Oriented Object Detection

[Xavier Bou](https://xavibou.github.io/), [Gabriele Facciolo](http://gfacciol.github.io/), [Rafael Grompone](https://scholar.google.fr/citations?user=GLovf4UAAAAJ&hl=en), [Jean-Michel Morel](https://sites.google.com/site/jeanmichelmorelcmlaenscachan/), [Thibaud Ehret](https://tehret.github.io)

Centre Borelli, ENS Paris-Saclay
---

[![arXiv](https://img.shields.io/badge/paper-arxiv-brightgreen)](https://arxiv.org/abs/2411.10497)
[![Google Drive](https://img.shields.io/badge/files-Google_Drive-blueviolet)](https://drive.google.com/drive/folders/1AnMQrW5UsMA6Hx-PM78iOnCPAzmgzPy3?usp=sharing)
[![Project](https://img.shields.io/badge/project%20web-github.io-red)]()

This repository is the official implementation of the paper [Structure Tensor Representation for Robust Oriented Object Detection](https://arxiv.org/abs/2411.10497).

---


Oriented object detection predicts orientation in addition to object location and bounding box. Precisely predicting orientation remains challenging due to angular periodicity, which introduces boundary discontinuity issues and symmetry ambiguities. Inspired by classical works on edge and corner detection, this paper proposes to represent orientation in oriented bounding boxes as a structure tensor. This representation combines the strengths of Gaussian-based methods and angle-coder solutions, providing a simple yet efficient approach that is robust to angular periodicity issues without additional hyperparameters. Extensive evaluations across five datasets demonstrate that the proposed structure tensor representation outperforms previous methods in both fully-supervised and weakly supervised tasks, achieving high precision in angular prediction with minimal computational overhead. Thus, this work establishes structure tensors as a robust and modular alternative for encoding orientation in oriented object detection. We make our code publicly available, allowing for seamless integration into existing object detectors.

![Alt text](./demo/general_diagram.png)

## 🛠️ Usage

More instructions on installation, pretrained models, training and evaluation, please refer to [MMRotate 0.3.4](README_en.md).
  
- Clone this repo:

  ```bash
  git clone https://github.com/yangxue0827/RSG-MMRotate
  cd RSG-MMRotate/
  ```

- Create a conda virtual environment and activate it:
  
  ```bash
  conda create -n rsg-mmrotate python=3.8 -y
  conda activate rsg-mmrotate
  ```

- Install Pytorch:

  ```bash
  pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
  ```

- Install requirements:

  ```bash
  pip install openmim
  mim install mmcv-full
  mim install mmdet
  
  cd mmrotate
  pip install -r requirements/build.txt
  pip install -v -e .

  pip install timm
  pip install ipdb

  # Optional, only for G-Rep
  git clone git@github.com:KinglittleQ/torch-batch-svd.git
  cd torch-batch-svd/
  python setup.py install
  ```

## 🚀 Released Models

### Oriented Object Detection

|  Detector  | mAP | Configs | Download | Note |
| :--------: |:---:|:-------:|:--------:|:----:|
| Deformable DETR | 17.1 | [deformable_detr_r50_1x_rsg](configs/ars_detr/deformable_detr_r50_1x_rsg.py) | [log](https://huggingface.co/yangxue/RSG-MMRotate/raw/main/deformable_detr_r50_1x_rsg.log) \| [ckpt](https://huggingface.co/yangxue/RSG-MMRotate/resolve/main/deformable_detr_r50_1x_rsg-fe862bb3.pth?download=true) |
| ARS-DETR | 28.1 | [dn_arw_arm_arcsl_rdetr_r50_1x_rsg](configs/ars_detr/dn_arw_arm_arcsl_rdetr_r50_1x_rsg.py) | [log](https://huggingface.co/yangxue/RSG-MMRotate/raw/main/dn_arw_arm_arcsl_rdetr_r50_1x_rsg.log) \| [ckpt](https://huggingface.co/yangxue/RSG-MMRotate/resolve/main/dn_arw_arm_arcsl_rdetr_r50_1x_rsg-cbb34897.pth?download=true) |
| RetinaNet | 21.8 | [rotated_retinanet_hbb_r50_fpn_1x_rsg_oc](configs/rotated_retinanet/rotated_retinanet_hbb_r50_fpn_1x_rsg_oc.py) | [log](https://huggingface.co/yangxue/RSG-MMRotate/raw/main/rotated_retinanet_hbb_r50_fpn_1x_rsg_oc.log) \| [ckpt](https://huggingface.co/yangxue/RSG-MMRotate/resolve/main/rotated_retinanet_hbb_r50_fpn_1x_rsg_oc-3ec35d77.pth?download=true) |
| ATSS | 20.4 | [rotated_atss_hbb_r50_fpn_1x_rsg_oc](configs/rotated_atss/rotated_atss_hbb_r50_fpn_1x_rsg_oc.py) | [log](https://huggingface.co/yangxue/RSG-MMRotate/raw/main/rotated_atss_hbb_r50_fpn_1x_rsg_oc.log) \| [ckpt](https://huggingface.co/yangxue/RSG-MMRotate/resolve/main/rotated_atss_hbb_r50_fpn_1x_rsg_oc-f65f07c2.pth?download=true) | 
|  KLD  |  25.0  |   [rotated_retinanet_hbb_kld_r50_fpn_1x_rsg_oc](configs/kld/rotated_retinanet_hbb_kld_r50_fpn_1x_rsg_oc.py)  |  [log](https://huggingface.co/yangxue/RSG-MMRotate/raw/main/rotated_retinanet_hbb_kld_r50_fpn_1x_rsg_oc.log) \| [ckpt](https://huggingface.co/yangxue/RSG-MMRotate/resolve/main/rotated_retinanet_hbb_kld_r50_fpn_1x_rsg_oc-343a0b83.pth?download=true) |
|  GWD  |  25.3  |   [rotated_retinanet_hbb_gwd_r50_fpn_1x_rsg_oc](configs/gwd/rotated_retinanet_hbb_gwd_r50_fpn_1x_rsg_oc.py)  |  [log](https://huggingface.co/yangxue/RSG-MMRotate/raw/main/rotated_retinanet_hbb_gwd_r50_fpn_1x_rsg_oc.log) \| [ckpt](https://huggingface.co/yangxue/RSG-MMRotate/resolve/main/rotated_retinanet_hbb_gwd_r50_fpn_1x_rsg_oc-566d2398.pth?download=true) |
| KFIoU |  25.5  |   [rotated_retinanet_hbb_kfiou_r50_fpn_1x_rsg_oc](configs/kfiou/rotated_retinanet_hbb_kfiou_r50_fpn_1x_rsg_oc.py)  |  [log](https://huggingface.co/yangxue/RSG-MMRotate/raw/main/rotated_retinanet_hbb_kfiou_r50_fpn_1x_rsg_oc.log) \| [ckpt](https://huggingface.co/yangxue/RSG-MMRotate/resolve/main/rotated_retinanet_hbb_kfiou_r50_fpn_1x_rsg_oc-198081a6.pth?download=true) |
| DCFL | 29.0 | [dcfl_r50_fpn_1x_rsg_le135](configs/dcfl/dcfl_r50_fpn_1x_rsg_le135.py) | [log](https://huggingface.co/yangxue/RSG-MMRotate/raw/main/dcfl_r50_fpn_1x_rsg_le135.log) \| [ckpt](https://huggingface.co/yangxue/RSG-MMRotate/resolve/main/dcfl_r50_fpn_1x_rsg_le135-a5945790.pth?download=true) |
| R<sup>3</sup>Det | 23.7 | [r3det_r50_fpn_1x_rsg_oc](configs/r3det/r3det_r50_fpn_1x_rsg_oc.py) | [log](https://huggingface.co/yangxue/RSG-MMRotate/raw/main/r3det_r50_fpn_1x_rsg_oc.log) \| [ckpt](https://huggingface.co/yangxue/RSG-MMRotate/resolve/main/r3det_r50_fpn_1x_rsg_oc-c8c4a5e5.pth?download=true) |
| S2A-Net | 27.3 | [s2anet_r50_fpn_1x_rsg_le135](configs/s2anet/s2anet_r50_fpn_1x_rsg_le135.py) | [log](https://huggingface.co/yangxue/RSG-MMRotate/raw/main/s2anet_r50_fpn_1x_rsg_le135.log) \| [ckpt](https://huggingface.co/yangxue/RSG-MMRotate/resolve/main/s2anet_r50_fpn_1x_rsg_le135-42887a81.pth?download=true) |
| FCOS  |  28.1  |   [rotated_fcos_r50_fpn_1x_rsg_le90](configs/rotated_fcos/rotated_fcos_r50_fpn_1x_rsg_le90.py)  |  [log](https://huggingface.co/yangxue/RSG-MMRotate/raw/main/rotated_fcos_r50_fpn_1x_rsg_le90.log) \| [ckpt](https://huggingface.co/yangxue/RSG-MMRotate/resolve/main/rotated_fcos_r50_fpn_1x_rsg_le90-a579fbf7.pth?download=true) | 
| CSL | 27.4 | [rotated_fcos_csl_gaussian_r50_fpn_1x_rsg_le90](configs/rotated_fcos/rotated_fcos_csl_gaussian_r50_fpn_1x_rsg_le90.py) | [log](https://huggingface.co/yangxue/RSG-MMRotate/raw/main/rotated_fcos_csl_gaussian_r50_fpn_1x_rsg_le90.log) \| [ckpt](https://huggingface.co/yangxue/RSG-MMRotate/resolve/main/rotated_fcos_csl_gaussian_r50_fpn_1x_rsg_le90-6ab9a42a.pth?download=true) | 
| PSC | 30.5 | [rotated_fcos_psc_r50_fpn_1x_rsg_le90](configs/psc/rotated_fcos_psc_r50_fpn_1x_rsg_le90.py) | [log](https://huggingface.co/yangxue/RSG-MMRotate/raw/main/rotated_fcos_psc_r50_fpn_1x_rsg_le90.log) \| [ckpt](https://huggingface.co/yangxue/RSG-MMRotate/resolve/main/rotated_fcos_psc_r50_fpn_1x_rsg_le90-7acce1be.pth?download=true) |
| H2RBox-v2 | 27.3 | [h2rbox_v2p_r50_fpn_1x_rsg_le90](configs/h2rbox_v2p/h2rbox_v2p_r50_fpn_1x_rsg_le90.py) | [log](https://huggingface.co/yangxue/RSG-MMRotate/raw/main/h2rbox_v2p_r50_fpn_1x_rsg_le90.log) \| [ckpt](https://huggingface.co/yangxue/RSG-MMRotate/resolve/main/h2rbox_v2p_r50_fpn_1x_rsg_le90-25409050.pth?download=true) |
| RepPoints  | 19.7 | [rotated_reppoints_r50_fpn_1x_rsg_oc](configs/rotated_reppoints/rotated_reppoints_r50_fpn_1x_rsg_oc.py) | [log](https://huggingface.co/yangxue/RSG-MMRotate/raw/main/rotated_reppoints_r50_fpn_1x_rsg_oc.log) \| [ckpt](https://huggingface.co/yangxue/RSG-MMRotate/resolve/main/rotated_reppoints_r50_fpn_1x_rsg_oc-7a6c59b9.pth?download=true) |
| CFA | 25.1 | [cfa_r50_fpn_1x_rsg_le135](configs/cfa/cfa_r50_fpn_1x_rsg_le135.py) | [log](https://huggingface.co/yangxue/RSG-MMRotate/raw/main/cfa_r50_fpn_1x_rsg_le135.log) \| [ckpt](https://huggingface.co/yangxue/RSG-MMRotate/resolve/main/cfa_r50_fpn_1x_rsg_le135-287f6b84.pth?download=true) |
| Oriented RepPoints  |  27.0  |   [oriented_reppoints_r50_fpn_1x_rsg_le135](configs/oriented_reppoints/oriented_reppoints_r50_fpn_1x_rsg_le135.py)  |  [log](https://huggingface.co/yangxue/RSG-MMRotate/raw/main/oriented_reppoints_r50_fpn_1x_rsg_le135.log) \| [ckpt](https://huggingface.co/yangxue/RSG-MMRotate/resolve/main/oriented_reppoints_r50_fpn_1x_rsg_le135-06389ea6.pth?download=true) | |
| G-Rep | 26.9 | [g_reppoints_r50_fpn_1x_rsg_le135](configs/g_reppoints/g_reppoints_r50_fpn_1x_rsg_le135.py) | [log](https://huggingface.co/yangxue/RSG-MMRotate/raw/main/g_reppoints_r50_fpn_1x_rsg_le135.log) \| [ckpt](https://huggingface.co/yangxue/RSG-MMRotate/resolve/main/g_reppoints_r50_fpn_1x_rsg_le135-ec243141.pth?download=true) |
| SASM  |  28.2  |   [sasm_reppoints_r50_fpn_1x_rsg_oc](configs/sasm_reppoints/sasm_reppoints_r50_fpn_1x_rsg_oc.py)  |  [log](https://huggingface.co/yangxue/RSG-MMRotate/raw/main/sasm_reppoints_r50_fpn_1x_rsg_oc.log) \| [ckpt](https://huggingface.co/yangxue/RSG-MMRotate/resolve/main/sasm_reppoints_r50_fpn_1x_rsg_oc-4f1ca558.pth?download=true) | [p_bs=2](https://github.com/yangxue0827/RSG-MMRotate/blob/05c0064cbcd5c44437321b50e1d2d4ee9b4445db/mmrotate/models/detectors/single_stage_crop.py#L310) |
| Faster RCNN | 32.6 | [rotated_faster_rcnn_r50_fpn_1x_rsg_le90](configs/rotated_faster_rcnn/rotated_faster_rcnn_r50_fpn_1x_rsg_le90.py) | [log](https://huggingface.co/yangxue/RSG-MMRotate/raw/main/rotated_faster_rcnn_r50_fpn_1x_rsg_le90.log) \| [ckpt](https://huggingface.co/yangxue/RSG-MMRotate/resolve/main/rotated_faster_rcnn_r50_fpn_1x_rsg_le90-9a832bc2.pth?download=true) |
| Gliding Vertex | 30.7 | [gliding_vertex_r50_fpn_1x_rsg_le90](configs/gliding_vertex/gliding_vertex_r50_fpn_1x_rsg_le90.py) | [log](https://huggingface.co/yangxue/RSG-MMRotate/raw/main/gliding_vertex_r50_fpn_1x_rsg_le90.log) \| [ckpt](https://huggingface.co/yangxue/RSG-MMRotate/resolve/main/gliding_vertex_r50_fpn_1x_rsg_le90-5c0bc879.pth?download=true) |
| Oriented RCNN | 33.2 | [oriented_rcnn_r50_fpn_1x_rsg_le90](configs/oriented_rcnn/oriented_rcnn_r50_fpn_1x_rsg_le90.py) | [log](https://huggingface.co/yangxue/RSG-MMRotate/raw/main/oriented_rcnn_r50_fpn_1x_rsg_le90.log) \| [ckpt](https://huggingface.co/yangxue/RSG-MMRotate/resolve/main/oriented_rcnn_r50_fpn_1x_rsg_le90-0b66f6a4.pth?download=true) |
| RoI Transformer | 35.7 | [roi_trans_r50_fpn_1x_rsg_le90](configs/roi_trans/roi_trans_r50_fpn_1x_rsg_le90.py) | [log](https://huggingface.co/yangxue/RSG-MMRotate/raw/main/roi_trans_r50_fpn_1x_rsg_le90.log) \| [ckpt](https://huggingface.co/yangxue/RSG-MMRotate/resolve/main/roi_trans_r50_fpn_1x_rsg_le90-e42f64d6.pth?download=true) |
| LSKNet-T | 34.7 | [lsk_t_fpn_1x_rsg_le90](configs/lsknet/lsk_t_fpn_1x_rsg_le90.py) | [log](https://huggingface.co/yangxue/RSG-MMRotate/raw/main/lsk_t_fpn_1x_rsg_le90.log) \| [ckpt](https://huggingface.co/yangxue/RSG-MMRotate/resolve/main/lsk_t_fpn_1x_rsg_le90-19635614.pth?download=true) |
| LSKNet-S | 37.8 | [lsk_s_fpn_1x_rsg_le90](configs/lsknet/lsk_s_fpn_1x_rsg_le90.py) | [log](https://huggingface.co/yangxue/RSG-MMRotate/raw/main/lsk_s_fpn_1x_rsg_le90.log) \| [ckpt](https://huggingface.co/yangxue/RSG-MMRotate/resolve/main/lsk_s_fpn_1x_rsg_le90-b77cdbc2.pth?download=true) |
| PKINet-S | 32.8 | [pkinet_s_fpn_1x_rsg_le90](configs/pkinet/pkinet_s_fpn_1x_rsg_le90.py) | [log](https://huggingface.co/yangxue/RSG-MMRotate/raw/main/pkinet_s_fpn_1x_rsg_le90.log) \| [ckpt](https://huggingface.co/yangxue/RSG-MMRotate/resolve/main/pkinet_s_fpn_1x_rsg_le90-e1459201.pth?download=true) |
| ReDet | 39.1 | [redet_re50_refpn_1x_rsg_le90](configs/redet/redet_re50_refpn_1x_rsg_le90.py) | [log](https://huggingface.co/yangxue/RSG-MMRotate/raw/main/redet_re50_refpn_1x_rsg_le90.log) \| [ckpt](https://huggingface.co/yangxue/RSG-MMRotate/resolve/main/redet_re50_refpn_1x_rsg_le90-d163f450.pth?download=true) | [ReResNet50](https://huggingface.co/yangxue/RSG-MMRotate/resolve/main/re_resnet50_c8_batch256-25b16846.pth?download=true) |
| Oriented RCNN | 40.7 | [oriented_rcnn_swin-l_fpn_1x_rsg_le90](configs/oriented_rcnn/oriented_rcnn_swin-l_fpn_1x_rsg_le90.py) | [log](https://huggingface.co/yangxue/RSG-MMRotate/raw/main/oriented_rcnn_swin-l_fpn_1x_rsg_le90.log) \| [ckpt](https://huggingface.co/yangxue/RSG-MMRotate/resolve/main/oriented_rcnn_swin-l_fpn_1x_rsg_le90-fe6f9e2d.pth?download=true) | [Swin-L](https://huggingface.co/yangxue/RSG-MMRotate/resolve/main/swin_large_patch4_window7_224_22k_20220412-aeecf2aa.pth?download=true) |

## 🖊️ Citation

If you find this work helpful for your research, please consider giving this repo a star ⭐ and citing our paper:

```bibtex
@article{li2024scene,
  title={Scene Graph Generation in Large-Size VHR Satellite Imagery: A Large-Scale Dataset and A Context-Aware Approach},
  author={Li, Yansheng and Wang, Linlin and Wang, Tingzhu and Yang, Xue and Luo, Junwei and Wang, Qi and Deng, Youming and Wang, Wenbin and Sun, Xian and Li, Haifeng and Dang, Bo and Zhang, Yongjun and Yi, Yu and Yan, Junchi},
  journal={arXiv preprint arXiv:},
  year={2024}
}
```

## 📃 License

This project is released under the [Apache license](LICENSE). Parts of this project contain code and models from other sources, which are subject to their respective licenses.

## Acknowledgements

This repository is heavily borrowed from the MMRotate framework. For more details, see:  
**STAR: A First-Ever Dataset and A Large-Scale Benchmark for Scene Graph Generation in Large-Size Satellite Imagery.**
