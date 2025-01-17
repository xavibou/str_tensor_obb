# Copyright (c) OpenMMLab. All rights reserved.
from .builder import build_dataset  # noqa: F401, F403
from .dota import DOTADataset, DOTAv15Dataset, DOTAv2Dataset  # noqa: F401, F403
from .hrsc import HRSCDataset  # noqa: F401, F403
from .pipelines import *  # noqa: F401, F403
from .sar import SARDataset  # noqa: F401, F403
from .rsg import RSGDataset
from .dior import DIORDataset
from .icdar import IcdarDataset
from .magentine import MagentineDataset
from .msra import MSRADataset
from .h2rbox import MSRADatasetWSOODDataset, HRSCWSOODDataset, DIORWSOODDataset, DOTAWSOODDataset, DOTAv15WSOODDataset, DOTAv2WSOODDataset, SARWSOODDataset, RSGWSOODDataset, MagentineWSOODDataset, IcdarWSOODDataset

__all__ = ['SARDataset', 'DOTADataset', 'DOTAv15Dataset', 'DOTAv2Dataset', 
           'build_dataset', 'HRSCDataset', 'RSGDataset', 'DIORDataset',
           'HRSCWSOODDataset', 'DIORWSOODDataset', 'DOTAWSOODDataset', 
           'DOTAv15WSOODDataset', 'DOTAv2WSOODDataset', 'MSRADataset', 
           'SARWSOODDataset', 'RSGWSOODDataset', 'IcdarDataset', 'MagentineDataset', 
           'MSRADatasetWSOODDataset', 'MagentineWSOODDataset', 'IcdarWSOODDataset']
