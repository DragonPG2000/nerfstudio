# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Data manager that outputs cameras / images instead of raybundles

Good for things like gaussian splatting which require full cameras instead of the standard ray
paradigm
"""

from __future__ import annotations

import random
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Dict, ForwardRef, Generic, List, Literal, Optional, Tuple, Type, Union, cast, get_args, get_origin

import cv2
import numpy as np
import torch
from rich.progress import track
from torch.nn import Parameter
from typing_extensions import assert_never

from nerfstudio.cameras.camera_utils import fisheye624_project, fisheye624_unproject_helper
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.configs.dataparser_configs import AnnotatedDataParserUnion
from nerfstudio.data.datamanagers.base_datamanager import DataManager, DataManagerConfig, TDataset
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.datasets.hyperspectral_dataset import HyperspectralDataset
from nerfstudio.utils.misc import get_orig_class
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanagerConfig, FullImageDatamanager
from nerfstudio.data.datamanagers.base_datamanager import DataManager, DataManagerConfig, TDataset

#Inherit from FullImageDatamanagerConfig
@dataclass
class HyperspectralDatamanagerConfig(FullImageDatamanagerConfig):
    _target_: str = field(default_factory=lambda: HyperspectralDatamanager)

    



#Inherit from FullImageDatamanager to avoid reimplementing the same logic

class HyperspectralDatamanager(FullImageDatamanager):
    """
    A datamanager that outputs hyperspectral images and cameras instead of raybundles. This makes the
    datamanager more lightweight since we don't have to do generate rays. Useful for hyperspectral
    training e.g. rasterization pipelines
    """

    config: HyperspectralDatamanagerConfig
    train_dataset: TDataset
    eval_dataset: TDataset


    def create_train_dataset(self) -> TDataset:
        return HyperspectralDataset(
            dataparser_outputs=self.train_dataparser_outputs,
            scale_factor=self.config.camera_res_scale_factor,
        )
    
    def create_eval_dataset(self) -> TDataset:
        return HyperspectralDataset(
            dataparser_outputs=self.dataparser.get_dataparser_outputs(split=self.test_split),
            scale_factor=self.config.camera_res_scale_factor,
        )
    



# Path: nerfstudio/data/datamanagers/full_images_datamanager.py