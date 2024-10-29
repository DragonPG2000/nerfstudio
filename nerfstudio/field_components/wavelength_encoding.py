# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
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

"""Space distortions which occur as a function of time."""

import abc
from enum import Enum
from typing import Any, Dict, Tuple

import torch
from jaxtyping import Float
from torch import Tensor, nn

from nerfstudio.field_components.encodings import Encoding, NeRFEncoding
from nerfstudio.field_components.mlp import MLP

class Wavelength_encoding(nn.Module):
    """Wavelegth encoding for the Rendering
    Args:
        position_encoding: An encoding for the XYZ of distortion
        mlp_num_layers: Number of layers in distortion MLP
        mlp_layer_width: Size of hidden layer for the MLP
        skip_connections: Number of layers for skip connections in the MLP
    """

    def __init__(
        self,
        mlp_num_layers: int = 4,
        mlp_layer_width: int = 2048,
        skip_connections: Tuple[int] = (4,),
    ) -> None:
        super().__init__()
        
        self.position_encoding = NeRFEncoding(
            in_dim=1, num_frequencies=10, min_freq_exp=0.0, max_freq_exp=8.0, include_input=True
        )
        self.mlp = MLP(
            in_dim=self.position_encoding.get_out_dim(),
            out_dim=1,
            num_layers=mlp_num_layers,
            layer_width=mlp_layer_width,
            skip_connections=skip_connections,
        )

    def forward(self, wavelengths: Float[Tensor, "*bs 141 1"]) -> Float[Tensor, "*bs 141 1"]:
        """
        Args:
            wavelengths: Wavelengths for each sample
            times: times for each sample

        Returns:
            Translated positions.
        """
        encoded_positions = self.position_encoding(wavelengths)
        output = self.mlp(encoded_positions)
        return output