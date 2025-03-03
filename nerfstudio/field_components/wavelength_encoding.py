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

from nerfstudio.field_components.encodings import Encoding, NeRFEncoding, HashEncoding
from nerfstudio.field_components.mlp import MLP

# class Wavelength_encoding(nn.Module):
#     """Wavelegth encoding for the Rendering
#     Args:
#         position_encoding: An encoding for the XYZ of distortion
#         mlp_num_layers: Number of layers in distortion MLP
#         mlp_layer_width: Size of hidden layer for the MLP
#         skip_connections: Number of layers for skip connections in the MLP
#     """

#     def __init__(
#         self,
#         mlp_num_layers: int = 4,
#         mlp_layer_width: int = 2048,
#         skip_connections: Tuple[int] = (4,),
#     ) -> None:
#         super().__init__()
        
#         self.position_encoding = NeRFEncoding(
#             in_dim=1, num_frequencies=10, min_freq_exp=0.0, max_freq_exp=8.0, include_input=True
#         )
#         self.mlp = MLP(
#             in_dim=self.position_encoding.get_out_dim(),
#             out_dim=256,
#             num_layers=mlp_num_layers,
#             layer_width=mlp_layer_width,
#             skip_connections=skip_connections,
#         )

#         self.features_xyz_mlp = MLP(
#             in_dim=3,
#             out_dim=256,
#             num_layers=4,
#             layer_width=2048,
#             skip_connections=skip_connections,
#         )

#         self.final_mlp = MLP(
#             in_dim=512,
#             out_dim=1,
#             num_layers=1,
#             layer_width=2048,
#             skip_connections=skip_connections,
#         )

#     def forward(self, wavelengths: Float[Tensor, "*bs 141 1"], means: Float[Tensor, "*bs 3"]) -> Float[Tensor, "*bs 141 1"]:
#         """
#         Args:
#             wavelengths: Wavelengths for each sample
#             means: means for each sample

#         Returns:
#             Translated positions.
#         """
#         encoded_positions = self.position_encoding(wavelengths)
#         output = self.mlp(encoded_positions)

#         features_xyz = self.features_xyz_mlp(means)
#         features_xyz = features_xyz.mean(dim=0)
#         features_xyz = features_xyz.unsqueeze(-2).unsqueeze(-2).expand_as(output)

#         #concatenate the features
#         output = torch.cat((output, features_xyz), dim=-1)
#         output = self.final_mlp(output)


#         return output
    

# class Wavelength_encoding_without_position(nn.Module):
#     """Wavelegth encoding for the Rendering
#     Args:
#         position_encoding: An encoding for the XYZ of distortion
#         mlp_num_layers: Number of layers in distortion MLP
#         mlp_layer_width: Size of hidden layer for the MLP
#         skip_connections: Number of layers for skip connections in the MLP
#     """

#     def __init__(
#         self,
#         mlp_num_layers: int = 1,
#         mlp_layer_width: int = 2048,
#         skip_connections: Tuple[int] = (4,),
#     ) -> None:
#         super().__init__()
        
#         # self.position_encoding = NeRFEncoding(
#         #     in_dim=1, num_frequencies=10, min_freq_exp=0.0, max_freq_exp=8.0, include_input=True
#         # )
#         self.mlp = MLP(
#             in_dim=1,
#             out_dim=1,
#             num_layers=mlp_num_layers,
#             layer_width=mlp_layer_width,
#             skip_connections=skip_connections,
#         )

#     def forward(self, wavelengths: Float[Tensor, "*bs 141 1"]) -> Float[Tensor, "*bs 141 1"]:
#         """
#         Args:
#             wavelengths: Wavelengths for each sample
#             times: times for each sample

#         Returns:
#             Translated positions.
#         """
#         # encoded_positions = self.position_encoding(wavelengths)
#         encoded_positions = wavelengths
#         output = self.mlp(encoded_positions)
#         return output
    

# class Wavelength_encoding_v2(nn.Module):
#     """Wavelegth encoding for the Rendering
#     Args:
#         position_encoding: An encoding for the XYZ of distortion
#         mlp_num_layers: Number of layers in distortion MLP
#         mlp_layer_width: Size of hidden layer for the MLP
#         skip_connections: Number of layers for skip connections in the MLP
#     """

#     def __init__(
#         self,
#         mlp_num_layers: int = 4,
#         mlp_layer_width: int = 2048,
#         skip_connections: Tuple[int] = (4,),
#     ) -> None:
#         super().__init__()
        
#         self.position_encoding = NeRFEncoding(
#             in_dim=1, num_frequencies=10, min_freq_exp=0.0, max_freq_exp=8.0, include_input=True
#         )
#         self.mlp = MLP(
#             in_dim=self.position_encoding.get_out_dim(),
#             out_dim=1,
#             num_layers=mlp_num_layers,
#             layer_width=mlp_layer_width,
#             skip_connections=skip_connections,
#         )

#         self.features_dc_mlp = MLP(
#             in_dim=141,
#             out_dim=256,
#             num_layers=4,
#             layer_width=2048,
#             skip_connections=skip_connections,
#         )

#     def forward(self, wavelengths: Float[Tensor, "*bs 141 1"], features_dc, features_rest) -> Float[Tensor, "*bs 141 1"]:
#         """
#         Args:
#             wavelengths: Wavelengths for each sample
#             times: times for each sample

#         Returns:
#             Translated positions.
#         """
#         encoded_positions = self.position_encoding(wavelengths)
#         output = self.mlp(encoded_positions)

#         features_dc = features_dc.unsqueeze(-1)
#         features_rest = features_rest.unsqueeze(-1)

#         #
#         return output
    


class View_MLP(nn.Module):
    def __init__(
        self,
        mlp_num_layers: int = 4,
        mlp_layer_width: int = 2048,
        skip_connections: Tuple[int] = (4,),
    ) -> None:
        super().__init__()

        """
        Viewdirs shape:  torch.Size([50000, 3])
        Opacities crop shape:  torch.Size([50000, 1])
        Colors crop shape:  torch.Size([50000, 16, 141])
        Means crop shape:  torch.Size([50000, 3])
        Predicted RGB shape: torch.Size([128, 160, 141])
        """

        #Hash encoding
        num_levels = 8
        min_res = 2
        max_res = 128
        log2_hashmap_size = 4  # Typically much larger tables are used

        resolution = 128
        slice = 0

        # Fixing features_per_level to 3 for easy RGB visualization. Typical value is 2 in networks
        features_per_level = 6

        self.encoder = HashEncoding(
        num_levels=num_levels,
        min_res=min_res,
        max_res=max_res,
        log2_hashmap_size=log2_hashmap_size,
        features_per_level=features_per_level,
        hash_init_scale=0.001,
        implementation="torch",
        )

        self.opacities_mlp = MLP(
            in_dim=512,
            out_dim=1,
            num_layers=1,
            layer_width=2048,
            skip_connections=skip_connections,
        )

        self.colors_mlp = MLP(
            in_dim=512,
            out_dim=16*141,
            num_layers=1,
            layer_width=2048,
            skip_connections=skip_connections,
        )

        self.mlp = MLP(
            in_dim=2305, # 48+16*141+1
            out_dim=512,
            num_layers=mlp_num_layers,
            layer_width=mlp_layer_width,
            skip_connections=skip_connections,
        )

    def forward(self, views,means, opacities, colors):
        """
        Args:
        views: View directions
        means: Centers of each gaussian
        opacities: Opacities of each gaussian
        colors: Colors of each gaussian

        Returns:
            colors: Colors of each gaussian
            opacities: Opacities of each gaussian
        """
        # view_mean = torch.cat((views, means), dim=-1)
        view_mean = views + means
        encoded_view = self.encoder(view_mean)
        print("encoded_view shape: ", encoded_view.shape)

        colors = colors.view(-1, 16*141)

        concatenated_features = torch.cat((encoded_view, colors, opacities), dim=-1)

        output = self.mlp(concatenated_features)

        color = self.colors_mlp(output).view(-1, 16, 141)
        opacities = self.opacities_mlp(output)
        return color, opacities