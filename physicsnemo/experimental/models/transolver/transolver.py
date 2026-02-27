
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

r"""Experimental Transolver with configurable attention type.

Experimental Transolver that inherits from core Transolver and supports
multiple attention backends via ``attention_type``: physics attention or FLARE.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from jaxtyping import Float

from physicsnemo.models.transolver import Transolver as CoreTransolver
from physicsnemo.models.transolver.transolver import _TransolverMlp

from physicsnemo.experimental.nn import FLARE


class _FLAREBlock(nn.Module):
    r"""Transformer block with FLARE attention instead of physics attention.

    Mirrors TransolverBlock structure but uses FLARE for the attention layer.
    FLARE does not support Transformer Engine.
    """

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        dropout: float,
        act: str = "gelu",
        mlp_ratio: int = 4,
        last_layer: bool = False,
        out_dim: int = 1,
        n_global_queries: int = 32,
    ) -> None:
        super().__init__()
        self.last_layer = last_layer
        dim_head = hidden_dim // num_heads

        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.Attn = FLARE(
            dim=hidden_dim,
            heads=num_heads,
            dim_head=dim_head,
            dropout=dropout,
            n_global_queries=n_global_queries,
            use_te=False,
        )
        self.ln_mlp1 = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            _TransolverMlp(
                in_features=hidden_dim,
                hidden_features=hidden_dim * mlp_ratio,
                out_features=hidden_dim,
                act_layer=act,
                use_te=False,
            ),
        )
        if last_layer:
            self.ln_mlp2 = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, out_dim),
            )

    def forward(
        self, fx: Float[torch.Tensor, "B N C"]
    ) -> Float[torch.Tensor, "B N C_out"]:
        fx = self.Attn(self.ln_1(fx)) + fx
        fx = self.ln_mlp1(fx) + fx
        if self.last_layer:
            return self.ln_mlp2(fx)
        return fx


class Transolver(CoreTransolver):
    r"""Experimental Transolver with configurable attention type.

    Inherits from the core Transolver and adds ``attention_type`` to select
    between physics attention (default) and FLARE attention. When
    ``attention_type="flare"``, FLARE is used and Transformer Engine is
    disabled (use_te forced to False).

    Parameters
    ----------
    functional_dim : int
        Dimension of input values, not including embeddings.
    out_dim : int
        Dimension of model output.
    embedding_dim : int | None, optional
        Dimension of input embeddings. Required if ``unified_pos=False``.
    n_layers : int, optional
        Number of transformer blocks. Default is 4.
    n_hidden : int, optional
        Hidden dimension. Default is 256.
    dropout : float, optional
        Dropout rate. Default is 0.0.
    n_head : int, optional
        Number of attention heads. Default is 8.
    act : str, optional
        Activation function name. Default is ``"gelu"``.
    mlp_ratio : int, optional
        MLP hidden ratio. Default is 4.
    slice_num : int, optional
        Number of physics slices (physics) or global queries (flare).
        Default is 32.
    unified_pos : bool, optional
        Whether to use unified positional embeddings. Default is ``False``.
    ref : int, optional
        Reference grid size for unified position. Default is 8.
    structured_shape : None | tuple[int, ...], optional
        Shape of structured data. ``None`` for unstructured. Default is ``None``.
    use_te : bool, optional
        Whether to use Transformer Engine. Ignored when ``attention_type="flare"``.
        Default is ``True``.
    time_input : bool, optional
        Whether to include time embeddings. Default is ``False``.
    plus : bool, optional
        Whether to use Transolver++ variant (physics only). Default is ``False``.
    attention_type : str, optional
        Attention backend: ``"physics"`` (default) or ``"flare"``.

    Forward
    -------
    Same as :class:`~physicsnemo.models.transolver.Transolver`.

    Outputs
    -------
    Same as :class:`~physicsnemo.models.transolver.Transolver`.

    See Also
    --------
    :class:`~physicsnemo.models.transolver.Transolver` : Core Transolver model.
    :class:`~physicsnemo.experimental.nn.FLARE` : FLARE attention layer.
    """

    def __init__(
        self,
        functional_dim: int,
        out_dim: int,
        embedding_dim: int | None = None,
        n_layers: int = 4,
        n_hidden: int = 256,
        dropout: float = 0.0,
        n_head: int = 8,
        act: str = "gelu",
        mlp_ratio: int = 4,
        slice_num: int = 32,
        unified_pos: bool = False,
        ref: int = 8,
        structured_shape: None | tuple[int, ...] = None,
        use_te: bool = True,
        time_input: bool = False,
        plus: bool = False,
        attention_type: str = "physics",
    ) -> None:
        if attention_type not in ("physics", "flare"):
            raise ValueError(
                f"attention_type must be 'physics' or 'flare', got {attention_type!r}"
            )

        # FLARE does not support TE
        effective_use_te = use_te if attention_type == "physics" else False
        if attention_type == "flare" and use_te:
            import warnings

            from physicsnemo.core.warnings import ExperimentalFeatureWarning

            warnings.warn(
                "attention_type='flare' requires use_te=False; Transformer Engine "
                "is incompatible with FLARE. Forcing use_te=False.",
                ExperimentalFeatureWarning,
                stacklevel=2,
            )

        super().__init__(
            functional_dim=functional_dim,
            out_dim=out_dim,
            embedding_dim=embedding_dim,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout=dropout,
            n_head=n_head,
            act=act,
            mlp_ratio=mlp_ratio,
            slice_num=slice_num,
            unified_pos=unified_pos,
            ref=ref,
            structured_shape=structured_shape,
            use_te=effective_use_te,
            time_input=time_input,
            plus=plus,
        )

        self.attention_type = attention_type

        if attention_type == "flare":
            # Replace physics attention blocks with FLARE blocks
            self.blocks = nn.ModuleList(
                [
                    _FLAREBlock(
                        num_heads=n_head,
                        hidden_dim=n_hidden,
                        dropout=dropout,
                        act=act,
                        mlp_ratio=mlp_ratio,
                        last_layer=(i == n_layers - 1),
                        out_dim=out_dim,
                        n_global_queries=slice_num,
                    )
                    for i in range(n_layers)
                ]
            )
            self.initialize_weights()
