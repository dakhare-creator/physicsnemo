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

"""GAFLARE (Geometry-Aware FLARE) attention layer and transformer block.

This module provides the GAFLARE attention mechanism, 
an alternative to the GALE attention mechanism of the GeoTransolver.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Float

import physicsnemo  # noqa: F401 for docs
from physicsnemo.core.version_check import check_version_spec

# Check optional dependency availability
TE_AVAILABLE = check_version_spec("transformer_engine", "0.1.0", hard_fail=False)
if TE_AVAILABLE:
    import transformer_engine.pytorch as te


class GAFLARE(nn.Module):
    r"""GAFLARE: Geometry-Aware FLARE attention layer.
    Adopted:
    - FLARE attention: Fast Low-rank Attention Routing Engine
        paper: https://arxiv.org/abs/2508.12594
    - GeoTransolver context:
        paper: https://arxiv.org/abs/2512.20399

    GAFLARE is an alternative to the GALE attention mechanism of the GeoTransolver 
    It support cross-attention with a context vector, built from geometry and global embeddings.
    GAFLARE combines self-attention on learned physical state slices with cross-attention
    to geometry-aware context, using a learnable mixing weight to blend the two.

    Parameters
    ----------
    dim : int
        Input dimension of the features.
    heads : int, optional
        Number of attention heads. Default is 8.
    dim_head : int, optional
        Dimension of each attention head. Default is 64.
    dropout : float, optional
        Dropout rate. Default is 0.0.
    slice_num : int, optional
        Number of learned queries. Default is 64.
    use_te : bool, optional
        Whether to use Transformer Engine backend when available. Default is False.
    context_dim : int, optional
        Dimension of the context vector for cross-attention. Default is 0.

    Forward
    -------
    x : tuple[torch.Tensor, ...]
        Tuple of input tensors, each of shape :math:`(B, N, C)` where :math:`B` is
        batch size, :math:`N` is number of tokens, and :math:`C` is number of channels.
    context : tuple[torch.Tensor, ...] | None, optional
        Context tensor for cross-attention of shape :math:`(B, H, S_c, D_c)` where
        :math:`H` is number of heads, :math:`S_c` is number of context slices, and
        :math:`D_c` is context dimension. If ``None``, only self-attention is applied.
        Default is ``None``.

    Outputs
    -------
    list[torch.Tensor]
        List of output tensors, each of shape :math:`(B, N, C)`, same shape as inputs.

    Notes
    -----
    The mixing between self-attention and cross-attention is controlled by a learnable
    parameter ``state_mixing`` which is passed through a sigmoid function to ensure
    the mixing weight stays in :math:`[0, 1]`.

    See Also
    --------
    :class:`GALE` : Original GeoTransolver GALE attention class.
    :class:`GALE_block` : Transformer block using GAFLARE attention.

    Examples
    --------
    >>> import torch
    >>> gaflare = GAFLARE(dim=256, heads=8, dim_head=32, context_dim=32)
    >>> x = (torch.randn(2, 100, 256),)  # Single input tensor in tuple
    >>> context = torch.randn(2, 8, 64, 32)  # Context for cross-attention
    >>> outputs = gaflare(x, context)
    >>> len(outputs)
    1
    >>> outputs[0].shape
    torch.Size([2, 100, 256])
    """

    def __init__(
        self,
        dim,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        slice_num: int = 64,
        use_te: bool = True,
        context_dim: int = 0,
        **kwargs,
    ):
        self.use_te = False # te will disable FlashAttention for different size of q and k
        self.scale = 1.     # FLARE scale is 1.0
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = dim_head * heads

        linear_layer = te.Linear if self.use_te else nn.Linear

        # Global queries for FLARE self-attention
        self.q_global = nn.Parameter(torch.randn(1, heads, slice_num, dim_head))

        # Linear projections for self-attention
        self.in_project_x = linear_layer(dim, inner_dim)
        self.self_k = linear_layer(dim_head, dim_head)
        self.self_v = linear_layer(dim_head, dim_head)

        # Linear projections for cross-attention
        self.cross_q = linear_layer(dim_head, dim_head)
        self.cross_k = linear_layer(context_dim, dim_head)
        self.cross_v = linear_layer(context_dim, dim_head)

        # te attention
        if self.use_te:
            self.attn_fn = te.DotProductAttention(
                num_attention_heads=self.heads,
                kv_channels=self.dim_head,
                attention_dropout=dropout,
                qkv_format="bshd",
                softmax_scale=self.scale
            )

        # Linear projection for output
        self.out_linear = linear_layer(inner_dim, dim)
        self.out_dropout = nn.Dropout(dropout)

        # Learnable mixing weight between self and cross attention
        self.state_mixing = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        x: tuple[Float[torch.Tensor, "batch tokens channels"], ...],
        context: Float[torch.Tensor, "batch heads context_slices context_dim"]
        | None = None,
    ) -> list[Float[torch.Tensor, "batch tokens channels"]]:
        r"""Forward pass of the FLARE module.

        Applies FLARE attention to the input features.

        Parameters
        ----------
        x : tuple[torch.Tensor, ...]
            Tuple of input tensors, each of shape :math:`(B, N, C)` where :math:`B`
            is batch size, :math:`N` is number of tokens, and :math:`C` is number
            of channels.
        context : torch.Tensor | None, optional
            Context tensor for cross-attention of shape :math:`(B, H, S_c, D_c)`
            where :math:`H` is number of heads, :math:`S_c` is number of context
            slices, and :math:`D_c` is context dimension. If ``None``, only
            self-attention is applied. Default is ``None``.

        Returns
        -------
        list[torch.Tensor]
            List of output tensors, each of shape :math:`(B, N, C)``, same shape
            as inputs.
        """

        # with record_function("forward"):
        x_mid = [self.in_project_x(_x) for _x in x]
        x_mid = [rearrange(
            _x_mid, "B N (h d) -> B N h d", h=self.heads, d=self.dim_head
        ) for _x_mid in x_mid]
        x_mid = [_x_mid.permute(0, 2, 1, 3) for _x_mid in x_mid]  # [B, H, N, D]
        G = [self.q_global.to(dtype=x_mid[0].dtype)] * len(x) 
        k = [self.self_k(_x_mid) for _x_mid in x_mid]
        v = [self.self_v(_x_mid) for _x_mid in x_mid]

        # FLARE: Self Attention
        if self.use_te:
            # Transformer Engine expects (B, S, H, D) format
            G = [rearrange(_G, "b h s d -> b s h d") for _G in G]
            k = [rearrange(_k, "b h s d -> b s h d") for _k in k]
            v = [rearrange(_v, "b h s d -> b s h d") for _v in v]
            z = [self.attn_fn(_G, _k, _v) for _G, _k, _v in zip(G, k, v)]
            z = [rearrange(
                _z, "b s (h d) -> b s h d", h=self.heads, d=self.dim_head
            ) for _z in z]
            self_attention = [self.attn_fn(_k, _G, _z) for _k, _G, _z in zip(k, G, z)]
            self_attention = [rearrange(
                _self_attention, "b s (h d) -> b h s d", h=self.heads, d=self.dim_head
            ) for _self_attention in self_attention]
        else:
            # Use PyTorch's scaled dot-product attention
            z = [F.scaled_dot_product_attention(_G, _k, _v, scale=1.0) for _G, _k, _v in zip(G, k, v)]
            self_attention = [F.scaled_dot_product_attention(_k, _G, _z, scale=1.0) for _k, _G, _z in zip(k, G, z)]

        # apply cross-attention with physical states:
        if context is not None:
            q = [self.cross_q(_x_mid) for _x_mid in x_mid]
            k = self.cross_k(context)
            v = self.cross_v(context)

            if self.use_te:
                q = [rearrange(_q, "b h s d -> b s h d") for _q in q]
                k = rearrange(k, "b h s d -> b s h d")
                v = rearrange(v, "b h s d -> b s h d")
                cross_attention = [self.attn_fn(_q, k, v) for _q in q]
                cross_attention = [rearrange(
                    _cross_attention, "b s (h d) -> b h s d", h=self.heads, d=self.dim_head
                ) for _cross_attention in cross_attention]
            else:
                cross_attention = [F.scaled_dot_product_attention(_q, k, v, scale=1.0) for _q in q]

            # Apply learnable mixing:
            mixing_weight = torch.sigmoid(self.state_mixing)
            outputs = [mixing_weight * _ys + (1 - mixing_weight) * _yc for _ys, _yc in zip(self_attention, cross_attention)]
        else:
            outputs = self_attention

        outputs = [_y.permute(0, 2, 1, 3) for _y in outputs]  # [B, N, H, D]
        outputs = [rearrange(_out, "b n h d -> b n (h d)") for _out in outputs]
        outputs = [self.out_linear(_out) for _out in outputs]
        return [self.out_dropout(_out) for _out in outputs]


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = (torch.randn(2, 100, 256).to(device),)
    context = torch.randn(2, 8, 64, 32).to(device)
    gaflare = GAFLARE(dim=256, heads=8, dim_head=32, context_dim=32).to(device)
    outputs = gaflare(x, context)
    print(outputs[0].shape)
