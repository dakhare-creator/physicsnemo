# ignore_header_test
# ruff: noqa: E402
""""""

"""
Transolver model. This code was modified from, https://github.com/thuml/Transolver

The following license is provided from their source,

MIT License

Copyright (c) 2024 THUML @ Tsinghua University

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import transformer_engine.pytorch as te

    TE_AVAILABLE = True
except ImportError:
    TE_AVAILABLE = False

from abc import ABC, abstractmethod

from einops import rearrange
from torch.autograd.profiler import record_function
from torch.distributed.tensor.placement_types import Replicate

import physicsnemo  # noqa: F401 for docs
from physicsnemo.distributed import ShardTensor
from physicsnemo.models.meta import ModelMetaData
from physicsnemo.models.module import Module

# from physicsnemo.models.transolver.Embedding import timestep_embedding
# from .Physics_Attention import Physics_Attention_Structured_Mesh_2D
from physicsnemo.models.transolver.Physics_Attention import (
    PhysicsAttentionStructuredMesh2D,
    PhysicsAttentionStructuredMesh3D,
)

# from torch_geometric.nn import fps

ACTIVATION = {
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU(0.1),
    "softplus": nn.Softplus,
    "ELU": nn.ELU,
    "silu": nn.SiLU,
}


class MLP(nn.Module):
    def __init__(
        self, n_input, n_hidden, n_output, n_layers=1, act="gelu", res=True, use_te=True
    ):
        super(MLP, self).__init__()

        if act in ACTIVATION.keys():
            act = ACTIVATION[act]
        else:
            raise NotImplementedError
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_layers = n_layers
        self.res = res

        self.act = act()

        linear_layer = nn.Linear if not use_te else te.Linear

        self.linear_pre = linear_layer(n_input, n_hidden)
        self.linear_post = linear_layer(n_hidden, n_output)
        self.linears = nn.ModuleList(
            [
                nn.Sequential(linear_layer(n_hidden, n_hidden), act())
                for _ in range(n_layers)
            ]
        )

    def forward(self, x):
        x = self.act(self.linear_pre(x))
        for i in range(self.n_layers):
            if self.res:
                x = self.linears[i](x) + x
            else:
                x = self.linears[i](x)
        x = self.linear_post(x)
        return x


class PhysicsAttentionBase(nn.Module, ABC):
    """
    Base class for all physics attention modules.

    Implements key functionality that is common across domains:
    - Slice weighting and computation
    - Attention among slices
    - Deslicing
    - Output Projection

    Each subclass must implement it's own methods for projecting input domain tokens onto the slice space.

    Deliberately, there are not default values for any of the parameters.  It's assumed you will
    assign them in the subclass.

    """

    def __init__(self, dim, heads, dim_head, dropout, slice_num, use_te):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads

        self.scale = dim_head**-0.5

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.ones([1, 1, heads, 1]) * 0.5)
        self.use_te = use_te

        if self.use_te:
            self.in_project_slice = te.Linear(dim_head, slice_num)
        else:
            self.in_project_slice = nn.Linear(dim_head, slice_num)

        for l_i in [self.in_project_slice]:
            torch.nn.init.orthogonal_(l_i.weight)  # use a principled initialization
        if not use_te:
            self.qkv_project = nn.Linear(dim_head, 3 * dim_head, bias=False)
        else:
            # These are used in the transformer engine pass function:
            self.qkv_project = te.Linear(dim_head, 3 * dim_head, bias=False)
            self.attn_fn = te.DotProductAttention(
                num_attention_heads=self.heads,
                kv_channels=self.dim_head,
                attention_dropout=dropout,
                qkv_format="bshd",
                softmax_scale=self.scale,
            )

        if self.use_te:
            self.out_linear = te.Linear(inner_dim, dim)
        else:
            self.out_linear = nn.Linear(inner_dim, dim)

        self.out_dropout = nn.Dropout(dropout)

    @abstractmethod
    def project_input_onto_slices(
        self, x, embedding
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Project the input onto the slice space.
        """
        raise NotImplementedError("Subclasses must implement this method")

    def compute_slices_from_projections(
        self, slice_projections: torch.Tensor, fx: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute slice weights and slice tokens from input projections and latent features.

        In a domain-parallel setting, this function will do an implicit allreduce.
        When we sum over the slice_weights over a sharded dimension
        and use the output, it will resolve Partial->Replicated placement (aka
        allreduce) implicitly.

        Args:
            slice_projections (torch.Tensor):
                The projected input tensor of shape [Batch, N_tokens, N_heads, Slice_num],
                representing the projection of each token onto each slice for each attention head.
            fx (torch.Tensor):
                The latent feature tensor of shape [Batch, N_tokens, N_heads, Head_dim],
                representing the learned states to be aggregated by the slice weights.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - slice_weights: Tensor of shape [Batch, N_tokens, N_heads, Slice_num],
                representing the normalized weights for each slice per token and head.
                - slice_token: Tensor of shape [Batch, N_heads, Slice_num, Head_dim],
                representing the aggregated latent features for each slice, head, and batch.

        Notes:
            - The function first computes a temperature-scaled softmax over the slice projections to obtain slice weights.
            - It then aggregates the latent features (fx) for each slice using these weights.
            - The aggregated features are normalized by the sum of weights for numerical stability.
        """

        with record_function("compute_slices_from_projections"):
            # Project the latent space vectors on to the weight computation space,
            # and compute a temperature adjusted softmax.
            clamped_temp = torch.clamp(self.temperature, min=0.5, max=5).to(
                slice_projections.dtype
            )

            slice_weights = nn.functional.softmax(
                slice_projections / clamped_temp, dim=-1
            )  # [Batch, N_tokens, N_heads, Slice_num]

            # Cast to the computation type (since the parameter is probably fp32)
            slice_weights = slice_weights.to(slice_projections.dtype)

            # This does the projection of the latent space fx by the weights:

            # Computing the slice tokens is a matmul followed by a normalization.
            # It can, unfortunately, overflow in reduced precision, so normalize first:
            slice_norm = slice_weights.sum(1)  # [Batch, N_heads, Slice_num]
            # Sharded note: slice_norm will be a partial sum at this point.
            # That's because the we're summing over the tokens, which are distributed
            normed_weights = slice_weights / (slice_norm[:, None, :, :] + 1e-2)
            # Normed weights has shape
            # (batch, n_tokens, n_heads, slice_num)

            # Sharded note: normed_weights will resolve the partial slice_norm
            # and the output normed_weights will be sharded.
            # fx has shape (Batch, n_tokens, n_heads, head_dim)
            # This matmul needs to contract over the tokens
            # This should produce an output with shape
            # [Batch, N_heads, Slice_num, Head_dim]

            # Like the weight norm, this sum is a **partial** sum since we are summing
            # over the tokens

            slice_token = torch.matmul(
                normed_weights.permute(0, 2, 3, 1), fx.permute(0, 2, 1, 3)
            )

            # Return the original weights, not the normed weights:

            return slice_weights, slice_token

    def compute_slice_attention_te(self, slice_tokens: torch.Tensor) -> torch.Tensor:
        """
        TE implementation of slice attention
        """

        qkv = self.qkv_project(slice_tokens)
        qkv = rearrange(qkv, " b h s (t d) -> t b s h d", t=3, d=self.dim_head)
        q_slice_token, k_slice_token, v_slice_token = qkv.unbind(0)

        out_slice_token2 = self.attn_fn(q_slice_token, k_slice_token, v_slice_token)
        out_slice_token2 = rearrange(
            out_slice_token2, "b s (h d) -> b h s d", h=self.heads, d=self.dim_head
        )

        return out_slice_token2

    def compute_slice_attention_sdpa(self, slice_tokens: torch.Tensor) -> torch.Tensor:
        """
        Torch SDPA implementation of slice attention

        Args:
            slice_tokens (torch.Tensor):
                The slice tokens tensor of shape [Batch, N_heads, Slice_num, Head_dim].

        Returns:
            torch.Tensor:
                The output tensor of shape [Batch, N_heads, Slice_num, Head_dim].
        """
        with record_function("compute_slice_attention_sdpa"):
            # In this case we're using ShardTensor, ensure slice_token is *replicated*

            qkv = self.qkv_project(slice_tokens)

            qkv = rearrange(qkv, " b h s (t d) -> b h s t d", t=3, d=self.dim_head)

            if isinstance(qkv, ShardTensor):
                # This will be a differentiable allreduce
                qkv = qkv.redistribute(placements=[Replicate()])

            q_slice_token, k_slice_token, v_slice_token = qkv.unbind(3)

            out_slice_token = torch.nn.functional.scaled_dot_product_attention(
                q_slice_token, k_slice_token, v_slice_token, is_causal=False
            )

            return out_slice_token

    def project_attention_outputs(
        self, out_slice_token: torch.Tensor, slice_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Project the attended slice tokens back onto the original token space.

        Note that in the distributed case, this will have a replicated and
        sharded inputs.  Slice tokens will be replicated, and slice weights will be sharded.

        Args:
            out_slice_token (torch.Tensor):
                The output tensor from the attention mechanism over slices,
                of shape [Batch, N_heads, Slice_num, Head_dim].
            slice_weights (torch.Tensor):
                The slice weights tensor of shape [Batch, N_tokens, N_heads, Slice_num],
                representing the contribution of each slice to each token.

        Returns:
            torch.Tensor:
                The reconstructed output tensor of shape [Batch, N_tokens, N_heads * Head_dim],
                representing the attended features for each token, with all heads concatenated.

        Notes:
            - The function projects the attended slice tokens back to the token space using the slice weights.
            - The output is reshaped to concatenate all attention heads for each token.
        """
        with record_function("project_attention_outputs"):
            # Slice weights has shape (Batch, n_tokens, n_heads, slice_num)
            # Out slice tokens has shape (Batch, n_heads, slice_num, head_dim)
            # The output of this function needs to have shape
            # (Batch, n_tokens, n_channels) == (Batch, n_tokens, n_heads * head_dim)
            # Note that tokens may be sharded, in which case slice_weights
            # is a sharded tensor and out_slice_token is a replicated tensor

            out_x = torch.einsum("bths,bhsd->bthd", slice_weights, out_slice_token)

            # Condense the last two dimensions:
            out_x = rearrange(out_x, "b t h d -> b t (h d)")

            out_x = self.out_linear(out_x)
            return self.out_dropout(out_x)

    def forward(self, x: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Physics Attention module.

        Input x should have shape of [Batch, N_tokens, N_Channels] ([B, N, C])
        """

        with record_function("forward"):
            # Project the inputs onto learned spaces:
            embedding_mid, fx_mid = self.project_input_onto_slices(x, embedding)

            # Perform the linear projection of learned latent space onto slices:

            slice_projections = self.in_project_slice(embedding_mid)

            # Slice projections has shape [B, N_tokens, N_head, Head_dim], but head_dim may have changed!

            # Use the slice projections and learned spaces to compute the slices, and their weights:
            slice_weights, slice_tokens = self.compute_slices_from_projections(
                slice_projections, fx_mid
            )
            # slice_weights has shape [Batch, N_tokens, N_heads, Slice_num]
            # slice_tokens has shape  [Batch, N_tokens, N_heads, head_dim]

            # Apply attention to the slice tokens
            if self.use_te:
                out_slice_token = self.compute_slice_attention_te(slice_tokens)
            else:
                out_slice_token = self.compute_slice_attention_sdpa(slice_tokens)

            # Shape unchanged

            # Deslice:
            outputs = self.project_attention_outputs(out_slice_token, slice_weights)

            # Outputs now has the same shape as the original input x

            return outputs


class PhysicsAttentionIrregularMesh(PhysicsAttentionBase):
    """
    Specialization of PhysicsAttention to Irregular Meshes
    """

    def __init__(
        self, dim, heads=8, dim_head=64, dropout=0.0, slice_num=64, use_te=True
    ):
        super().__init__(dim, heads, dim_head, dropout, slice_num, use_te)
        inner_dim = dim_head * heads
        if use_te:
            # self.in_project_x = nn.Linear(dim, inner_dim)
            self.in_project_embd = te.Linear(dim, inner_dim)
            self.in_project_fx = te.Linear(dim, inner_dim)
        else:
            # self.in_project_x = nn.Linear(dim, inner_dim)
            self.in_project_embd = nn.Linear(dim, inner_dim)
            self.in_project_fx = nn.Linear(dim, inner_dim)

    def project_input_onto_slices(
        self, x, embedding
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Project the input onto the slice space.

        Args:
            x (torch.Tensor): The input tensor of shape [Batch, N_tokens, N_Channels]

        Returns:
            tuple[torch.Tensor, torch.Tensor]: The projected x and fx tensors of shape [Batch, N_tokens, N_Channels], [Batch, N_tokens, N_heads, Head_dim]

        """
        fx = self.in_project_fx(x)
        fx_mid = rearrange(fx, "B N (h d) -> B N h d", h=self.heads, d=self.dim_head)

        # x_mid = rearrange(
        #     self.in_project_x(x), "B N (h d) -> B N h d", h=self.heads, d=self.dim_head
        # )
        embedding_mid = rearrange(
            self.in_project_embd(embedding),
            "B N (h d) -> B N h d",
            h=self.heads,
            d=self.dim_head,
        )

        return embedding_mid, fx_mid


class FlareAttention(nn.Module):
    """
    Specialization of PhysicsAttention to Irregular Meshes
    """

    def __init__(
        self,
        dim,
        heads=8,
        dim_head=64,
        dropout=0.0,
        slice_num=64,
        n_global_feat=128,
        use_te=True,
    ):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = dim_head * heads
        self.in_project_x = nn.Linear(dim, inner_dim)
        self.kv_project1 = nn.Linear(dim_head, 2 * dim_head)
        # self.qkv_project_global1 = nn.Linear(dim_head, 3 * dim_head)
        self.q_global = nn.Parameter(torch.randn(1, heads, n_global_feat, dim_head))
        self.out_linear = nn.Linear(inner_dim, dim)
        self.out_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, parts: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Input x should have shape of [Batch, N_tokens, N_Channels] ([B, N, C])
        """

        with record_function("forward"):
            x_mid = self.in_project_x(x)
            x_mid = rearrange(
                x_mid, "B N (h d) -> B N h d", h=self.heads, d=self.dim_head
            )
            x_mid = x_mid.permute(0, 2, 1, 3)  # [B, H, N, D]
            kv = self.kv_project1(x_mid)
            kv = rearrange(kv, " B h N (t d) -> B h N t d", t=2, d=self.dim_head)
            k, v = kv.unbind(-2)
            q = self.q_global

            # FLARE: Fast Low-rank Attention Routing Engine
            z = F.scaled_dot_product_attention(q, k, v, scale=1.0)
            y = F.scaled_dot_product_attention(k, q, z, scale=1.0)

            out_x = y.permute(0, 2, 1, 3)  # [B, N, H, D]
            out_x = rearrange(out_x, "b n h d -> b n (h d)")
            out_x = self.out_linear(out_x)
            return self.out_dropout(out_x)


class LatentAttention_topk(nn.Module):
    """
    Specialization of PhysicsAttention to Irregular Meshes
    """

    def __init__(
        self,
        dim,
        heads=8,
        dim_head=64,
        dropout=0.0,
        slice_num=64,
        n_global_feat=128,
        use_te=True,
    ):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = dim_head * heads
        self.in_project_x = nn.Linear(dim, inner_dim)
        self.qkv = nn.Linear(dim_head, 3 * dim_head)

        self.out_linear = nn.Linear(inner_dim, dim)
        self.out_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, parts: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Input x should have shape of [Batch, N_tokens, N_Channels] ([B, N, C])
        """

        with record_function("forward"):
            x_mid = self.in_project_x(x)
            x_mid = rearrange(
                x_mid, "B N (h d) -> B N h d", h=self.heads, d=self.dim_head
            )
            x_mid = x_mid.permute(0, 2, 1, 3)  # [B, H, N, D]
            qkv = self.qkv(x_mid)
            qkv = rearrange(qkv, " B h N (t d) -> B h N t d", t=3, d=self.dim_head)
            q, k, v = qkv.unbind(-2)

            # option 5:
            topk = torch.topk(
                torch.linalg.norm(q, dim=-1), k=128, dim=-1, largest=False
            )
            qg = torch.gather(
                q, -2, topk[1].unsqueeze(-1).expand(-1, -1, -1, q.shape[-1])
            )  # [B, H, K, D]
            kg = torch.gather(
                k, -2, topk[1].unsqueeze(-1).expand(-1, -1, -1, k.shape[-1])
            )  # [B, H, K, D]

            # FLARE: Fast Low-rank Attention Routing Engine
            vg = F.scaled_dot_product_attention(qg, k, v, scale=1.0)
            y = F.scaled_dot_product_attention(q, kg, vg, scale=1.0)

            out_x = y.permute(0, 2, 1, 3)  # [B, N, H, D]
            out_x = rearrange(out_x, "b n h d -> b n (h d)")
            out_x = self.out_linear(out_x)
            return self.out_dropout(out_x)


class LatentAttention_QGGK(nn.Module):
    """
    Specialization of PhysicsAttention to Irregular Meshes
    """

    def __init__(
        self,
        dim,
        heads=8,
        dim_head=64,
        dropout=0.0,
        slice_num=64,
        n_global_feat=128,
        use_te=True,
    ):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = dim_head * heads
        self.in_project_x = nn.Linear(dim, inner_dim)
        self.qkv = nn.Linear(dim_head, 3 * dim_head)

        self.x_global = nn.Parameter(torch.randn(1, heads, n_global_feat, dim_head))

        self.out_linear = nn.Linear(inner_dim, dim)
        self.out_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, parts: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Input x should have shape of [Batch, N_tokens, N_Channels] ([B, N, C])
        """

        with record_function("forward"):
            x_mid = self.in_project_x(x)
            x_mid = rearrange(
                x_mid, "B N (h d) -> B N h d", h=self.heads, d=self.dim_head
            )
            x_mid = x_mid.permute(0, 2, 1, 3)  # [B, H, N, D]
            qkv = self.qkv(x_mid)
            qkv = rearrange(qkv, " B h N (t d) -> B h N t d", t=3, d=self.dim_head)
            q, k, v = qkv.unbind(-2)

            # Option 7: FLARE: Fast Low-rank Attention Routing Engine
            G = self.x_global
            z = F.scaled_dot_product_attention(G, k, v, scale=1.0)
            y = F.scaled_dot_product_attention(q, G, z, scale=1.0)

            out_x = y.permute(0, 2, 1, 3)  # [B, N, H, D]
            out_x = rearrange(out_x, "b n h d -> b n (h d)")
            out_x = self.out_linear(out_x)
            return self.out_dropout(out_x)


class LatentAttention_QGGK_Gorth(nn.Module):
    """
    Specialization of PhysicsAttention to Irregular Meshes
    """

    def __init__(
        self,
        dim,
        heads=8,
        dim_head=64,
        dropout=0.0,
        slice_num=64,
        n_global_feat=128,
        use_te=True,
    ):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = dim_head * heads
        self.in_project_x = nn.Linear(dim, inner_dim)
        self.qkv = nn.Linear(dim_head, 3 * dim_head)

        self.x_global = [
            torch.nn.utils.parametrizations.orthogonal(
                nn.Linear(dim_head, n_global_feat, bias=False), "weight"
            )
            for _ in range(heads)
        ]
        self.col_norms = nn.Parameter(torch.ones(1, heads, 1, dim_head))

        self.out_linear = nn.Linear(inner_dim, dim)
        self.out_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, parts: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Input x should have shape of [Batch, N_tokens, N_Channels] ([B, N, C])
        """

        with record_function("forward"):
            x_mid = self.in_project_x(x)
            x_mid = rearrange(
                x_mid, "B N (h d) -> B N h d", h=self.heads, d=self.dim_head
            )
            x_mid = x_mid.permute(0, 2, 1, 3)  # [B, H, N, D]
            qkv = self.qkv(x_mid)
            qkv = rearrange(qkv, " B h N (t d) -> B h N t d", t=3, d=self.dim_head)
            q, k, v = qkv.unbind(-2)

            # Option 7:
            G = (
                torch.stack([self.x_global[i].weight for i in range(self.heads)], dim=0)
                .unsqueeze(0)
                .to(q.device)
            )  # TODO Make it GPU compatable
            G = G * self.col_norms**2
            # FLARE: Fast Low-rank Attention Routing Engine
            z = F.scaled_dot_product_attention(G, k, v, scale=1.0)
            y = F.scaled_dot_product_attention(q, G, z, scale=1.0)

            out_x = y.permute(0, 2, 1, 3)  # [B, N, H, D]
            out_x = rearrange(out_x, "b n h d -> b n (h d)")
            out_x = self.out_linear(out_x)
            return self.out_dropout(out_x)


class LatentAttention(nn.Module):
    """
    Specialization of PhysicsAttention to Irregular Meshes
    """

    def __init__(
        self,
        dim,
        heads=8,
        dim_head=64,
        dropout=0.0,
        slice_num=64,
        n_global_feat=128,
        use_te=True,
    ):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = dim_head * heads
        self.in_project_x = nn.Linear(dim, inner_dim)
        self.qkv = nn.Linear(dim_head, 3 * dim_head)

        # self.x_global = nn.Parameter(torch.randn(1, heads, n_global_feat, dim_head))
        # self.x_global = torch.nn.utils.parametrizations.orthogonal(nn.Linear(dim_head, n_global_feat, bias=False), "weight")
        self.x_global = [
            torch.nn.utils.parametrizations.orthogonal(
                nn.Linear(dim_head, n_global_feat, bias=False), "weight"
            )
            for _ in range(heads)
        ]
        # self.qk_global = nn.Linear(dim_head, 2 * dim_head)
        self.col_norms = nn.Parameter(torch.ones(1, heads, 1, dim_head))

        self.out_linear = nn.Linear(inner_dim, dim)
        self.out_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, parts: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Input x should have shape of [Batch, N_tokens, N_Channels] ([B, N, C])
        """

        with record_function("forward"):
            x_mid = self.in_project_x(x)
            x_mid = rearrange(
                x_mid, "B N (h d) -> B N h d", h=self.heads, d=self.dim_head
            )
            x_mid = x_mid.permute(0, 2, 1, 3)  # [B, H, N, D]
            qkv = self.qkv(x_mid)
            qkv = rearrange(qkv, " B h N (t d) -> B h N t d", t=3, d=self.dim_head)
            q, k, v = qkv.unbind(-2)

            # # option 1: Winner
            # qkg = self.qk_global(self.x_global)
            # qkg = rearrange(qkg, " B h N (t d) -> B h N t d", t=2, d=self.dim_head)
            # qg, kg = qkg.unbind(-2)

            # # # option 2:
            # # x_global = F.scaled_dot_product_attention(self.x_global, x_mid, x_mid, scale=1.0)
            # # qkg = self.qk_global(x_global)
            # # qkg = rearrange(qkg, " B h N (t d) -> B h N t d", t=2, d=self.dim_head)
            # # qg, kg = qkg.unbind(-2)

            # # # option 3:
            # # qg = F.scaled_dot_product_attention(self.x_global, k, q, scale=1.0)
            # # kg = F.scaled_dot_product_attention(self.x_global, k, k, scale=1.0)

            # # # option 4:
            # # topk = torch.topk(torch.linalg.norm(q, dim=-1), k=128, dim=-1)
            # # qg = torch.gather(q, -2, topk[1].unsqueeze(-1).expand(-1,-1,-1,q.shape[-1])) # [B, H, K, D]
            # # kg = torch.gather(k, -2, topk[1].unsqueeze(-1).expand(-1,-1,-1,k.shape[-1])) # [B, H, K, D]

            # # # option 5:
            # # topk = torch.topk(torch.linalg.norm(q, dim=-1), k=128, dim=-1, largest=False)
            # # qg = torch.gather(q, -2, topk[1].unsqueeze(-1).expand(-1,-1,-1,q.shape[-1])) # [B, H, K, D]
            # # kg = torch.gather(k, -2, topk[1].unsqueeze(-1).expand(-1,-1,-1,k.shape[-1])) # [B, H, K, D]

            # # # option 6: min and max
            # # topkMax = torch.topk(torch.linalg.norm(q, dim=-1), k=64, dim=-1, largest=True)
            # # topkMin = torch.topk(torch.linalg.norm(q, dim=-1), k=64, dim=-1, largest=False)
            # # topk = torch.cat([topkMax[1], topkMin[1]], dim=-1)
            # # qg = torch.gather(q, -2, topk.unsqueeze(-1).expand(-1,-1,-1,q.shape[-1])) # [B, H, K, D]
            # # kg = torch.gather(k, -2, topk.unsqueeze(-1).expand(-1,-1,-1,k.shape[-1])) # [B, H, K, D]

            # # FLARE: Fast Low-rank Attention Routing Engine
            # vg = F.scaled_dot_product_attention(qg, k,  v,  scale=1.0)
            # y  = F.scaled_dot_product_attention(q,  kg, vg, scale=1.0)

            # Option 7:
            # G = torch.linalg.qr(self.x_global.to('cpu'), mode="reduced")[0].to(q.device)
            # G = deterministic_orthonormal_columns(self.x_global)
            # G = self.x_global.weight[None, None, :, :].expand(-1, 8, -1, -1) # TODO Make it GPU compatable
            G = (
                torch.stack([self.x_global[i].weight for i in range(self.heads)], dim=0)
                .unsqueeze(0)
                .to(q.device)
            )  # TODO Make it GPU compatable
            G = G * self.col_norms**2
            # FLARE: Fast Low-rank Attention Routing Engine
            z = F.scaled_dot_product_attention(G, k, v, scale=1.0)
            y = F.scaled_dot_product_attention(q, G, z, scale=1.0)

            out_x = y.permute(0, 2, 1, 3)  # [B, N, H, D]
            out_x = rearrange(out_x, "b n h d -> b n (h d)")
            out_x = self.out_linear(out_x)
            return self.out_dropout(out_x)


class Transolver_block(nn.Module):
    """Transformer encoder block, replacing standard attention with physics attention."""

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        dropout: float,
        act="gelu",
        mlp_ratio=4,
        last_layer=False,
        out_dim=1,
        slice_num=32,
        spatial_shape: tuple[int, ...] | None = None,
        use_te=True,
        attention_type: str = "LatentAttention",
    ):
        super().__init__()

        if use_te and not TE_AVAILABLE:
            raise ImportError(
                "Transformer Engine is not installed. Please install it with `pip install transformer-engine`."
            )

        self.last_layer = last_layer
        if use_te:
            self.ln_1 = te.LayerNorm(hidden_dim)
            self.ln_2 = lambda x: x  # te.LayerNorm(hidden_dim)
        else:
            self.ln_1 = nn.LayerNorm(hidden_dim)
            self.ln_2 = lambda x: x  # nn.LayerNorm(hidden_dim)

        if spatial_shape is None:
            # self.Attn = PhysicsAttentionIrregularMesh(
            if attention_type in globals():
                self.Attn = globals()[attention_type](
                    hidden_dim,
                    heads=num_heads,
                    dim_head=hidden_dim // num_heads,
                    dropout=dropout,
                    slice_num=slice_num,
                    use_te=use_te,
                )
        else:
            if len(spatial_shape) == 2:
                self.Attn = PhysicsAttentionStructuredMesh2D(
                    hidden_dim,
                    spatial_shape=spatial_shape,
                    heads=num_heads,
                    dim_head=hidden_dim // num_heads,
                    dropout=dropout,
                    slice_num=slice_num,
                    use_te=use_te,
                )
            elif len(spatial_shape) == 3:
                self.Attn = PhysicsAttentionStructuredMesh3D(
                    hidden_dim,
                    spatial_shape=spatial_shape,
                    heads=num_heads,
                    dim_head=hidden_dim // num_heads,
                    dropout=dropout,
                    slice_num=slice_num,
                    use_te=use_te,
                )
            else:
                raise Exception(
                    f"Unexpected length of spatial shape encountered in Transolver_block: {len(spatial_shape)}"
                )

        if use_te:
            self.ln_mlp1 = te.LayerNormMLP(
                hidden_size=hidden_dim,
                ffn_hidden_size=hidden_dim * mlp_ratio,
            )
        else:
            self.ln_mlp1 = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                MLP(
                    hidden_dim,
                    hidden_dim * mlp_ratio,
                    hidden_dim,
                    n_layers=0,
                    res=False,
                    act=act,
                    use_te=False,
                ),
            )
        if self.last_layer:
            if use_te:
                self.ln_mlp2 = te.LayerNormLinear(
                    in_features=hidden_dim, out_features=out_dim
                )
            else:
                self.ln_mlp2 = nn.Sequential(
                    nn.LayerNorm(hidden_dim),
                    nn.Linear(hidden_dim, out_dim),
                )

    def forward(self, fx, embedding):
        fx = self.Attn(self.ln_1(fx), self.ln_2(embedding)) + fx
        fx = self.ln_mlp1(fx) + fx
        if self.last_layer:
            return self.ln_mlp2(fx)
        else:
            return fx


@dataclass
class MetaData(ModelMetaData):
    name: str = "Transolver"
    # Optimization
    jit: bool = False
    cuda_graphs: bool = False
    amp: bool = True
    # Inference
    onnx_cpu: bool = False  # No FFT op on CPU
    onnx_gpu: bool = True
    onnx_runtime: bool = True
    # Physics informed
    var_dim: int = 1
    func_torch: bool = False
    auto_grad: bool = False


class Transolver_crash(Module):
    """
    Transolver model, adapted from original transolver code.

    Transolver is an adaptation of the transformer architecture, with a physics-attention
    mechanism replacing the standard attention mechanism.

    For more architecture details, see: https://arxiv.org/pdf/2402.02366 and https://arxiv.org/pdf/2502.02414

    Transolver can work on structured or unstructured data points as a model construction choice:
    - unstructured data (like a mesh) should provide some sort of positional encoding to accompany inputs
    - structured data (2D and 3D grids) can provide positional encodings optionally

    When constructing Transolver, you can choose to use "unified position" or not.  If you select "unified
    position" (`unified_pos=True`), then

    If using structured data, pass the structured shape as a tuple in the model constructor.
    Length 2 tuples are assumed to be image-like, length 3 tuples are assumed to be 3D voxel like.
    Other structured shape sizes are not supported.  Passing a structured_shape of None assumes irregular data.

    Output shape will have the same spatial shape as the input shape, with potentially more features

    Also can support Transolver++ implementation.  When using the distributed algorithm
    of Transolver++, use PhysicsNeMo's ShardTensor implementation to support automatic
    domain parallelism and 2D parallelization (data parallel + domain parallel, for example).

    Note
    ----


    Parameters
    ----------
    functional_dim : int
        The dimension of the input values, not including any embeddings.  No Default.
        Input will be concatenated with embeddings or unified position before processing
        with PhysicsAttention blocks.  Originally known as "fun_dim"
    out_dim : int
        The dimension of the output of the model.  This is a mandatory parameter.
    embedding_dim : int | None
        The spatial dimension of the input data embeddings.  Should include not just
        position but all computed embedding features.  Default is None, but if
        `unified_pos=False` this is a mandatory parameter.  Originally named "space_dim"
    n_layers : int
        The number of transformer PhysicsAttention layers in the model.  Default of 4.
    n_hidden : int
        The hidden dimension of the transformer.  Default of 256.  Projection is made
        from the input data + embeddings in the early preprocessing, before the
        PhysicsAttention layers.
    dropout : float
        The dropout rate, applied across the PhysicsAttention Layers.  Default is 0.0
    n_head : int
        The number of attention heads in each PhysicsAttention Layer.  Default is 8.  Note
        that the number of heads must evenly divide the `n_hidden` parameter to yield an
        integer head dimension.
    act : str
        The activation function, default is gelu.
    mlp_ratio : int
        The ratio of hidden dimension in the MLP, default is 4.  Used in the MLPs in the
        PhysicsAttention Layers.
    slice_num : int
        The number of slices in the PhysicsAttention layers.  Default is 32.  Represents the
        number of learned states each layer should project inputs onto.
    unified_pos : bool
        Whether to use unified positional embeddings.  Unified positions are only available for
        structured data (2D grids, 3D grids).  They are computed once initially, and reused through
        training in place of embeddings.
    ref : int
        The reference dimension size when using unified positions.  Default is 8.  Will be
        used to create a linear grid in spatial dimensions to serve as spatial embeddings.
        If `unified_pos=False`, this value is unused.
    structured_shape : None | tuple(int)
        The shape of the latent space.  If None, assumes irregular latent space.  If not
        `None`, this parameter can only be a length-2 or length-3 tuple of ints.
    use_te: bool
        Whether to use transformer engine backend when possible.
    time_input : bool
        Whether to include time embeddings. Default is false
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
        structured_shape: None | tuple[int] = None,
        use_te: bool = True,
        time_input: bool = False,
        attention_type: str = "LatentAttention",
    ) -> None:
        super().__init__(meta=MetaData())
        self.__name__ = "Transolver"

        self.use_te = use_te
        # Check that the hidden dimension and head dimensions are compatible:
        if not n_hidden % n_head == 0:
            raise ValueError(
                f"Transolver requires n_hidden % n_head == 0, but instead got {n_hidden % n_head}"
            )

        # Check the shape of the data, if it's structured data:
        if structured_shape is not None:
            # Has to be 2D or 3D data:
            if len(structured_shape) not in [2, 3]:
                raise ValueError(
                    f"Transolver can only use structured data in 2D or 3D, got {structured_shape}"
                )

            # Ensure it's all integers > 0:
            if not all([s > 0 and s == int(s) for s in structured_shape]):
                raise ValueError(
                    f"Transolver can only use integer shapes > 0, got {structured_shape}"
                )
        else:
            # It's mandatory for unified position:
            if unified_pos:
                raise ValueError(
                    "Transolver requires structured_shape to be passed if using unified_pos=True"
                )

        self.structured_shape = structured_shape

        # If we're using the unified position, create and save the position embeddings:
        self.unified_pos = unified_pos

        if unified_pos:
            if structured_shape is None:
                raise ValueError(
                    "Transolver can not use unified position without a structured_shape argument (got None)"
                )

            # This ensures embedding is tracked by torch and moves to the GPU, and saves/loads
            self.register_buffer("embedding", self.get_grid(ref))
            self.embedding_dim = ref * ref
            # mlp_input_dimension = functional_dim + ref * ref

        else:
            self.embedding_dim = embedding_dim
            # mlp_input_dimension = functional_dim + embedding_dim

        # This MLP is the initial projection onto the hidden space
        self.preprocess_fx = MLP(
            functional_dim + embedding_dim,
            n_hidden * 2,
            n_hidden,
            n_layers=0,
            res=False,
            act=act,
            use_te=use_te,
        )

        # self.preprocess_embedding = MLP(
        #     embedding_dim,
        #     n_hidden * 2,
        #     n_hidden,
        #     n_layers=0,
        #     res=False,
        #     act=act,
        #     use_te=use_te,
        # )
        self.time_input = time_input
        self.n_hidden = n_hidden
        if time_input:
            self.time_fc = nn.Sequential(
                nn.Linear(n_hidden, n_hidden), nn.SiLU(), nn.Linear(n_hidden, n_hidden)
            )

        self.blocks = nn.ModuleList(
            [
                Transolver_block(
                    num_heads=n_head,
                    hidden_dim=n_hidden,
                    dropout=dropout,
                    act=act,
                    mlp_ratio=mlp_ratio,
                    out_dim=out_dim,
                    slice_num=slice_num,
                    spatial_shape=structured_shape,
                    last_layer=(_ == n_layers - 1),
                    use_te=use_te,
                    attention_type=attention_type,
                )
                for _ in range(n_layers)
            ]
        )
        # n_nodes = 384862
        # self.embedding_vec = nn.Parameter(torch.randn(len(self.blocks), 1, n_nodes, n_hidden))
        # self.embedding = nn.Parameter(torch.zeros([1, n_nodes, self.n_head, self.slice_num,]), requires_grad=False)
        self.initialize_weights()

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        linear_layers = (nn.Linear,)
        if self.use_te:
            linear_layers = linear_layers + (te.Linear,)

        if isinstance(m, linear_layers):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, linear_layers) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        norm_layers = (nn.LayerNorm, nn.BatchNorm1d)
        if self.use_te:
            norm_layers = norm_layers + (te.LayerNorm,)
        if isinstance(m, norm_layers):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_grid(self, ref: int, batchsize: int = 1) -> torch.Tensor:
        """
        Generate a unified positional encoding grid for structured 2D data.

        Parameters
        ----------
        ref : int
            The reference grid size for the unified position encoding.
        batchsize : int, optional
            The batch size for the generated grid (default is 1).

        Returns
        -------
        torch.Tensor
            A tensor of shape (batchsize, H*W, ref*ref) containing the positional encodings,
            where H and W are the spatial dimensions from self.structured_shape.
        """
        size_x, size_y = self.structured_shape
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        grid = torch.cat((gridx, gridy), dim=-1)  # B H W 2

        gridx = torch.tensor(np.linspace(0, 1, ref), dtype=torch.float)
        gridx = gridx.reshape(1, ref, 1, 1).repeat([batchsize, 1, ref, 1])
        gridy = torch.tensor(np.linspace(0, 1, ref), dtype=torch.float)
        gridy = gridy.reshape(1, 1, ref, 1).repeat([batchsize, ref, 1, 1])
        grid_ref = torch.cat((gridx, gridy), dim=-1)  # B H W 8 8 2

        pos = (
            torch.sqrt(
                torch.sum(
                    (grid[:, :, :, None, None, :] - grid_ref[:, None, None, :, :, :])
                    ** 2,
                    dim=-1,
                )
            )
            .reshape(batchsize, -1, ref * ref)  # Flatten spatial dims
            .contiguous()
        )
        return pos

    def forward(
        self,
        fx: torch.Tensor | None,
        embedding: torch.Tensor | None = None,
        time: torch.Tensor | None = None,
        parts: list[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of the transolver model.

        Args:
            fx (torch.Tensor | None): Functional input tensor. For structured data,
                shape should be [B, N, C] or [B, *structure, C]. For unstructured data,
                shape should be [B, N, C]. Can be None if not used.
            embedding (torch.Tensor | None, optional): Embedding tensor. For structured
                data, shape should be [B, N, C] or [B, *structure, C]. For unstructured
                data, shape should be [B, N, C]. Defaults to None.
            time (torch.Tensor | None, optional): Optional time tensor. Shape and usage
                depend on the model configuration. Defaults to None.

        Returns:
            torch.Tensor: Output tensor with the same shape as the input.

        """
        # if self.unified_pos:
        #     # Extend the embedding to the batch size:
        #     embedding = self.embedding.repeat(fx.shape[0], 1, 1)

        # # Reshape automatically, if necessary:
        # if self.structured_shape is not None:
        #     unflatten_output = False
        #     if len(fx.shape) != 3:
        #         unflatten_output = True
        #         fx = fx.reshape(fx.shape[0], -1, fx.shape[-1])
        #     if embedding is not None and len(embedding.shape) != 3:
        #         embedding = embedding.reshape(
        #             embedding.shape[0], *self.structured_shape, -1
        #         )
        # else:
        #     if embedding is None:
        #         raise ValueError("Embedding is required for unstructured data")

        # Combine the embedding and functional input:
        if embedding is not None:
            fx = torch.cat((embedding, fx), -1)

        # Apply preprocessing
        fx = self.preprocess_fx(fx)

        # if time is not None:
        #     time_emb = timestep_embedding(time, self.n_hidden).repeat(
        #         1, embedding.shape[1], 1
        #     )
        #     time_emb = self.time_fc(time_emb)
        #     fx = fx + time_emb

        for i, block in enumerate(self.blocks):
            fx = block(fx, parts)

        # if self.structured_shape is not None:
        #     if unflatten_output:
        #         fx = fx.reshape(fx.shape[0], *self.structured_shape, -1)

        return fx
