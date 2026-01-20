# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
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

import torch
from torch.utils.checkpoint import checkpoint as ckpt

from physicsnemo.models.transolver import Transolver
from physicsnemo.experimental.models.transolver_crash.transolver_crash import (
    Transolver_crash,
)
from physicsnemo.experimental.models.transolver_crash.transolver_flare import (
    Transolver_flare,
)
from physicsnemo.experimental.models.geotransolver import GeoTransolver
from physicsnemo.models.meshgraphnet import MeshGraphNet
# from physicsnemo.models.figconvnet.figconvunet import FIGConvUNet

from datapipe import SimSample

EPS = 1e-8


def sample_time_integration(
    epoch, total_epochs, rollout_steps=None, device=None, ar_type="increasing_rollout"
):
    rollout_steps = (1,) if rollout_steps is None else torch.Size((rollout_steps,))
    if ar_type == "increasing_rollout":
        rollout_len = min(epoch // 100 + 1, rollout_steps)
        GT_or_pred = torch.arange(rollout_steps, device=device) % rollout_len == 0
        return GT_or_pred
    elif ar_type == "scheduled_rollout_binary":
        frac = torch.tensor(epoch / total_epochs, device=device)
        prob = 1 - torch.exp(-10 * frac)
        GT_or_pred = torch.rand(rollout_steps, device=device)
        GT_or_pred = GT_or_pred >= prob
        return GT_or_pred
    elif ar_type == "scheduled_rollout_aggregate":
        frac = torch.tensor(epoch / total_epochs, device=device)
        prob = 1 - torch.exp(-30 * frac)
        return 1 - prob.expand(rollout_steps)
    elif ar_type == "TeacherForcing":
        return torch.ones(rollout_steps, device=device)
    else:
        return torch.zeros(rollout_steps, device=device)


def time_integration(y_t1, y_t0, dt, outf, data_stats, acc_or_vel="acc"):
    if acc_or_vel == "acc":
        vel = (y_t1 - y_t0) / dt
        acc = (
            outf * data_stats["node"]["norm_acc_std"]
            + data_stats["node"]["norm_acc_mean"]
        )
        vel = dt * acc + vel
    elif acc_or_vel == "vel":
        vel = (
            outf * data_stats["node"]["norm_vel_std"]
            + data_stats["node"]["norm_vel_mean"]
        )
    else:
        raise ValueError(f"Invalid acc_or_vel: {acc_or_vel}")

    y_t2 = dt * vel + y_t1
    return y_t2


class TransolverAutoregressiveRolloutTraining(Transolver):
    """
    Transolver model with autoregressive rollout training.

    Predicts sequence by autoregressively updating velocity and position
    using predicted accelerations. Supports gradient checkpointing during training.
    """

    def __init__(self, *args, **kwargs):
        self.dt: float = kwargs.pop("dt")
        self.initial_vel: torch.Tensor = kwargs.pop("initial_vel")
        self.rollout_steps: int = kwargs.pop("num_time_steps") - 1
        super().__init__(*args, **kwargs)

    def forward(self, sample: SimSample, data_stats: dict, **kwargs) -> torch.Tensor:
        """
        Args:
            sample: SimSample containing node_features and node_target
            data_stats: dict containing normalization stats
        Returns:
            [T, N, 3] rollout of predicted positions
        """
        inputs = sample.node_features
        coords = inputs["coords"]  # [N,3]
        features = inputs.get("features", coords.new_zeros((coords.size(0), 0)))
        N = coords.size(0)

        # Initial states
        y_t1 = coords  # [N,3]
        y_t0 = y_t1 - self.initial_vel * self.dt  # backstep using initial velocity

        outputs: list[torch.Tensor] = []
        for t in range(self.rollout_steps):
            # Velocity normalization
            vel = (y_t1 - y_t0) / self.dt
            vel_norm = (vel - data_stats["node"]["norm_vel_mean"]) / (
                data_stats["node"]["norm_vel_std"] + EPS
            )

            # Model input
            fx_t = torch.cat([vel_norm, features], dim=-1)  # [N, 3+F+1]

            def step_fn(fx, embedding):
                return super(TransolverAutoregressiveRolloutTraining, self).forward(
                    fx=fx, embedding=embedding
                )

            if self.training:
                outf = ckpt(
                    step_fn, fx_t.unsqueeze(0), y_t1.unsqueeze(0), use_reentrant=False
                ).squeeze(0)
            else:
                outf = step_fn(fx_t.unsqueeze(0), y_t1.unsqueeze(0)).squeeze(0)

            # De-normalize acceleration
            acc = (
                outf * data_stats["node"]["norm_acc_std"]
                + data_stats["node"]["norm_acc_mean"]
            )
            vel = self.dt * acc + vel
            y_t2 = self.dt * vel + y_t1

            outputs.append(y_t2)
            y_t1, y_t0 = y_t2, y_t1

        return torch.stack(outputs, dim=0)  # [T,N,3]


class TransolverAutoregressiveRolloutTraining_vel(Transolver):
    """
    Transolver model with autoregressive rollout training.

    Predicts sequence by autoregressively updating velocity and position
    using predicted accelerations. Supports gradient checkpointing during training.
    """

    def __init__(self, *args, **kwargs):
        self.dt: float = kwargs.pop("dt")
        self.initial_vel: torch.Tensor = kwargs.pop("initial_vel")
        self.rollout_steps: int = kwargs.pop("num_time_steps") - 1
        super().__init__(*args, **kwargs)

    def forward(self, sample: SimSample, data_stats: dict, **kwargs) -> torch.Tensor:
        """
        Args:
            sample: SimSample containing node_features and node_target
            data_stats: dict containing normalization stats
        Returns:
            [T, N, 3] rollout of predicted positions
        """
        inputs = sample.node_features
        coords = inputs["coords"]  # [N,3]
        features = inputs.get("features", coords.new_zeros((coords.size(0), 0)))
        N = coords.size(0)

        # Initial states
        y_t1 = coords  # [N,3]
        y_t0 = y_t1 - self.initial_vel * self.dt  # backstep using initial velocity

        outputs: list[torch.Tensor] = []
        for t in range(self.rollout_steps):
            # Velocity normalization
            vel = (y_t1 - y_t0) / self.dt
            vel_norm = (vel - data_stats["node"]["norm_vel_mean"]) / (
                data_stats["node"]["norm_vel_std"] + EPS
            )

            # Model input
            fx_t = torch.cat([vel_norm, features], dim=-1)  # [N, 3+F+1]

            def step_fn(fx, embedding):
                return super(TransolverAutoregressiveRolloutTraining_vel, self).forward(
                    fx=fx, embedding=embedding
                )

            if self.training:
                outf = ckpt(
                    step_fn, fx_t.unsqueeze(0), y_t1.unsqueeze(0), use_reentrant=False
                ).squeeze(0)
            else:
                outf = step_fn(fx_t.unsqueeze(0), y_t1.unsqueeze(0)).squeeze(0)

            # De-normalize velocity
            vel = (
                outf * data_stats["node"]["norm_vel_std"]
                + data_stats["node"]["norm_vel_mean"]
            )
            y_t2 = self.dt * vel + y_t1

            outputs.append(y_t2)
            y_t1, y_t0 = y_t2, y_t1

        return torch.stack(outputs, dim=0)  # [T,N,3]


class TransolverAutoregressiveRolloutTraining_flareX(Transolver_crash):
    """
    Transolver model with autoregressive rollout training.

    Predicts sequence by autoregressively updating velocity and position
    using predicted accelerations. Supports gradient checkpointing during training.
    """

    def __init__(self, *args, **kwargs):
        self.dt: float = kwargs.pop("dt")
        self.initial_vel: torch.Tensor = kwargs.pop("initial_vel")
        self.rollout_steps: int = kwargs.pop("num_time_steps") - 1
        super().__init__(*args, **kwargs)

    def forward(self, sample: SimSample, data_stats: dict, **kwargs) -> torch.Tensor:
        """
        Args:
            sample: SimSample containing node_features and node_target
            data_stats: dict containing normalization stats
        Returns:
            [T, N, 3] rollout of predicted positions
        """
        inputs = sample.node_features
        coords = inputs["coords"]  # [N,3]
        features = inputs.get("features", coords.new_zeros((coords.size(0), 0)))
        N = coords.size(0)

        # Initial states
        y_t1 = coords  # [N,3]
        y_t0 = y_t1 - self.initial_vel * self.dt  # backstep using initial velocity

        outputs: list[torch.Tensor] = []
        for t in range(self.rollout_steps):
            # Velocity normalization
            vel = (y_t1 - y_t0) / self.dt
            vel_norm = (vel - data_stats["node"]["norm_vel_mean"]) / (
                data_stats["node"]["norm_vel_std"] + EPS
            )

            # Model input
            fx_t = torch.cat([vel_norm, features], dim=-1)  # [N, 3+F+1]

            def step_fn(fx, embedding):
                return super(
                    TransolverAutoregressiveRolloutTraining_flareX, self
                ).forward(fx=fx, embedding=embedding)

            if self.training:
                outf = ckpt(
                    step_fn, fx_t.unsqueeze(0), y_t1.unsqueeze(0), use_reentrant=False
                ).squeeze(0)
            else:
                outf = step_fn(fx_t.unsqueeze(0), y_t1.unsqueeze(0)).squeeze(0)

            y_t2 = time_integration(
                y_t1, y_t0, self.dt, outf, data_stats, acc_or_vel="acc"
            )

            outputs.append(y_t2)
            y_t1, y_t0 = y_t2, y_t1

        return torch.stack(outputs, dim=0)  # [T,N,3]


class GeoTransolverAutoregressiveRolloutTraining(GeoTransolver):
    """
    Transolver model with autoregressive rollout training.

    Predicts sequence by autoregressively updating velocity and position
    using predicted accelerations. Supports gradient checkpointing during training.
    """

    def __init__(self, *args, **kwargs):
        self.dt: float = kwargs.pop("dt")
        self.initial_vel: torch.Tensor = kwargs.pop("initial_vel")
        self.rollout_steps: int = kwargs.pop("num_time_steps") - 1
        super().__init__(*args, **kwargs)

    def forward(self, sample: SimSample, data_stats: dict, **kwargs) -> torch.Tensor:
        """
        Args:
            sample: SimSample containing node_features and node_target
            data_stats: dict containing normalization stats
        Returns:
            [T, N, 3] rollout of predicted positions
        """
        inputs = sample.node_features
        sampled_geom_idx = inputs["sampled_geom_idx"]
        coords = inputs["coords"]  # [N,3]
        features = inputs.get("features", coords.new_zeros((coords.size(0), 0)))
        N = coords.size(0)

        # Initial states
        y_t1 = coords  # [N,3]
        y_t0 = y_t1 - self.initial_vel * self.dt  # backstep using initial velocity

        outputs: list[torch.Tensor] = []
        for t in range(self.rollout_steps):
            # Velocity normalization
            vel = (y_t1 - y_t0) / self.dt
            vel_norm = (vel - data_stats["node"]["norm_vel_mean"]) / (
                data_stats["node"]["norm_vel_std"] + EPS
            )

            # Model input
            fx_t = torch.cat([vel_norm, features], dim=-1)  # [N, 3+F+1]

            def step_fn(fx, y_t1):
                return super(GeoTransolverAutoregressiveRolloutTraining, self).forward(
                    local_embedding=torch.cat([y_t1, fx], dim=-1),
                    local_positions=y_t1,
                    geometry=y_t1[:, sampled_geom_idx],
                    global_embedding=None,
                )

            if self.training:
                outf = ckpt(
                    step_fn, fx_t.unsqueeze(0), y_t1.unsqueeze(0), use_reentrant=False
                ).squeeze(0)
            else:
                outf = step_fn(fx_t.unsqueeze(0), y_t1.unsqueeze(0)).squeeze(0)

            y_t2 = time_integration(
                y_t1, y_t0, self.dt, outf, data_stats, acc_or_vel="acc"
            )

            outputs.append(y_t2)
            y_t1, y_t0 = y_t2, y_t1

        return torch.stack(outputs, dim=0)  # [T,N,3]


class TransolverAutoregressiveRolloutTraining_flareX_dmgcoll(Transolver_crash):
    """
    Transolver model with autoregressive rollout training.

    Predicts sequence by autoregressively updating velocity and position
    using predicted accelerations. Supports gradient checkpointing during training.
    """

    def __init__(self, *args, **kwargs):
        self.dt: float = kwargs.pop("dt")
        self.initial_vel: torch.Tensor = kwargs.pop("initial_vel")
        self.rollout_steps: int = kwargs.pop("num_time_steps") - 1
        super().__init__(*args, **kwargs)

    def forward(self, sample: SimSample, data_stats: dict, **kwargs) -> torch.Tensor:
        """
        Args:
            sample: SimSample containing node_features and node_target
            data_stats: dict containing normalization stats
        Returns:
            [T, N, 3] rollout of predicted positions
        """
        inputs = sample.node_features
        coords = inputs["coords"]  # [N,3]
        features = inputs.get("features", coords.new_zeros((coords.size(0), 0)))
        N = coords.size(0)

        # Initial states
        y_t1 = coords  # [N,3]
        y_t0 = y_t1 - self.initial_vel * self.dt  # backstep using initial velocity

        # Additional features for car collision detection
        wall_x = (-30 - data_stats["node"]["pos_mean"][0]) / data_stats["node"][
            "pos_std"
        ][0]
        damage_emb = torch.zeros(N, 1, device=coords.device)
        ref_idx = coords.argmin(dim=0)[
            0
        ].item()  # ref idx for min x point on car geometry
        y_t1_diff_0 = y_t1 - y_t1[ref_idx]

        outputs: list[torch.Tensor] = []
        for t in range(self.rollout_steps):
            # Velocity normalization
            vel = (y_t1 - y_t0) / self.dt
            vel_norm = (vel - data_stats["node"]["norm_vel_mean"]) / (
                data_stats["node"]["norm_vel_std"] + EPS
            )

            # Additional features for car collision detection
            nowall_y_t2 = self.dt * vel + y_t1
            collision_emb = torch.nn.functional.elu(10 * (nowall_y_t2[:, 0:1] - wall_x))
            damage_emb += torch.nn.functional.softmax(collision_emb**3, dim=0)
            y_t1_diff = y_t1 - y_t1[ref_idx]
            diff = y_t1_diff - y_t1_diff_0

            # Model input
            fx_t = torch.cat(
                [vel_norm, features, damage_emb, collision_emb, diff], dim=-1
            )  # [N, 3+F+2+3]

            def step_fn(fx, embedding):
                return super(
                    TransolverAutoregressiveRolloutTraining_flareX_dmgcoll, self
                ).forward(fx=fx, embedding=embedding)

            if self.training:
                outf = ckpt(
                    step_fn, fx_t.unsqueeze(0), y_t1.unsqueeze(0), use_reentrant=False
                ).squeeze(0)
            else:
                outf = step_fn(fx_t.unsqueeze(0), y_t1.unsqueeze(0)).squeeze(0)

            y_t2 = time_integration(
                y_t1, y_t0, self.dt, outf, data_stats, acc_or_vel="acc"
            )

            outputs.append(y_t2)
            y_t1, y_t0 = y_t2, y_t1

        return torch.stack(outputs, dim=0)  # [T,N,3]


class TransolverAutoregressiveRolloutTraining_xrel(Transolver):
    """
    Transolver model with autoregressive rollout training.

    Predicts sequence by autoregressively updating velocity and position
    using predicted accelerations. Supports gradient checkpointing during training.
    """

    def __init__(self, *args, **kwargs):
        self.dt: float = kwargs.pop("dt")
        self.initial_vel: torch.Tensor = kwargs.pop("initial_vel")
        self.rollout_steps: int = kwargs.pop("num_time_steps") - 1
        super().__init__(*args, **kwargs)

    def forward(self, sample: SimSample, data_stats: dict, **kwargs) -> torch.Tensor:
        """
        Args:
            sample: SimSample containing node_features and node_target
            data_stats: dict containing normalization stats
        Returns:
            [T, N, 3] rollout of predicted positions
        """
        inputs = sample.node_features
        coords = inputs["coords"]  # [N,3]
        features = inputs.get("features", coords.new_zeros((coords.size(0), 0)))
        N = coords.size(0)
        # device = coords.device

        # Initial states
        y_t1 = coords  # [N,3]
        y_t0 = y_t1 - self.initial_vel * self.dt  # backstep using initial velocity

        outputs: list[torch.Tensor] = []
        wall_x = (-30 - data_stats["node"]["pos_mean"][0]) / data_stats["node"][
            "pos_std"
        ][0]
        damage_emb = torch.zeros(N, 1, device=coords.device)
        ref_idx = coords.argmin(dim=0)[
            0
        ].item()  # ref idx for min x point on car geometry
        y_t1_diff_0 = y_t1 - y_t1[ref_idx]
        for t in range(self.rollout_steps):
            # Velocity normalization
            vel = (y_t1 - y_t0) / self.dt
            vel_norm = (vel - data_stats["node"]["norm_vel_mean"]) / (
                data_stats["node"]["norm_vel_std"] + EPS
            )

            nowall_y_t2 = self.dt * vel + y_t1
            collision_emb = torch.nn.functional.elu(10 * (nowall_y_t2[:, 0:1] - wall_x))
            damage_emb += torch.nn.functional.softmax(collision_emb**3, dim=0)
            y_t1_diff = y_t1 - y_t1[ref_idx]
            diff = y_t1_diff - y_t1_diff_0

            # Model input
            fx_t = torch.cat(
                [y_t1_diff, vel_norm, damage_emb, collision_emb], dim=-1
            )  # [N, 3+F+1]

            embedding = torch.cat([features, diff], dim=-1)

            def step_fn(fx, embedding):
                return super(
                    TransolverAutoregressiveRolloutTraining_xrel, self
                ).forward(fx=fx, embedding=embedding)

            if self.training:
                outf = ckpt(
                    step_fn,
                    fx_t.unsqueeze(0),
                    embedding.unsqueeze(0),
                    use_reentrant=False,
                ).squeeze(0)
            else:
                outf = step_fn(fx_t.unsqueeze(0), embedding.unsqueeze(0)).squeeze(0)

            # De-normalize acceleration
            acc = (
                outf * data_stats["node"]["norm_acc_std"]
                + data_stats["node"]["norm_acc_mean"]
            )
            vel = self.dt * acc + vel
            y_t2 = self.dt * vel + y_t1

            outputs.append(y_t2)
            y_t1, y_t0 = y_t2, y_t1

        return torch.stack(outputs, dim=0)  # [T,N,3]


class TransolverAutoregressiveIncreasingRolloutTraining(Transolver):
    """
    Transolver model with autoregressive rollout training.

    Predicts sequence by autoregressively updating velocity and position
    using predicted accelerations. Supports gradient checkpointing during training.
    """

    def __init__(self, *args, **kwargs):
        self.dt: float = kwargs.pop("dt")
        self.initial_vel: torch.Tensor = kwargs.pop("initial_vel")
        self.rollout_steps: int = kwargs.pop("num_time_steps") - 1
        super().__init__(*args, **kwargs)

    def forward(self, sample: SimSample, data_stats: dict, **kwargs) -> torch.Tensor:
        """
        Args:
            sample: SimSample containing node_features and node_target
            data_stats: dict containing normalization stats
        Returns:
            [T, N, 3] rollout of predicted positions
        """
        epoch = kwargs.get("epoch", 1e6)
        rollout_len = min(epoch // 100 + 1, self.rollout_steps)

        inputs = sample.node_features
        coords0 = inputs["coords"]  # [N,3]
        features = inputs.get("features", coords0.new_zeros((coords0.size(0), 0)))

        # Ground truth sequence [T,N,3]
        N = coords0.size(0)
        gt_seq = torch.cat(
            [coords0.unsqueeze(0), sample.node_target.view(N, -1, 3).transpose(0, 1)],
            dim=0,
        )

        # Initial states
        y_t0 = gt_seq[0] - self.initial_vel * self.dt
        y_t1 = gt_seq[0]

        outputs: list[torch.Tensor] = []
        wall_x = (-30 - data_stats["node"]["pos_mean"][0]) / data_stats["node"][
            "pos_std"
        ][0]
        damage_emb = torch.zeros(N, 1, device=coords0.device)
        ref_idx = coords0.argmin(dim=0)[
            0
        ].item()  # ref idx for min x point on car geometry
        y_t1_diff_0 = y_t1 - y_t1[ref_idx]
        for t in range(self.rollout_steps):
            # Velocity normalization
            vel = (y_t1 - y_t0) / self.dt
            vel_norm = (vel - data_stats["node"]["norm_vel_mean"]) / (
                data_stats["node"]["norm_vel_std"] + EPS
            )

            nowall_y_t2 = self.dt * vel + y_t1
            collision_emb = torch.nn.functional.elu(10 * (nowall_y_t2[:, 0:1] - wall_x))
            damage_emb += torch.nn.functional.softmax(collision_emb**3, dim=0)
            y_t1_diff = y_t1 - y_t1[ref_idx]
            diff = y_t1_diff - y_t1_diff_0

            # Model input
            fx_t = torch.cat(
                [y_t1_diff, vel_norm, damage_emb, collision_emb], dim=-1
            )  # [N, 3+F+1]

            embedding = torch.cat([features, diff], dim=-1)

            def step_fn(fx, embedding):
                return super(
                    TransolverAutoregressiveIncreasingRolloutTraining, self
                ).forward(fx=fx, embedding=embedding)

            if self.training:
                outf = ckpt(
                    step_fn,
                    fx_t.unsqueeze(0),
                    embedding.unsqueeze(0),
                    use_reentrant=False,
                ).squeeze(0)
            else:
                outf = step_fn(fx_t.unsqueeze(0), embedding.unsqueeze(0)).squeeze(0)

            # De-normalize acceleration
            acc = (
                outf * data_stats["node"]["norm_acc_std"]
                + data_stats["node"]["norm_acc_mean"]
            )
            vel = self.dt * acc + vel
            y_t2 = self.dt * vel + y_t1

            outputs.append(y_t2)
            # y_t1, y_t0 = y_t2, y_t1
            if self.training and (t + 1) % rollout_len == 0:
                y_t1, y_t0 = gt_seq[t + 1], gt_seq[t]
            else:
                y_t1, y_t0 = y_t2, y_t1

        return torch.stack(outputs, dim=0)  # [T,N,3]


class TransolverAutoregressiveSheduledRolloutTraining_binary(Transolver):
    """
    Transolver model with autoregressive rollout training.

    Predicts sequence by autoregressively updating velocity and position
    using predicted accelerations. Supports gradient checkpointing during training.
    """

    def __init__(self, *args, **kwargs):
        self.dt: float = kwargs.pop("dt")
        self.initial_vel: torch.Tensor = kwargs.pop("initial_vel")
        self.rollout_steps: int = kwargs.pop("num_time_steps") - 1
        super().__init__(*args, **kwargs)

    def forward(self, sample: SimSample, data_stats: dict, **kwargs) -> torch.Tensor:
        """
        Args:
            sample: SimSample containing node_features and node_target
            data_stats: dict containing normalization stats
        Returns:
            [T, N, 3] rollout of predicted positions
        """
        inputs = sample.node_features
        coords0 = inputs["coords"]  # [N,3]
        features = inputs.get("features", coords0.new_zeros((coords0.size(0), 0)))

        epoch = kwargs.get("epoch", 1e6)
        GT_or_pred = sample_time_integration(
            epoch,
            10_000,
            self.rollout_steps,
            device=coords0.device,
            ar_type="scheduled_rollout_binary",
        )

        # Ground truth sequence [T,N,3]
        N = coords0.size(0)
        gt_seq = torch.cat(
            [coords0.unsqueeze(0), sample.node_target.view(N, -1, 3).transpose(0, 1)],
            dim=0,
        )

        # Initial states
        y_t0 = gt_seq[0] - self.initial_vel * self.dt
        y_t1 = gt_seq[0]

        outputs: list[torch.Tensor] = []
        wall_x = (-30 - data_stats["node"]["pos_mean"][0]) / data_stats["node"][
            "pos_std"
        ][0]
        damage_emb = torch.zeros(N, 1, device=coords0.device)
        ref_idx = coords0.argmin(dim=0)[
            0
        ].item()  # ref idx for min x point on car geometry
        y_t1_diff_0 = y_t1 - y_t1[ref_idx]
        for t in range(self.rollout_steps):
            # Velocity normalization
            vel = (y_t1 - y_t0) / self.dt
            vel_norm = (vel - data_stats["node"]["norm_vel_mean"]) / (
                data_stats["node"]["norm_vel_std"] + EPS
            )

            nowall_y_t2 = self.dt * vel + y_t1
            collision_emb = torch.nn.functional.elu(10 * (nowall_y_t2[:, 0:1] - wall_x))
            damage_emb += torch.nn.functional.softmax(collision_emb**3, dim=0)
            y_t1_diff = y_t1 - y_t1[ref_idx]
            diff = y_t1_diff - y_t1_diff_0

            # Model input
            fx_t = torch.cat(
                [y_t1_diff, vel_norm, damage_emb, collision_emb], dim=-1
            )  # [N, 3+F+1]

            embedding = torch.cat([features, diff], dim=-1)

            def step_fn(fx, embedding):
                return super(
                    TransolverAutoregressiveSheduledRolloutTraining_binary, self
                ).forward(fx=fx, embedding=embedding)

            if self.training:
                outf = ckpt(
                    step_fn,
                    fx_t.unsqueeze(0),
                    embedding.unsqueeze(0),
                    use_reentrant=False,
                ).squeeze(0)
            else:
                outf = step_fn(fx_t.unsqueeze(0), embedding.unsqueeze(0)).squeeze(0)

            y_t2 = time_integration(
                y_t1, y_t0, self.dt, outf, data_stats, acc_or_vel="acc"
            )

            outputs.append(y_t2)
            y_t0 = y_t1
            if self.training:
                # Scheduled rollout: from ground truth to predicted
                y_t1 = y_t2 + GT_or_pred[t] * (gt_seq[t + 1] - y_t2)
            else:
                y_t1 = y_t2

        return torch.stack(outputs, dim=0)  # [T,N,3]


class TransolverAutoregressiveSheduledRolloutTraining_aggregate(Transolver):
    """
    Transolver model with autoregressive rollout training.

    Predicts sequence by autoregressively updating velocity and position
    using predicted accelerations. Supports gradient checkpointing during training.
    """

    def __init__(self, *args, **kwargs):
        self.dt: float = kwargs.pop("dt")
        self.initial_vel: torch.Tensor = kwargs.pop("initial_vel")
        self.rollout_steps: int = kwargs.pop("num_time_steps") - 1
        super().__init__(*args, **kwargs)

    def forward(self, sample: SimSample, data_stats: dict, **kwargs) -> torch.Tensor:
        """
        Args:
            sample: SimSample containing node_features and node_target
            data_stats: dict containing normalization stats
        Returns:
            [T, N, 3] rollout of predicted positions
        """
        inputs = sample.node_features
        coords0 = inputs["coords"]  # [N,3]
        features = inputs.get("features", coords0.new_zeros((coords0.size(0), 0)))

        epoch = kwargs.get("epoch", 1e6)
        GT_or_pred = sample_time_integration(
            epoch,
            10_000,
            self.rollout_steps,
            device=coords0.device,
            ar_type="scheduled_rollout_aggregate",
        )

        # Ground truth sequence [T,N,3]
        N = coords0.size(0)
        gt_seq = torch.cat(
            [coords0.unsqueeze(0), sample.node_target.view(N, -1, 3).transpose(0, 1)],
            dim=0,
        )

        # Initial states
        y_t0 = gt_seq[0] - self.initial_vel * self.dt
        y_t1 = gt_seq[0]

        outputs: list[torch.Tensor] = []
        wall_x = (-30 - data_stats["node"]["pos_mean"][0]) / data_stats["node"][
            "pos_std"
        ][0]
        damage_emb = torch.zeros(N, 1, device=coords0.device)
        ref_idx = coords0.argmin(dim=0)[
            0
        ].item()  # ref idx for min x point on car geometry
        y_t1_diff_0 = y_t1 - y_t1[ref_idx]
        for t in range(self.rollout_steps):
            # Velocity normalization
            vel = (y_t1 - y_t0) / self.dt
            vel_norm = (vel - data_stats["node"]["norm_vel_mean"]) / (
                data_stats["node"]["norm_vel_std"] + EPS
            )

            nowall_y_t2 = self.dt * vel + y_t1
            collision_emb = torch.nn.functional.elu(10 * (nowall_y_t2[:, 0:1] - wall_x))
            damage_emb += torch.nn.functional.softmax(collision_emb**3, dim=0)
            y_t1_diff = y_t1 - y_t1[ref_idx]
            diff = y_t1_diff - y_t1_diff_0

            # Model input
            fx_t = torch.cat(
                [y_t1_diff, vel_norm, damage_emb, collision_emb], dim=-1
            )  # [N, 3+F+1]

            embedding = torch.cat([features, diff], dim=-1)

            def step_fn(fx, embedding):
                return super(
                    TransolverAutoregressiveSheduledRolloutTraining_aggregate, self
                ).forward(fx=fx, embedding=embedding)

            if self.training:
                outf = ckpt(
                    step_fn,
                    fx_t.unsqueeze(0),
                    embedding.unsqueeze(0),
                    use_reentrant=False,
                ).squeeze(0)
            else:
                outf = step_fn(fx_t.unsqueeze(0), embedding.unsqueeze(0)).squeeze(0)

            y_t2 = time_integration(
                y_t1, y_t0, self.dt, outf, data_stats, acc_or_vel="acc"
            )

            outputs.append(y_t2)
            y_t0 = y_t1
            if self.training:
                # Scheduled rollout: from ground truth to predicted
                y_t1 = y_t2 + GT_or_pred[t] * (gt_seq[t + 1] - y_t2)
            else:
                y_t1 = y_t2

        return torch.stack(outputs, dim=0)  # [T,N,3]


class TransolverAutoregressiveRolloutTraining_com(Transolver):
    """
    Transolver model with autoregressive rollout training.

    Predicts sequence by autoregressively updating velocity and position
    using predicted accelerations. Supports gradient checkpointing during training.
    """

    def __init__(self, *args, **kwargs):
        self.dt: float = kwargs.pop("dt")
        self.initial_vel: torch.Tensor = kwargs.pop("initial_vel")
        self.rollout_steps: int = kwargs.pop("num_time_steps") - 1
        super().__init__(*args, **kwargs)
        self.node_encoder = torch.nn.Sequential(
            torch.nn.Linear(7, 64),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(64, 256),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(256, 512),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(512, 1024),
        )
        self.acc_net = torch.nn.Sequential(
            torch.nn.Linear(2 * 1024, 512),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(512, 128),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(128, 3),
        )

    def com_trajectory(self, y_t1, vel_norm, features):
        nfeat = torch.cat([y_t1, vel_norm, features], dim=-1)
        nfeat = self.node_encoder(nfeat)  # [N, embed_dim]
        nfeat_agg = torch.cat(
            [nfeat.max(-2)[0], nfeat.mean(-2)], dim=-1
        )  # [2*embed_dim]

        com_acc = self.acc_net(nfeat_agg)
        return com_acc

    def forward(self, sample: SimSample, data_stats: dict, **kwargs) -> torch.Tensor:
        """
        Args:
            sample: SimSample containing node_features and node_target
            data_stats: dict containing normalization stats
        Returns:
            [T, N, 3] rollout of predicted positions
        """
        epoch = 1e6  # kwargs.get("epoch", 1e6) # TODO: Need attention befor committing this change
        rollout_len = min(epoch // 100 + 1, self.rollout_steps)

        inputs = sample.node_features
        coords0 = inputs["coords"]  # [N,3]
        features = inputs.get("features", coords0.new_zeros((coords0.size(0), 0)))

        # Ground truth sequence [T,N,3]
        N = coords0.size(0)
        gt_seq = torch.cat(
            [coords0.unsqueeze(0), sample.node_target.view(N, -1, 3).transpose(0, 1)],
            dim=0,
        )

        # Initial states
        y_t0 = gt_seq[0] - self.initial_vel * self.dt
        y_t1 = gt_seq[0]

        outputs: list[torch.Tensor] = []
        wall_x = (-30 - data_stats["node"]["pos_mean"][0]) / data_stats["node"][
            "pos_std"
        ][0]
        damage_emb = torch.zeros(N, 1, device=coords0.device)
        pred_nodes = coords0[:, 0] >= 0  # TODO: change it for partial prediction
        ref_idx = coords0.argmin(dim=0)[
            0
        ].item()  # ref idx for min x point on car geometry
        y_t1_diff_0 = y_t1 - y_t1[ref_idx]
        rel_acc = torch.zeros_like(y_t1)
        ones = torch.ones_like(y_t1)
        for t in range(self.rollout_steps):
            # Velocity normalization
            vel = (y_t1 - y_t0) / self.dt
            vel_norm = (vel - data_stats["node"]["norm_vel_mean"]) / (
                data_stats["node"]["norm_vel_std"] + EPS
            )

            _com_acc = self.com_trajectory(
                y_t1[~pred_nodes], vel_norm[~pred_nodes], features[~pred_nodes]
            )
            com_acc = (
                _com_acc * data_stats["node"]["norm_acc_std"]
                + data_stats["node"]["norm_acc_mean"]
            )
            nowall_y_t2 = self.dt * (self.dt * com_acc + vel) + y_t1
            # nowall_y_t2 = self.dt * vel + y_t1
            collision_emb = torch.nn.functional.elu(10 * (nowall_y_t2[:, 0:1] - wall_x))
            damage_emb += torch.nn.functional.softmax(collision_emb**3, dim=0)
            y_t1_diff = y_t1 - y_t1[ref_idx]
            diff = y_t1_diff - y_t1_diff_0

            # Model input
            fx_t = torch.cat(
                [y_t1_diff, vel_norm, damage_emb, collision_emb, ones * _com_acc],
                dim=-1,
            )  # [N, 3+F+1]

            embedding = torch.cat([features, diff], dim=-1)

            def step_fn(fx, embedding):
                return super(TransolverAutoregressiveRolloutTraining_com, self).forward(
                    fx=fx, embedding=embedding
                )

            if self.training:
                outf = ckpt(
                    step_fn,
                    fx_t[pred_nodes].unsqueeze(0),
                    embedding[pred_nodes].unsqueeze(0),
                    use_reentrant=False,
                ).squeeze(0)
            else:
                outf = step_fn(
                    fx_t[pred_nodes].unsqueeze(0), embedding[pred_nodes].unsqueeze(0)
                ).squeeze(0)

            # De-normalize acceleration
            rel_acc.zero_()
            rel_acc[pred_nodes] = (
                outf * data_stats["node"]["norm_acc_std"]
                + data_stats["node"]["norm_acc_mean"]
            )
            acc = rel_acc + com_acc
            # rel_acc[~pred_nodes] = com_acc

            y_t2 = self.dt * (self.dt * acc + vel) + y_t1

            outputs.append(y_t2)
            # y_t1, y_t0 = y_t2, y_t1
            if self.training and (t + 1) % rollout_len == 0:
                y_t1, y_t0 = gt_seq[t + 1], gt_seq[t]
            else:
                y_t1, y_t0 = y_t2, y_t1

        return torch.stack(outputs, dim=0)  # [T,N,3]


class TransolverAutoregressiveRolloutTraining_front(Transolver):
    """
    Transolver model with autoregressive rollout training.

    Predicts sequence by autoregressively updating velocity and position
    using predicted accelerations. Supports gradient checkpointing during training.
    """

    def __init__(self, *args, **kwargs):
        self.dt: float = kwargs.pop("dt")
        self.initial_vel: torch.Tensor = kwargs.pop("initial_vel")
        self.rollout_steps: int = kwargs.pop("num_time_steps") - 1
        super().__init__(*args, **kwargs)

    def forward(self, sample: SimSample, data_stats: dict, **kwargs) -> torch.Tensor:
        """
        Args:
            sample: SimSample containing node_features and node_target
            data_stats: dict containing normalization stats
        Returns:
            [T, N, 3] rollout of predicted positions
        """
        epoch = 1e6  # kwargs.get("epoch", 1e6) # TODO: Need attention befor committing this change
        rollout_len = min(epoch // 100 + 1, self.rollout_steps)

        inputs = sample.node_features
        coords0 = inputs["coords"]  # [N,3]
        features = inputs.get("features", coords0.new_zeros((coords0.size(0), 0)))

        # Ground truth sequence [T,N,3]
        N = coords0.size(0)
        gt_seq = torch.cat(
            [coords0.unsqueeze(0), sample.node_target.view(N, -1, 3).transpose(0, 1)],
            dim=0,
        )

        # Initial states
        y_t0 = gt_seq[0] - self.initial_vel * self.dt
        y_t1 = gt_seq[0]

        outputs: list[torch.Tensor] = []
        wall_x = (-30 - data_stats["node"]["pos_mean"][0]) / data_stats["node"][
            "pos_std"
        ][0]
        damage_emb = torch.zeros(N, 1, device=coords0.device)
        pred_nodes = coords0[:, 0] >= 0
        ref_idx = coords0.argmin(dim=0)[
            0
        ].item()  # ref idx for min x point on car geometry
        y_t1_diff_0 = y_t1 - y_t1[ref_idx]
        rel_acc = torch.zeros_like(y_t1)
        for t in range(self.rollout_steps):
            # Velocity normalization
            vel = (y_t1 - y_t0) / self.dt
            vel_norm = (vel - data_stats["node"]["norm_vel_mean"]) / (
                data_stats["node"]["norm_vel_std"] + EPS
            )

            nowall_y_t2 = self.dt * vel + y_t1
            collision_emb = torch.nn.functional.elu(10 * (nowall_y_t2[:, 0:1] - wall_x))
            damage_emb += torch.nn.functional.softmax(collision_emb**3, dim=0)
            y_t1_diff = y_t1 - y_t1[ref_idx]
            diff = y_t1_diff - y_t1_diff_0

            # Model input
            fx_t = torch.cat(
                [y_t1_diff, vel_norm, damage_emb, collision_emb], dim=-1
            )  # [N, 3+F+1]

            embedding = torch.cat([features, diff], dim=-1)

            def step_fn(fx, embedding):
                return super(
                    TransolverAutoregressiveRolloutTraining_front, self
                ).forward(fx=fx, embedding=embedding)

            if self.training:
                outf = ckpt(
                    step_fn,
                    fx_t[pred_nodes].unsqueeze(0),
                    embedding[pred_nodes].unsqueeze(0),
                    use_reentrant=False,
                ).squeeze(0)
            else:
                outf = step_fn(
                    fx_t[pred_nodes].unsqueeze(0), embedding[pred_nodes].unsqueeze(0)
                ).squeeze(0)

            # De-normalize acceleration
            rel_acc.zero_()
            rel_acc[pred_nodes] = (
                outf * data_stats["node"]["norm_acc_std"]
                + data_stats["node"]["norm_acc_mean"]
            )

            y_t2 = self.dt * (self.dt * rel_acc + vel) + y_t1
            # rest is copied from ground truth sequence
            y_t2[~pred_nodes] = gt_seq[t + 1, ~pred_nodes]

            outputs.append(y_t2)
            # y_t1, y_t0 = y_t2, y_t1
            if self.training and (t + 1) % rollout_len == 0:
                y_t1, y_t0 = gt_seq[t + 1], gt_seq[t]
            else:
                y_t1, y_t0 = y_t2, y_t1

        return torch.stack(outputs, dim=0)  # [T,N,3]


class TransolverAutoregressiveRolloutTraining_flare_xrel(Transolver_flare):
    """
    Transolver model with autoregressive rollout training.

    Predicts sequence by autoregressively updating velocity and position
    using predicted accelerations. Supports gradient checkpointing during training.
    """

    def __init__(self, *args, **kwargs):
        self.dt: float = kwargs.pop("dt")
        self.initial_vel: torch.Tensor = kwargs.pop("initial_vel")
        self.rollout_steps: int = kwargs.pop("num_time_steps") - 1
        super().__init__(*args, **kwargs)

    def forward(self, sample: SimSample, data_stats: dict, **kwargs) -> torch.Tensor:
        """
        Args:
            sample: SimSample containing node_features and node_target
            data_stats: dict containing normalization stats
        Returns:
            [T, N, 3] rollout of predicted positions
        """
        inputs = sample.node_features
        coords = inputs["coords"]  # [N,3]
        features = inputs.get("features", coords.new_zeros((coords.size(0), 0)))
        thickness = features[:, :1]
        # part_ids = features[:, 1]
        # unique_part_ids = torch.unique(part_ids)
        # parts = []
        # for unique_part_id in unique_part_ids:
        #     idx = torch.where(unique_part_id == part_ids)[0]
        #     parts.append(idx)

        N = coords.size(0)
        # device = coords.device

        # Initial states
        y_t1 = coords  # [N,3]
        y_t0 = y_t1 - self.initial_vel * self.dt  # backstep using initial velocity

        outputs: list[torch.Tensor] = []
        wall_x = (-30 - data_stats["node"]["pos_mean"][0]) / data_stats["node"][
            "pos_std"
        ][0]
        damage_emb = torch.zeros(N, 1, device=coords.device)
        ref_idx = coords.argmin(dim=0)[
            0
        ].item()  # ref idx for min x point on car geometry
        y_t1_diff_0 = y_t1 - y_t1[ref_idx]
        for t in range(self.rollout_steps):
            # Velocity normalization
            vel = (y_t1 - y_t0) / self.dt
            vel_norm = (vel - data_stats["node"]["norm_vel_mean"]) / (
                data_stats["node"]["norm_vel_std"] + EPS
            )

            nowall_y_t2 = self.dt * vel + y_t1
            collision_emb = torch.nn.functional.elu(10 * (nowall_y_t2[:, 0:1] - wall_x))
            damage_emb += torch.nn.functional.softmax(collision_emb**3, dim=0)
            y_t1_diff = y_t1 - y_t1[ref_idx]
            diff = y_t1_diff_0 - y_t1_diff

            # Model input
            fx_t = torch.cat(
                [y_t1_diff, vel_norm, collision_emb, damage_emb], dim=-1
            )  # [N, 3+F+1]
            embedding = torch.cat([thickness, diff], dim=-1)

            def step_fn(fx, embedding, parts=None):
                return super(
                    TransolverAutoregressiveRolloutTraining_flare_xrel, self
                ).forward(fx=fx, embedding=embedding, parts=parts)

            if self.training:
                outf = ckpt(
                    step_fn,
                    fx_t.unsqueeze(0),
                    embedding.unsqueeze(0),
                    use_reentrant=False,
                ).squeeze(0)
            else:
                outf = step_fn(fx_t.unsqueeze(0), embedding.unsqueeze(0)).squeeze(0)

            y_t2 = time_integration(
                y_t1, y_t0, self.dt, outf, data_stats, acc_or_vel="acc"
            )

            outputs.append(y_t2)
            y_t1, y_t0 = y_t2, y_t1

        return torch.stack(outputs, dim=0)  # [T,N,3]


class TransolverAutoregressiveRolloutTraining_flareX_xrel(Transolver_crash):
    """
    Transolver model with autoregressive rollout training.

    Predicts sequence by autoregressively updating velocity and position
    using predicted accelerations. Supports gradient checkpointing during training.
    """

    def __init__(self, *args, **kwargs):
        self.dt: float = kwargs.pop("dt")
        self.initial_vel: torch.Tensor = kwargs.pop("initial_vel")
        self.rollout_steps: int = kwargs.pop("num_time_steps") - 1
        super().__init__(*args, **kwargs)

    def forward(self, sample: SimSample, data_stats: dict, **kwargs) -> torch.Tensor:
        """
        Args:
            sample: SimSample containing node_features and node_target
            data_stats: dict containing normalization stats
        Returns:
            [T, N, 3] rollout of predicted positions
        """
        inputs = sample.node_features
        coords = inputs["coords"]  # [N,3]
        features = inputs.get("features", coords.new_zeros((coords.size(0), 0)))
        thickness = features[:, :1]
        # part_ids = features[:, 1]
        # unique_part_ids = torch.unique(part_ids)
        # parts = []
        # for unique_part_id in unique_part_ids:
        #     idx = torch.where(unique_part_id == part_ids)[0]
        #     parts.append(idx)

        N = coords.size(0)
        # device = coords.device

        # Initial states
        y_t1 = coords  # [N,3]
        y_t0 = y_t1 - self.initial_vel * self.dt  # backstep using initial velocity

        outputs: list[torch.Tensor] = []
        wall_x = (-30 - data_stats["node"]["pos_mean"][0]) / data_stats["node"][
            "pos_std"
        ][0]
        damage_emb = torch.zeros(N, 1, device=coords.device)
        ref_idx = coords.argmin(dim=0)[
            0
        ].item()  # ref idx for min x point on car geometry
        y_t1_diff_0 = y_t1 - y_t1[ref_idx]
        for t in range(self.rollout_steps):
            # Velocity normalization
            vel = (y_t1 - y_t0) / self.dt
            vel_norm = (vel - data_stats["node"]["norm_vel_mean"]) / (
                data_stats["node"]["norm_vel_std"] + EPS
            )

            nowall_y_t2 = self.dt * vel + y_t1
            collision_emb = torch.nn.functional.elu(10 * (nowall_y_t2[:, 0:1] - wall_x))
            damage_emb += torch.nn.functional.softmax(collision_emb**3, dim=0)
            y_t1_diff = y_t1 - y_t1[ref_idx]
            diff = y_t1_diff_0 - y_t1_diff

            # Model input
            fx_t = torch.cat(
                [y_t1_diff, vel_norm, collision_emb, damage_emb], dim=-1
            )  # [N, 3+F+1]
            embedding = torch.cat([thickness, diff], dim=-1)

            def step_fn(fx, embedding, parts=None):
                return super(
                    TransolverAutoregressiveRolloutTraining_flareX_xrel, self
                ).forward(fx=fx, embedding=embedding, parts=parts)

            if self.training:
                outf = ckpt(
                    step_fn,
                    fx_t.unsqueeze(0),
                    embedding.unsqueeze(0),
                    use_reentrant=False,
                ).squeeze(0)
            else:
                outf = step_fn(fx_t.unsqueeze(0), embedding.unsqueeze(0)).squeeze(0)

            y_t2 = time_integration(
                y_t1, y_t0, self.dt, outf, data_stats, acc_or_vel="acc"
            )

            outputs.append(y_t2)
            y_t1, y_t0 = y_t2, y_t1

        return torch.stack(outputs, dim=0)  # [T,N,3]


class TransolverTimeConditionalRollout(Transolver):
    """
    Transolver model with time-conditional rollout.

    Predicts each time step independently, conditioned on normalized time.
    """

    def __init__(self, *args, **kwargs):
        self.rollout_steps: int = kwargs.pop("num_time_steps") - 1
        super().__init__(*args, **kwargs)

    def forward(
        self,
        sample: SimSample,
        data_stats: dict,
    ) -> torch.Tensor:
        """
        Args:
            Sample: SimSample containing node_features and node_target
            data_stats: dict containing normalization stats
        Returns:
            [T, N, 3] rollout of predicted positions
        """
        inputs = sample.node_features
        x = inputs["coords"]  # [N,3]
        features = inputs.get("features", x.new_zeros((x.size(0), 0)))  # [N,F]

        outputs: list[torch.Tensor] = []
        time_seq = torch.linspace(0.0, 1.0, self.rollout_steps, device=x.device)

        for time in time_seq:
            fx_t = features  # [N,F]

            def step_fn(fx, embedding, time_t):
                return super(TransolverTimeConditionalRollout, self).forward(
                    fx=fx, embedding=embedding, time=time_t
                )

            if self.training:
                outf = ckpt(
                    step_fn,
                    fx_t.unsqueeze(0),
                    x.unsqueeze(0),
                    time.unsqueeze(0),
                    use_reentrant=False,
                ).squeeze(0)
            else:
                outf = step_fn(
                    fx_t.unsqueeze(0), x.unsqueeze(0), time.unsqueeze(0)
                ).squeeze(0)

            y_t2 = x + outf
            outputs.append(y_t2)

        return torch.stack(outputs, dim=0)  # [T,N,3]


class MeshGraphNetAutoregressiveRolloutTraining(MeshGraphNet):
    """MeshGraphNet with autoregressive rollout training."""

    def __init__(self, *args, **kwargs):
        self.dt: float = kwargs.pop("dt")
        self.initial_vel: torch.Tensor = kwargs.pop("initial_vel")
        self.rollout_steps: int = kwargs.pop("num_time_steps") - 1
        super().__init__(*args, **kwargs)

    def forward(self, sample: SimSample, data_stats: dict, **kwargs) -> torch.Tensor:
        """
        Args:
            Sample: SimSample containing node_features and node_target
            data_stats: dict containing normalization stats
        Returns:
            [T, N, 3] rollout of predicted positions
        """
        inputs = sample.node_features
        coords = inputs["coords"]  # [N,3]
        features = inputs.get(
            "features", coords.new_zeros((coords.size(0), 0))
        )  # [N,F]
        edge_features = sample.graph.edge_attr
        graph = sample.graph

        N = coords.size(0)
        y_t1 = coords
        y_t0 = y_t1 - self.initial_vel * self.dt

        outputs: list[torch.Tensor] = []
        for _ in range(self.rollout_steps):
            vel = (y_t1 - y_t0) / self.dt
            vel_norm = (vel - data_stats["node"]["norm_vel_mean"]) / (
                data_stats["node"]["norm_vel_std"] + EPS
            )
            fx_t = torch.cat([y_t1, vel_norm, features], dim=-1)

            def step_fn(nf, ef, g):
                return super(MeshGraphNetAutoregressiveRolloutTraining, self).forward(
                    node_features=nf, edge_features=ef, graph=g
                )

            outf = (
                ckpt(step_fn, fx_t, edge_features, graph, use_reentrant=False)
                if self.training
                else step_fn(fx_t, edge_features, graph)
            )

            acc = (
                outf * data_stats["node"]["norm_acc_std"]
                + data_stats["node"]["norm_acc_mean"]
            )

            vel = self.dt * acc + vel
            y_t2 = self.dt * vel + y_t1

            outputs.append(y_t2)
            y_t1, y_t0 = y_t2, y_t1

        return torch.stack(outputs, dim=0)


class MeshGraphNetTimeConditionalRollout(MeshGraphNet):
    """MeshGraphNet with time-conditional rollout."""

    def __init__(self, *args, **kwargs):
        self.rollout_steps: int = kwargs.pop("num_time_steps") - 1
        super().__init__(*args, **kwargs)

    def forward(self, sample: SimSample, data_stats: dict, **kwargs) -> torch.Tensor:
        """
        Args:
            Sample: SimSample containing node_features and node_target
            data_stats: dict containing normalization stats
        Returns:
            [T, N, 3] rollout of predicted positions
        """
        inputs = sample.node_features
        x = inputs["coords"]  # [N,3]
        features = inputs.get("features", x.new_zeros((x.size(0), 0)))  # [N,F]
        edge_features = sample.graph.edge_attr
        graph = sample.graph

        outputs: list[torch.Tensor] = []
        time_seq = torch.linspace(0.0, 1.0, self.rollout_steps, device=x.device)

        for time in time_seq:
            fx_t = torch.cat([x, features, time.expand(x.size(0), 1)], dim=-1)

            def step_fn(nf, ef, g):
                return super(MeshGraphNetTimeConditionalRollout, self).forward(
                    node_features=nf, edge_features=ef, graph=g
                )

            outf = (
                ckpt(step_fn, fx_t, edge_features, graph, use_reentrant=False)
                if self.training
                else step_fn(fx_t, edge_features, graph)
            )

            y_t2 = x + outf
            outputs.append(y_t2)

        return torch.stack(outputs, dim=0)


class TransolverOneStepRollout(
    Transolver
):  # TODO this can be merged with TransolverAutoregressiveRolloutTraining
    """
    One-step rollout:
      - Training: teacher forcing (uses GT for each step, but first step needs backstep)
      - Inference: autoregressive (uses predictions)
    """

    def __init__(self, *args, **kwargs):
        self.dt: float = kwargs.pop("dt", 5e-3)
        self.initial_vel: torch.Tensor = kwargs.pop("initial_vel")
        self.rollout_steps: int = kwargs.pop("num_time_steps") - 1
        super().__init__(*args, **kwargs)

    def forward(self, sample: SimSample, data_stats: dict, **kwargs) -> torch.Tensor:
        inputs = sample.node_features
        coords0 = inputs["coords"]  # [N,3]
        features = inputs.get("features", coords0.new_zeros((coords0.size(0), 0)))

        # Ground truth sequence [T,N,3]
        N = coords0.size(0)
        gt_seq = torch.cat(
            [coords0.unsqueeze(0), sample.node_target.view(N, -1, 3).transpose(0, 1)],
            dim=0,
        )

        outputs: list[torch.Tensor] = []

        # First step: backstep to create y_-1
        y_t0 = gt_seq[0] - self.initial_vel * self.dt
        y_t1 = gt_seq[0]

        for t in range(self.rollout_steps):
            if self.training and t > 0:
                # teacher forcing uses GT pairs
                y_t0, y_t1 = gt_seq[t - 1], gt_seq[t]

            vel = (y_t1 - y_t0) / self.dt
            vel_norm = (vel - data_stats["node"]["norm_vel_mean"]) / (
                data_stats["node"]["norm_vel_std"] + EPS
            )
            fx_t = torch.cat([vel_norm, features], dim=-1)

            def step_fn(fx, embedding):
                return super(TransolverOneStepRollout, self).forward(
                    fx=fx, embedding=embedding
                )

            if self.training:
                outf = ckpt(
                    step_fn, fx_t.unsqueeze(0), y_t1.unsqueeze(0), use_reentrant=False
                ).squeeze(0)
            else:
                outf = step_fn(fx_t.unsqueeze(0), y_t1.unsqueeze(0)).squeeze(0)

            acc = (
                outf * data_stats["node"]["norm_acc_std"]
                + data_stats["node"]["norm_acc_mean"]
            )
            vel_pred = self.dt * acc + vel
            y_t2_pred = self.dt * vel_pred + y_t1

            outputs.append(y_t2_pred)

            if not self.training:
                # autoregressive update for inference
                y_t0, y_t1 = y_t1, y_t2_pred

        return torch.stack(outputs, dim=0)  # [T,N,3]


class MeshGraphNetOneStepRollout(MeshGraphNet):
    """
    MeshGraphNet with one-step rollout:
      - Training: teacher forcing (uses GT positions at each step, first step needs backstep)
      - Inference: autoregressive (uses predictions)
    """

    def __init__(self, *args, **kwargs):
        self.dt: float = kwargs.pop("dt", 5e-3)
        self.initial_vel: torch.Tensor = kwargs.pop("initial_vel")
        self.rollout_steps: int = kwargs.pop("num_time_steps") - 1
        super().__init__(*args, **kwargs)

    def forward(self, sample: SimSample, data_stats: dict, **kwargs) -> torch.Tensor:
        inputs = sample.node_features
        coords0 = inputs["coords"]  # [N,3]
        features = inputs.get(
            "features", coords0.new_zeros((coords0.size(0), 0))
        )  # [N,F]
        edge_features = sample.graph.edge_attr
        graph = sample.graph

        # Full ground truth trajectory [T,N,3]
        N = coords0.size(0)
        gt_seq = torch.cat(
            [coords0.unsqueeze(0), sample.node_target.view(N, -1, 3).transpose(0, 1)],
            dim=0,
        )

        outputs: list[torch.Tensor] = []

        # First step: construct backstep
        y_t0 = gt_seq[0] - self.initial_vel * self.dt
        y_t1 = gt_seq[0]

        for t in range(self.rollout_steps):
            if self.training and t > 0:
                # Teacher forcing: use GT sequence
                y_t0, y_t1 = gt_seq[t - 1], gt_seq[t]

            vel = (y_t1 - y_t0) / self.dt
            vel_norm = (vel - data_stats["node"]["norm_vel_mean"]) / (
                data_stats["node"]["norm_vel_std"] + EPS
            )

            fx_t = torch.cat([y_t1, vel_norm, features], dim=-1)

            def step_fn(nf, ef, g):
                return super(MeshGraphNetOneStepRollout, self).forward(
                    node_features=nf, edge_features=ef, graph=g
                )

            if self.training:
                outf = ckpt(step_fn, fx_t, edge_features, graph, use_reentrant=False)
            else:
                outf = step_fn(fx_t, edge_features, graph)

            acc = (
                outf * data_stats["node"]["norm_acc_std"]
                + data_stats["node"]["norm_acc_mean"]
            )
            vel_pred = self.dt * acc + vel
            y_t2_pred = self.dt * vel_pred + y_t1

            outputs.append(y_t2_pred)

            if not self.training:
                # Autoregressive update
                y_t0, y_t1 = y_t1, y_t2_pred

        return torch.stack(outputs, dim=0)  # [T,N,3]


# class FIGConvUNetTimeConditionalRollout(FIGConvUNet):
#     """
#     FIGConvUNet with time-conditional rollout for crash simulation.

#     Predicts each time step independently, conditioned on normalized time.
#     """

#     def __init__(self, *args, **kwargs):
#         self.rollout_steps: int = kwargs.pop("num_time_steps") - 1
#         super().__init__(*args, **kwargs)

#     def forward(
#         self,
#         sample: SimSample,
#         data_stats: dict,
#     ) -> torch.Tensor:
#         """
#         Args:
#             Sample: SimSample containing node_features and node_target
#             data_stats: dict containing normalization stats
#         Returns:
#             [T, N, 3] rollout of predicted positions
#         """
#         inputs = sample.node_features
#         x = inputs["coords"]  # initial pos [N, 3]
#         features = inputs.get("features", x.new_zeros((x.size(0), 0)))  # [N, F]

#         outputs: list[torch.Tensor] = []
#         time_seq = torch.linspace(0.0, 1.0, self.rollout_steps, device=x.device)

#         for time_t in time_seq:
#             # Prepare vertices for FIGConvUNet: [1, N, 3]
#             vertices = x.unsqueeze(0)  # [1, N, 3]

#             # Prepare features: features + time [N, F+1]
#             time_expanded = time_t.expand(x.size(0), 1)  # [N, 1]
#             features_t = torch.cat([features, time_expanded], dim=-1)  # [N, F+1]
#             features_t = features_t.unsqueeze(0)  # [1, N, F+1]

#             def step_fn(verts, feats):
#                 out, _ = super(FIGConvUNetTimeConditionalRollout, self).forward(
#                     vertices=verts, features=feats
#                 )
#                 return out

#             if self.training:
#                 outf = ckpt(
#                     step_fn,
#                     vertices,
#                     features_t,
#                     use_reentrant=False,
#                 ).squeeze(0)  # [N, 3]
#             else:
#                 outf = step_fn(vertices, features_t).squeeze(0)  # [N, 3]

#             y_t = x + outf
#             outputs.append(y_t)

#         return torch.stack(outputs, dim=0)  # [T, N, 3]


# class FIGConvUNetOneStepRollout(FIGConvUNet):
#     """
#     FIGConvUNet with one-step rollout for crash simulation.

#     - Training: teacher forcing (uses GT positions at each step)
#     - Inference: autoregressive (uses predictions)
#     """

#     def __init__(self, *args, **kwargs):
#         self.dt: float = kwargs.pop("dt", 5e-3)
#         self.initial_vel: torch.Tensor = kwargs.pop("initial_vel")
#         self.rollout_steps: int = kwargs.pop("num_time_steps") - 1
#         super().__init__(*args, **kwargs)

#     def forward(self, sample: SimSample, data_stats: dict) -> torch.Tensor:
#         """
#         Args:
#             Sample: SimSample containing node_features and node_target
#             data_stats: dict containing normalization stats
#         Returns:
#             [T, N, 3] rollout of predicted positions
#         """
#         inputs = sample.node_features
#         x0 = inputs["coords"]  # initial pos [N, 3]
#         features = inputs.get("features", x0.new_zeros((x0.size(0), 0)))  # [N, F]

#         # Ground truth sequence [T, N, 3]
#         N = x0.size(0)
#         gt_seq = torch.cat(
#             [x0.unsqueeze(0), sample.node_target.view(N, -1, 3).transpose(0, 1)],
#             dim=0,
#         )

#         outputs: list[torch.Tensor] = []
#         # First step: backstep to create y_-1
#         y_t0 = gt_seq[0] - self.initial_vel * self.dt
#         y_t1 = gt_seq[0]

#         for t in range(self.rollout_steps):
#             # In training mode (except first step), use ground truth positions
#             if self.training and t > 0:
#                 y_t0, y_t1 = gt_seq[t - 1], gt_seq[t]

#             # Prepare vertices for FIGConvUNet: [1, N, 3]
#             vertices = y_t1.unsqueeze(0)  # [1, N, 3]

#             vel = (y_t1 - y_t0) / self.dt
#             vel_norm = (vel - data_stats["node"]["norm_vel_mean"]) / (
#                 data_stats["node"]["norm_vel_std"] + EPS
#             )

#             # [1, N, 3 + F]
#             fx_t = torch.cat([vel_norm, features], dim=-1).unsqueeze(0)

#             def step_fn(verts, feats):
#                 out, _ = super(FIGConvUNetOneStepRollout, self).forward(
#                     vertices=verts, features=feats
#                 )
#                 return out

#             if self.training:
#                 outf = ckpt(
#                     step_fn,
#                     vertices,
#                     fx_t,
#                     use_reentrant=False,
#                 ).squeeze(0)  # [N, 3]
#             else:
#                 outf = step_fn(vertices, fx_t).squeeze(0)  # [N, 3]

#             acc = (
#                 outf * data_stats["node"]["norm_acc_std"]
#                 + data_stats["node"]["norm_acc_mean"]
#             )
#             vel_pred = self.dt * acc + vel
#             y_t2_pred = self.dt * vel_pred + y_t1

#             outputs.append(y_t2_pred)

#             if not self.training:
#                 # autoregressive update for inference
#                 y_t0, y_t1 = y_t1, y_t2_pred

#         return torch.stack(outputs, dim=0)  # [T, N, 3]


# class FIGConvUNetAutoregressiveRolloutTraining(FIGConvUNet):
#     """
#     FIGConvUNet with autoregressive rollout training for crash simulation.

#     Predicts sequence by autoregressively updating velocity and position
#     using predicted accelerations. Supports gradient checkpointing during training.
#     """

#     def __init__(self, *args, **kwargs):
#         self.dt: float = kwargs.pop("dt")
#         self.initial_vel: torch.Tensor = kwargs.pop("initial_vel")
#         self.rollout_steps: int = kwargs.pop("num_time_steps") - 1
#         super().__init__(*args, **kwargs)

#     def forward(self, sample: SimSample, data_stats: dict) -> torch.Tensor:
#         """
#         Args:
#             sample: SimSample containing node_features and node_target
#             data_stats: dict containing normalization stats
#         Returns:
#             [T, N, 3] rollout of predicted positions
#         """
#         inputs = sample.node_features
#         coords = inputs["coords"]  # [N, 3]
#         features = inputs.get("features", coords.new_zeros((coords.size(0), 0)))
#         N = coords.size(0)
#         device = coords.device

#         # Initial states
#         y_t1 = coords  # [N, 3]
#         y_t0 = y_t1 - self.initial_vel * self.dt  # backstep using initial velocity

#         outputs: list[torch.Tensor] = []
#         for t in range(self.rollout_steps):
#             time_t = 0.0 if self.rollout_steps <= 1 else t / (self.rollout_steps - 1)
#             time_t = torch.tensor([time_t], device=device, dtype=torch.float32)

#             # Velocity normalization
#             vel = (y_t1 - y_t0) / self.dt
#             vel_norm = (vel - data_stats["node"]["norm_vel_mean"]) / (
#                 data_stats["node"]["norm_vel_std"] + EPS
#             )

#             # Prepare vertices for FIGConvUNet: [1, N, 3]
#             vertices = y_t1.unsqueeze(0)  # [1, N, 3]

#             # Prepare features: vel_norm + features + time [N, 3+F+1]
#             fx_t = torch.cat(
#                 [vel_norm, features, time_t.expand(N, 1)], dim=-1
#             )  # [N, 3+F+1]
#             fx_t = fx_t.unsqueeze(0)  # [1, N, 3+F+1]

#             def step_fn(verts, feats):
#                 out, _ = super(FIGConvUNetAutoregressiveRolloutTraining, self).forward(
#                     vertices=verts, features=feats
#                 )
#                 return out

#             if self.training:
#                 outf = ckpt(
#                     step_fn,
#                     vertices,
#                     fx_t,
#                     use_reentrant=False,
#                 ).squeeze(0)  # [N, 3]
#             else:
#                 outf = step_fn(vertices, fx_t).squeeze(0)  # [N, 3]

#             # De-normalize acceleration
#             acc = (
#                 outf * data_stats["node"]["norm_acc_std"]
#                 + data_stats["node"]["norm_acc_mean"]
#             )
#             vel = self.dt * acc + vel
#             y_t2 = self.dt * vel + y_t1

#             outputs.append(y_t2)
#             y_t1, y_t0 = y_t2, y_t1

#         return torch.stack(outputs, dim=0)  # [T, N, 3]
