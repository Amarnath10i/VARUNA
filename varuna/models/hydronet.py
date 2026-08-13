"""HydroNet -- Graph-Neural-ODE for river discharge with per-step uncertainty.

Novel contributions
-------------------
1. **Learnable routing.** A GNN over the river-network graph performs message
   passing between gauges/catchments so upstream rainfall is *routed*
   downstream; the node state is then evolved by a Neural-ODE whose
   right-hand side is itself a small network conditioned on the routed state.

2. **ODE-as-depth, not fixed stacks.** A discrete-step RK4/Midpoint solver
   replaces fixed residual depth: temporal resolution of the hydrologic
   response is a model decision, giving continuous-time dynamics that match
   the physics of flood-wave travel time.

3. **Per-step conformal wrapper.** The ODE output feeds a per-node Gaussian
   head; split conformal prediction (see `varuna.uncertainty.conformal`)
   then converts it to a 1-alpha coverage interval at *every* lead step.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MidpointODE:
    """Second-order (Midpoint / improved Euler) explicit ODE integrator.

    Kept dependency-free (no torchdiffeq) and parallelisable across the batch.
    """

    @staticmethod
    def integrate(dynamics, z0: torch.Tensor, t_span: torch.Tensor) -> list[torch.Tensor]:
        zs = [z0]
        z = z0
        for i in range(len(t_span) - 1):
            h = t_span[i + 1] - t_span[i]
            k1 = dynamics(z)
            k2 = dynamics(z + h * k1 / 2.0)
            z = z + h * k2
            zs.append(z)
        return zs


class GNNEncoder(nn.Module):
    """Message-passing encoder that routes rainfall along the river network.

    edge_index: [2, E] directed edges (upstream -> downstream).
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, n_layers: int = 2) -> None:
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """x: [N, in_dim] node features; edge_index [2, E]."""
        h = F.relu(self.in_proj(x))
        src, dst = edge_index
        for lin in self.layers:
            msg = F.relu(lin(h[src]))  # [E, hidden]
            agg = torch.zeros_like(h)
            agg.index_add_(0, dst, msg)
            h = h + agg
            h = self.norm(h)
        return self.out_proj(h)


class HydroNet(nn.Module):
    """Graph-Neural-ODE discharge forecaster.

    Args:
        num_nodes: number of gauges / catchments.
        node_feat_dim: per-node feature width (static = dynamic = node_feat_dim).
        hidden_dim: hidden state width of the ODE.
        out_dim: 1 (discharge) per node.
    """

    def __init__(
        self,
        num_nodes: int,
        node_feat_dim: int,
        hidden_dim: int = 64,
        out_dim: int = 1,
        n_gnn_layers: int = 2,
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.node_feat_dim = node_feat_dim
        self.encoder = GNNEncoder(node_feat_dim, hidden_dim, hidden_dim, n_gnn_layers)
        self.ode_dynamics = nn.Sequential(
            nn.Linear(hidden_dim + node_feat_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.decode = nn.Linear(hidden_dim, out_dim)
        self.decode_logvar = nn.Linear(hidden_dim, out_dim)

    def _dynamics(self, u: torch.Tensor, x_dyn: torch.Tensor) -> torch.Tensor:
        return self.ode_dynamics(torch.cat([u, x_dyn], dim=-1))

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        t_span: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """x: [T, N, 2*node_feat_dim]; returns per-step (mu, logvar, state) over nodes.

        x[:, :, :node_feat_dim] is time-varying exogenous forcing (rainfall, etc.),
        x[:, :, node_feat_dim:] is static attributes used to condition the encoder.
        """
        dyn_dim = self.node_feat_dim
        x_dyn = x[..., :dyn_dim]
        x_static = x[..., dyn_dim:]
        x_static = x[..., dyn_dim:]

        h0 = self.encoder(x_static[0], edge_index)  # [N, hidden]
        steps = [h0]
        u = h0
        # roll the ODE forward, reconditioning with the exogenous forcing at
        # each step (a control-affine structure: dynamics(u, forcing)).
        for i in range(len(t_span) - 1):
            forcing = x_dyn[i + 1]  # [N, dyn_dim]
            # shift time window handled by caller; here force is per-step
            u = self._midpoint_step(u, forcing, t_span[i + 1] - t_span[i])
            steps.append(u)

        states = torch.stack(steps)  # [T, N, hidden]
        mu = self.decode(states)
        logvar = self.decode_logvar(states)
        return mu, logvar, states

    def _midpoint_step(self, u: torch.Tensor, forcing: torch.Tensor, h: float) -> torch.Tensor:
        k1 = self._dynamics(u, forcing)
        k2 = self._dynamics(u + h * k1 / 2.0, forcing)
        return u + h * k2