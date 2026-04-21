from __future__ import annotations

import torch
import torch.nn as nn


class GraphConv(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x: [batch, nodes, features]
        support = self.linear(x)
        return torch.einsum("ij,bjf->bif", adj, support)


class STGNN(nn.Module):
    def __init__(self, num_features: int, num_horizons: int, hidden_dim: int, gcn_out_dim: int, dropout: float):
        super().__init__()
        self.gcn1 = GraphConv(num_features, hidden_dim)
        self.gcn2 = GraphConv(hidden_dim, gcn_out_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.temporal = nn.LSTM(
            input_size=gcn_out_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.mean_head = nn.Linear(hidden_dim, num_horizons)
        self.logvar_head = nn.Linear(hidden_dim, num_horizons)

    def forward(self, x: torch.Tensor, adj: torch.Tensor):
        # x: [batch, window, nodes, features]
        batch_size, window, nodes, features = x.shape
        spatial_outputs = []
        for t in range(window):
            xt = x[:, t, :, :]
            h = self.relu(self.gcn1(xt, adj))
            h = self.dropout(h)
            h = self.relu(self.gcn2(h, adj))
            spatial_outputs.append(h)

        spatial_seq = torch.stack(spatial_outputs, dim=1)
        spatial_seq = spatial_seq.permute(0, 2, 1, 3).contiguous()
        spatial_seq = spatial_seq.view(batch_size * nodes, window, -1)

        temporal_out, _ = self.temporal(spatial_seq)
        last_hidden = temporal_out[:, -1, :]

        pred_mean = self.mean_head(last_hidden)
        pred_logvar = self.logvar_head(last_hidden)

        pred_mean = pred_mean.view(batch_size, nodes, -1).permute(0, 2, 1)
        pred_logvar = pred_logvar.view(batch_size, nodes, -1).permute(0, 2, 1)
        return pred_mean, pred_logvar


def gaussian_nll_loss(mean: torch.Tensor, logvar: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean(0.5 * (logvar + ((target - mean) ** 2) / torch.exp(logvar)))