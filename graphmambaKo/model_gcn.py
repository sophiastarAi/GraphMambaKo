from torch import nn
import torch
import numpy as np
from torch_geometric.nn import ChebConv, GATConv
import torch.nn.functional as F
from einops import rearrange


class CELU(nn.Module):
    def __init__(self, alpha=1.0):
        super(CELU, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        return torch.where(x > 0, x, self.alpha * (torch.exp(x / self.alpha) - 1))


class ImprovedMambaBlock(nn.Module):
    def __init__(self, seq_len, d_model, state_size, device, batch, dropout=0.1):
        super(ImprovedMambaBlock, self).__init__()
        self.d_model = d_model
        self.state_size = state_size
        self.device = device


        self.lift_fn = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.feature_transform = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.state_init = nn.Parameter(torch.randn(1, d_model, state_size) * 0.02)

        self.A = nn.Parameter(torch.empty(d_model, state_size))
        nn.init.xavier_uniform_(self.A, gain=0.1)

        self.delta_net = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, d_model)
        )

        self.B_net = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, state_size)
        )

        self.C_net = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, state_size)
        )

        self.output_norm = nn.LayerNorm(d_model)

    def discretization(self, delta, B):
        delta_A = torch.einsum("bld,dn->bldn", delta, self.A)
        dA = torch.exp(-F.celu(delta_A))
        dB = torch.einsum("bld,bln->bldn", delta, B)
        return dA, dB

    def forward(self, x, h_prev=None):
        batch_size, seq_len, _ = x.shape

        # 1. Koopman enhanced
        lifted = self.lift_fn(x)

        # 2. feature cat and transform
        combined = torch.cat([x, lifted], dim=-1)
        x_trans = self.feature_transform(combined)

        # 3. state init
        if h_prev is None:
            h_prev = self.state_init.repeat(batch_size, 1, 1)

        # 4. dynamic parm
        delta = F.softplus(self.delta_net(x_trans)) + 1e-8  # 避免除零
        B = torch.tanh(self.B_net(x_trans))  # 使用tanh限制范围
        C = self.C_net(x_trans)

        # 5. discretization
        dA, dB = self.discretization(delta, B)

        h_prev_expanded = h_prev.unsqueeze(1)
        x_expanded = x.unsqueeze(-1)

        h_new = dA * h_prev_expanded + x_expanded * dB
        #
        output = torch.einsum('bldn,bldn->bld', C.unsqueeze(2), h_new)

        output = output + x
        output = self.output_norm(output)

        last_h_state = h_new[:, -1]

        return output, last_h_state


class ImprovedEncoder(nn.Module):
    def __init__(self, m, n, b, ALPHA=1, batch=128, dropout=0.1):
        super(ImprovedEncoder, self).__init__()
        self.N = m * n
        self.gcn_out = 16
        self.dim = 1
        self.batch = batch
        self.dropout_rate = dropout

        # 改进的图卷积
        self.static_conv = ChebConv(batch * 24, self.gcn_out, K=2)  #batch*seq_len

        # 动态图卷积改进
        self.dynamic_conv = GATConv(batch * 24, self.gcn_out, heads=2, concat=True)


        self.fusion_net = nn.Sequential(
            nn.Linear(3 * self.gcn_out, 24),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(24, 1)
        )

        self.encoder_net = nn.Sequential(
            nn.Linear(self.N, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, b)
        )


        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if hasattr(m, 'activation') and m.activation == 'relu':
                    gain = nn.init.calculate_gain('relu')
                elif hasattr(m, 'activation') and m.activation == 'tanh':
                    gain = nn.init.calculate_gain('tanh')
                else:
                    gain = 1.0

                nn.init.xavier_uniform_(m.weight, gain=gain)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def build_dynamic_adjacency(self, x):
        B, T, N = x.shape

        node_features = x.permute(2, 0, 1).reshape(N, -1)

        norms = torch.norm(node_features, dim=1, keepdim=True) + 1e-8
        normalized_features = node_features / norms

        sim_matrix = torch.mm(normalized_features, normalized_features.t())
        sim_matrix = F.softplus(sim_matrix)

        sim_matrix = (sim_matrix + sim_matrix.t()) / 2
        sim_matrix = sim_matrix.fill_diagonal_(1.0)

        degree = sim_matrix.sum(dim=1)
        degree_inv_sqrt = torch.diag(1.0 / torch.sqrt(degree + 1e-8))
        adj_dynamic = torch.mm(torch.mm(degree_inv_sqrt, sim_matrix), degree_inv_sqrt)

        return adj_dynamic

    def forward(self, x, edge_index, edge_attr):
        batchsize, seq_len, _ = x.shape

        x_static = x.contiguous().view(-1, self.N).t()
        static_out = self.static_conv(x_static, edge_index, edge_attr.float())
        static_out = static_out.unsqueeze(0).unsqueeze(0).repeat(batchsize, seq_len, 1, 1)

        adj_dynamic = self.build_dynamic_adjacency(x)
        rows, cols = torch.nonzero(adj_dynamic, as_tuple=True)
        dynamic_edge_index = torch.stack([rows, cols], dim=0)
        dynamic_edge_attr = adj_dynamic[rows, cols]

        x_dynamic = x.reshape(-1, self.N).t()
        dynamic_out = self.dynamic_conv(x_dynamic, dynamic_edge_index, dynamic_edge_attr)
        dynamic_out = dynamic_out.unsqueeze(0).unsqueeze(0).repeat(batchsize, seq_len, 1, 1)

        combined = torch.cat([static_out, dynamic_out], dim=-1)
        output = self.fusion_net(combined).squeeze(-1)

        encoded = self.encoder_net(output.view(batchsize * seq_len, -1))
        encoded = encoded.view(batchsize, seq_len, -1)

        return encoded


class GraphMambaKo(nn.Module):
    def __init__(self, m, n, b, steps, steps_back, seq_len, d_model, state_size, device,
                 alpha=1, init_scale=1, batch=32, dropout=0.1):
        super(GraphMambaKo, self).__init__()
        self.steps = steps
        self.steps_back = steps_back
        self.device = device
        self.seq_len = seq_len
        self.d_model = d_model

        self.encoder = ImprovedEncoder(m, n, b, ALPHA=alpha, batch=batch, dropout=dropout)

        self.encoder_dynamics = ImprovedMambaBlock(seq_len, d_model, state_size, device, batch, dropout)

        self.decoder_dynamics = ImprovedMambaBlock(steps, d_model, state_size, device, batch, dropout)

        self.encoder_adapter = nn.Sequential(
            nn.Linear(b, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, d_model)
        )

        self.decoder_adapter = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, d_model)
        )

        self.context_net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.output_net = nn.Sequential(
            nn.Linear(d_model, 48),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(48, m)
        )

    def forward(self, x, edge_index, edge_attr, mode='forward'):
        batch_size = x.size(0)

        z_encoded = self.encoder(x, edge_index, edge_attr)
        z_encoded = self.encoder_adapter(z_encoded)

        encoded_seq, h_state = self.encoder_dynamics(z_encoded, None)
        # print("encoded seq", encoded_seq.shape)  #64,24,128

        context = encoded_seq[:, -1:, :]
        context = self.context_net(context)

        decoder_input = torch.zeros(batch_size, self.steps, self.d_model, device=self.device)


        decoded_seq, _ = self.decoder_dynamics(decoder_input, h_state)

        outputs = self.output_net(decoded_seq)  # [batch_size, steps, m]

        return outputs, []
