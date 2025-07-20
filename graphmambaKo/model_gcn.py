from torch import nn
import torch
# from layers import *
import numpy as np
from torch_geometric.nn import ChebConv,GATConv
import torch.nn.functional as F

import torch.nn.functional as F
from einops import rearrange

class CELU(nn.Module):
    def __init__(self, alpha=1.0):
        super(CELU, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        # CELU
        return torch.where(x > 0, -x, -self.alpha * (torch.exp(x / self.alpha) - 1))


class mamba_insert(nn.Module):
    def __init__(self, seq_len, d_model, state_size, device,batch):
        super(mamba_insert, self).__init__()
        self.fc1 = nn.Linear(d_model, d_model, device=device)
        self.fc2 = nn.Linear(d_model, state_size, device=device)
        self.fc3 = nn.Linear(d_model, state_size, device=device)

        # 关键参数设置
        self.input_dim = d_model  # 128
        self.lift_dim = d_model  # 128
        self.d_model = d_model  # 128
        self.state_size = state_size  # 64
        self.hidden_dim = 128  #

        # Koopman increase dimension
        self.lift_fn = nn.Sequential(
            nn.Linear(self.input_dim, 128),  # 128 → 128
            nn.ReLU(),
            nn.Linear(128, self.lift_dim),  # 128 → 128
            nn.Tanh()
        )

        # feature transform
        self.fc_net = nn.Sequential(
            nn.Linear(self.input_dim + self.lift_dim, self.hidden_dim),  # 128+128=256 → 128
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim, d_model)  # 128 → 128
        )

        # state parm
        self.state_init = nn.Parameter(torch.zeros(1, d_model, state_size, device=device))
        self.A = nn.Parameter(torch.empty(d_model, state_size, device=device))
        nn.init.kaiming_normal_(self.A)

    def discretization(self, delta, B):
        #softplus = torch.nn.Softplus()
        dB = torch.einsum("bld,bln->bldn", delta, B)  # [b,l,d,n]
        dA = torch.exp(-F.celu(torch.einsum("bld,dn->bldn", delta, self.A)))  # [b,l,d,n]
        #dA = torch.exp(softplus(torch.einsum("bld,dn->bldn", delta, self.A)))  # [b,l,d,n]
        return dA, dB

    def forward(self, x, h_prev=None):
        batch_size, seq_len, _ = x.shape

        # 1. Koopman enhance [256,12,128]->[256,12,128]
        lifted = self.lift_fn(x)  # [b,l,d] = [256,12,128]

        # 2. feature emerge transform [256,12,256]->[256,12,128]
        combined = torch.cat([x, lifted], dim=-1)  # [256,12,256]
        x_trans = self.fc_net(combined)  # [256,12,128]

        # 3. state init [256,128,64]
        if h_prev is None:
            h_prev = self.state_init.repeat(batch_size, 1, 1)  # [256,128,64]

        # 4. dynamic parm compute
        B = self.fc2(x_trans)  # [256,12,64]
        C = self.fc3(x_trans)  # [256,12,64]
        delta = F.softplus(self.fc1(x_trans))  # [256,12,128]

        # 5. discret
        dA, dB = self.discretization(delta, B)  # 两者均为 [256,12,128,64]

        # 6. update
        h_prev_expanded = h_prev.unsqueeze(1).repeat(1, seq_len, 1, 1)  # [256,12,128,64]
        x_trans_expanded = rearrange(x_trans, "b l d -> b l d 1")  # [256,12,128,1]
        h_new = dA * h_prev_expanded + x_trans_expanded * dB  # [256,12,128,64]

        # 7. output generate [256,12,128]
        output = torch.einsum('bln,bldn->bld', C, h_new)  # [256,12,128]

        # 8. return state [256,128,64]
        last_h_state = h_new[:, -1]  # [256,128,64]

        return output, last_h_state


def gaussian_init_(n_units, std=1):
    sampler = torch.distributions.Normal(torch.Tensor([0]), torch.Tensor([std/n_units]))
    Omega = sampler.sample((n_units, n_units))[..., 0]
    return Omega

def build_dynamic_adjacency(self, x):
    """
    input: x - (B, T, N)
    output: adj_dynamic - (N, N)
    """
    B, T, N = x.shape


    node_features = x.permute(2, 0, 1)  # (N, B, T)
    node_features = node_features.reshape(N, -1)  # (N, B*T)

    eps = 1e-8
    norms = torch.norm(node_features, dim=1, keepdim=True) + eps
    normalized_features = node_features / norms

    sim_matrix = torch.mm(normalized_features, normalized_features.t())  # (N, N)

    sim_matrix = F.relu(sim_matrix)


    sim_matrix = sim_matrix.fill_diagonal_(1.0)


    degree = sim_matrix.sum(dim=1)  #
    degree_inv_sqrt = torch.diag(1.0 / torch.sqrt(degree + eps))
    adj_dynamic = torch.mm(torch.mm(degree_inv_sqrt, sim_matrix), degree_inv_sqrt)

    return adj_dynamic


def dynamic_graph_conv(self, x):

    adj_dynamic = self.build_dynamic_adjacency(x)  # (N, N)

    rows, cols = torch.nonzero(adj_dynamic, as_tuple=True)
    dynamic_edge_index = torch.stack([rows, cols], dim=0)
    dynamic_edge_attr = adj_dynamic[rows, cols]

    B, T, N = x.shape
    x_dynamic = x.reshape(-1, N).t()  # (N, B*T)

    dynamic_out = self.dynamic_conv(x_dynamic, dynamic_edge_index, dynamic_edge_attr)

    dynamic_out = dynamic_out.view(N, B, T, self.gcn_out).permute(1, 2, 0, 3)
    return dynamic_out  # (B, T, N, gcn_out)



class encoderNet(nn.Module):
    def __init__(self, m, n, b, ALPHA = 1, batch=128):
        super(encoderNet, self).__init__()
        self.N = m * n
        self.tanh = nn.LeakyReLU()
        self.gcn_out = 8  #gcn
        self.dim = 512  #512
        self.batch = batch
        self.l = 24
        self.dropout = 0.2
        # self.conv = GATConv(1, self.gcn_out)
        self.conv = ChebConv(self.batch*self.l,self.gcn_out,K=1)
        # self.dropout = nn.Dropout(self.dropout)
        self.fc1 = nn.Linear(self.N, self.dim*ALPHA)
        self.fc2 = nn.Linear(self.dim*ALPHA, self.dim*self.l)
        self.fc3 = nn.Linear(self.dim*ALPHA, b)
        self.bn = nn.BatchNorm1d(self.N,momentum=0.5)
        self.fc_out = nn.Linear(self.dim, m)

        # self.dynamic_graph_conv=dynamic_graph_conv()
        # add
        self.fusion = nn.Linear(2 * self.gcn_out, self.gcn_out)
        self.norm = nn.LayerNorm([self.N, self.gcn_out])

        self.feature_compressor = nn.Sequential(
            nn.Linear(2 * self.gcn_out, 32),  # 16 -> 32  2->1
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)  #
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def build_dynamic_adjacency(self, x):

        B, T, N = x.shape

        node_features = x.permute(2, 0, 1)  # (N, B, T)
        node_features = node_features.reshape(N, -1)  # (N, B*T)


        eps = 1e-8
        norms = torch.norm(node_features, dim=1, keepdim=True) + eps
        normalized_features = node_features / norms

        sim_matrix = torch.mm(normalized_features, normalized_features.t())  # (N, N)
        sim_matrix = F.relu(sim_matrix)
        # b.
        sim_matrix = sim_matrix.fill_diagonal_(1.0)

        degree = sim_matrix.sum(dim=1)  #
        degree_inv_sqrt = torch.diag(1.0 / torch.sqrt(degree + eps))
        adj_dynamic = torch.mm(torch.mm(degree_inv_sqrt, sim_matrix), degree_inv_sqrt)

        return adj_dynamic

    def dynamic_graph_conv(self, x):
        adj_dynamic = self.build_dynamic_adjacency(x)  # (N, N)

        rows, cols = torch.nonzero(adj_dynamic, as_tuple=True)
        dynamic_edge_index = torch.stack([rows, cols], dim=0)
        dynamic_edge_attr = adj_dynamic[rows, cols]

        B, T, N = x.shape
        x_dynamic = x.reshape(-1, N).t()  # (N, B*T)

        dynamic_out = self.conv(x_dynamic, dynamic_edge_index, dynamic_edge_attr)

        # dynamic_out = dynamic_out.view(N, B, T, self.gcn_out).permute(1, 2, 0, 3)
        return dynamic_out  # (B, T, N, gcn_out)

    def forward(self, x, edge_index, edge_attr):
        batchsize = x.size()[0]  # 65
        l = x.size()[1]
        # x = x.view(batchsize,self.N,-1)
        x_gcn = x.contiguous()
        x_gcn = x_gcn.view(-1, self.N).t()
        # print("x_gcn---", x_gcn.shape)   #7865,24     #121,1536
        edge_attr = edge_attr.to(torch.float32)

        static_out = self.conv(x_gcn, edge_index, edge_attr)
        static_out = static_out.unsqueeze(0).unsqueeze(0)
        static_out = static_out.repeat(batchsize, l, 1, 1)
        dynamic_out = self.dynamic_graph_conv(x)  # [B, T, gcn_out]
        # print("dynamic out", dynamic_out.shape)   #121 256
        dynamic_out = dynamic_out.unsqueeze(0).unsqueeze(0)  #  [1, 1, 121, 256]
        dynamic_out = dynamic_out.repeat(batchsize, l, 1, 1)  #  [B, T, 121, 256]

        combined = torch.cat([static_out, dynamic_out], dim=-1)  # [B, T, N, 2*gcn_out]
        # print("combined", combined.shape)  # 256,12,121,16


        output = self.feature_compressor(combined)  # [B, T, N, 1]
        # output = self.feature_compressor(static_out)  # [B, T, N, 1]
        output = output.squeeze(-1)

        return output


class GraphMambaKo(nn.Module):
    def __init__(self, m, n, b, steps, steps_back, seq_len, d_model, state_size, device, alpha = 1, init_scale=1, batch=32):
        ''''
        m=121(num station)
        n=1  (feature dimension)
        '''
        super(GraphMambaKo, self).__init__()   #The super() function is a method used to call the parent class (superclass)
        self.steps = steps
        self.steps_back = steps_back
        # encodernet
        self.encoder = encoderNet(m, n, b, ALPHA = alpha, batch=batch)

        self.dynamics = mamba_insert(seq_len, d_model, state_size, device, batch)
        self.fc_out = nn.Linear(d_model, m, device=device)

        self.adapter = nn.Sequential(
            nn.Linear(121, d_model),
            nn.LayerNorm(d_model)
        )

    def predict_chunk(self, q, h_state, steps):
        chunk = []
        for _ in range(steps):
            q, h_state = self.dynamics(q, h_state)
            q = self.fc_out(q)
            chunk.append(q)
        return torch.cat(chunk, dim=1), h_state

    def forward(self, x, edge_index, edge_attr, mode='forward'):
        out_back = []
        z = self.encoder(x.contiguous(), edge_index, edge_attr)
        z = self.adapter(z)  # [B, T, d_model]

        h_state = None
        out = []

        with torch.no_grad():
            _, h_state = self.dynamics(z, None)

        if self.steps <= 10:
            for _ in range(self.steps):
                z, h_state = self.dynamics(z, h_state)
                pred = self.fc_out(z)
                out.append(pred)
                z = pred
        else:
            chunks = 3
            chunk_size = self.steps // chunks
            for i in range(chunks):
                steps = min(chunk_size, self.steps - i * chunk_size)
                chunk_pred, h_state = self.predict_chunk(z, h_state, steps)
                out.append(chunk_pred)
                z = chunk_pred[:, -1:]
        return out, out_back
