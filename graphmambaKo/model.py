from torch import nn
import torch

def gaussian_init_(n_units, std=1):    
    sampler = torch.distributions.Normal(torch.Tensor([0]), torch.Tensor([std/n_units]))
    Omega = sampler.sample((n_units, n_units))[..., 0]  
    return Omega

#输入张量 x 的维度: (64, 1, 64, 1)。 简单的全连接神经网络，用于将输入数据编码为一个低维表示
#经过 view 操作后，维度变为 (64, 1, 64)。
#经过第一层全连接层 self.fc1，维度变为 (64, 1, 16)。
#经过第二层全连接层 self.fc2，维度仍为 (64, 1, 16)。
#经过第三层全连接层 self.fc3，最终输出维度为 (64, 1, 8)。
class encoderNet(nn.Module):
    def __init__(self, m, n, b, ALPHA = 1):
        super(encoderNet, self).__init__()
        self.N = m * n
        self.tanh = nn.Tanh()

        self.fc1 = nn.Linear(self.N, 16*ALPHA)
        self.fc2 = nn.Linear(16*ALPHA, 16*ALPHA)
        self.fc3 = nn.Linear(16*ALPHA, b)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)          

    def forward(self, x):
        x = x.view(-1, 1, self.N)
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x))        
        x = self.fc3(x)
        
        return x

#将低维表示还原为原始数据的形状,64,1,64,1
#encoder和decoder通常连起来使用，自编码器（Autoencoder），用于学习数据的有效低维表示，并能够从这些表示中重建原始数据。
class decoderNet(nn.Module):
    def __init__(self, m, n, b, ALPHA = 1):
        super(decoderNet, self).__init__()

        self.m = m
        self.n = n
        self.b = b

        self.tanh = nn.Tanh()

        self.fc1 = nn.Linear(b, 16*ALPHA)
        self.fc2 = nn.Linear(16*ALPHA, 16*ALPHA)
        self.fc3 = nn.Linear(16*ALPHA, m*n)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)          

    def forward(self, x):
        x = x.view(-1, 1, self.b)
        x = self.tanh(self.fc1(x)) 
        x = self.tanh(self.fc2(x)) 
        x = self.tanh(self.fc3(x))
        x = x.view(-1, 1, self.m, self.n)
        return x


#线性变换模块，用于对输入数据进行动态变换,目的是：
# 生成正交矩阵：通过奇异值分解（SVD）将权重矩阵转换为正交矩阵。
#控制矩阵的谱范数：通过缩放因子 init_scale 调整正交矩阵的大小，从而控制矩阵的谱范数（最大奇异值）。
#改善模型的训练稳定性和性能：正交初始化有助于避免梯度消失或爆炸问题，同时缩放因子可以进一步调整模型的动态范围。
class dynamics(nn.Module):
    def __init__(self, b, init_scale):
        super(dynamics, self).__init__()
        self.dynamics = nn.Linear(b, b, bias=False)
        self.dynamics.weight.data = gaussian_init_(b, std=1)           
        U, _, V = torch.svd(self.dynamics.weight.data)
        self.dynamics.weight.data = torch.mm(U, V.t()) * init_scale

        
    def forward(self, x):
        x = self.dynamics(x)
        return x

#通过计算 omega 模型权重矩阵的伪逆来实现逆动态变换。这个模型的输入和输出维度均为 (batch_size, 1, b)
#逆动态变换通常用于以下场景：
#逆问题求解：在某些物理系统或动态系统中，已知系统的输出，需要求解系统的输入。逆动态变换可以帮助实现这一目标。
#数据恢复：在某些情况下，数据经过某种动态变换后被压缩或编码，逆动态变换可以用于恢复原始数据
class dynamics_back(nn.Module):
    def __init__(self, b, omega):
        super(dynamics_back, self).__init__()
        self.dynamics = nn.Linear(b, b, bias=False)
        self.dynamics.weight.data = torch.pinverse(omega.dynamics.weight.data.t())     

    def forward(self, x):
        x = self.dynamics(x)
        return x




class koopmanAE(nn.Module):
    def __init__(self, m, n, b, steps, steps_back, alpha = 1, init_scale=1):
        super(koopmanAE, self).__init__()
        self.steps = steps
        self.steps_back = steps_back
        
        self.encoder = encoderNet(m, n, b, ALPHA = alpha)
        self.dynamics = dynamics(b, init_scale)
        self.backdynamics = dynamics_back(b, self.dynamics)
        self.decoder = decoderNet(m, n, b, ALPHA = alpha)


    def forward(self, x, mode='forward'):
        out = []
        out_back = []
        z = self.encoder(x.contiguous())   #x:64,1,64,1,z:64,1,8
        q = z.contiguous()

        
        if mode == 'forward':
            for _ in range(self.steps):
                q = self.dynamics(q)
                out.append(self.decoder(q))

            out.append(self.decoder(z.contiguous())) 
            return out, out_back    

        if mode == 'backward':
            for _ in range(self.steps_back):
                q = self.backdynamics(q)
                out_back.append(self.decoder(q))
                
            out_back.append(self.decoder(z.contiguous()))
            return out, out_back
