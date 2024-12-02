import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class GeolocationalEncoding(nn.Module):

    def __init__(self, d_model, device='cuda', constant=100, lmd_init=10000.):
        super(GeolocationalEncoding, self).__init__()
        self.constant = constant
        self.d_model = d_model                
        self.k_even = torch.arange(0, d_model, 2).to(device)
        self.k_odd = torch.arange(1, d_model, 2).to(device)
        self.lmd = nn.Parameter(torch.tensor(lmd_init))
        self.device = device

    def w(self, k):
        return torch.pow(self.constant, k/self.d_model)

    def v(self, k):
        return torch.pow(self.constant, (self.d_model-k-1)/self.d_model)

    def forward(self, x, lon, lat):

        pe = torch.zeros(lon.shape[0], lon.shape[1], lon.shape[2], self.d_model).to(self.device)
        pe[:, :, :, 0::2] = torch.sin((lon.unsqueeze(-1)/self.w(self.k_even))) + torch.sin((lat.unsqueeze(-1)/self.v(self.k_even)))
        pe[:, :, :, 1::2] = torch.cos((lon.unsqueeze(-1)/self.w(self.k_odd))) + torch.cos((lat.unsqueeze(-1)/self.v(self.k_odd)))
        x += torch.mul(torch.div(pe, 2), self.lmd)
        
        return x


class SpatioformerModel(nn.Module):

    def __init__(self, device='cuda', nbands=6, dim_out=1, patchsize=9, d_model=16, nhead=8, dim_feedforward=1024, nlayers=3, dropout=0.1, if_encode=True, nodes_hidden=1024):
        super(SpatioformerModel, self).__init__()
        self.geolocation_encoder = GeolocationalEncoding(d_model=d_model, device=device)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Linear(nbands, d_model)
        self.feedforward = nn.Sequential(nn.Linear((patchsize*patchsize+1)*d_model, nodes_hidden), nn.Linear(nodes_hidden, dim_out))
        self.patchsize = patchsize
        self.d_model = torch.tensor(d_model)
        self.nbands = nbands
        self.if_encode = if_encode
        self.independent = nn.Parameter(torch.ones((1, d_model), dtype=torch.float32))

    def forward(self, src, lon, lat):

        src = self.encoder(src.reshape(-1, self.patchsize*self.patchsize, self.nbands)) * torch.sqrt(self.d_model)
        if self.if_encode:
            src = self.geolocation_encoder(src.view(-1, self.patchsize, self.patchsize, self.d_model), lon, lat)
        src = torch.cat((self.independent.unsqueeze(0).expand(src.shape[0], -1, -1),
                         src.view(-1, self.patchsize*self.patchsize, self.d_model)), 1)
        src = self.transformer_encoder(src)
        output = self.feedforward(src.view(-1, (self.patchsize*self.patchsize+1)*self.d_model))

        return output
