from numpy import sqrt
import torch
from torch import nn
from einops import einsum

class LinearModule(nn.Module):
    def __init__(self,in_feature, out_feature,device=None,dtype=None) -> None:
        super().__init__()
        d = 2 /(in_feature + out_feature)
        d_sqrt = sqrt(d)
        trunc = 3 * d_sqrt
        w = torch.empty(out_feature,in_feature,device=device,dtype=dtype)
        self.W = nn.Parameter(torch.nn.init.trunc_normal_(w,mean=0.0,a = -trunc,b = trunc))


    def forward(self,x:torch.Tensor):
        return einsum(self.W,x,"dout din, ... din -> ... dout",)
