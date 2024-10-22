import torch
import torch.nn as nn
import torch.fft
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        self.mean = None
        self.stdev = None
        self.last = None
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Backbone(nn.Module):
    def __init__(self, configs):
        super(Backbone, self).__init__()

        self.seq_len = seq_len = configs.seq_len
        self.pred_len = pred_len = configs.pred_len

        # Patching
        self.patch_len = patch_len = configs.patch_len  # 16
        self.stride = stride = configs.stride  # 8
        self.patch_num = patch_num = int((seq_len - patch_len) / stride + 1)
        self.padding_patch = configs.padding_patch
        if configs.padding_patch == 'end':  # can be modified to general case
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
            self.patch_num = patch_num = patch_num + 1

        d_model = patch_len * patch_len
        self.embed = nn.Linear(patch_len, d_model)
        self.dropout_embed = nn.Dropout(0.3)

        self.lin_res = nn.Linear(patch_num * d_model, pred_len)
        self.dropout_res = nn.Dropout(0.3)

        self.Local_Relational_Block = Local_Relational_Block(d_model=d_model, patch_num=patch_num, patch_len=patch_len)

        self.block = nn.ModuleList(
            [GLRBlock(configs=configs, patch_num=patch_num, patch_len=patch_len, d_model=d_model, dim=32,
                      num_heads=configs.n_heads, mlp_ratio=8, norm_layer=nn.LayerNorm)
             for i in range(1)])
        self.mlp = Mlp(patch_len * patch_num, pred_len * 2, pred_len)


    def forward(self, x):  # B, L, D -> B, H, D
        B, _, D = x.shape
        L = self.patch_num
        P = self.patch_len

        if self.padding_patch == 'end':
            z = self.padding_patch_layer(x.permute(0, 2, 1))  # B, L, D -> B, D, L -> B, D, L
        z = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)  # B, D, L, P
        z = z.reshape(B * D, L, P, 1).squeeze(-1)
        z = self.embed(z)  # B * D, L, P -> # B * D, L, d
        z = self.dropout_embed(z)

        z_res = self.lin_res(z.reshape(B, D, -1))  # B * D, L, d -> B, D, L * d -> B, D, H
        z_res = self.dropout_res(z_res)
        z = self.Local_Relational_Block(z)
        for i, blk in enumerate(self.block):
            z = blk(z)

        z_point = z.reshape(B, D, -1)  # B * D, L, P -> B, D, L * P
        z_mlp = self.mlp(z_point)  # B, D, L * P -> B, D, H

        return (z_res + z_mlp).permute(0, 2, 1)


class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super(CausalConv1d, self).__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=self.padding, dilation=dilation)

    def forward(self, x):
        x = self.conv(x)
        if self.padding != 0:
            x = x[:, :, :-self.padding]  # Remove the padded values
        return x


class GLRBlock(nn.Module):
    """
    Global Local Relational Block
    """

    def __init__(self, configs, patch_num, patch_len, d_model, dim, num_heads, mlp_ratio=4., drop=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(patch_num)
        self.Global_Relational_Block = Global_Relational_Block(configs=configs, patch_num=patch_num,
                                                               patch_len=patch_len, dim=dim, num_heads=num_heads)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            fan_out = m.kernel_size[0] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):

        x = x + self.Global_Relational_Block(self.norm1(x.permute(0, 2, 1)))

        return x


class Local_Relational_Block(nn.Module):
    def __init__(self, d_model, patch_num, patch_len):
        super().__init__()
        self.depth_res = nn.Linear(d_model, patch_len)

        self.depth_conv = nn.Conv1d(patch_num, patch_num, kernel_size=8, stride=patch_len, groups=patch_num)
        self.depth_activation = nn.GELU()
        self.depth_norm = nn.BatchNorm1d(patch_num)

        self.point_conv = nn.Conv1d(patch_num, patch_num, kernel_size=1, stride=1)

    def forward(self, x):
        res = self.depth_res(x)

        z_depth = self.depth_conv(x)
        z_depth = self.depth_activation(z_depth)
        z_depth = self.depth_norm(z_depth)
        z_depth = z_depth + res
        z_point = self.point_conv(z_depth)

        return z_point


class Global_Relational_Block(nn.Module):
    def __init__(self, configs, patch_num, patch_len, dim, num_heads=8):
        super().__init__()

        assert patch_num % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        # self.dim = dim
        self.num_heads = num_heads
        head_dim = patch_num // num_heads
        self.scale = None or head_dim ** -0.5
        self.q = nn.Linear(patch_num, patch_num)
        self.kv = nn.Linear(patch_num, patch_num * 2)
        self.proj = nn.Linear(patch_num, patch_num)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            fan_out = m.kernel_size[0] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = x.permute(0, 2, 1)
        return x


class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.rev = RevIN(configs.enc_in)

        self.backbone = Backbone(configs)

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

    def forward(self, x, batch_x_mark, dec_inp, batch_y_mark):
        z = self.rev(x, 'norm')  # B, L, D -> B, L, D
        z = self.backbone(z)  # B, L, D -> B, H, D
        z = self.rev(z, 'denorm')  # B, L, D -> B, H, D
        return z

