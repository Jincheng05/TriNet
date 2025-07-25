import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as nnf
from models.Dynamic_Conv import DynamicConv3D
import models.configs_TriNet as configs


class SpatialTransformer(nn.Module):
    """
    Spatial Transformer
    https://github.com/voxelmorph/voxelmorph
    """
    def __init__(self, size, mode='bilinear'):
        super().__init__()
        self.mode = mode
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.to(torch.float32)
        self.register_buffer('grid', grid, persistent=False) 

    def forward(self, src, flow):
        new_locs = self.grid + flow
        shape = flow.shape[2:]
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)
        if len(shape) == 3: # 3D
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]
        elif len(shape) == 2: 
             new_locs = new_locs.permute(0, 2, 3, 1)
             new_locs = new_locs[..., [1, 0]]
        return nnf.grid_sample(src, new_locs, align_corners=False, mode=self.mode, padding_mode='border') 


class DyT(nn.Module):
    def __init__(self, dim, init_alpha=0.5):
        super(DyT, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1) * init_alpha)
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        output = self.gamma.view(1, -1, *([1] * (x.ndim - 2))) * x + self.beta.view(1, -1, *([1] * (x.ndim - 2)))
        return output

class ExtractNet(nn.Module):
    def __init__(self, dim, Dyt = False):
        super(ExtractNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(1, 2 * dim, kernel_size=3, stride=1, padding=1),
            nn.Conv3d(2 * dim, 2 * dim, kernel_size=3, stride=2, padding=1)
        )
        self.dyt1 = DyT(2 * dim) if Dyt else nn.Identity()
        self.conv2 = nn.Sequential(
            nn.Conv3d(2 * dim, 4 * dim, kernel_size=3, stride=1, padding=1),
            nn.Conv3d(4 * dim, 4 * dim, kernel_size=3, stride=2, padding=1)
        )
        self.dyt2 = DyT(4 * dim) if Dyt else nn.Identity()
        self.conv3 = nn.Sequential(
            nn.Conv3d(4 * dim, 6 * dim, kernel_size=3, stride=1, padding=1),
            nn.Conv3d(6 * dim, 6 * dim, kernel_size=3, stride=2, padding=1)
        )
        self.dyt3 = DyT(6 * dim) if Dyt else nn.Identity()
        self.conv4 = nn.Sequential(
            nn.Conv3d(6 * dim, 8 * dim, kernel_size=3, stride=1, padding=1),
            nn.Conv3d(8 * dim, 8 * dim, kernel_size=3, stride=2, padding=1)
        )
        self.dyt4 = DyT(8 * dim) if Dyt else nn.Identity()
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        x_2x = self.lrelu(self.dyt1(self.conv1(x)))
        x_4x = self.lrelu(self.dyt2(self.conv2(x_2x)))
        x_8x = self.lrelu(self.dyt3(self.conv3(x_4x)))
        x_16x = self.lrelu(self.dyt4(self.conv4(x_8x)))
        return {'1/2': x_2x, '1/4': x_4x, '1/8': x_8x, '1/16': x_16x}

class SelfAttention3D(nn.Module):
    def __init__(self, in_channels: int, num_heads: int, dim_head: int | None = None, qkv_bias: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.dim_head = dim_head if dim_head is not None else in_channels // num_heads
        self.inner_dim = self.dim_head * self.num_heads
        self.to_qkv = nn.Linear(in_channels, self.inner_dim * 3, bias=qkv_bias)
        self.to_out = nn.Linear(self.inner_dim, in_channels)
        self.scale = self.dim_head ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, d, w, h = x.shape
        x = x.view(b, c, -1).transpose(1, 2)
        N = x.shape[1]
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(b, N, self.num_heads, self.dim_head).transpose(1, 2), qkv)
        dots = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = nnf.softmax(dots, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(b, N, self.inner_dim)
        out = self.to_out(out)
        out = out.transpose(1, 2).view(b, c, d, w, h)
        return out

class Shuffle(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale
    def forward(self, input):
        batch_size, channels, in_depth, in_height, in_width = input.size()
        Cout = channels // self.scale ** 3
        out_depth = in_depth * self.scale
        out_height = in_height * self.scale
        out_width = in_width * self.scale
        input_view = input.reshape(batch_size, Cout, self.scale, self.scale, self.scale, in_depth, in_height, in_width)
        output = input_view.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
        return output.view(batch_size, Cout, out_depth, out_height, out_width)

class MRHU(nn.Module):
    def __init__(self, in_c, up_scale, skip_c = 0):
        super().__init__()
        out_c = in_c * 2
        self.conv1 = nn.Conv3d(in_c + skip_c, out_c, 3, 1, 1, dilation=1)
        self.conv2 = nn.Conv3d(in_c + skip_c, out_c, 3, 1, 2, dilation=2)
        self.conv3 = nn.Conv3d(in_c + skip_c, out_c, 3, 1, 3, dilation=3)
        self.conv4 = nn.Conv3d(in_c + skip_c, out_c, 3, 1, 4, dilation=4)

        self.up = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.trans_conv = nn.ConvTranspose3d(2 * out_c, in_c // 2, 4, 2, 1)
        self.shuffle = Shuffle(up_scale)

        self.fusion = nn.Conv3d(in_c, in_c, 3, 1, 1)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x, skip = None):
        h_up = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x1 = self.act(self.conv1(x))
        x2 = self.act(self.conv2(x))
        x3 = self.act(self.conv3(x))
        x4 = self.act(self.conv4(x))

        x1_ = self.trans_conv(torch.cat([x1, x3], dim=1))
        x2_ = self.shuffle(torch.cat([x2, x4], dim=1))
        # print(f"x1_:{x1_.shape}, x2_:{x2_.shape}")
        x = torch.cat([x1_, x2_], dim=1)
        x = self.fusion(x)
        x = x + h_up
        x = self.fusion(x)

        return x

class Decoder(nn.Module):
    def __init__(self, inputc, hidden_dim, up_scale = 2, skip_dim = 0):
        super().__init__()
        up_dim = (up_scale ** 3) * hidden_dim
        self.convz1 = nn.Conv3d(inputc, hidden_dim, 3, 1, padding=1)
        self.convr1 = nn.Conv3d(inputc, hidden_dim, 3, 1, padding=1)
        self.convq1 = nn.Conv3d(inputc, hidden_dim, 3, 1, padding=1)

        self.mrhu = MRHU(hidden_dim, skip_c=skip_dim, up_scale=up_scale)
        self.dyt = DyT(hidden_dim)

        self.conv1 = nn.Conv3d(hidden_dim, hidden_dim, 3, 1, 1)
        self.conv2 = nn.Conv3d(hidden_dim, 3, kernel_size=3, stride=1, padding=1)
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, h, x, skip = None):
        hx = torch.cat([h, x], dim=1)

        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = self.dyt(self.convq1(torch.cat([r*h, x], dim=1)))

        h = (1-z) * h + z * q

        if skip is None:
            h_merge = self.mrhu(h)
        else:
            h_merge = self.mrhu(h, skip)
            
        flow = self.conv2(self.lrelu(self.conv1(h_merge)))

        return flow, h_merge

class TriNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.if_skip = config.if_skip
        embed_dim = config.embed_dim
        hidden_dim = config.hidden_dim
        inshape = config.inshape
        if_Dyt = config.if_Dyt
        num_heads = config.num_heads
        qkv_bias = config.qkv_bias

        self.extract_moving = ExtractNet(embed_dim, Dyt=if_Dyt)
        self.extract_fixed = ExtractNet(embed_dim, Dyt=if_Dyt)

        if self.if_skip:
            self.decoder_2x = Decoder(2 * 2 * embed_dim + hidden_dim, hidden_dim, skip_dim=2 * embed_dim)
            self.decoder_4x = Decoder(2 * 4 * embed_dim + hidden_dim, hidden_dim, skip_dim=4 * embed_dim)
        else:
            self.decoder_2x = Decoder(2 * 2 * embed_dim + hidden_dim, hidden_dim)
            self.decoder_4x = Decoder(2 * 4 * embed_dim + hidden_dim, hidden_dim)

        self.decoder_8x = Decoder(2 * 6 * embed_dim + hidden_dim, hidden_dim)
        self.decoder_16x = Decoder(2 * 8 * embed_dim + hidden_dim, hidden_dim)

        self.warp = nn.ModuleList()
        for i in range(4):
            current_shape = tuple(s // (2**i) for s in inshape)
            self.warp.append(SpatialTransformer(current_shape))

        self.attention = SelfAttention3D(3 * 6 * embed_dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.down_conv = nn.Conv3d(3 * 6 * embed_dim, 6 * embed_dim, 3, 2, 1)

        self.DC1 = DynamicConv3D(3 * 6 * embed_dim, 6 * embed_dim, kernel_size=3, stride=2, padding=1)
        self.up_dc = nn.Conv3d(6 * embed_dim, 8 * embed_dim, 1, 1)

        self.DC2 = DynamicConv3D(3 * 8 * embed_dim , 8 * embed_dim, kernel_size=3, stride=1, padding=1)

        self.to_hidden_dim = nn.Conv3d(8 * embed_dim, hidden_dim, 1, 1)

        self.dconv1 = nn.Conv3d(2 * 2 * embed_dim, 2 * embed_dim, 3, 1, 1, dilation=1)
        self.dconv2 = nn.Conv3d(2 * embed_dim, 4 * embed_dim, 3, 2, 2, dilation=2)

        self.dconv3 = nn.Conv3d(3 * 4 * embed_dim, 4 * embed_dim, 3, 1, 1, dilation=1)
        self.dconv4 = nn.Conv3d(4 * embed_dim, 6 * embed_dim, 3, 2, 2, dilation=2)

        self.lrelu = nn.LeakyReLU(0.1, inplace=True)

        self.use_checkpointing_internally = True

    def forward(self, f, m):
        # ==========================
        # 1.Encoder 1, 2
        # ==========================
        m_feat = self.extract_moving(m)
        f_feat = self.extract_fixed(f)

        m_2x, m_4x, m_8x, m_16x = m_feat['1/2'], m_feat['1/4'], m_feat['1/8'], m_feat['1/16']
        f_2x, f_4x, f_8x, f_16x = f_feat['1/2'], f_feat['1/4'], f_feat['1/8'], f_feat['1/16']

        # ==========================
        # 2. Encoder 3
        # ==========================
        cat_2x = torch.cat([m_2x, f_2x], dim=1)
        fusion1 = self.lrelu(self.dconv1(cat_2x))
        fusion1_ = self.lrelu(self.dconv2(fusion1))

        cat_4x = torch.cat([fusion1_, m_4x, f_4x], dim=1)
        fusion2 = self.lrelu(self.dconv3(cat_4x))
        fusion2_ = self.lrelu(self.dconv4(fusion2))

        fusion2_cat_8x = torch.cat([fusion2_, m_8x, f_8x], dim=1)

        fusion3_dc = self.DC1(fusion2_cat_8x)
        fusion3_att = self.attention(fusion2_cat_8x)
        fusion3_att_down = self.down_conv(fusion3_att)
        fusion3 = fusion3_dc  + fusion3_att_down
        fusion3_ = self.lrelu(self.up_dc(fusion3))

        fusion3_cat_16x = torch.cat([fusion3_, m_16x, f_16x], dim=1)
        fusion4 = self.DC2(fusion3_cat_16x)
        fusion4_hidden = self.lrelu(self.to_hidden_dim(fusion4))

        # ==========================
        # 3. Decoding Stage
        # ==========================
        hid = fusion4_hidden

        x_16x = torch.cat([m_16x, f_16x], dim=1)
        flow1, hid1 = self.decoder_16x(hid, x_16x)
        m_8x_warped = self.warp[3](m_8x, flow1)

        x_8x = torch.cat([m_8x_warped, f_8x], dim=1)
        flow2, hid2 = self.decoder_8x(hid1, x_8x)
        m_4x_warped = self.warp[2](m_4x, flow2)

        x_4x = torch.cat([m_4x_warped, f_4x], dim=1)
        skip_4x = fusion2 if self.if_skip else None
        flow3, hid3 = self.decoder_4x(hid2, x_4x, skip=skip_4x)
        m_2x_warped = self.warp[1](m_2x, flow3)

        x_2x = torch.cat([m_2x_warped, f_2x], dim=1)
        skip_2x = fusion1 if self.if_skip else None
        flow4, hid4 = self.decoder_2x(hid3, x_2x, skip=skip_2x)
        m_warped = self.warp[0](m, flow4)

        del hid1, hid2, hid3, hid4

        return flow4, m_warped


CONFIGS = {
    'TriNet': configs.get_trinet_config(),
    'TriNet_small':configs.get_trinet_small_config(),
    'TriNet_large':configs.get_trinet_large_config()
}
