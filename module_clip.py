from collections import OrderedDict
from typing import Tuple, Union
from mmcv.cnn import build_activation_layer, build_norm_layer

import hashlib
import os
import urllib
import warnings
from mmengine.model import BaseModule
from PIL.ImageOps import scale
from cv2 import dilate
from iopath.common.download import download
from sympy.tensor.tensor import tensor_heads
from torchgen.native_function_generation import self_to_out_signature
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from timm.layers import DropPath, trunc_normal_
from einops import rearrange

from util import get_logger, parallel_apply

_MODELS = {
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
}
_PT_NAME = {
    "ViT-B/32": "ViT-B-32.pt",
}

def _download(url: str, root: str = os.path.expanduser("~/.cache/clip")):
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)

    expected_sha256 = url.split("/")[-2]
    download_target = os.path.join(root, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        if hashlib.sha256(open(download_target, "rb").read()).hexdigest() == expected_sha256:
            return download_target
        else:
            warnings.warn(f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file")

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    if hashlib.sha256(open(download_target, "rb").read()).hexdigest() != expected_sha256:
        raise RuntimeError(f"Model has been downloaded but the SHA256 checksum does not not match")

    return download_target

def available_models():
    """Returns the names of available CLIP models"""
    return list(_MODELS.keys())

# =============================

class Adapter(nn.Module):
    def __init__(self, D_features=768, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        self.act = act_layer()
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)

    def forward(self, x):
        # x is (BT, HW+1, D)
        xs = self.D_fc1(x)
        xs = self.act(xs)
        xs = self.D_fc2(xs)
        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x

class adapter(nn.Module):
    def __init__(self, feature = 768 ,mlp_ratio=0.25, act_layer=nn.GELU):
        super().__init__()
        down = int(feature * mlp_ratio)
        self.act = act_layer()
        self.relu = nn.ReLU()
        self.D_fc1 = nn.Linear(feature, down)
        self.D_fc2 = nn.Linear(down, feature)

        self.fc = nn.Sequential(
            nn.Linear(feature, down, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(down, feature, bias=False),
            nn.ReLU(inplace=True)
    )
    def forward(self, x):
        x = self.fc(x)

        return x


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )

        return x[0]


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class MultiScaleConv(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        # self.conv3 = nn.Conv1d(d_model, width//4, kernel_size=3, padding='same')
        # self.conv5 = nn.Conv1d(d_model, width//4, kernel_size=5, padding='same')
        # self.conv7 = nn.Conv1d(d_model, width//2, kernel_size=7, padding='same')

        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding='same', groups=in_channels)# 深度可分离卷积
        self.conv5 = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding='same', groups=in_channels)
        self.conv7 = nn.Conv2d(in_channels, in_channels, kernel_size=7, padding='same', groups=in_channels)
        # 1x1卷积融合多尺度特征
        self.fusion = nn.Conv2d(3 * in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        """
        输入: x [B, C, H, W] (e.g. [128,3,224,224])
        输出: [B, C, H, W] (维度不变)
        """
        x3 = self.conv3(x)
        x5 = self.conv5(x)
        x7 = self.conv7(x)
        # x_out = torch.cat([x3, x5, x7], dim=1)  # [B, width, T]
        # 拼接多尺度特征
        combined = torch.cat([x3, x5, x7], dim=1)  # [B, 3D, T]
        # 特征融合降维
        out = self.fusion(combined)  # [B, D, T]
        return out

class ParallelSTConv3D(nn.Module):
    def __init__(self,d_model ,width):
        super().__init__()

        self.temporal_conv = nn.Conv1d(in_channels=50, out_channels=width, kernel_size=3,
            padding='same', groups = 1 )
        # 深度可分离空间卷积（模拟局部关系）
        self.spatial_conv = nn.Conv1d( in_channels=d_model, out_channels=width, kernel_size=1,  # 等价于线性变换
            groups = 1 )

    def forward(self, x):
        """
        输入: x [batch_size, seq_len, d_model]    # 时间卷积分支
                128  768  50
        输出: [batch_size, seq_len, d_model]      # 维度转换: [B, T, D] -> [B, D, T] (Conv1d要求通道在前)
        """
        x_t = x.permute(0, 2, 1)
        x_t = self.temporal_conv(x_t)  # [B, D, T]
        x_t = x_t.permute(0, 2, 1)  # [B, T, D]

        # 空间卷积分支（实际为逐点线性变换）
        x_s = x.permute(0, 2, 1)  # [B, D, T]
        x_s = self.spatial_conv(x_s)
        x_s = x_s.permute(0, 2, 1)  # [B, T, D]

        # 融合并残差连接
        return  0.5 * (x_t + x_s)

class PoolingMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, pool_ratio=4, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads          #   8
        self.head_dim = embed_dim // num_heads
        self.pool_ratio = pool_ratio
        self.dropout = dropout
        # 线性变换层（与标准多头注意力一致）
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim // pool_ratio, embed_dim)
        # 缩放因子
        self.scale = self.head_dim ** -0.5
        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)  # Xavier 初始化
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)  # 偏置初始化为 0

    def forward(  self,
            x: torch.Tensor,  # 输入张量（与原始CLIP的attention(x)调用方式一致）
            key: torch.Tensor = None,
            value: torch.Tensor = None,
            need_weights: bool = False,
            attn_mask: torch.Tensor = None,  # 新增attn_mask参数
  ):
        """
        输入格式对齐原始CLIP的attention模块：  - 默认自注意力模式：key/value = x   - 兼容交叉注意力调用：显式传递key/value
        """
      # 自注意力模式兼容
        if key is None:
            key = x
        if value is None:
            value = x

        B, T, C = x.shape       #[50,128,768]
        # 1. 线性投影 + 多头分割
        # q = self.q_proj(x).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, T, D/H]
        q = self.q_proj(key).view(B, -1, self.num_heads, self.head_dim).permute(0, 2, 3, 1)  # [B, H, D/H, T]
        k = self.k_proj(key).view(B, -1, self.num_heads, self.head_dim).permute(0, 2, 3, 1)  # [B, H, D/H, T]
        v = self.v_proj(value).view(B, -1, self.num_heads, self.head_dim).permute(0, 2, 3, 1)  # [B, H, D/H, T] [50,128,12,64]
        # print(q.shape,k.shape,v.shape,"qkv")#[50,12,64,128]
        # 2. 动态调整池化参数
        def apply_pool(tensor: torch.Tensor):
            """输入形状: [B, H, D/H, T]"""
            B, H, D_h, T = tensor.shape #torch.Size([50, 12, 64, 128])
            # print(tensor.shape,"tensor.shape")
            tensor = tensor.permute(3, 1, 2, 0)  # [128, 12, 64, 50]
            tensor = tensor.reshape(128, 12, 50, 8, 8)  # [128, 12, 50, 8, 8]
            # 定义 3D 池化层（核大小为 2，步长为 2）
            pooling = nn.AvgPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

            output_data = pooling(tensor)  # [128, 12, 64, 25]
            # print(output_data.shape,"out")  #torch.Size([128, 12, 25, 4, 4])
            # 还原为原始格式 [25, 12, 16, 128]
            output_data = output_data.reshape(128, 12, 25, 16)  # [128, 12, 25, 16]
            output_data = output_data.permute(2, 1, 3, 0)  # [25, 12, 16, 128]
            # print(output_data.shape, "out1")    #torch.Size([25, 12, 16, 128])
            return output_data

        k_pooled = apply_pool(k)  # [B, H, D/H, T_pooled]
        v_pooled = apply_pool(v)
        q_pooled = apply_pool(q)
        k_pooled = k_pooled.transpose(-2, -1)  # 将最后两个维度转置
        # print(q_pooled.shape, k_pooled.shape, v_pooled.shape, "qkv")
        #([25, 12, 16, 128]) ([25, 12, 16, 128]) ([25, 12, 16, 128]) qkv
        # 3. 计算注意力分数
        attn = (q_pooled @ k_pooled) * self.scale  # [B, H, T, T//pool_ratio]
        # 4. 处理attn_mask（对齐原始CLIP逻辑）
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn.masked_fill_(attn_mask, float('-inf'))
            else:
                attn += attn_mask

        attn = F.softmax(attn, dim=-1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)
        # 5. 聚合Value并合并多头
        output = (attn @ v_pooled)  # [B, H, T, D/H]
        output = output + q_pooled  #[25,12,16,128]
        # print(output.shape,"1111")
        output = output.permute(3, 1, 0, 2).unsqueeze(-1)  # [128, 12, 25, 16, 1]
        # 使用三线性插值将时间维度从 25 恢复到 50
        output = F.interpolate(output, size=(50, 16, 1), mode='trilinear')  # [128, 12, 50, 16, 1]
        # 恢复原始格式 [50, 12, 16, 128]
        output = output.squeeze(-1).permute(2, 1, 3, 0)  # [50, 12, 16, 128]
        # 合并注意力头和每个头的维度
        output = output.permute(0, 3, 1, 2)  # [50, 128, 12, 16]
        output = output.reshape(50, 128, -1)  # [50, 128, 192]
        # print(output.shape,"2222")
        # 6. 最终投影
        output = self.out_proj(output)

        return output  # 保持与原始CLIP一致的返回格式（output, None）


class FusionGate(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(3 * d_model, 2 * d_model),
            nn.GELU(),
            nn.Linear(2 * d_model, 2),
            nn.Softmax(dim=-1)
        )

    def forward(self, x, xt, xs):
        gate_weights = self.gate(torch.cat([x, xt, xs], dim=-1))  # [n, bt, 2]
        return x + gate_weights[..., 0:1] * xt + gate_weights[..., 1:2] * xs

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, a:bool, attn_mask=None, scale=0.3,drop_path=0.,dropout=0.3):
        super().__init__()
        # self.att = ParallelSTConv3D(d_model, n_head)
        # self.att = PoolingMultiheadAttention(768, n_head)
        self.dropout = nn.Dropout(dropout)
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        self.MLP_Adapter = Adapter(d_model, skip_connect=False)
        self.S_Adapter = Adapter(d_model)
        self.T_Adapter = Adapter(d_model, skip_connect=False)
        self.n_head = n_head
        self.a=a
        self.scale = scale
        self.fusion_gate = FusionGate(d_model)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()


    def attention(self, x: torch.Tensor):
        attn_mask_ = self.attn_mask
        if self.attn_mask is not None and hasattr(self.attn_mask, '__call__'):
            attn_mask_ = self.attn_mask(x.size(0))   # LND
            # print(attn_mask_,"attn_mask_")
        # if not isinstance(attn_mask_, torch.Tensor):
        #     attn_mask_ = None
            # 处理 attn_mask_ 不是张量的情况

        attn_mask_ = attn_mask_.to(dtype=x.dtype, device=x.device) if attn_mask_ is not None else None

        return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask_)[0]


    def forward(self, x_tuple: tuple):
        x, video_frame = x_tuple
        # print("x.shape",x.shape)  #文本尺寸 [32, 32, 512]  视觉尺寸 [50, 256, 768]
        l,n,d = x.shape#[50,256,768]
        m = int(n/ video_frame)
        v = int(video_frame)
        if self.a:
            # print(x.shape,"x")
            #[8,50*32,768]
            #时间
            xt = x.reshape(l, v, m, d)#[50,8,32,768]
            xt = xt.permute(1, 0, 2, 3)  # [T, L, N', D] = [8, 50, 32, 768]
            xt = xt.reshape(v, l * m, d)  # [T, L*N', D] = [8, 1600, 768]
            xt = self.attention(self.ln_1(xt))
            xt = xt.reshape(v, l, m, d)
            xt = xt.permute(1, 0, 2, 3)
            xt = xt.reshape(l, v * m, d)

            # x = x + self.drop_path(xt)
            xt = self.drop_path(xt)
            #空间
            # x = x + self.S_Adapter(self.attention(self.ln_1(x)))
            xs = self.attention(self.ln_1(x))
            xs = self.drop_path(xs)

            # x = self.fusion_gate(x,xt,xs)
            x = x  + xs + 0.2 * xt

            ## joint adaptation
            xn = self.ln_2(x)
            # x = x + self.mlp(xn) + self.drop_path(self.scale * self.MLP_Adapter(xn))
            x = x + self.mlp(xn)
        else:
            x = x +self.attention(self.ln_1(x))
            xn = self.ln_2(x)
            x = x + self.mlp(xn)

        return (x, video_frame)

class Transformer(nn.Module):
    def __init__(self,  width: int, layers: int, heads: int, a , attn_mask: torch.Tensor = None,
                 scale=1, drop_path=0.1):
        super().__init__()
        self.width = width
        self.layers = layers
        self.a = a
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, a, attn_mask )
                                         for _ in range(layers)])

    def forward(self, x: torch.Tensor, video_frame=-1):
        return self.resblocks((x, video_frame))[0]

class VisualTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int,
                   a , drop_path_rate,  adapter_scale=0.5, linear_patch: str = '2d',):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        # self.malt = MultiScaleConv(in_channels=3)
        self.a = a
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)


        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads, a=True,
                                       drop_path=drop_path_rate, scale=adapter_scale)


        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))
        # self.parallel = ParallelSTConv3D(in_channels=width)#111

        # For 3D
        assert linear_patch in ['2d', '3d']
        self.linear_patch = linear_patch
        if self.linear_patch == '3d':
            self.conv2 = nn.Conv3d(in_channels=3, out_channels=width, kernel_size=(3, patch_size, patch_size),
                                   stride=(1, patch_size, patch_size), padding=(1, 0, 0), bias=False)

    def forward(self, x: torch.Tensor, video_frame=-1):
        # print(x.shape) #[(b*p)*(bs*ts),3,224,224][256,3,224,224]

        if self.linear_patch == '3d':
            assert video_frame != -1
            x_3d = x.reshape(-1, video_frame, x.shape[-3], x.shape[-2], x.shape[-1])
            x_3d = x_3d.permute(0, 2, 1, 3, 4)
            x_3d = self.conv2(x_3d)     # shape = [*, width, frame, grid, grid]
            x_3d = x_3d.permute(0, 2, 1, 3, 4)      # shape = [*, frame, width, grid, grid]
            x = x_3d.reshape(-1, x_3d.shape[-3], x_3d.shape[-2], x_3d.shape[-1]).contiguous() # shape = [*, width, grid, grid]
        else:
            x = self.conv1(x)  # shape = [*, width, grid, grid]----[256,768,7,7]
            # print(x.shape,"x")
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) +
                       torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        # print(x.shape,"x")#[256,50,768]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, video_frame=video_frame)
        x = x.permute(1, 0, 2)  # LND -> NLD
        #[50,128,768]
        return x


class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 drop_path_rate: float,
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 # vision linear of patch
                 linear_patch: str = '2d',
                 ):
        super().__init__()

        self.context_length = context_length

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisualTransformer(
                drop_path_rate =drop_path_rate,
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim,
                a = True,
                linear_patch=linear_patch,
            )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            a=False,
            attn_mask=self.build_attention_mask
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]))
        # 添加 VideoSpecificPrompt 模块
        # self.video_specific_prompt = VideoSpecificPrompt(layers=prompt_layers, embed_dim=embed_dim, alpha=prompt_alpha, )
        # self.t = nn.Parameter(torch.randn(1))
        # self.b = nn.Parameter(torch.randn(1))

        # self.adapter = adapter()
        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5

        # 确认self.transformer的实际类型
        # print(type(self.transformer))
        # for n, m in self.transformer.named_modules():
        #     if 'adapter' in n:
        #         for n2, m2 in m.named_modules():
        #             if 'D_fc2' in n2:
        #                 if isinstance(m2, nn.Linear):
        #                     nn.init.constant_(m2.weight, 0)
        #                     nn.init.constant_(m2.bias, 0)

        for n, m in self.transformer.named_modules():
            if 'S_Adapter' in n:
                for n2, m2 in m.named_modules():
                    if 'D_fc2' in n2:
                        if isinstance(m2, nn.Linear):
                            nn.init.constant_(m2.weight, 0)
                            nn.init.constant_(m2.bias, 0)

            ## initialize T_Adapter
        for n, m in self.transformer.named_modules():
            if 'T_Adapter' in n:
                for n2, m2 in m.named_modules():
                    if 'D_fc2' in n2:
                        if isinstance(m2, nn.Linear):
                            nn.init.constant_(m2.weight, 0)
                            nn.init.constant_(m2.bias, 0)

            ## initialize MLP_Adapter
        for n, m in self.transformer.named_modules():
            if 'MLP_Adapter' in n:
                for n2, m2 in m.named_modules():
                    if 'D_fc2' in n2:
                        if isinstance(m2, nn.Linear):
                            nn.init.constant_(m2.weight, 0)
                            nn.init.constant_(m2.bias, 0)

        for block in self.transformer.resblocks:               #11
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    @staticmethod
    def get_config(pretrained_clip_name="ViT-B/32"):
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ViT-B-32.pt")
        if pretrained_clip_name in _MODELS and pretrained_clip_name in _PT_NAME:
            model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), _PT_NAME[pretrained_clip_name])

        if pretrained_clip_name in ["ViT-B/32", "ViT-B/16"] and os.path.exists(model_path):
            pass
        else:
            if pretrained_clip_name in _MODELS:
                model_path = _download(_MODELS[pretrained_clip_name])
            elif os.path.isfile(pretrained_clip_name):
                model_path = pretrained_clip_name
            else:
                raise RuntimeError(f"Model {pretrained_clip_name} not found; available models = {available_models()}")

        try:
            # loading JIT archive
            model = torch.jit.load(model_path, map_location="cpu").eval()
            state_dict = model.state_dict()
        except RuntimeError:
            state_dict = torch.load(model_path, map_location="cpu")

        return state_dict

    def build_attention_mask(self, context_length):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.zeros(context_length, context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image, return_hidden=False, video_frame=-1):
        hidden = self.visual(image.type(self.dtype), video_frame=video_frame)
        hidden = self.visual.ln_post(hidden) @ self.visual.proj

        x = hidden[:, 0, :]

        if return_hidden:
            return x, hidden
        # print(x.shape,hidden.shape,"encode image")#[256,512]
        return x

    def encode_text(self, text, return_hidden=False):
        #[32,32]
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
        #  [32,32,512]
        pos_emd = self.positional_embedding[:x.size(1), :].type(self.dtype)
        x = x + pos_emd
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        # print(x.shape,"vvvvv")
        hidden = self.ln_final(x).type(self.dtype) @ self.text_projection

        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = hidden[torch.arange(hidden.shape[0]), text.argmax(dim=-1)]

        if return_hidden:
            return x, hidden

        return x            #[32,512]

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # x = self.adapter(image_features)
        #
        # ratio = 0.2
        # image_features = ratio * x + (1 - ratio) * image_features

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()
        # logits_per_text = torch.matmul(text_features, image_features.t()) # * self.t.exp()  + self.b
        # logits_per_image = logits_per_text.t()


        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model(state_dict: dict):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    convert_weights(model)
    model.load_state_dict(state_dict)
    return model.eval()
