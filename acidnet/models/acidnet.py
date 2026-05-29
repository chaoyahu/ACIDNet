import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin

from acidnet.models.hvi_transform import RGB_HVI
from acidnet.models.transformer_utils import LayerNorm
from acidnet.models.lca import *

# =========================================================
# 1. 基础组件 (支持大核卷积动态 Padding)
# =========================================================

class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, kernel_size=7, layer_scale_init_value=1e-6):
        super().__init__()
        # 动态计算 padding，确保任意大核尺寸输入输出分辨率一致
        pad_size = kernel_size // 2 
        self.pad = nn.ReflectionPad2d(pad_size)
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=0, groups=dim) 
        
        self.norm = LayerNorm(dim, data_format="channels_first")
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(4 * dim, dim, kernel_size=1)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((1, dim, 1, 1)), 
                                  requires_grad=True) if layer_scale_init_value > 0 else None

    def forward(self, x):
        input = x
        x = self.pad(x)
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = input + x
        return x

class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1)
        )
    def forward(self, x):
        return x + self.body(x)

# =========================================================
# Context Blocks (安全除法加固)
# =========================================================

class StatContextBlock(nn.Module):
    def __init__(self, dim, reduction=4, use_mean=True, use_std=True):
        super().__init__()
        self.use_mean = use_mean
        self.use_std = use_std
        
        input_dim = 0
        if use_mean: input_dim += dim
        if use_std: input_dim += dim
        
        # 防止极端消融下通道数变为 0
        mid_dim = max(1, dim // reduction) 
        
        if input_dim > 0:
            self.mlp = nn.Sequential(
                nn.Conv2d(input_dim, mid_dim, 1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_dim, dim, 1, bias=False),
                nn.Sigmoid()
            )
        else:
            self.mlp = None

    def forward(self, x):
        if self.mlp is None:
            return x

        stats = []
        if self.use_mean:
            stats.append(F.adaptive_avg_pool2d(x, 1))
        if self.use_std:
            mu = F.adaptive_avg_pool2d(x, 1)
            stats.append(torch.sqrt(F.adaptive_avg_pool2d((x - mu)**2, 1) + 1e-6))
        
        stat = torch.cat(stats, dim=1)
        scale = self.mlp(stat)
        return x * scale

class DualPoolContext(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        mid_dim = max(1, channels // reduction)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels * 2, mid_dim, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_dim, channels, 1, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        x_avg = F.adaptive_avg_pool2d(x, 1)
        x_max = F.adaptive_max_pool2d(x, 1)
        x_cat = torch.cat([x_avg, x_max], dim=1)
        return x * self.mlp(x_cat)

# =========================================================
# 2. Attention Utils (添加头数校验)
# =========================================================

def window_partition(x, window_size):
    B, H, W, C = x.shape
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    
    H_pad, W_pad = x.shape[1], x.shape[2]
    x = x.view(B, H_pad // window_size, window_size, W_pad // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size * window_size, C)
    return windows, H_pad, W_pad

def window_reverse(windows, window_size, H_pad, W_pad, H, W):
    B = int(windows.shape[0] / (H_pad * W_pad / window_size / window_size))
    x = windows.view(B, H_pad // window_size, W_pad // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H_pad, W_pad, -1)
    if H_pad > H or W_pad > W:
        x = x[:, :H, :W, :]
    return x

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        
        # 确保通道数可以被头数整除，防止运行报错
        assert dim % num_heads == 0, f"Channel dimension {dim} must be divisible by num_heads {num_heads}"
        head_dim = dim // num_heads
        
        self.scale = head_dim ** -0.5
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))

        coords = torch.stack(torch.meshgrid([torch.arange(window_size), torch.arange(window_size)], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, q, k, v):
        B_, N, C = q.shape
        q = q.reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = k.reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = v.reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size * self.window_size, self.window_size * self.window_size, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        attn = F.softmax(attn, dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return x

class NoiseRobust_DDA(nn.Module):
    def __init__(self, up_channels, skip_channels, out_channels, in_channels=None, 
                 window_size=8, num_heads=4, 
                 use_guidance=True, use_context=True, 
                 attn_mode='cross', smooth_v=True):
        super().__init__()
        self.window_size = window_size
        if in_channels is None: in_channels = up_channels
        self.use_guidance = use_guidance
        self.use_context = use_context
        self.attn_mode = attn_mode

        self.up_pre_conv = nn.Conv2d(in_channels, up_channels, 3, 1, 1, bias=False)
        self.align_conv = nn.Conv2d(skip_channels, up_channels, 1) if skip_channels != up_channels else nn.Identity()

        if use_guidance:
            self.guidance_map = nn.Sequential(
                nn.Conv2d(up_channels, up_channels, 5, padding=2, groups=up_channels, bias=False),
                nn.Conv2d(up_channels, up_channels, 1),
                nn.Sigmoid()
            )
        else:
            self.guidance_map = None

        self.context_path = DualPoolContext(up_channels) if use_context else nn.Identity()

        self.norm_q = nn.GroupNorm(1, up_channels)
        self.norm_k = nn.GroupNorm(1, up_channels)
        self.norm_v = nn.GroupNorm(1, up_channels)

        self.q_conv = nn.Conv2d(up_channels, up_channels, 1, bias=False)
        self.k_conv = nn.Conv2d(up_channels, up_channels, 1, bias=False)
        self.v_conv = nn.Identity() if smooth_v else nn.Conv2d(up_channels, up_channels, 1, bias=False)

        self.local_attention = WindowAttention(up_channels, window_size, num_heads)
        self.fidelity_proj = nn.Conv2d(up_channels, up_channels, 1, bias=False)

        self.fusion_module = nn.Sequential(
            nn.Conv2d(up_channels * 2, up_channels, 3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(up_channels, out_channels, 3, padding=1, bias=False)
        )

    def forward(self, F_main, F_skip, guidance=None):
        F_up = self.up_pre_conv(F_main)
        F_up = F.interpolate(F_up, size=F_skip.shape[2:], mode='bilinear', align_corners=False)
        F_skip_aligned = self.align_conv(F_skip)

        if self.use_guidance and guidance is not None:
            g_map = self.guidance_map(guidance)
            self.last_g_map = g_map.detach().cpu()
            F_up = F_up * (1 + g_map)
        
        F_fused = F_up + F_skip_aligned
        F_context = self.context_path(F_fused) if self.use_context else torch.zeros_like(F_fused)

        B, C, H, W = F_up.shape
        
        if self.attn_mode == 'self':
            q_src, k_src, v_src = F_fused, F_fused, F_fused
        elif self.attn_mode == 'reverse':
            q_src, k_src, v_src = F_skip_aligned, F_up, F_up
        else:
            q_src, k_src, v_src = F_up, F_skip_aligned, F_skip_aligned

        Q = self.q_conv(self.norm_q(q_src)).permute(0, 2, 3, 1)
        K = self.k_conv(self.norm_k(k_src)).permute(0, 2, 3, 1)
        V = self.v_conv(self.norm_v(v_src)).permute(0, 2, 3, 1)

        Q_win, H_pad, W_pad = window_partition(Q, self.window_size)
        K_win, _, _ = window_partition(K, self.window_size)
        V_win, _, _ = window_partition(V, self.window_size)

        attn_win = self.local_attention(Q_win, K_win, V_win)
        
        F_fidelity = window_reverse(attn_win, self.window_size, H_pad, W_pad, H, W).permute(0, 3, 1, 2)
        F_fidelity = self.fidelity_proj(F_fidelity)

        F_refined = F_context + F_fidelity
        out = self.fusion_module(torch.cat([F_refined, F_skip_aligned], dim=1))
        
        return out, None

class ACIDNet(nn.Module, PyTorchModelHubMixin):
    def __init__(self,
        channels=[36, 36, 72, 144],
        heads=[1, 2, 4, 8],
        norm=False
    ):
        super().__init__()

        cfg = dict(
            i_blk='convnext', hv_blk='resblock', dda_type='robust', 
            ws=8, stats='full', use_guidance=True, use_context=True,
            attn_mode='cross', smooth_v=True, kernel_size=7, scb_k=4
        )
        self.cfg = cfg
        current_heads = heads

        def blk_factory(type_name, dim):
            k_size = cfg.get('kernel_size', 7)
            return ResBlock(dim) if type_name == 'resblock' else ConvNeXtBlock(dim, kernel_size=k_size)

        def dda_factory(up_c, skip_c, out_c, in_c, head_num):
            ws = cfg.get('ws', 8)
            return NoiseRobust_DDA(
                up_channels=up_c, skip_channels=skip_c, out_channels=out_c, in_channels=in_c,
                window_size=ws, num_heads=head_num,
                use_guidance=cfg.get('use_guidance', True),
                use_context=cfg.get('use_context', True),
                attn_mode=cfg.get('attn_mode', 'cross'),
                smooth_v=cfg.get('smooth_v', True)
            )

        [ch1, ch2, ch3, ch4] = channels
        [head1, head2, head3, head4] = current_heads
        
        i_type, hv_type = cfg.get('i_blk', 'convnext'), cfg.get('hv_blk', 'resblock')

        # --- Color Stream (HV) ---
        self.HVE_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(3, ch1, 3, stride=1, padding=0, bias=False)
        )
        self.HVE_block1 = NormDownsample(ch1, ch2, use_norm=norm)
        self.HVE_context1 = blk_factory(hv_type, ch2)
        
        self.HVE_block2 = NormDownsample(ch2, ch3, use_norm=norm)
        self.HVE_context2 = blk_factory(hv_type, ch3)
        
        self.HVE_block3 = NormDownsample(ch3, ch4, use_norm=norm)
        self.HVE_context3 = blk_factory(hv_type, ch4)
        
        # Stat Block Logic (注入 scb_k)
        stat_mode = cfg.get('stats', 'full')
        scb_reduction = cfg.get('scb_k', 4) 
        
        if stat_mode == 'none':
            self.HV_GlobalContext = StatContextBlock(ch4, reduction=scb_reduction, use_mean=False, use_std=False)
        elif stat_mode == 'mean':
            self.HV_GlobalContext = StatContextBlock(ch4, reduction=scb_reduction, use_mean=True, use_std=False)
        elif stat_mode == 'std':  
            self.HV_GlobalContext = StatContextBlock(ch4, reduction=scb_reduction, use_mean=False, use_std=True)
        else: # full
            self.HV_GlobalContext = StatContextBlock(ch4, reduction=scb_reduction, use_mean=True, use_std=True)
        
        self.HVD_block3 = NormUpsample(ch4, ch3, use_norm=norm)
        self.HVD_block2 = NormUpsample(ch3, ch2, use_norm=norm)
        self.HVD_block1 = NormUpsample(ch2, ch1, use_norm=norm)
        self.HVD_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(ch1, 2, 3, stride=1, padding=0, bias=False)
        )
        
        # --- Intensity Stream (I) ---
        self.IE_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(1, ch1, 3, stride=1, padding=0, bias=False),
        )
        self.IE_block1 = NormDownsample(ch1, ch2, use_norm=norm)
        self.IE_context1 = blk_factory(i_type, ch2)
        
        self.IE_block2 = NormDownsample(ch2, ch3, use_norm=norm)
        self.IE_context2 = blk_factory(i_type, ch3)
        
        self.IE_block3 = NormDownsample(ch3, ch4, use_norm=norm)
        self.IE_context3 = blk_factory(i_type, ch4)
        
        # --- Decoders with Noise Robust DDA ---
        self.ID_block3_DDA = dda_factory(ch3, ch3, ch3, ch4, head3)
        self.ID_block2_DDA = dda_factory(ch2, ch2, ch2, ch3, head2)
        self.ID_block1_DDA = dda_factory(ch1, ch1, ch1, ch2, head1)

        self.ID_block0 =  nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(ch1, 1, 3, stride=1, padding=0, bias=False),
        )
        
        # --- Interactions (LCA) ---
        self.HV_LCA1, self.HV_LCA2, self.HV_LCA3 = HV_LCA(ch2, head2), HV_LCA(ch3, head3), HV_LCA(ch4, head4)
        self.HV_LCA4, self.HV_LCA5, self.HV_LCA6 = HV_LCA(ch4, head4), HV_LCA(ch3, head3), HV_LCA(ch2, head2)
        
        self.I_LCA1, self.I_LCA2, self.I_LCA3 = I_LCA(ch2, head2), I_LCA(ch3, head3), I_LCA(ch4, head4)
        self.I_LCA4, self.I_LCA5, self.I_LCA6 = I_LCA(ch4, head4), I_LCA(ch3, head3), I_LCA(ch2, head2)
        
        self.trans = RGB_HVI()
        self.residual_scale = nn.Parameter(torch.ones(1) * 0.1, requires_grad=True)

    def forward(self, x, return_features=False):
        dtypes = x.dtype
        hvi = self.trans.HVIT(x)
        i = hvi[:,2,:,:].unsqueeze(1).to(dtypes)
        
        # === Encoder ===
        i_enc0 = self.IE_block0(i)
        hv_0 = self.HVE_block0(hvi)
        i_jump0 = i_enc0
        hv_jump0 = hv_0
        
        i_enc1 = self.IE_block1(i_enc0)
        i_enc1 = self.IE_context1(i_enc1)
        hv_1 = self.HVE_block1(hv_0)
        hv_1 = self.HVE_context1(hv_1)
        
        i_enc2_pre = self.I_LCA1(i_enc1, hv_1)
        hv_2_pre = self.HV_LCA1(hv_1, i_enc1)
        v_jump1 = i_enc2_pre
        hv_jump1 = hv_2_pre
        
        i_enc2 = self.IE_block2(i_enc2_pre)
        i_enc2 = self.IE_context2(i_enc2)
        hv_2 = self.HVE_block2(hv_2_pre)
        hv_2 = self.HVE_context2(hv_2)
        
        i_enc3_pre = self.I_LCA2(i_enc2, hv_2)
        hv_3_pre = self.HV_LCA2(hv_2, i_enc2)
        v_jump2 = i_enc3_pre
        hv_jump2 = hv_3_pre
        
        i_enc3 = self.IE_block3(i_enc3_pre)
        i_enc3 = self.IE_context3(i_enc3)
        hv_3 = self.HVE_block3(hv_3_pre)
        hv_3 = self.HVE_context3(hv_3)
        
        i_enc4 = self.I_LCA3(i_enc3, hv_3)
        hv_4 = self.HV_LCA3(hv_3, i_enc3)
        
        # === Decoder ===
        i_dec4 = self.I_LCA4(i_enc4, hv_4) 
        hv_4 = self.HV_LCA4(hv_4, i_enc4)
        
        hv_4_before_scb = hv_4
        hv_4_after_scb = self.HV_GlobalContext(hv_4)
        hv_4 = hv_4_after_scb
        
        hv_3_dec = self.HVD_block3(hv_4, hv_jump2) 
        i_dec3, _ = self.ID_block3_DDA(i_dec4, v_jump2, guidance=hv_3_dec)
        
        i_dec2_pre = self.I_LCA5(i_dec3, hv_3_dec)
        hv_2_pre = self.HV_LCA5(hv_3_dec, i_dec3)
        
        hv_2_dec = self.HVD_block2(hv_2_pre, hv_jump1) 
        i_dec2, _ = self.ID_block2_DDA(i_dec2_pre, v_jump1, guidance=hv_2_dec)
        
        i_dec1_pre = self.I_LCA6(i_dec2, hv_2_dec)
        hv_1_pre = self.HV_LCA6(hv_2_dec, i_dec2)
        
        hv_1_dec = self.HVD_block1(hv_1_pre, hv_jump0) 
        i_dec1, _ = self.ID_block1_DDA(i_dec1_pre, i_jump0, guidance=hv_1_dec)
        
        # Output
        i_dec0 = self.ID_block0(i_dec1)
        hv_0_dec = self.HVD_block0(hv_1_dec)

        hv_0_dec = torch.clamp(hv_0_dec, -1.0, 1.0)
        i_dec0 = torch.clamp(i_dec0, -1.0, 1.0)

        output_hvi = torch.cat([hv_0_dec, i_dec0], dim=1) + hvi * self.residual_scale
        
        output_rgb = self.trans.PHVIT(output_hvi)
        output_rgb = torch.clamp(output_rgb, 0, 1)

        if return_features and not self.training:
            features_dict = {
                'intensity_shallow': i_enc2,
                'color_shallow': hv_2,
                'intensity_deep': i_enc4,             
                'color_before_scb': hv_4_before_scb,  
                'color_after_scb': hv_4_after_scb,    
                'intensity_output': i_dec0,           
                'color_output': hv_0_dec,
                'cgda_output': i_dec1

            }
            return output_rgb, features_dict

        if self.training:
            return output_rgb, output_hvi
        else:
            return output_rgb

    def HVIT(self,x):
        hvi = self.trans.HVIT(x)
        return hvi
