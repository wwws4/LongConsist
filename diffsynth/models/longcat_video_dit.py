from typing import List, Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.amp as amp

import numpy as np
import torch.nn.functional as F
from einops import rearrange, repeat
from .wan_video_dit import flash_attention
from ..vram_management import gradient_checkpoint_forward


class RMSNorm_FP32(torch.nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def broadcat(tensors, dim=-1):
    num_tensors = len(tensors)
    shape_lens = set(list(map(lambda t: len(t.shape), tensors)))
    assert len(shape_lens) == 1, "tensors must all have the same number of dimensions"
    shape_len = list(shape_lens)[0]
    dim = (dim + shape_len) if dim < 0 else dim
    dims = list(zip(*map(lambda t: list(t.shape), tensors)))
    expandable_dims = [(i, val) for i, val in enumerate(dims) if i != dim]
    assert all(
        [*map(lambda t: len(set(t[1])) <= 2, expandable_dims)]
    ), "invalid dimensions for broadcastable concatentation"
    max_dims = list(map(lambda t: (t[0], max(t[1])), expandable_dims))
    expanded_dims = list(map(lambda t: (t[0], (t[1],) * num_tensors), max_dims))
    expanded_dims.insert(dim, (dim, dims[dim]))
    expandable_shapes = list(zip(*map(lambda t: t[1], expanded_dims)))
    tensors = list(map(lambda t: t[0].expand(*t[1]), zip(tensors, expandable_shapes)))
    return torch.cat(tensors, dim=dim)


def rotate_half(x):
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d r -> ... (d r)")


class RotaryPositionalEmbedding(nn.Module):

    def __init__(self,
                 head_dim,
                 cp_split_hw=None
                 ):
        """Rotary positional embedding for 3D
        Reference : https://blog.eleuther.ai/rotary-embeddings/
        Paper: https://arxiv.org/pdf/2104.09864.pdf
        Args:
            dim: Dimension of embedding
            base: Base value for exponential
        """
        super().__init__()
        self.head_dim = head_dim
        assert self.head_dim % 8 == 0, 'Dim must be a multiply of 8 for 3D RoPE.'
        self.cp_split_hw = cp_split_hw
        # We take the assumption that the longest side of grid will not larger than 512, i.e, 512 * 8 = 4098 input pixels
        self.base = 10000
        self.freqs_dict = {}
        self.freqs_with_id_dict = {}

    def register_grid_size(self, grid_size):
        if grid_size not in self.freqs_dict:
            self.freqs_dict.update({
                grid_size: self.precompute_freqs_cis_3d(grid_size)
            })

    def precompute_freqs_cis_3d(self, grid_size):
        num_frames, height, width = grid_size     
        dim_t = self.head_dim - 4 * (self.head_dim // 6)
        dim_h = 2 * (self.head_dim // 6)
        dim_w = 2 * (self.head_dim // 6)
        freqs_t = 1.0 / (self.base ** (torch.arange(0, dim_t, 2)[: (dim_t // 2)].float() / dim_t))
        freqs_h = 1.0 / (self.base ** (torch.arange(0, dim_h, 2)[: (dim_h // 2)].float() / dim_h))
        freqs_w = 1.0 / (self.base ** (torch.arange(0, dim_w, 2)[: (dim_w // 2)].float() / dim_w))
        grid_t = np.linspace(0, num_frames, num_frames, endpoint=False, dtype=np.float32)
        grid_h = np.linspace(0, height, height, endpoint=False, dtype=np.float32)
        grid_w = np.linspace(0, width, width, endpoint=False, dtype=np.float32)
        grid_t = torch.from_numpy(grid_t).float()
        grid_h = torch.from_numpy(grid_h).float()
        grid_w = torch.from_numpy(grid_w).float()
        freqs_t = torch.einsum("..., f -> ... f", grid_t, freqs_t)
        freqs_h = torch.einsum("..., f -> ... f", grid_h, freqs_h)
        freqs_w = torch.einsum("..., f -> ... f", grid_w, freqs_w)
        freqs_t = repeat(freqs_t, "... n -> ... (n r)", r=2)
        freqs_h = repeat(freqs_h, "... n -> ... (n r)", r=2)
        freqs_w = repeat(freqs_w, "... n -> ... (n r)", r=2)
        freqs = broadcat((freqs_t[:, None, None, :], freqs_h[None, :, None, :], freqs_w[None, None, :, :]), dim=-1)
        # (T H W D)
        freqs = rearrange(freqs, "T H W D -> (T H W) D")
        # if self.cp_split_hw[0] * self.cp_split_hw[1] > 1:
        #     with torch.no_grad():
        #         freqs = rearrange(freqs, "(T H W) D -> T H W D", T=num_frames, H=height, W=width)
        #         freqs = context_parallel_util.split_cp_2d(freqs, seq_dim_hw=(1, 2), split_hw=self.cp_split_hw)
        #         freqs = rearrange(freqs, "T H W D -> (T H W) D")

        return freqs

    def precompute_freqs_cis_3d_with_id_offset(self, video_grid_size, num_id_frames):
        """
        Generate position encodings with offset for video and reference images.
        
        Position encoding offset strategy for reference images:
        - Temporal dimension: Use position encoding of the last video frame
        - Height dimension: Use normal range [0, height)
        - Width dimension: Each reference image uses different offset to distinguish them
          - 1st reference image: [width, 2*width)
          - 2nd reference image: [2*width, 3*width)
          - 3rd reference image: [3*width, 4*width)
        
        Args:
            video_grid_size: Video grid size (num_frames, height, width)
            num_id_frames: Number of reference image frames (1-3 images)
        
        Returns:
            freqs: Concatenated position encodings with shape ((T + num_id_frames) * H * W, D)
        """
        num_frames, height, width = video_grid_size
        
        dim_t = self.head_dim - 4 * (self.head_dim // 6)
        dim_h = 2 * (self.head_dim // 6)
        dim_w = 2 * (self.head_dim // 6)
        
        freqs_t = 1.0 / (self.base ** (torch.arange(0, dim_t, 2)[: (dim_t // 2)].float() / dim_t))
        freqs_h = 1.0 / (self.base ** (torch.arange(0, dim_h, 2)[: (dim_h // 2)].float() / dim_h))
        freqs_w = 1.0 / (self.base ** (torch.arange(0, dim_w, 2)[: (dim_w // 2)].float() / dim_w))
        grid_t = np.linspace(0, num_frames, num_frames, endpoint=False, dtype=np.float32)
        grid_h = np.linspace(0, height, height, endpoint=False, dtype=np.float32)
        grid_w = np.linspace(0, width, width, endpoint=False, dtype=np.float32)
        grid_t = torch.from_numpy(grid_t).float()
        grid_h = torch.from_numpy(grid_h).float()
        grid_w = torch.from_numpy(grid_w).float()
        
        freqs_t_video = torch.einsum("..., f -> ... f", grid_t, freqs_t)
        freqs_h_video = torch.einsum("..., f -> ... f", grid_h, freqs_h)
        freqs_w_video = torch.einsum("..., f -> ... f", grid_w, freqs_w)
        freqs_t_video = repeat(freqs_t_video, "... n -> ... (n r)", r=2)
        freqs_h_video = repeat(freqs_h_video, "... n -> ... (n r)", r=2)
        freqs_w_video = repeat(freqs_w_video, "... n -> ... (n r)", r=2)
        
        freqs_video = broadcat((freqs_t_video[:, None, None, :], freqs_h_video[None, :, None, :], freqs_w_video[None, None, :, :]), dim=-1)
        freqs_video = rearrange(freqs_video, "T H W D -> (T H W) D")
        
        freqs_ref_list = []
        
        grid_h_ref = np.linspace(0, height, height, endpoint=False, dtype=np.float32)
        grid_h_ref = torch.from_numpy(grid_h_ref).float()
        freqs_h_ref = torch.einsum("..., f -> ... f", grid_h_ref, freqs_h)
        freqs_h_ref = repeat(freqs_h_ref, "... n -> ... (n r)", r=2)
        
        for ref_idx in range(num_id_frames):
            grid_t_single = np.array([num_frames - 1], dtype=np.float32)
            grid_t_single = torch.from_numpy(grid_t_single).float()
            freqs_t_single = torch.einsum("..., f -> ... f", grid_t_single, freqs_t)
            freqs_t_single = repeat(freqs_t_single, "... n -> ... (n r)", r=2)
            
            w_offset_start = (ref_idx + 1) * width
            w_offset_end = (ref_idx + 2) * width
            grid_w_single = np.linspace(w_offset_start, w_offset_end, width, endpoint=False, dtype=np.float32)
            grid_w_single = torch.from_numpy(grid_w_single).float()
            freqs_w_single = torch.einsum("..., f -> ... f", grid_w_single, freqs_w)
            freqs_w_single = repeat(freqs_w_single, "... n -> ... (n r)", r=2)
            
            freqs_single = broadcat((
                freqs_t_single[:, None, None, :],
                freqs_h_ref[None, :, None, :],
                freqs_w_single[None, None, :, :]
            ), dim=-1)
            freqs_single = rearrange(freqs_single, "T H W D -> (T H W) D")
            freqs_ref_list.append(freqs_single)
        
        freqs_ref = torch.cat(freqs_ref_list, dim=0)
        
        freqs = torch.cat([freqs_video, freqs_ref], dim=0)
        
        return freqs

    def forward(self, q, k, grid_size, num_id_latents=0):
        """3D RoPE.

        Args:
            query: [B, head, seq, head_dim]
            key: [B, head, seq, head_dim]
            grid_size: Complete grid size (T_total, H, W) where T_total includes video and reference frames
            num_id_latents: Number of reference image frames (for consistency I2V), default 0
        Returns:
            query and key with the same shape as input.
        """
        
        if num_id_latents > 0:
            video_grid_size = (grid_size[0] - num_id_latents, grid_size[1], grid_size[2])
            
            cache_key = (video_grid_size, num_id_latents)
            if cache_key not in self.freqs_with_id_dict:
                self.freqs_with_id_dict[cache_key] = self.precompute_freqs_cis_3d_with_id_offset(video_grid_size, num_id_latents)
            freqs_cis = self.freqs_with_id_dict[cache_key].to(q.device)
        else:
            if grid_size not in self.freqs_dict:
                self.register_grid_size(grid_size)
            freqs_cis = self.freqs_dict[grid_size].to(q.device)

        q_, k_ = q.float(), k.float()
        freqs_cis = freqs_cis.float().to(q.device)
        cos, sin = freqs_cis.cos(), freqs_cis.sin()
        cos, sin = rearrange(cos, 'n d -> 1 1 n d'), rearrange(sin, 'n d -> 1 1 n d')
        q_ = (q_ * cos) + (rotate_half(q_) * sin)
        k_ = (k_ * cos) + (rotate_half(k_) * sin)

        return q_.type_as(q), k_.type_as(k)


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        enable_flashattn3: bool = False,
        enable_flashattn2: bool = False,
        enable_xformers: bool = False,
        enable_bsa: bool = False,
        bsa_params: dict = None,
        cp_split_hw: Optional[List[int]] = None
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.enable_flashattn3 = enable_flashattn3
        self.enable_flashattn2 = enable_flashattn2
        self.enable_xformers = enable_xformers
        self.enable_bsa = enable_bsa
        self.bsa_params = bsa_params
        self.cp_split_hw = cp_split_hw

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.q_norm = RMSNorm_FP32(self.head_dim, eps=1e-6)
        self.k_norm = RMSNorm_FP32(self.head_dim, eps=1e-6)
        self.proj = nn.Linear(dim, dim)

        self.rope_3d = RotaryPositionalEmbedding(
            self.head_dim,
            cp_split_hw=cp_split_hw
        )

    def _process_attn(self, q, k, v, shape):
        q = rearrange(q, "B H S D -> B S (H D)")
        k = rearrange(k, "B H S D -> B S (H D)")
        v = rearrange(v, "B H S D -> B S (H D)")
        x = flash_attention(q, k, v, num_heads=self.num_heads)
        x = rearrange(x, "B S (H D) -> B H S D", H=self.num_heads)
        return x

    def forward(self, x: torch.Tensor, shape=None, num_cond_latents=None, return_kv=False, num_id_latents=0) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, N, C]
            shape: Video grid size (T, H, W)
            num_cond_latents: Number of condition frames (for LongCat long video generation)
            return_kv: Whether to return KV cache
            num_id_latents: Number of reference image frames (for consistency I2V), default 0
        """
        B, N, C = x.shape
        qkv = self.qkv(x)

        qkv_shape = (B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.view(qkv_shape).permute((2, 0, 3, 1, 4))
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if return_kv:
            k_cache, v_cache = k.clone(), v.clone()

        q, k = self.rope_3d(q, k, shape, num_id_latents=num_id_latents)

        # cond mode
        if num_cond_latents is not None and num_cond_latents > 0:
            num_cond_latents_thw = num_cond_latents * (N // shape[0])
            # process the condition tokens
            q_cond = q[:, :, :num_cond_latents_thw].contiguous()
            k_cond = k[:, :, :num_cond_latents_thw].contiguous()
            v_cond = v[:, :, :num_cond_latents_thw].contiguous()
            x_cond = self._process_attn(q_cond, k_cond, v_cond, shape)
            # process the noise tokens
            q_noise = q[:, :, num_cond_latents_thw:].contiguous()
            x_noise = self._process_attn(q_noise, k, v, shape)
            # merge x_cond and x_noise
            x = torch.cat([x_cond, x_noise], dim=2).contiguous()
        else:
            x = self._process_attn(q, k, v, shape)

        x_output_shape = (B, N, C)
        x = x.transpose(1, 2) # [B, H, N, D] --> [B, N, H, D]
        x = x.reshape(x_output_shape) # [B, N, H, D] --> [B, N, C]
        x = self.proj(x)

        if return_kv:
            return x, (k_cache, v_cache)
        else:
            return x

    def forward_with_kv_cache(self, x: torch.Tensor, shape=None, num_cond_latents=None, kv_cache=None, num_id_latents=0) -> torch.Tensor:
        """
        Forward pass with KV cache (for long video continuation).
        
        Args:
            x: Input tensor [B, N, C], contains only current noise frames (and optional reference frames)
            shape: Grid size of current generation frames (T, H, W)
            num_cond_latents: Number of condition frames in KV cache
            kv_cache: (k_cache, v_cache) tuple containing KV of condition frames
            num_id_latents: Number of reference image frames (for consistency I2V), default 0
        """
        B, N, C = x.shape
        qkv = self.qkv(x)
        
        qkv_shape = (B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.view(qkv_shape).permute((2, 0, 3, 1, 4)) # [3, B, H, N, D]
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        T, H, W = shape
        k_cache, v_cache = kv_cache
        assert k_cache.shape[0] == v_cache.shape[0] and k_cache.shape[0] in [1, B]
        if k_cache.shape[0] == 1:
            k_cache = k_cache.repeat(B, 1, 1, 1)
            v_cache = v_cache.repeat(B, 1, 1, 1)
        
        if num_cond_latents is not None and num_cond_latents > 0:
            k_full = torch.cat([k_cache, k], dim=2).contiguous()
            v_full = torch.cat([v_cache, v], dim=2).contiguous()
            q_padding = torch.cat([torch.empty_like(k_cache), q], dim=2).contiguous()
            q_padding, k_full = self.rope_3d(q_padding, k_full, (T + num_cond_latents, H, W), num_id_latents=num_id_latents)
            q = q_padding[:, :, -N:].contiguous()
            
        x = self._process_attn(q, k_full, v_full, shape)
        
        x_output_shape = (B, N, C)
        x = x.transpose(1, 2) # [B, H, N, D] --> [B, N, H, D]
        x = x.reshape(x_output_shape) # [B, N, H, D] --> [B, N, C]
        x = self.proj(x)

        return x


class MultiHeadCrossAttention(nn.Module):
    def __init__(
            self,
            dim,
            num_heads,
            enable_flashattn3=False,
            enable_flashattn2=False,
            enable_xformers=False,
        ):
        super(MultiHeadCrossAttention, self).__init__()
        assert dim % num_heads == 0, "d_model must be divisible by num_heads"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q_linear = nn.Linear(dim, dim)
        self.kv_linear = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)

        self.q_norm = RMSNorm_FP32(self.head_dim, eps=1e-6)
        self.k_norm = RMSNorm_FP32(self.head_dim, eps=1e-6)

        self.enable_flashattn3 = enable_flashattn3
        self.enable_flashattn2 = enable_flashattn2
        self.enable_xformers = enable_xformers

    def _process_cross_attn(self, x, cond, kv_seqlen):
        B, N, C = x.shape
        assert C == self.dim and cond.shape[2] == self.dim

        q = self.q_linear(x).view(1, -1, self.num_heads, self.head_dim)
        kv = self.kv_linear(cond).view(1, -1, 2, self.num_heads, self.head_dim)
        k, v = kv.unbind(2)

        q, k = self.q_norm(q), self.k_norm(k)

        q = rearrange(q, "B S H D -> B S (H D)")
        k = rearrange(k, "B S H D -> B S (H D)")
        v = rearrange(v, "B S H D -> B S (H D)")
        x = flash_attention(q, k, v, num_heads=self.num_heads)

        x = x.view(B, -1, C)
        x = self.proj(x)
        return x

    def forward(self, x, cond, kv_seqlen, num_cond_latents=None, shape=None):
        """
            x: [B, N, C]
            cond: [B, M, C]
        """
        if num_cond_latents is None or num_cond_latents == 0:
            return self._process_cross_attn(x, cond, kv_seqlen)
        else:
            B, N, C = x.shape
            if num_cond_latents is not None and num_cond_latents > 0:
                assert shape is not None, "SHOULD pass in the shape"
                num_cond_latents_thw = num_cond_latents * (N // shape[0])
                x_noise = x[:, num_cond_latents_thw:] # [B, N_noise, C]
                output_noise = self._process_cross_attn(x_noise, cond, kv_seqlen) # [B, N_noise, C]
                output = torch.cat([
                    torch.zeros((B, num_cond_latents_thw, C), dtype=output_noise.dtype, device=output_noise.device),
                    output_noise
                ], dim=1).contiguous()
            else:
                raise NotImplementedError
                
            return output


class LayerNorm_FP32(nn.LayerNorm):
    def __init__(self, dim, eps, elementwise_affine):
        super().__init__(dim, eps=eps, elementwise_affine=elementwise_affine)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        origin_dtype = inputs.dtype
        out = F.layer_norm(
            inputs.float(), 
            self.normalized_shape, 
            None if self.weight is None else self.weight.float(), 
            None if self.bias is None else self.bias.float() ,
            self.eps
        ).to(origin_dtype)
        return out


def modulate_fp32(norm_func, x, shift, scale):
    # Suppose x is (B, N, D), shift is (B, -1, D), scale is (B, -1, D)
    # ensure the modulation params be fp32
    assert shift.dtype == torch.float32, scale.dtype == torch.float32
    dtype = x.dtype
    x = norm_func(x.to(torch.float32))
    x = x * (scale + 1) + shift
    x = x.to(dtype)
    return x


class FinalLayer_FP32(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, hidden_size, num_patch, out_channels, adaln_tembed_dim):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_patch = num_patch
        self.out_channels = out_channels
        self.adaln_tembed_dim = adaln_tembed_dim

        self.norm_final = LayerNorm_FP32(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, num_patch * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(adaln_tembed_dim, 2 * hidden_size, bias=True))

    def forward(self, x, t, latent_shape):
        # timestep shape: [B, T, C]
        assert t.dtype == torch.float32
        B, N, C = x.shape
        T, _, _ = latent_shape

        with amp.autocast('cuda', dtype=torch.float32):
            shift, scale = self.adaLN_modulation(t).unsqueeze(2).chunk(2, dim=-1) # [B, T, 1, C]
            x = modulate_fp32(self.norm_final, x.view(B, T, -1, C), shift, scale).view(B, N, C)
            x = self.linear(x)
        return x


class FeedForwardSwiGLU(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int = 256,
        ffn_dim_multiplier: Optional[float] = None,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.dim = dim
        self.hidden_dim = hidden_dim
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, t_embed_dim, frequency_embedding_size=256):
        super().__init__()
        self.t_embed_dim = t_embed_dim
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, t_embed_dim, bias=True),
            nn.SiLU(),
            nn.Linear(t_embed_dim, t_embed_dim, bias=True),
        )

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half)
        freqs = freqs.to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t, dtype):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        if t_freq.dtype != dtype:
            t_freq = t_freq.to(dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


class CaptionEmbedder(nn.Module):
    """
    Embeds class labels into vector representations.
    """

    def __init__(self, in_channels, hidden_size):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.y_proj = nn.Sequential(
            nn.Linear(in_channels, hidden_size, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    def forward(self, caption):
        B, _, N, C = caption.shape
        caption = self.y_proj(caption)
        return caption


class PatchEmbed3D(nn.Module):
    """Video to Patch Embedding.

    Args:
        patch_size (int): Patch token size. Default: (2,4,4).
        in_chans (int): Number of input video channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(
        self,
        patch_size=(2, 4, 4),
        in_chans=3,
        embed_dim=96,
        norm_layer=None,
        flatten=True,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.flatten = flatten

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, D, H, W = x.size()
        if W % self.patch_size[2] != 0:
            x = F.pad(x, (0, self.patch_size[2] - W % self.patch_size[2]))
        if H % self.patch_size[1] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[1] - H % self.patch_size[1]))
        if D % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - D % self.patch_size[0]))

        B, C, T, H, W = x.shape
        x = self.proj(x)  # (B C T H W)
        if self.norm is not None:
            D, Wh, Ww = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, D, Wh, Ww)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCTHW -> BNC
        return x


class LongCatSingleStreamBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: int,
        adaln_tembed_dim: int,
        enable_flashattn3: bool = False,
        enable_flashattn2: bool = False,
        enable_xformers: bool = False,
        enable_bsa: bool = False,
        bsa_params=None,
        cp_split_hw=None
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.adaln_tembed_dim = adaln_tembed_dim

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(adaln_tembed_dim, 6 * hidden_size, bias=True)
        )
        
        self.adaLN_modulation_ref = nn.Sequential(
            nn.SiLU(),
            nn.Linear(adaln_tembed_dim, 6 * hidden_size, bias=True)
        )

        self.mod_norm_attn = LayerNorm_FP32(hidden_size, eps=1e-6, elementwise_affine=False)
        self.mod_norm_ffn  = LayerNorm_FP32(hidden_size, eps=1e-6, elementwise_affine=False)
        self.pre_crs_attn_norm = LayerNorm_FP32(hidden_size, eps=1e-6, elementwise_affine=True)

        self.attn = Attention(
            dim=hidden_size,
            num_heads=num_heads,
            enable_flashattn3=enable_flashattn3,
            enable_flashattn2=enable_flashattn2,
            enable_xformers=enable_xformers,
            enable_bsa=enable_bsa,
            bsa_params=bsa_params,
            cp_split_hw=cp_split_hw
        )
        self.cross_attn = MultiHeadCrossAttention(
            dim=hidden_size,
            num_heads=num_heads,
            enable_flashattn3=enable_flashattn3,
            enable_flashattn2=enable_flashattn2,
            enable_xformers=enable_xformers,
        )
        self.ffn = FeedForwardSwiGLU(dim=hidden_size, hidden_dim=int(hidden_size * mlp_ratio))

    def forward(self, x, y, t, y_seqlen, latent_shape, num_cond_latents=None, return_kv=False, kv_cache=None, skip_crs_attn=False, num_id_latents=0):
        """
            x: [B, N, C]
            y: [1, N_valid_tokens, C]
            t: [B, T, C_t]
            y_seqlen: [B]; type of a list
            latent_shape: latent shape of a single item (note: includes reference images)
            num_cond_latents: Number of condition frames (for LongCat long video generation)
            num_id_latents: Number of reference image frames (for consistency I2V), default 0
        """
        x_dtype = x.dtype

        B, N, C = x.shape
        T, _, _ = latent_shape

        with amp.autocast(device_type='cuda', dtype=torch.float32):
            modulation_output = self.adaLN_modulation(t).unsqueeze(2)
            
            if num_id_latents > 0:
                t_ref = t[:, -num_id_latents:, :]
                modulation_output_ref = self.adaLN_modulation_ref(t_ref).unsqueeze(2)
                modulation_output = torch.cat([
                    modulation_output[:, :-num_id_latents, :, :],
                    modulation_output_ref
                ], dim=1)
            
            shift_msa, scale_msa, gate_msa, \
            shift_mlp, scale_mlp, gate_mlp = modulation_output.chunk(6, dim=-1)

        # self attn with modulation
        x_m = modulate_fp32(self.mod_norm_attn, x.view(B, T, -1, C), shift_msa, scale_msa).view(B, N, C)

        if kv_cache is not None:
            kv_cache = (kv_cache[0].to(x.device), kv_cache[1].to(x.device))
            attn_outputs = self.attn.forward_with_kv_cache(x_m, shape=latent_shape, num_cond_latents=num_cond_latents, kv_cache=kv_cache, num_id_latents=num_id_latents)
        else:
            attn_outputs = self.attn(x_m, shape=latent_shape, num_cond_latents=num_cond_latents, return_kv=return_kv, num_id_latents=num_id_latents)
        
        if return_kv:
            x_s, kv_cache = attn_outputs
        else:
            x_s = attn_outputs

        with amp.autocast(device_type='cuda', dtype=torch.float32):
            if num_id_latents > 0:
                num_tokens_per_frame = N // T
                num_video_tokens = (T - num_id_latents) * num_tokens_per_frame
                num_video_frames = T - num_id_latents
                
                x_s_video = x_s[:, :num_video_tokens, :].view(B, num_video_frames, num_tokens_per_frame, C)
                gate_video = gate_msa[:, :num_video_frames, :, :]
                gated_video = (gate_video * x_s_video).view(B, num_video_tokens, C)
                output_video = x[:, :num_video_tokens, :] + gated_video
                
                x_s_ref = x_s[:, num_video_tokens:, :].view(B, num_id_latents, num_tokens_per_frame, C)
                gate_ref = gate_msa[:, num_video_frames:, :, :]
                gated_ref = (gate_ref * x_s_ref).view(B, num_id_latents * num_tokens_per_frame, C)
                output_ref = x[:, num_video_tokens:, :] + gated_ref
                
                x = torch.cat([output_video, output_ref], dim=1)
            else:
                x = x + (gate_msa * x_s.view(B, -1, N//T, C)).view(B, -1, C)
        x = x.to(x_dtype)

        if not skip_crs_attn:
            if num_id_latents > 0:
                num_tokens_per_frame = N // T
                num_video_tokens = (T - num_id_latents) * num_tokens_per_frame
                
                x_video = x[:, :num_video_tokens, :]
                x_ref = x[:, num_video_tokens:, :]
                
                video_shape = (T - num_id_latents, latent_shape[1], latent_shape[2])
                
                num_cond_latents_for_cross = None if kv_cache is not None else num_cond_latents
                
                x_video = x_video + self.cross_attn(
                    self.pre_crs_attn_norm(x_video), y, y_seqlen, 
                    num_cond_latents=num_cond_latents_for_cross, 
                    shape=video_shape
                )
                
                x = torch.cat([x_video, x_ref], dim=1)
            else:
                if kv_cache is not None:
                    num_cond_latents = None
                x = x + self.cross_attn(self.pre_crs_attn_norm(x), y, y_seqlen, num_cond_latents=num_cond_latents, shape=latent_shape)

        x_m = modulate_fp32(self.mod_norm_ffn, x.view(B, -1, N//T, C), shift_mlp, scale_mlp).view(B, -1, C)
        x_s = self.ffn(x_m)
        
        with amp.autocast(device_type='cuda', dtype=torch.float32):
            if num_id_latents > 0:
                num_tokens_per_frame = N // T
                num_video_tokens = (T - num_id_latents) * num_tokens_per_frame
                num_video_frames = T - num_id_latents
                
                x_s_video = x_s[:, :num_video_tokens, :].view(B, num_video_frames, num_tokens_per_frame, C)
                gate_video = gate_mlp[:, :num_video_frames, :, :]
                gated_video = (gate_video * x_s_video).view(B, num_video_tokens, C)
                output_video = x[:, :num_video_tokens, :] + gated_video
                
                x_s_ref = x_s[:, num_video_tokens:, :].view(B, num_id_latents, num_tokens_per_frame, C)
                gate_ref = gate_mlp[:, num_video_frames:, :, :]
                gated_ref = (gate_ref * x_s_ref).view(B, num_id_latents * num_tokens_per_frame, C)
                output_ref = x[:, num_video_tokens:, :] + gated_ref
                
                x = torch.cat([output_video, output_ref], dim=1)
            else:
                x = x + (gate_mlp * x_s.view(B, -1, N//T, C)).view(B, -1, C)
        x = x.to(x_dtype)

        if return_kv:
            return x, kv_cache
        else:
            return x


class LongCatVideoTransformer3DModel(torch.nn.Module):
    def __init__(
        self,
        in_channels: int = 16,
        out_channels: int = 16,
        hidden_size: int = 4096,
        depth: int = 48,
        num_heads: int = 32,
        caption_channels: int = 4096,
        mlp_ratio: int = 4,
        adaln_tembed_dim: int = 512,
        frequency_embedding_size: int = 256,
        # default params
        patch_size: Tuple[int] = (1, 2, 2),
        # attention config
        enable_flashattn3: bool = False,
        enable_flashattn2: bool = True,
        enable_xformers: bool = False,
        enable_bsa: bool = False,
        bsa_params: dict = {'sparsity': 0.9375, 'chunk_3d_shape_q': [4, 4, 4], 'chunk_3d_shape_k': [4, 4, 4]},
        cp_split_hw: Optional[List[int]] = [1, 1],
        text_tokens_zero_pad: bool = True,
    ) -> None:
        super().__init__()

        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cp_split_hw = cp_split_hw

        self.x_embedder = PatchEmbed3D(patch_size, in_channels, hidden_size)
        self.t_embedder = TimestepEmbedder(t_embed_dim=adaln_tembed_dim, frequency_embedding_size=frequency_embedding_size)
        self.y_embedder = CaptionEmbedder(
            in_channels=caption_channels,
            hidden_size=hidden_size,
        )

        self.blocks = nn.ModuleList(
            [
                LongCatSingleStreamBlock(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    adaln_tembed_dim=adaln_tembed_dim,
                    enable_flashattn3=enable_flashattn3,
                    enable_flashattn2=enable_flashattn2,
                    enable_xformers=enable_xformers,
                    enable_bsa=enable_bsa,
                    bsa_params=bsa_params,
                    cp_split_hw=cp_split_hw
                )
                for i in range(depth)
            ]
        )

        self.final_layer = FinalLayer_FP32(
            hidden_size,
            np.prod(self.patch_size),
            out_channels,
            adaln_tembed_dim,
        )

        self.gradient_checkpointing = False
        self.text_tokens_zero_pad = text_tokens_zero_pad

        self.lora_dict = {}
        self.active_loras = []

    def enable_loras(self, lora_key_list=[]):
        self.disable_all_loras()
    
        module_loras = {}  # {module_name: [lora1, lora2, ...]}
        model_device = next(self.parameters()).device
        model_dtype = next(self.parameters()).dtype
        
        for lora_key in lora_key_list:
            if lora_key in self.lora_dict:
                for lora in self.lora_dict[lora_key].loras:
                    lora.to(model_device, dtype=model_dtype, non_blocking=True)
                    module_name = lora.lora_name.replace("lora___lorahyphen___", "").replace("___lorahyphen___", ".")
                    if module_name not in module_loras:
                        module_loras[module_name] = []
                    module_loras[module_name].append(lora)
                self.active_loras.append(lora_key)
    
        for module_name, loras in module_loras.items():
            module = self._get_module_by_name(module_name)
            if not hasattr(module, 'org_forward'):
                module.org_forward = module.forward
            module.forward = self._create_multi_lora_forward(module, loras)
    
    def _create_multi_lora_forward(self, module, loras):
        def multi_lora_forward(x, *args, **kwargs):
            weight_dtype = x.dtype
            org_output = module.org_forward(x, *args, **kwargs)
            
            total_lora_output = 0
            for lora in loras:
                if lora.use_lora:
                    lx = lora.lora_down(x.to(lora.lora_down.weight.dtype))
                    lx = lora.lora_up(lx)
                    lora_output = lx.to(weight_dtype) * lora.multiplier * lora.alpha_scale
                    total_lora_output += lora_output
            
            return org_output + total_lora_output
        
        return multi_lora_forward
    
    def _get_module_by_name(self, module_name):
        try:
            module = self
            for part in module_name.split('.'):
                module = getattr(module, part)
            return module
        except AttributeError as e:
            raise ValueError(f"Cannot find module: {module_name}, error: {e}")
    
    def disable_all_loras(self):
        for name, module in self.named_modules():
            if hasattr(module, 'org_forward'):
                module.forward = module.org_forward
                delattr(module, 'org_forward')
        
        for lora_key, lora_network in self.lora_dict.items():
            for lora in lora_network.loras:
                lora.to("cpu")
        
        self.active_loras.clear()

    def enable_bsa(self,):
        for block in self.blocks:
            block.attn.enable_bsa = True
    
    def disable_bsa(self,):
        for block in self.blocks:
            block.attn.enable_bsa = False
    
    def initialize_ref_modulation_weights(self):
        """
        Initialize reference image temporal modulation module weights.
        
        Copy weights from adaLN_modulation to adaLN_modulation_ref.
        
        Benefits:
        1. Reference image modulation starts from video modulation weights
        2. Can learn reference-specific modulation parameters during training
        3. Original functionality remains unaffected when no reference images
        """
        for block in self.blocks:
            block.adaLN_modulation_ref.load_state_dict(
                block.adaLN_modulation.state_dict(), strict=True
            )
        print("Initialized adaLN_modulation_ref weights (copied from adaLN_modulation)")

    def forward(
        self, 
        hidden_states, 
        timestep, 
        encoder_hidden_states, 
        encoder_attention_mask=None, 
        num_cond_latents=0,
        num_id_latents=0,
        return_kv=False, 
        kv_cache_dict={},
        skip_crs_attn=False, 
        offload_kv_cache=False,
        use_gradient_checkpointing=False,
        use_gradient_checkpointing_offload=False,
    ):
        """
        Args:
            hidden_states: Input latents with shape [B, C, T, H, W]
            timestep: Timestep
            encoder_hidden_states: Text encoding
            encoder_attention_mask: Text attention mask
            num_cond_latents: Number of condition frames (for LongCat long video generation)
            num_id_latents: Number of reference image frames (for consistency I2V), default 0
            return_kv: Whether to return KV cache
            kv_cache_dict: KV cache dictionary
            skip_crs_attn: Whether to skip cross attention
            offload_kv_cache: Whether to offload KV cache to CPU
            use_gradient_checkpointing: Whether to use gradient checkpointing
            use_gradient_checkpointing_offload: Whether to offload gradient checkpointing to CPU
        """
        B, _, T, H, W = hidden_states.shape

        N_t = T // self.patch_size[0]
        N_h = H // self.patch_size[1]
        N_w = W // self.patch_size[2]

        assert self.patch_size[0]==1, "Currently, 3D x_embedder should not compress the temporal dimension."

        # expand the shape of timestep from [B] to [B, T]
        if len(timestep.shape) == 1:
            timestep = timestep.unsqueeze(1).expand(-1, N_t).clone() # [B, T]
        timestep[:, :num_cond_latents] = 0
        
        if num_id_latents > 0:
            timestep[:, -num_id_latents:] = 0

        dtype = hidden_states.dtype
        hidden_states = hidden_states.to(dtype)
        timestep = timestep.to(dtype)
        encoder_hidden_states = encoder_hidden_states.to(dtype)

        hidden_states = self.x_embedder(hidden_states)  # [B, N, C]

        with amp.autocast(device_type='cuda', dtype=torch.float32):
            t = self.t_embedder(timestep.float().flatten(), dtype=torch.float32).reshape(B, N_t, -1)  # [B, T, C_t]

        encoder_hidden_states = self.y_embedder(encoder_hidden_states)  # [B, 1, N_token, C]

        if self.text_tokens_zero_pad and encoder_attention_mask is not None:
            encoder_hidden_states = encoder_hidden_states * encoder_attention_mask[:, None, :, None]
            encoder_attention_mask = (encoder_attention_mask * 0 + 1).to(encoder_attention_mask.dtype)

        if encoder_attention_mask is not None:
            encoder_attention_mask = encoder_attention_mask.squeeze(1).squeeze(1)
            encoder_hidden_states = encoder_hidden_states.squeeze(1).masked_select(encoder_attention_mask.unsqueeze(-1) != 0).view(1, -1, hidden_states.shape[-1]) # [1, N_valid_tokens, C]
            y_seqlens = encoder_attention_mask.sum(dim=1).tolist() # [B]
        else:
            y_seqlens = [encoder_hidden_states.shape[2]] * encoder_hidden_states.shape[0]
            encoder_hidden_states = encoder_hidden_states.squeeze(1).view(1, -1, hidden_states.shape[-1])

        # if self.cp_split_hw[0] * self.cp_split_hw[1] > 1:
        #     hidden_states = rearrange(hidden_states, "B (T H W) C -> B T H W C", T=N_t, H=N_h, W=N_w)
        #     hidden_states = context_parallel_util.split_cp_2d(hidden_states, seq_dim_hw=(2, 3), split_hw=self.cp_split_hw)
        #     hidden_states = rearrange(hidden_states, "B T H W C -> B (T H W) C")

        N_t_video = N_t - num_id_latents
        
        # blocks
        kv_cache_dict_ret = {}
        for i, block in enumerate(self.blocks):
            block_outputs = gradient_checkpoint_forward(
                block,
                use_gradient_checkpointing=use_gradient_checkpointing,
                use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
                x=hidden_states,
                y=encoder_hidden_states,
                t=t,
                y_seqlen=y_seqlens,
                latent_shape=(N_t, N_h, N_w),
                num_cond_latents=num_cond_latents,
                return_kv=return_kv,
                kv_cache=kv_cache_dict.get(i, None),
                skip_crs_attn=skip_crs_attn,
                num_id_latents=num_id_latents,
            )
            
            if return_kv:
                hidden_states, kv_cache = block_outputs
                if offload_kv_cache:
                    kv_cache_dict_ret[i] = (kv_cache[0].cpu(), kv_cache[1].cpu())
                else:
                    kv_cache_dict_ret[i] = (kv_cache[0].contiguous(), kv_cache[1].contiguous())
            else:
                hidden_states = block_outputs

        hidden_states = self.final_layer(hidden_states, t, (N_t, N_h, N_w))  # [B, N, C=T_p*H_p*W_p*C_out]

        # if self.cp_split_hw[0] * self.cp_split_hw[1] > 1:
        #     hidden_states = context_parallel_util.gather_cp_2d(hidden_states, shape=(N_t, N_h, N_w), split_hw=self.cp_split_hw)

        hidden_states = self.unpatchify(hidden_states, N_t, N_h, N_w)  # [B, C_out, H, W]

        # cast to float32 for better accuracy
        hidden_states = hidden_states.to(torch.float32)

        if return_kv:
            return hidden_states, kv_cache_dict_ret
        else:
            return hidden_states
    

    def unpatchify(self, x, N_t, N_h, N_w):
        """
        Args:
            x (torch.Tensor): of shape [B, N, C]

        Return:
            x (torch.Tensor): of shape [B, C_out, T, H, W]
        """
        T_p, H_p, W_p = self.patch_size
        x = rearrange(
            x,
            "B (N_t N_h N_w) (T_p H_p W_p C_out) -> B C_out (N_t T_p) (N_h H_p) (N_w W_p)",
            N_t=N_t,
            N_h=N_h,
            N_w=N_w,
            T_p=T_p,
            H_p=H_p,
            W_p=W_p,
            C_out=self.out_channels,
        )
        return x

    @staticmethod
    def state_dict_converter():
        return LongCatVideoTransformer3DModelDictConverter()


class LongCatVideoTransformer3DModelDictConverter:
    """
    State dict converter for handling model weight loading.
    
    Main features:
    1. Compatible with diffusers and civitai formats
    2. Generate initial weights for adaLN_modulation_ref (copied from adaLN_modulation)
    """
    def __init__(self):
        pass

    def _add_ref_modulation_weights(self, state_dict):
        """
        Add weights for adaLN_modulation_ref module.
        
        If state_dict doesn't contain ref module weights, copy from corresponding original module.
        This ensures ref module is properly initialized when loading old checkpoints.
        
        Note: Only add ref weights for LongCatSingleStreamBlock in blocks,
        final_layer (FinalLayer_FP32) doesn't have adaLN_modulation_ref module.
        """
        new_state_dict = state_dict.copy()
        
        keys_to_add = {}
        for key in state_dict.keys():
            if 'blocks.' in key and 'adaLN_modulation.' in key and 'adaLN_modulation_ref.' not in key:
                ref_key = key.replace('adaLN_modulation.', 'adaLN_modulation_ref.')
                if ref_key not in state_dict:
                    keys_to_add[ref_key] = state_dict[key].clone()
        
        new_state_dict.update(keys_to_add)
        
        if keys_to_add:
            print(f"Added {len(keys_to_add)} weights for adaLN_modulation_ref module (copied from adaLN_modulation)")
        
        return new_state_dict

    def from_diffusers(self, state_dict):
        state_dict = self._add_ref_modulation_weights(state_dict)
        return state_dict

    def from_civitai(self, state_dict):
        state_dict = self._add_ref_modulation_weights(state_dict)
        return state_dict

