import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from einops import rearrange
from ..control.dino_control_module import DINO2WanLatentAdapter, WanControlFusionBridge

try:
    from flash_attn.cute import flash_attn_func as flash_attn_4_func
    FLASH_ATTN_4_AVAILABLE = True
    FLASH_ATTN_4_IMPORT_ERROR = None
except Exception as exc:
    flash_attn_4_func = None
    FLASH_ATTN_4_AVAILABLE = False
    FLASH_ATTN_4_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

try:
    from flash_attn_interface import flash_attn_func as flash_attn_3_func
    FLASH_ATTN_3_AVAILABLE = True
    FLASH_ATTN_3_IMPORT_ERROR = None
except Exception as exc:
    flash_attn_3_func = None
    FLASH_ATTN_3_AVAILABLE = False
    FLASH_ATTN_3_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

try:
    from flash_attn import flash_attn_func as flash_attn_2_func
    FLASH_ATTN_2_AVAILABLE = True
    FLASH_ATTN_2_IMPORT_ERROR = None
except Exception as exc:
    flash_attn_2_func = None
    FLASH_ATTN_2_AVAILABLE = False
    FLASH_ATTN_2_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

try:
    from sageattention import sageattn
    SAGE_ATTN_AVAILABLE = True
    SAGE_ATTN_IMPORT_ERROR = None
except Exception as exc:
    sageattn = None
    SAGE_ATTN_AVAILABLE = False
    SAGE_ATTN_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"


_ATTENTION_BACKENDS = ("fa4", "fa3", "fa2", "sage", "sdpa")
_FLASH_ATTENTION_BACKENDS = {"fa4", "fa3", "fa2"}
_SELECTED_ATTENTION_BACKEND = "sdpa"
_ATTENTION_BACKEND_REPORT = "Attention backend selected: sdpa (default before pipeline configuration)."
_ATTENTION_RUNTIME_FALLBACK_WARNED: set[str] = set()


def _rank0() -> bool:
    try:
        import torch.distributed as dist

        return not dist.is_initialized() or dist.get_rank() == 0
    except Exception:
        return True


def _device_capability(device: Optional[torch.device | str] = None) -> Optional[tuple[int, int]]:
    if not torch.cuda.is_available():
        return None
    torch_device = torch.device(device) if device is not None else torch.device("cuda")
    if torch_device.type != "cuda":
        return None
    index = torch_device.index
    if index is None:
        index = torch.cuda.current_device()
    return torch.cuda.get_device_capability(index)


def _device_label(device: Optional[torch.device | str] = None) -> str:
    capability = _device_capability(device)
    if capability is None:
        return str(device or "cpu")
    return f"{device or f'cuda:{torch.cuda.current_device()}'} sm{capability[0]}{capability[1]}"


def _backend_unavailable_reason(backend: str, device: Optional[torch.device | str] = None) -> Optional[str]:
    capability = _device_capability(device)
    if backend == "fa4":
        if not FLASH_ATTN_4_AVAILABLE:
            return f"not installed/importable ({FLASH_ATTN_4_IMPORT_ERROR})"
        if capability is None or capability[0] < 9:
            return f"requires Hopper/Blackwell GPU (sm90+), got {_device_label(device)}"
        return None
    if backend == "fa3":
        if not FLASH_ATTN_3_AVAILABLE:
            return f"not installed/importable ({FLASH_ATTN_3_IMPORT_ERROR})"
        if capability is None or capability[0] != 9:
            return f"requires Hopper GPU (sm90), got {_device_label(device)}"
        return None
    if backend == "fa2":
        if not FLASH_ATTN_2_AVAILABLE:
            return f"not installed/importable ({FLASH_ATTN_2_IMPORT_ERROR})"
        if capability is None or capability[0] < 8:
            return f"requires Ampere/Ada/Hopper/Blackwell CUDA GPU, got {_device_label(device)}"
        return None
    if backend == "sage":
        if not SAGE_ATTN_AVAILABLE:
            return f"not installed/importable ({SAGE_ATTN_IMPORT_ERROR})"
        if capability is None:
            return f"requires CUDA GPU, got {_device_label(device)}"
        return None
    if backend == "sdpa":
        return None
    return f"unknown attention backend '{backend}'"


def _backend_is_usable(backend: str, device: Optional[torch.device | str] = None) -> bool:
    return _backend_unavailable_reason(backend, device) is None


def _architecture_label(device: Optional[torch.device | str] = None) -> str:
    capability = _device_capability(device)
    if capability is None:
        return _device_label(device)
    major, minor = capability
    if major >= 10:
        family = "Blackwell or newer"
    elif major == 9:
        family = "Hopper"
    elif major == 8 and minor == 9:
        family = "Ada"
    elif major == 8:
        family = "Ampere"
    else:
        family = "pre-Ampere"
    return f"{family} (sm{major}{minor})"


def _flash_attention_guidance(
    selected: str,
    device: Optional[torch.device | str] = None,
) -> str | None:
    if selected in _FLASH_ATTENTION_BACKENDS:
        return None

    capability = _device_capability(device)
    if capability is None:
        return (
            f"No FlashAttention backend is active; using '{selected}'. "
            "No CUDA GPU was detected for attention backend selection."
        )

    if capability[0] < 9:
        return (
            f"No FlashAttention backend is active; using '{selected}'. "
            "The default environment installs FA4, but FA4 requires Hopper/Blackwell GPUs (sm90+). "
            f"This GPU is {_architecture_label(device)}. "
            "For Ampere/Ada GPUs, install FA2 instead, for example `pip install flash-attn --no-build-isolation`, "
            "or install a matching prebuilt `flash_attn` wheel, then rerun inference."
        )

    return (
        f"No FlashAttention backend is active; using '{selected}'. "
        f"This GPU is {_architecture_label(device)} and can use FA4, but `flash_attn.cute` was not usable. "
        "Install FA4 with `pip install flash-attn-4` for CUDA 12, or `pip install \"flash-attn-4[cu13]\"` "
        "for CUDA 13, then rerun inference."
    )


def _fallback_order(requested: str) -> tuple[str, ...]:
    if requested == "auto":
        return _ATTENTION_BACKENDS
    return (requested,) + tuple(backend for backend in _ATTENTION_BACKENDS if backend != requested)


def configure_attention_backend(
    device: Optional[torch.device | str] = None,
    verbose: bool = True,
) -> str:
    global _SELECTED_ATTENTION_BACKEND, _ATTENTION_BACKEND_REPORT

    selected = "sdpa"
    skipped: list[str] = []
    for backend in _fallback_order("auto"):
        reason = _backend_unavailable_reason(backend, device)
        if reason is None:
            selected = backend
            break
        skipped.append(f"{backend}: {reason}")

    _SELECTED_ATTENTION_BACKEND = selected
    _ATTENTION_BACKEND_REPORT = (
        f"Attention backend selected: {selected} "
        f"(automatic, device={_device_label(device)})."
    )

    if verbose and _rank0():
        print(_ATTENTION_BACKEND_REPORT)
        if skipped:
            print("Attention backend fallbacks skipped: " + "; ".join(skipped))
        guidance = _flash_attention_guidance(selected, device)
        if guidance:
            print(guidance)
    return selected


def get_attention_backend_report() -> str:
    return _ATTENTION_BACKEND_REPORT


def _normalize_attention_output(x):
    if isinstance(x, tuple):
        return x[0]
    return x


def _run_attention_backend(
    backend: str,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_heads: int,
) -> torch.Tensor:
    if backend == "fa4":
        return _normalize_attention_output(flash_attn_4_func(q, k, v))
    if backend == "fa3":
        return _normalize_attention_output(flash_attn_3_func(q, k, v))
    if backend == "fa2":
        return _normalize_attention_output(flash_attn_2_func(q, k, v))
    if backend == "sage":
        q_nsd = rearrange(q, "b s n d -> b n s d")
        k_nsd = rearrange(k, "b s n d -> b n s d")
        v_nsd = rearrange(v, "b s n d -> b n s d")
        return rearrange(sageattn(q_nsd, k_nsd, v_nsd), "b n s d -> b s n d")
    if backend == "sdpa":
        q_nsd = rearrange(q, "b s n d -> b n s d")
        k_nsd = rearrange(k, "b s n d -> b n s d")
        v_nsd = rearrange(v, "b s n d -> b n s d")
        return rearrange(F.scaled_dot_product_attention(q_nsd, k_nsd, v_nsd), "b n s d -> b s n d")
    raise ValueError(f"Unsupported attention backend: {backend}")


def _run_attention_with_runtime_fallback(
    preferred_backend: str,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_heads: int,
) -> torch.Tensor:
    last_error: Exception | None = None
    for backend in _fallback_order(preferred_backend):
        if not _backend_is_usable(backend, q.device):
            continue
        try:
            result = _run_attention_backend(backend, q, k, v, num_heads)
            if backend not in _FLASH_ATTENTION_BACKENDS and preferred_backend in _FLASH_ATTENTION_BACKENDS:
                warn_key = f"runtime-no-fa:{preferred_backend}->{backend}"
                if warn_key not in _ATTENTION_RUNTIME_FALLBACK_WARNED:
                    _ATTENTION_RUNTIME_FALLBACK_WARNED.add(warn_key)
                    guidance = _flash_attention_guidance(backend, q.device)
                    if guidance and _rank0():
                        print(f"{guidance} This happened after '{preferred_backend}' failed at runtime.")
            return result
        except Exception as exc:
            if "out of memory" in str(exc).lower():
                raise
            last_error = exc
            warn_key = f"{backend}->{preferred_backend}"
            if warn_key not in _ATTENTION_RUNTIME_FALLBACK_WARNED:
                _ATTENTION_RUNTIME_FALLBACK_WARNED.add(warn_key)
                if _rank0():
                    print(
                        f"Attention backend '{backend}' failed at runtime "
                        f"({type(exc).__name__}: {exc}). Trying the next fallback."
                    )
    if last_error is not None:
        raise last_error
    raise RuntimeError("No usable attention backend is available.")


def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int, compatibility_mode=False):
    q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
    k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
    v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
    backend = "sdpa" if compatibility_mode else _SELECTED_ATTENTION_BACKEND
    x = _run_attention_with_runtime_fallback(backend, q, k, v, num_heads)
    return rearrange(x, "b s n d -> b s (n d)")


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return (x * (1 + scale) + shift)


def sinusoidal_embedding_1d(dim, position):
    sinusoid = torch.outer(position.type(torch.float64), torch.pow(
        10000, -torch.arange(dim//2, dtype=torch.float64, device=position.device).div(dim//2)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)


def precompute_freqs_cis_3d(dim: int, end: int = 1024, theta: float = 10000.0):
    # 3d rope precompute
    f_freqs_cis = precompute_freqs_cis(dim - 2 * (dim // 3), end, theta)
    h_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    w_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    return f_freqs_cis, h_freqs_cis, w_freqs_cis


def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0):
    # 1d rope precompute
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)
                   [: (dim // 2)].double() / dim))
    freqs = torch.outer(torch.arange(end, device=freqs.device), freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def rope_apply(x, freqs, num_heads):
    x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)
    x_out = torch.view_as_complex(x.to(torch.float64).reshape(
        x.shape[0], x.shape[1], x.shape[2], -1, 2))
    x_out = torch.view_as_real(x_out * freqs).flatten(2)
    return x_out.to(x.dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        dtype = x.dtype
        return self.norm(x.float()).to(dtype) * self.weight


class AttentionModule(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads
        
    def forward(self, q, k, v):
        x = flash_attention(q=q, k=k, v=v, num_heads=self.num_heads)
        return x


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = (-1, -1)

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        
        self.attn = AttentionModule(self.num_heads)

    def forward(self, x, freqs):
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)
        q = rope_apply(q, freqs, self.num_heads)
        k = rope_apply(k, freqs, self.num_heads)
        x = self.attn(q, k, v)
        return self.o(x)


class CrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        self.attn = AttentionModule(self.num_heads)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(y))
        v = self.v(y)
        x = self.attn(q, k, v)
        return self.o(x)


class GateModule(nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self, x, gate, residual):
        return x + gate * residual

class DiTBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, ffn_dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim

        self.self_attn = SelfAttention(dim, num_heads, eps)
        self.cross_attn = CrossAttention(dim, num_heads, eps)
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim, eps=eps)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(
            approximate='tanh'), nn.Linear(ffn_dim, dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        self.gate = GateModule()

    def forward(self, x, context, t_mod, freqs):
        # msa: multi-head self-attention  mlp: multi-layer perceptron
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(6, dim=1)
        input_x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = self.gate(
            x,
            gate_msa,
            self.self_attn(input_x, freqs),
        )
        x = x + self.cross_attn(self.norm3(x), context)
        input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = self.gate(x, gate_mlp, self.ffn(input_x))
        return x


class Head(nn.Module):
    def __init__(self, dim: int, out_dim: int, patch_size: Tuple[int, int, int], eps: float):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size))
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, t_mod):
        shift, scale = (self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(2, dim=1)
        x = (self.head(self.norm(x) * (1 + scale) + shift))
        return x


class WanModel(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        in_dim: int,
        ffn_dim: int,
        out_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        patch_size: Tuple[int, int, int],
        num_heads: int,
        num_layers: int,
    ):
        super().__init__()
        self.dim = dim
        self.freq_dim = freq_dim
        self.patch_size = patch_size

        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim)
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))
        self.blocks = nn.ModuleList([
            DiTBlock(dim, num_heads, ffn_dim, eps)
            for _ in range(num_layers)
        ])
        self.head = Head(dim, out_dim, patch_size, eps)
        head_dim = dim // num_heads
        self.freqs = precompute_freqs_cis_3d(head_dim)
        
        # Trained C2R control architecture: 0c + F + A3 + D + E.
        self.dino_patch_adapter = DINO2WanLatentAdapter(
            Cd=768,
            C=dim,
            Td=81,
            T=21,
            gated_control=False,
            mlp_hidden_mult=4,
        )
        self.dino_fusion_bridge = WanControlFusionBridge(
            C=dim,
            hidden_mult=0.5,
            gated=False,
        )
        self.dino_strength = 1.0
        self.repeat_dino_in_blocks = 13
        self.dino_repeat_blocks_frac = 0.333
        self.dino_block_decay_min = 0.3

    def patchify(self, x: torch.Tensor):
        x = self.patch_embedding(x)
        grid_size = x.shape[2:]
        x = rearrange(x, 'b c f h w -> b (f h w) c').contiguous()
        return x, grid_size  # x, grid_size: (f, h, w)

    def unpatchify(self, x: torch.Tensor, grid_size: torch.Tensor):
        return rearrange(
            x, 'b (f h w) (x y z c) -> b c (f x) (h y) (w z)',
            f=grid_size[0], h=grid_size[1], w=grid_size[2], 
            x=self.patch_size[0], y=self.patch_size[1], z=self.patch_size[2]
        )

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
    ):
        t = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timestep))
        t_mod = self.time_projection(t).unflatten(1, (6, self.dim))
        context = self.text_embedding(context)

        x, (f, h, w) = self.patchify(x)
        
        freqs = torch.cat([
            self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)

        for block in self.blocks:
            x = block(x, context, t_mod, freqs)

        x = self.head(x, t)
        x = self.unpatchify(x, (f, h, w))
        return x

    @staticmethod
    def state_dict_converter():
        return WanModelStateDictConverter()
    
    
class WanModelStateDictConverter:
    def __init__(self):
        pass

    @staticmethod
    def _infer_num_layers(state_dict: dict) -> int:
        layer_ids = set()
        for name in state_dict.keys():
            if not name.startswith("blocks."):
                continue
            parts = name.split(".")
            if len(parts) > 1 and parts[1].isdigit():
                layer_ids.add(int(parts[1]))
        if not layer_ids:
            raise ValueError("Could not infer WAN DiT layer count from checkpoint.")
        return max(layer_ids) + 1

    @staticmethod
    def _infer_num_heads(dim: int) -> int:
        if dim == 5120:
            return 40
        raise ValueError(f"Unsupported WAN DiT hidden size: {dim}. Expected 5120 for the 14B release model.")

    @classmethod
    def _infer_release_config(cls, state_dict: dict) -> dict:
        patch_weight = state_dict.get("patch_embedding.weight")
        if patch_weight is None:
            raise ValueError("Checkpoint is missing `patch_embedding.weight`.")

        dim = int(patch_weight.shape[0])
        in_dim = int(patch_weight.shape[1])
        if in_dim != 16:
            raise ValueError(
                f"This C2R release expects WAN video latent input channels in_dim=16, but got in_dim={in_dim}."
            )

        ffn_weight = state_dict.get("blocks.0.ffn.0.weight")
        if ffn_weight is None:
            raise ValueError("Checkpoint is missing `blocks.0.ffn.0.weight`.")

        time_weight = state_dict.get("time_embedding.0.weight")
        text_weight = state_dict.get("text_embedding.0.weight")
        if time_weight is None or text_weight is None:
            raise ValueError("Checkpoint is missing time/text embedding weights.")

        return {
            "patch_size": [1, 2, 2],
            "in_dim": in_dim,
            "dim": dim,
            "ffn_dim": int(ffn_weight.shape[0]),
            "freq_dim": int(time_weight.shape[1]),
            "text_dim": int(text_weight.shape[1]),
            "out_dim": 16,
            "num_heads": cls._infer_num_heads(dim),
            "num_layers": cls._infer_num_layers(state_dict),
            "eps": 1e-6,
        }

    def from_diffusers(self, state_dict):
        if any(name.startswith("transformer.") for name in state_dict):
            state_dict = {
                name.split("transformer.", 1)[1] if name.startswith("transformer.") else name: param
                for name, param in state_dict.items()
            }

        rename_dict = {
            "blocks.0.attn1.norm_k.weight": "blocks.0.self_attn.norm_k.weight",
            "blocks.0.attn1.norm_q.weight": "blocks.0.self_attn.norm_q.weight",
            "blocks.0.attn1.to_k.bias": "blocks.0.self_attn.k.bias",
            "blocks.0.attn1.to_k.weight": "blocks.0.self_attn.k.weight",
            "blocks.0.attn1.to_out.0.bias": "blocks.0.self_attn.o.bias",
            "blocks.0.attn1.to_out.0.weight": "blocks.0.self_attn.o.weight",
            "blocks.0.attn1.to_q.bias": "blocks.0.self_attn.q.bias",
            "blocks.0.attn1.to_q.weight": "blocks.0.self_attn.q.weight",
            "blocks.0.attn1.to_v.bias": "blocks.0.self_attn.v.bias",
            "blocks.0.attn1.to_v.weight": "blocks.0.self_attn.v.weight",
            "blocks.0.attn2.norm_k.weight": "blocks.0.cross_attn.norm_k.weight",
            "blocks.0.attn2.norm_q.weight": "blocks.0.cross_attn.norm_q.weight",
            "blocks.0.attn2.to_k.bias": "blocks.0.cross_attn.k.bias",
            "blocks.0.attn2.to_k.weight": "blocks.0.cross_attn.k.weight",
            "blocks.0.attn2.to_out.0.bias": "blocks.0.cross_attn.o.bias",
            "blocks.0.attn2.to_out.0.weight": "blocks.0.cross_attn.o.weight",
            "blocks.0.attn2.to_q.bias": "blocks.0.cross_attn.q.bias",
            "blocks.0.attn2.to_q.weight": "blocks.0.cross_attn.q.weight",
            "blocks.0.attn2.to_v.bias": "blocks.0.cross_attn.v.bias",
            "blocks.0.attn2.to_v.weight": "blocks.0.cross_attn.v.weight",
            "blocks.0.ffn.net.0.proj.bias": "blocks.0.ffn.0.bias",
            "blocks.0.ffn.net.0.proj.weight": "blocks.0.ffn.0.weight",
            "blocks.0.ffn.net.2.bias": "blocks.0.ffn.2.bias",
            "blocks.0.ffn.net.2.weight": "blocks.0.ffn.2.weight",
            "blocks.0.norm2.bias": "blocks.0.norm3.bias",
            "blocks.0.norm2.weight": "blocks.0.norm3.weight",
            "blocks.0.scale_shift_table": "blocks.0.modulation",
            "condition_embedder.text_embedder.linear_1.bias": "text_embedding.0.bias",
            "condition_embedder.text_embedder.linear_1.weight": "text_embedding.0.weight",
            "condition_embedder.text_embedder.linear_2.bias": "text_embedding.2.bias",
            "condition_embedder.text_embedder.linear_2.weight": "text_embedding.2.weight",
            "condition_embedder.time_embedder.linear_1.bias": "time_embedding.0.bias",
            "condition_embedder.time_embedder.linear_1.weight": "time_embedding.0.weight",
            "condition_embedder.time_embedder.linear_2.bias": "time_embedding.2.bias",
            "condition_embedder.time_embedder.linear_2.weight": "time_embedding.2.weight",
            "condition_embedder.time_proj.bias": "time_projection.1.bias",
            "condition_embedder.time_proj.weight": "time_projection.1.weight",
            "patch_embedding.bias": "patch_embedding.bias",
            "patch_embedding.weight": "patch_embedding.weight",
            "scale_shift_table": "head.modulation",
            "proj_out.bias": "head.head.bias",
            "proj_out.weight": "head.head.weight",
        }
        state_dict_ = {}
        for name, param in state_dict.items():
            if name in rename_dict:
                state_dict_[rename_dict[name]] = param
            else:
                name_ = ".".join(name.split(".")[:1] + ["0"] + name.split(".")[2:])
                if name_ in rename_dict:
                    name_ = rename_dict[name_]
                    name_ = ".".join(name_.split(".")[:1] + [name.split(".")[1]] + name_.split(".")[2:])
                    state_dict_[name_] = param
        config = self._infer_release_config(state_dict_)
        return state_dict_, config
    
    def from_civitai(self, state_dict):
        state_dict = {name: param for name, param in state_dict.items() if not name.startswith("vace")}
        config = self._infer_release_config(state_dict)
        return state_dict, config
