import torch
import torch.distributed as dist

from ..control.dino_control_module import (
    apply_initial_dino_fusion,
    dino_repeat_block_count,
    dino_repeat_scale,
)
from ..models.wan_video_dit import sinusoidal_embedding_1d


def _get_sequence_parallel_rank() -> int:
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def _get_sequence_parallel_world_size() -> int:
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def _all_gather_along_sequence(x: torch.Tensor) -> torch.Tensor:
    world_size = _get_sequence_parallel_world_size()
    if world_size == 1:
        return x

    if hasattr(dist, "all_gather_into_tensor"):
        input_tensor = x.movedim(1, 0).contiguous()
        output_shape = (world_size * input_tensor.shape[0],) + input_tensor.shape[1:]
        gathered = torch.empty(output_shape, dtype=x.dtype, device=x.device)
        dist.all_gather_into_tensor(gathered, input_tensor)
        gathered = gathered.view(world_size, input_tensor.shape[0], *input_tensor.shape[1:])
        permute_order = [2, 0, 1] + list(range(3, gathered.dim()))
        gathered = gathered.permute(*permute_order).contiguous()
        return gathered.view(x.shape[0], world_size * x.shape[1], *x.shape[2:])

    gathered = [torch.empty_like(x) for _ in range(world_size)]
    dist.all_gather(gathered, x.contiguous())
    return torch.cat(gathered, dim=1)


def _pad_sequence_freqs(freqs: torch.Tensor, target_len: int) -> torch.Tensor:
    seq_len, dim1, dim2 = freqs.shape
    pad_size = target_len - seq_len
    if pad_size <= 0:
        return freqs
    padding_tensor = torch.ones(
        pad_size,
        dim1,
        dim2,
        dtype=freqs.dtype,
        device=freqs.device,
    )
    return torch.cat([freqs, padding_tensor], dim=0)


def _local_sequence_freqs(freqs: torch.Tensor, seq_len_per_rank: int) -> torch.Tensor:
    start = _get_sequence_parallel_rank() * seq_len_per_rank
    return freqs.narrow(0, start, seq_len_per_rank)


def _resolve_dino_condition_latents(self, dino_patch_features: torch.Tensor | None) -> torch.Tensor | None:
    if dino_patch_features is None:
        return None
    if dino_patch_features.ndim == 3:
        return dino_patch_features
    return self.dino_patch_adapter(dino_patch_features)


def _usp_dit_forward_impl(
    self,
    x: torch.Tensor,
    timestep: torch.Tensor,
    context: torch.Tensor,
    dino_patch_features: torch.Tensor = None,
):
    t = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep))
    t_mod = self.time_projection(t).unflatten(1, (6, self.dim))
    context = self.text_embedding(context)

    dino_latents = _resolve_dino_condition_latents(self, dino_patch_features)

    x, (f, h, w) = self.patchify(x)
    if dino_latents is not None:
        x = apply_initial_dino_fusion(self, x, dino_latents)
    repeat_ctrl = dino_latents

    if x.shape[0] != context.shape[0]:
        x = torch.cat([x] * context.shape[0], dim=0)

    freqs = torch.cat(
        [
            self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
        ],
        dim=-1,
    ).reshape(f * h * w, 1, -1).to(x.device)

    original_seq_len = x.shape[1]
    sp_world_size = _get_sequence_parallel_world_size()
    padded_seq_len = ((original_seq_len + sp_world_size - 1) // sp_world_size) * sp_world_size
    if padded_seq_len != original_seq_len:
        x = torch.cat(
            [x, x.new_zeros(x.shape[0], padded_seq_len - original_seq_len, x.shape[2])],
            dim=1,
        )
        if repeat_ctrl is not None:
            repeat_ctrl = torch.cat(
                [
                    repeat_ctrl,
                    repeat_ctrl.new_zeros(
                        repeat_ctrl.shape[0],
                        padded_seq_len - original_seq_len,
                        repeat_ctrl.shape[2],
                    ),
                ],
                dim=1,
            )
        freqs = _pad_sequence_freqs(freqs, padded_seq_len)

    seq_len_per_rank = padded_seq_len // sp_world_size
    local_freqs = _local_sequence_freqs(freqs, seq_len_per_rank)
    x = torch.chunk(x, sp_world_size, dim=1)[_get_sequence_parallel_rank()]
    if repeat_ctrl is not None:
        repeat_ctrl = torch.chunk(
            repeat_ctrl,
            sp_world_size,
            dim=1,
        )[_get_sequence_parallel_rank()]

    repeat_blocks = dino_repeat_block_count(self) if repeat_ctrl is not None else 0
    dino_strength = float(getattr(self, "dino_strength", 1.0))
    for block_id, block in enumerate(self.blocks):
        if block_id < repeat_blocks:
            block_scale = dino_repeat_scale(self, block_id, repeat_blocks)
            x = x + dino_strength * block_scale * repeat_ctrl
        x = block(x, context, t_mod, local_freqs)

    x = self.head(x, t)
    x = _all_gather_along_sequence(x)
    x = x[:, :original_seq_len]
    return self.unpatchify(x, (f, h, w))


try:
    _compile_disable = torch.compiler.disable
except AttributeError:
    _compile_disable = None

if _compile_disable is None:
    try:
        import torch._dynamo as _torch_dynamo

        usp_dit_forward = _torch_dynamo.disable(_usp_dit_forward_impl)
    except Exception:
        usp_dit_forward = _usp_dit_forward_impl
else:
    usp_dit_forward = _compile_disable(_usp_dit_forward_impl)
