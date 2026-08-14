import os
import inspect
import time
import tempfile
import hashlib
import types
from collections import OrderedDict
from datetime import timedelta
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import torch
from einops import reduce
from PIL import Image
from tqdm import tqdm

from ..control.apg_utils import APGMomentum, apg_delta_x0, flow_pred_to_x0, x0_to_flow_pred
from ..control.dino_control_module import (
    DINOFeaturesExtractor,
    apply_initial_dino_fusion,
    dino_repeat_block_count,
    dino_repeat_scale,
)
from ..models import ModelLoader
from ..models.wan_video_dit import WanModel, sinusoidal_embedding_1d
from ..models.wan_video_text_encoder import WanTextEncoder
from ..models.wan_video_vae import WanVideoVAE
from ..prompters import WanPrompter
from ..schedulers.flow_match import FlowMatchScheduler


@dataclass
class ModelConfig:
    path: Union[str, list[str]] = None
    model_name: Optional[str] = None
    model_resource: str = "auto"
    offload_device: Optional[Union[str, torch.device]] = None
    offload_dtype: Optional[torch.dtype] = None


class WanVideoPipeline(torch.nn.Module):
    def __init__(self, device="cuda", torch_dtype=torch.bfloat16):
        super().__init__()
        self.device = device
        self.torch_dtype = torch_dtype

        # Size constraints for WAN video latents.
        self.height_division_factor = 16
        self.width_division_factor = 16
        self.time_division_factor = 4
        self.time_division_remainder = 1

        # Emergency VRAM fallback settings.
        self.vram_management_enabled = False
        self.vram_buffer_gb = 1.0

        self.scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
        self.prompter = WanPrompter()
        self.text_encoder: WanTextEncoder = None
        self.dit: WanModel = None
        self.vae: WanVideoVAE = None
        self.dino_features_extractor = None

        self.denoise_step = wan_video_denoise_step
        self.use_unified_sequence_parallel = False
        self._usp_attention_backend = "gather"
        self._prompt_embedding_cache = OrderedDict()
        self._control_condition_cache = OrderedDict()
        self._prompt_embedding_cache_limit = 128
        self._control_condition_cache_limit = 2
        self._prompt_cache_encode_batch_size = 1

    def to(self, *args, **kwargs):
        device, dtype, _, _ = torch._C._nn._parse_to(*args, **kwargs)
        if device is not None:
            self.device = device
        if dtype is not None:
            self.torch_dtype = dtype
        self._prompt_embedding_cache.clear()
        self._control_condition_cache.clear()
        super().to(*args, **kwargs)
        return self

    def check_resize_height_width(self, height, width, num_frames=None):
        if height % self.height_division_factor != 0:
            height = (height + self.height_division_factor - 1) // self.height_division_factor * self.height_division_factor
            print(f"height % {self.height_division_factor} != 0. Rounded up to {height}.")
        if width % self.width_division_factor != 0:
            width = (width + self.width_division_factor - 1) // self.width_division_factor * self.width_division_factor
            print(f"width % {self.width_division_factor} != 0. Rounded up to {width}.")

        if num_frames is None:
            return height, width

        if num_frames % self.time_division_factor != self.time_division_remainder:
            num_frames = (num_frames + self.time_division_factor - 1) // self.time_division_factor * self.time_division_factor + self.time_division_remainder
            print(f"num_frames % {self.time_division_factor} != {self.time_division_remainder}. Rounded up to {num_frames}.")
        return height, width, num_frames

    def vae_output_to_image(self, vae_output, pattern="B C H W", min_value=-1, max_value=1):
        if pattern != "H W C":
            vae_output = reduce(vae_output, f"{pattern} -> H W C", reduction="mean")
        image = ((vae_output - min_value) * (255 / (max_value - min_value))).clip(0, 255)
        image = image.to(device="cpu", dtype=torch.uint8)
        return Image.fromarray(image.numpy())

    def vae_output_to_video(self, vae_output, pattern="B C T H W", min_value=-1, max_value=1):
        if pattern != "T H W C":
            vae_output = reduce(vae_output, f"{pattern} -> T H W C", reduction="mean")
        return [self.vae_output_to_image(frame, pattern="H W C", min_value=min_value, max_value=max_value) for frame in vae_output]

    def generate_noise(self, shape, seed=None, rand_device="cpu", rand_torch_dtype=torch.float32, device=None, torch_dtype=None):
        generator = None if seed is None else torch.Generator(rand_device).manual_seed(seed)
        noise = torch.randn(shape, generator=generator, device=rand_device, dtype=rand_torch_dtype)
        noise = noise.to(dtype=torch_dtype or self.torch_dtype, device=device or self.device)
        return noise

    def enable_vram_management(self, enabled: bool = True, vram_buffer_gb: float = 1.0):
        self.vram_management_enabled = enabled
        self.vram_buffer_gb = vram_buffer_gb
        return self

    def _offload_models_except(self, keep: set[str]):
        if not self.vram_management_enabled:
            return
        if not torch.cuda.is_available():
            return
        if self.vram_buffer_gb is not None and self.vram_buffer_gb > 0:
            free_gb = torch.cuda.mem_get_info()[0] / (1024 ** 3)
            # Emergency-only behavior: offload only when free VRAM is below the threshold.
            if free_gb >= self.vram_buffer_gb:
                return
        managed = ("text_encoder", "dit", "vae", "dino_features_extractor")
        for name in managed:
            model = getattr(self, name, None)
            if model is None:
                continue
            if name in keep:
                model.to(self.device)
            else:
                model.to("cpu")
        torch.cuda.empty_cache()

    def initialize_usp(self):
        self._disable_torch_compile_for_usp()

        import torch.distributed as dist

        os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
        os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")
        if "TORCH_FR_BUFFER_SIZE" not in os.environ and "TORCH_NCCL_TRACE_BUFFER_SIZE" in os.environ:
            os.environ["TORCH_FR_BUFFER_SIZE"] = os.environ["TORCH_NCCL_TRACE_BUFFER_SIZE"]
        os.environ.setdefault("TORCH_FR_BUFFER_SIZE", "20000")
        os.environ.setdefault("TORCH_NCCL_DUMP_ON_TIMEOUT", "1")
        os.environ.setdefault("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", "120")

        required = ("RANK", "WORLD_SIZE", "LOCAL_RANK")
        if any(name not in os.environ for name in required):
            raise RuntimeError("USP mode requires torchrun environment variables: RANK, WORLD_SIZE, LOCAL_RANK.")

        local_rank = int(os.environ.get("LOCAL_RANK", str(int(os.environ.get("RANK", "0")))))
        device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu")
        if torch.cuda.is_available():
            # Set the local device before NCCL init so collectives are associated with the correct GPU.
            torch.cuda.set_device(local_rank)

        if not dist.is_initialized():
            init_kwargs = {
                "backend": "nccl",
                "init_method": "env://",
                "timeout": timedelta(seconds=int(os.environ.get("C2R_USP_PG_TIMEOUT_SEC", "180"))),
            }
            try:
                if "device_id" in inspect.signature(dist.init_process_group).parameters and torch.cuda.is_available():
                    init_kwargs["device_id"] = device
            except (TypeError, ValueError):
                pass
            dist.init_process_group(**init_kwargs)
        os.environ["RANK"] = str(dist.get_rank())
        os.environ["WORLD_SIZE"] = str(dist.get_world_size())
        os.environ["LOCAL_RANK"] = str(local_rank)
        if dist.get_rank() == 0:
            print(
                "Initializing custom USP over torch.distributed with "
                f"sequence_parallel_degree={dist.get_world_size()}."
            )
            print(
                "NCCL transport overrides: "
                f"NCCL_P2P_DISABLE={os.environ.get('NCCL_P2P_DISABLE', '0')} "
                f"NCCL_SHM_DISABLE={os.environ.get('NCCL_SHM_DISABLE', '0')}."
            )
            print("Warming up NCCL collectives for USP...")

        if torch.cuda.is_available():
            warmup_reduce = torch.zeros(1, device=device)
            dist.all_reduce(warmup_reduce)

            warmup_gather = torch.zeros((1, 4, 8), device=device, dtype=self.torch_dtype)
            gathered = [torch.empty_like(warmup_gather) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered, warmup_gather.contiguous())
            torch.cuda.synchronize(device)

        if dist.get_rank() == 0:
            print("USP NCCL warm-up complete.")

    def _disable_torch_compile_for_usp(self):
        # This custom WAN USP path has shown first-step hangs on some clusters when
        # TorchDynamo is active. Prefer correctness and startup reliability here.
        os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
        try:
            import torch._dynamo as dynamo  # noqa: WPS433

            if hasattr(dynamo, "config") and hasattr(dynamo.config, "disable"):
                dynamo.config.disable = True
            if hasattr(dynamo, "reset"):
                dynamo.reset()
        except Exception:
            pass

    def enable_usp(self):
        from ..distributed import get_usp_backend

        usp_attn_forward, usp_dit_forward = get_usp_backend(self._usp_attention_backend)
        if self._usp_attention_backend == "ring":
            print(
                "Enabling experimental USP ring attention backend. "
                "Set `usp_attention_backend` back to `gather` to return to the stable path."
            )
            if os.environ.get("NCCL_P2P_DISABLE", "0") == "1":
                print(
                    "Note: NCCL safe mode is active (NCCL_P2P_DISABLE=1). "
                    "The experimental ring backend will still run, but its speedup may be limited on this node."
                )
        elif self._usp_attention_backend != "gather":
            raise ValueError(
                f"Unsupported USP backend '{self._usp_attention_backend}'. "
                "Use 'gather' or 'ring'."
            )

        for block in self.dit.blocks:
            block.self_attn.forward = types.MethodType(usp_attn_forward, block.self_attn)
        self.dit.forward = types.MethodType(usp_dit_forward, self.dit)
        self.use_unified_sequence_parallel = True

    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = "cuda",
        model_configs: Optional[list[ModelConfig]] = None,
        tokenizer_config: Optional[ModelConfig] = None,
        use_usp: bool = False,
        usp_attention_backend: str = "gather",
        dino_model_path: Optional[str] = None,
        verbose: bool = True,
    ):
        if model_configs is None:
            model_configs = []
        if tokenizer_config is None:
            raise ValueError("`tokenizer_config.path` is required in this inference-only build.")

        usp_attention_backend = (usp_attention_backend or "gather").lower().strip()
        if usp_attention_backend not in {"gather", "ring"}:
            raise ValueError(
                f"Unsupported USP backend '{usp_attention_backend}'. "
                "Use 'gather' or 'ring'."
            )

        pipe = WanVideoPipeline(device=device, torch_dtype=torch_dtype)
        pipe._usp_attention_backend = usp_attention_backend
        if use_usp:
            pipe.initialize_usp()
        model_loader = ModelLoader(verbose=verbose)

        for model_config in model_configs:
            if model_config.path is None:
                raise ValueError("Each model config must define `path` explicitly in this inference-only build.")
            if model_config.model_name is None:
                raise ValueError("Each model config must define `model_name` explicitly in this inference-only build.")
            model_loader.load_model(
                model_config.path,
                model_name=model_config.model_name,
                model_resource=model_config.model_resource,
                device=model_config.offload_device or device,
                torch_dtype=model_config.offload_dtype or torch_dtype,
            )

        pipe.text_encoder = model_loader.fetch_model("wan_video_text_encoder")
        pipe.dit = model_loader.fetch_model("wan_video_dit")
        pipe.vae = model_loader.fetch_model("wan_video_vae")
        if pipe.text_encoder is None or pipe.dit is None or pipe.vae is None:
            raise ValueError("Pipeline requires these weights: wan_video_dit, wan_video_text_encoder, wan_video_vae.")

        rank = 0
        dist_is_initialized = False
        try:
            import torch.distributed as dist

            dist_is_initialized = dist.is_initialized()
            if dist_is_initialized:
                rank = dist.get_rank()
        except Exception:
            dist = None

        run_namespace = (
            os.environ.get("TORCHELASTIC_RUN_ID")
            or os.environ.get("SLURM_JOB_ID")
            or os.environ.get("JOB_ID")
            or f"{os.environ.get('MASTER_ADDR', 'local')}_{os.environ.get('MASTER_PORT', '0')}"
        )
        run_key = hashlib.sha1(run_namespace.encode("utf-8")).hexdigest()[:12]
        runtime_root = Path(os.environ.get("C2R_RUNTIME_DIR", tempfile.gettempdir()))
        dino_ready_dir = runtime_root / "c2r_runtime" / run_key
        dino_ready_dir.mkdir(parents=True, exist_ok=True)
        dino_ready_file = dino_ready_dir / "dino_backbone_ready"

        if rank == 0 and verbose:
            print("Initializing DINO control backbone...")

        if dist_is_initialized and rank != 0:
            deadline = time.time() + 1800
            while not dino_ready_file.exists():
                if time.time() > deadline:
                    raise TimeoutError(
                        "Timed out waiting for rank 0 to initialize the DINO control backbone cache."
                    )
                time.sleep(1.0)

        pipe.dino_features_extractor = DINOFeaturesExtractor(
            model_id=dino_model_path or "facebook/dinov3-vitb16-pretrain-lvd1689m",
            device=device,
            dtype=torch_dtype,
            local_files_only=dist_is_initialized and rank != 0,
        )

        if rank == 0:
            try:
                dino_ready_file.write_text("ready\n", encoding="utf-8")
            except OSError:
                pass

        if rank == 0 and verbose:
            print("DINO control backbone ready.")
        if tokenizer_config.path is None:
            raise ValueError("Tokenizer config must provide a local `path`.")
        pipe.prompter.fetch_models(pipe.text_encoder)
        if verbose:
            print(f"Loading tokenizer from {tokenizer_config.path}...")
        pipe.prompter.fetch_tokenizer(tokenizer_config.path)
        if verbose:
            print("Tokenizer ready.")
        if use_usp:
            pipe.enable_usp()
        return pipe

    def _collect_control_frames(self, control_video, num_frames: int) -> list[Image.Image]:
        frames = [control_video[index] for index in range(min(len(control_video), num_frames))]
        if len(frames) < num_frames:
            raise ValueError(f"control_video has {len(frames)} frames but num_frames={num_frames}.")
        return frames

    @staticmethod
    def _store_lru_value(cache: OrderedDict, cache_key, value, max_size: int) -> None:
        if max_size <= 0:
            return
        cache.pop(cache_key, None)
        cache[cache_key] = value
        while len(cache) > max_size:
            cache.popitem(last=False)

    def _encode_prompt_cached(self, prompt: str, positive: bool) -> torch.Tensor:
        if self._prompt_embedding_cache_limit <= 0:
            return self.prompter.encode_prompt(prompt, positive=positive, device=self.device)

        cache_key = (prompt, positive)
        cached = self._prompt_embedding_cache.get(cache_key)
        if cached is not None:
            self._prompt_embedding_cache.move_to_end(cache_key)
            return cached.to(device=self.device, non_blocking=True)

        encoded = self.prompter.encode_prompt(prompt, positive=positive, device=self.device)
        self._store_lru_value(
            self._prompt_embedding_cache,
            cache_key,
            encoded.detach().to("cpu"),
            self._prompt_embedding_cache_limit,
        )
        return encoded

    def prime_prompt_cache(self, prompts: list[str], negative_prompt: Optional[str] = "") -> int:
        if self._prompt_embedding_cache_limit <= 0:
            return 0

        uncached_prompts: list[str] = []
        seen_prompts: set[str] = set()
        for prompt in prompts:
            if prompt in seen_prompts:
                continue
            seen_prompts.add(prompt)
            if (prompt, True) not in self._prompt_embedding_cache:
                uncached_prompts.append(prompt)

        reserve_slots = 0
        if negative_prompt is not None and (negative_prompt, False) not in self._prompt_embedding_cache:
            reserve_slots = 1
        preprime_budget = max(self._prompt_embedding_cache_limit - reserve_slots, 0)
        prompts_to_prime = uncached_prompts[:preprime_budget]

        if prompts_to_prime:
            batch_size = max(1, self._prompt_cache_encode_batch_size)
            for start in range(0, len(prompts_to_prime), batch_size):
                prompt_batch = prompts_to_prime[start : start + batch_size]
                batch_embeddings = self.prompter.encode_prompt(prompt_batch, positive=True, device=self.device)
                for index, prompt in enumerate(prompt_batch):
                    self._store_lru_value(
                        self._prompt_embedding_cache,
                        (prompt, True),
                        batch_embeddings[index : index + 1].detach().to("cpu").contiguous(),
                        self._prompt_embedding_cache_limit,
                    )
                del batch_embeddings
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        if negative_prompt is not None:
            self._encode_prompt_cached(negative_prompt, positive=False)
        return len(prompts_to_prime)

    def _control_condition_cache_key(
        self,
        control_video,
        height: int,
        width: int,
        num_frames: int,
    ) -> tuple[object, ...]:
        control_source = (
            getattr(control_video, "video_file", None)
            or getattr(control_video, "name", None)
            or id(control_video)
        )
        return (
            str(control_source),
            height,
            width,
            num_frames,
            str(self.device),
            str(self.torch_dtype),
        )

    def _prepare_control_condition_tokens(
        self,
        control_video,
        height: int,
        width: int,
        num_frames: int,
    ) -> torch.Tensor:
        cache_key = self._control_condition_cache_key(
            control_video=control_video,
            height=height,
            width=width,
            num_frames=num_frames,
        )
        cached = self._control_condition_cache.get(cache_key)
        if cached is not None:
            self._control_condition_cache.move_to_end(cache_key)
            return cached.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)

        self._offload_models_except({"dino_features_extractor"})
        control_frames = self._collect_control_frames(control_video, num_frames)
        self.dino_features_extractor.set_resolution(height, width)
        dino_patch_features = self.dino_features_extractor.get_features(control_frames)
        dino_patch_features = dino_patch_features.to(device=self.device, dtype=self.torch_dtype)

        self._offload_models_except({"dit"})
        control_condition_tokens = self.dit.dino_patch_adapter(dino_patch_features)

        if self._control_condition_cache_limit > 0:
            self._store_lru_value(
                self._control_condition_cache,
                cache_key,
                control_condition_tokens.detach().to("cpu"),
                self._control_condition_cache_limit,
            )
        return control_condition_tokens

    def _decode_latents(
        self,
        latents: torch.Tensor,
        tiled: bool,
        tile_size: tuple[int, int],
        tile_stride: tuple[int, int],
        vae_decode_mode: Optional[str],
    ) -> torch.Tensor:
        if vae_decode_mode is None:
            vae_decode_mode = "tiled" if tiled else "single"

        decode_mode = vae_decode_mode.lower().strip()
        if decode_mode not in {"auto", "tiled", "single"}:
            raise ValueError(
                f"Unsupported vae_decode_mode='{vae_decode_mode}'. Use 'auto', 'tiled', or 'single'."
            )

        if decode_mode == "tiled":
            return self.vae.decode(
                latents,
                device=self.device,
                tiled=True,
                tile_size=tile_size,
                tile_stride=tile_stride,
            )
        if decode_mode == "single":
            return self.vae.decode(latents, device=self.device, tiled=False)

        try:
            return self.vae.decode(latents, device=self.device, tiled=False)
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("Untiled VAE decode ran out of memory. Falling back to tiled decode.")
            return self.vae.decode(
                latents,
                device=self.device,
                tiled=True,
                tile_size=tile_size,
                tile_stride=tile_stride,
            )

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        negative_prompt: Optional[str] = "",
        control_video=None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        num_inference_steps: int = 50,
        sigma_shift: float = 5.0,
        control_video_scale: float = 1.0,
        text_scale: float = 10.0,
        guidance_mode: str = "cfg",
        apg_eta: float = 0.0,
        apg_momentum: float = -0.5,
        apg_norm_threshold: Optional[float] = None,
        apg_eps: float = 1e-6,
        apg_eta_control_video: Optional[float] = None,
        apg_eta_text: Optional[float] = None,
        apg_momentum_control_video: Optional[float] = None,
        apg_momentum_text: Optional[float] = None,
        apg_norm_threshold_control_video: Optional[float] = None,
        apg_norm_threshold_text: Optional[float] = None,
        control_video_guidance_end: float = 1.0,
        tiled: bool = True,
        tile_size: tuple[int, int] = (30, 52),
        tile_stride: tuple[int, int] = (15, 26),
        vae_decode_mode: Optional[str] = None,
        progress_bar_cmd=tqdm,
    ):
        if control_video is None:
            raise ValueError("C2R inference requires `control_video`.")

        guidance_mode = guidance_mode.lower().strip()
        if guidance_mode not in {"cfg", "apg"}:
            raise ValueError(f"Unsupported guidance_mode='{guidance_mode}'. Use 'cfg' or 'apg'.")

        height, width, num_frames = self.check_resize_height_width(height, width, num_frames)
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=1.0, shift=sigma_shift)

        self._offload_models_except({"text_encoder"})
        context_pos = self._encode_prompt_cached(prompt, positive=True)
        context_neg = self._encode_prompt_cached(negative_prompt, positive=False)

        if self.dino_features_extractor is None:
            raise ValueError("DINO control is not initialized in this pipeline.")
        control_condition_tokens = self._prepare_control_condition_tokens(
            control_video=control_video,
            height=height,
            width=width,
            num_frames=num_frames,
        )

        self._offload_models_except({"dit"})
        latents = self.generate_noise(
            (1, 16, (num_frames - 1) // 4 + 1, height // 8, width // 8),
            seed=seed,
            rand_device=rand_device,
        )

        apg_norm = 0.0 if apg_norm_threshold is None else float(apg_norm_threshold)
        apg_norm_control_video = apg_norm if apg_norm_threshold_control_video is None else float(apg_norm_threshold_control_video)
        apg_norm_text = apg_norm if apg_norm_threshold_text is None else float(apg_norm_threshold_text)
        apg_eta_control_video_value = apg_eta if apg_eta_control_video is None else float(apg_eta_control_video)
        apg_eta_text_value = apg_eta if apg_eta_text is None else float(apg_eta_text)
        apg_momentum_control_video_value = apg_momentum if apg_momentum_control_video is None else float(apg_momentum_control_video)
        apg_momentum_text_value = apg_momentum if apg_momentum_text is None else float(apg_momentum_text)
        total_steps = max(len(self.scheduler.timesteps), 1)
        if guidance_mode == "apg":
            apg_control_video_buffer = APGMomentum(momentum=apg_momentum_control_video_value)
            apg_text_buffer = APGMomentum(momentum=apg_momentum_text_value)

        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            timestep_model = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)
            current_progress = progress_id / total_steps
            current_control_video_condition_tokens = (
                control_condition_tokens if current_progress < control_video_guidance_end else None
            )

            if guidance_mode == "cfg":
                no_control_video_no_text = self.denoise_step(
                    dit=self.dit,
                    latents=latents,
                    timestep=timestep_model,
                    context=context_neg,
                    dino_patch_features=None,
                    use_unified_sequence_parallel=self.use_unified_sequence_parallel,
                )
                if current_control_video_condition_tokens is None:
                    control_video_no_text = no_control_video_no_text
                else:
                    control_video_no_text = self.denoise_step(
                        dit=self.dit,
                        latents=latents,
                        timestep=timestep_model,
                        context=context_neg,
                        dino_patch_features=current_control_video_condition_tokens,
                        use_unified_sequence_parallel=self.use_unified_sequence_parallel,
                    )
                control_video_text = self.denoise_step(
                    dit=self.dit,
                    latents=latents,
                    timestep=timestep_model,
                    context=context_pos,
                    dino_patch_features=current_control_video_condition_tokens,
                    use_unified_sequence_parallel=self.use_unified_sequence_parallel,
                )
                noise_pred = (
                    no_control_video_no_text
                    + control_video_scale * (control_video_no_text - no_control_video_no_text)
                    + text_scale * (control_video_text - control_video_no_text)
                )
            else:
                x_sigma = latents
                sigma = self.scheduler.sigmas[progress_id].to(device=self.device, dtype=self.torch_dtype)

                v_00 = self.denoise_step(
                    dit=self.dit,
                    latents=latents,
                    timestep=timestep_model,
                    context=context_neg,
                    dino_patch_features=None,
                    use_unified_sequence_parallel=self.use_unified_sequence_parallel,
                )

                if current_control_video_condition_tokens is None:
                    v_C0 = v_00
                else:
                    v_C0 = self.denoise_step(
                        dit=self.dit,
                        latents=latents,
                        timestep=timestep_model,
                        context=context_neg,
                        dino_patch_features=current_control_video_condition_tokens,
                        use_unified_sequence_parallel=self.use_unified_sequence_parallel,
                    )

                v_CT = self.denoise_step(
                    dit=self.dit,
                    latents=latents,
                    timestep=timestep_model,
                    context=context_pos,
                    dino_patch_features=current_control_video_condition_tokens,
                    use_unified_sequence_parallel=self.use_unified_sequence_parallel,
                )

                x0_00 = flow_pred_to_x0(v_00, x_sigma, sigma)
                x0_C0 = flow_pred_to_x0(v_C0, x_sigma, sigma)
                x0_CT = flow_pred_to_x0(v_CT, x_sigma, sigma)

                dC = apg_delta_x0(
                    x0_uncond=x0_00,
                    x0_cond=x0_C0,
                    scale=control_video_scale,
                    eta=apg_eta_control_video_value,
                    norm_threshold=apg_norm_control_video,
                    momentum=apg_control_video_buffer,
                    eps=apg_eps,
                )
                dT = apg_delta_x0(
                    x0_uncond=x0_C0,
                    x0_cond=x0_CT,
                    scale=text_scale,
                    eta=apg_eta_text_value,
                    norm_threshold=apg_norm_text,
                    momentum=apg_text_buffer,
                    eps=apg_eps,
                )

                x0_guided = x0_00 + dC + dT
                noise_pred = x0_to_flow_pred(x0_guided, x_sigma, sigma, eps=apg_eps)

            latents = self.scheduler.step(noise_pred, self.scheduler.timesteps[progress_id], latents)

        decode_on_this_rank = True
        if self.use_unified_sequence_parallel:
            try:
                import torch.distributed as dist

                if dist.is_initialized() and dist.get_rank() != 0:
                    decode_on_this_rank = False
            except Exception:
                pass

        if not decode_on_this_rank:
            return None

        self._offload_models_except({"vae"})
        video = self._decode_latents(
            latents,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
            vae_decode_mode=vae_decode_mode,
        )
        return self.vae_output_to_video(video)


def _resolve_dino_condition_latents(
    dit: WanModel,
    dino_patch_features: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    if dino_patch_features is None:
        return None
    if dino_patch_features.ndim == 3:
        return dino_patch_features
    return dit.dino_patch_adapter(dino_patch_features)


def wan_video_denoise_step(
    dit: WanModel,
    latents: torch.Tensor,
    timestep: torch.Tensor,
    context: torch.Tensor,
    dino_patch_features: Optional[torch.Tensor] = None,
    use_unified_sequence_parallel: bool = False,
):
    if use_unified_sequence_parallel:
        return dit(latents, timestep, context, dino_patch_features=dino_patch_features)

    t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
    t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))
    context = dit.text_embedding(context)

    x = latents
    dino_latents = _resolve_dino_condition_latents(dit, dino_patch_features)

    x, (f, h, w) = dit.patchify(x)
    if dino_latents is not None:
        x = apply_initial_dino_fusion(dit, x, dino_latents)

    if x.shape[0] != context.shape[0]:
        x = torch.cat([x] * context.shape[0], dim=0)

    freqs = torch.cat(
        [
            dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
        ],
        dim=-1,
    ).reshape(f * h * w, 1, -1).to(x.device)

    repeat_blocks = dino_repeat_block_count(dit) if dino_latents is not None else 0
    dino_strength = float(getattr(dit, "dino_strength", 1.0))
    for block_id, block in enumerate(dit.blocks):
        if block_id < repeat_blocks:
            block_scale = dino_repeat_scale(dit, block_id, repeat_blocks)
            x = x + dino_strength * block_scale * dino_latents
        x = block(x, context, t_mod, freqs)

    x = dit.head(x, t)
    return dit.unpatchify(x, (f, h, w))
