import glob
import inspect
import json
import os
from collections.abc import Iterable
from dataclasses import dataclass, fields
from pathlib import Path
from typing import List

import torch

from c2r import load_state_dict
from c2r.models.wan_video_dit import configure_attention_backend
from c2r.pipelines.wan_video_pipeline import ModelConfig, WanVideoPipeline
try:
    from inference.control_video_preprocess import (
        ControlVideoPrepConfig,
        prepare_control_videos as preprocess_control_videos,
    )
except ModuleNotFoundError:
    from control_video_preprocess import (  # type: ignore
        ControlVideoPrepConfig,
        prepare_control_videos as preprocess_control_videos,
    )


DEFAULT_NEGATIVE_PROMPT = ("色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走")

@dataclass
class InferenceConfig:
    base_model_dir: str | None = None
    dit_path: str | list[str] | None = None
    text_encoder_path: str | None = None
    vae_path: str | None = None
    tokenizer_path: str | None = None
    dino_adapter_path: str | None = None
    dino_model_path: str | None = None
    prompts_file: str = "inference/c2r-prompts.txt"
    control_videos_dir: str | None = None
    output_dir: str = "outputs/wan_dino"
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT
    prompt_enhancement_mode: str = "off"  # "off" or "enhanced"; C2R only
    prompt_enhancement_num_frames: int = 12
    prompt_enhancement_camera_num_frames: int = 6
    prompt_enhancement_vlm_model_id: str = "Qwen/Qwen3-VL-8B-Instruct"
    prompt_enhancement_llm_model_id: str = "Qwen/Qwen3-8B"
    prompt_enhancement_cache_dir: str = "inference/prompt_enhancement_cache"
    height: int = 480
    width: int = 832
    num_frames: int = 81
    steps: int = 50
    seed: int = 0
    fps: int = 16
    video_encoding_quality: int = 5
    vae_decode_mode: str = "auto"  # "auto", "tiled", or "single"
    prompt_embedding_cache_limit: int = 128
    control_condition_cache_limit: int = 2
    prompt_cache_autoprime_threshold: int | None = 256
    usp_attention_backend: str = "gather"  # "gather" or "ring"
    preprocess_control_videos: bool = True
    control_video_preprocess_dir: str = "inference/preprocessed_control_videos"
    control_video_target_fps: int = 16
    control_short_video_strategy: str = "pad"  # "pad" or "error"
    guidance_mode: str = "apg"  # "cfg" or "apg"
    control_video_scale: float = 1.0
    text_scale: float = 10.0
    apg_eta: float = 0.0
    apg_momentum: float = -0.5
    apg_norm_threshold: float | None = None
    apg_eps: float = 1e-6
    apg_eta_control_video: float | None = None
    apg_eta_text: float | None = None
    apg_momentum_control_video: float | None = None
    apg_momentum_text: float | None = None
    apg_norm_threshold_control_video: float | None = None
    apg_norm_threshold_text: float | None = None
    control_video_guidance_end: float = 1.0
    dtype: str = "bfloat16"
    parallel_mode: str = "single-gpu"  # "single-gpu", "dp", or "usp"
    enable_vram_management: bool = False
    vram_buffer_gb: float = 1.0

    def normalized_prompt_enhancement_mode(self) -> str:
        mode = self.prompt_enhancement_mode.lower().strip()
        if mode not in {"off", "enhanced"}:
            raise ValueError(
                f"Unsupported prompt_enhancement_mode='{self.prompt_enhancement_mode}'. "
                "Use 'off' or 'enhanced'."
            )
        return mode

    def normalized_parallel_mode(self) -> str:
        mode = self.parallel_mode.lower().strip()
        if mode not in {"single-gpu", "dp", "usp"}:
            raise ValueError(
                f"Unsupported parallel_mode='{self.parallel_mode}'. "
                "Use 'single-gpu', 'dp', or 'usp'."
            )
        return mode

    def normalized_vae_decode_mode(self) -> str:
        mode = self.vae_decode_mode.lower().strip()
        if mode not in {"auto", "tiled", "single"}:
            raise ValueError(
                f"Unsupported vae_decode_mode='{self.vae_decode_mode}'. "
                "Use 'auto', 'tiled', or 'single'."
            )
        return mode

    def normalized_usp_attention_backend(self) -> str:
        mode = self.usp_attention_backend.lower().strip()
        if mode not in {"gather", "ring"}:
            raise ValueError(
                f"Unsupported usp_attention_backend='{self.usp_attention_backend}'. "
                "Use 'gather' or 'ring'."
            )
        return mode

    def validate_runtime_settings(self) -> None:
        if not self.control_videos_dir or not str(self.control_videos_dir).strip():
            raise ValueError("`control_videos_dir` is required for C2R inference.")
        if not self.dino_adapter_path or not str(self.dino_adapter_path).strip():
            raise ValueError("`dino_adapter_path` is required for C2R inference.")
        if self.prompt_embedding_cache_limit < 0:
            raise ValueError("`prompt_embedding_cache_limit` must be >= 0.")
        if self.control_condition_cache_limit < 0:
            raise ValueError("`control_condition_cache_limit` must be >= 0.")
        if self.prompt_cache_autoprime_threshold is not None and self.prompt_cache_autoprime_threshold < 0:
            raise ValueError("`prompt_cache_autoprime_threshold` must be >= 0 or null.")
        if self.prompt_enhancement_num_frames <= 0:
            raise ValueError("`prompt_enhancement_num_frames` must be > 0.")
        if self.prompt_enhancement_camera_num_frames <= 0:
            raise ValueError("`prompt_enhancement_camera_num_frames` must be > 0.")


def load_config(config_path: str) -> InferenceConfig:
    payload = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a JSON object: {config_path}")
    allowed = {field.name for field in fields(InferenceConfig)}
    filtered = {key: value for key, value in payload.items() if key in allowed}
    cfg = InferenceConfig(**filtered)
    if isinstance(cfg.dino_model_path, str):
        cfg.dino_model_path = cfg.dino_model_path.strip()
        if cfg.dino_model_path.lower() in {"", "none", "null"}:
            cfg.dino_model_path = None
    cfg.normalized_prompt_enhancement_mode()
    cfg.normalized_parallel_mode()
    cfg.normalized_vae_decode_mode()
    cfg.normalized_usp_attention_backend()
    cfg.validate_runtime_settings()
    return cfg


def dtype_from_name(name: str) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    raise ValueError(f"Unsupported dtype: {name}")


def resolve_model_paths(cfg: InferenceConfig) -> tuple[List[str], str, str, Path, str]:
    base_dir = Path(cfg.base_model_dir) if cfg.base_model_dir else Path("models") / "wan"

    def _expand_dit_input(value: str | list[str] | None) -> list[str]:
        if value is None:
            return []
        items: Iterable[str] = value if isinstance(value, list) else [value]
        files: list[str] = []
        for item in items:
            has_glob = any(char in item for char in "*?[]")
            path = Path(item)
            if has_glob:
                files.extend(sorted(str(Path(match)) for match in glob.glob(item) if Path(match).is_file()))
            elif path.is_file():
                files.append(str(path))
            elif path.is_dir():
                files.extend(sorted(str(p) for p in path.rglob("*.safetensors") if p.is_file()))
            else:
                raise FileNotFoundError(f"DiT path not found: {item}")
        # deterministic dedup while preserving order
        return list(dict.fromkeys(files))

    def _default_wan_dit_paths() -> list[str]:
        paths = sorted(path for path in glob.glob(str(base_dir / "diffusion_pytorch_model*.safetensors")) if Path(path).is_file())
        if not paths:
            paths = sorted(str(path) for path in base_dir.glob("*WAN*weights*.safetensors"))
        return paths

    dit_candidates: list[str] = []
    if cfg.dit_path is not None:
        try:
            dit_candidates = _expand_dit_input(cfg.dit_path)
        except FileNotFoundError as exc:
            print(f"Warning: {exc}. Trying WAN default DiT weights under {base_dir} instead.")
        if dit_candidates:
            dit_source = "config.dit_path"
        else:
            dit_candidates = _default_wan_dit_paths()
            dit_source = f"default base_model_dir ({base_dir}) [fallback from dit_path]"
    else:
        dit_candidates = _default_wan_dit_paths()
        dit_source = f"default base_model_dir ({base_dir})"

    if not dit_candidates:
        raise FileNotFoundError(
            "Could not resolve WAN DiT weights. "
            f"Set `dit_path` in config or add DiT safetensors under {base_dir}."
        )

    text_encoder_path = cfg.text_encoder_path or str(base_dir / "models_t5_umt5-xxl-enc-bf16.pth")
    vae_path = cfg.vae_path or str(base_dir / "Wan2.1_VAE.pth")
    if not Path(text_encoder_path).is_file():
        raise FileNotFoundError(f"Text encoder file not found: {text_encoder_path}")
    if not Path(vae_path).is_file():
        raise FileNotFoundError(f"VAE file not found: {vae_path}")

    return dit_candidates, text_encoder_path, vae_path, base_dir, dit_source


def resolve_tokenizer_config(cfg: InferenceConfig, base_dir: Path) -> ModelConfig:
    def _looks_like_tokenizer_dir(path: Path) -> bool:
        expected_files = (
            "tokenizer.json",
            "tokenizer_config.json",
            "spiece.model",
            "special_tokens_map.json",
            "config.json",
        )
        return path.is_dir() and any((path / file_name).exists() for file_name in expected_files)

    def _resolve_local_tokenizer_dir(path: Path) -> Path | None:
        if _looks_like_tokenizer_dir(path):
            return path

        preferred_children = [path / "umt5-xxl"]
        for child in preferred_children:
            if _looks_like_tokenizer_dir(child):
                return child

        if path.is_dir():
            for child in sorted(candidate for candidate in path.iterdir() if candidate.is_dir()):
                if _looks_like_tokenizer_dir(child):
                    return child
        return None

    if cfg.tokenizer_path is not None:
        tok_dir = Path(cfg.tokenizer_path)
        if not tok_dir.exists():
            raise FileNotFoundError(f"Tokenizer path not found: {cfg.tokenizer_path}")
        resolved = _resolve_local_tokenizer_dir(tok_dir)
        if resolved is None:
            raise FileNotFoundError(
                "Tokenizer files were not found under the configured tokenizer_path. "
                f"Expected files like tokenizer.json or tokenizer_config.json under {cfg.tokenizer_path}."
            )
        return ModelConfig(path=str(resolved))

    local_candidates = [
        base_dir / "google" / "umt5-xxl",
        base_dir / "google",
        base_dir / "tokenizer",
        base_dir / "tokenizer_2",
    ]
    for candidate in local_candidates:
        resolved = _resolve_local_tokenizer_dir(candidate)
        if resolved is not None:
            return ModelConfig(path=str(resolved))
    raise FileNotFoundError(
        "Tokenizer files were not found locally. "
        "Set `tokenizer_path` in config (or place files under <base_model_dir>/google/umt5-xxl)."
    )


def _load_trained_c2r_control(pipe: WanVideoPipeline, checkpoint_path: str) -> None:
    state_dict = load_state_dict(checkpoint_path)

    normalized_state_dict = {}
    for key, value in state_dict.items():
        normalized_key = key.removeprefix("pipe.dit.")
        normalized_state_dict[normalized_key] = value

    adapter_marker = "dino_patch_adapter."
    bridge_marker = "dino_fusion_bridge."
    adapter_state_dict = {
        key.removeprefix(adapter_marker): value
        for key, value in normalized_state_dict.items()
        if key.startswith(adapter_marker)
    }
    bridge_state_dict = {
        key.removeprefix(bridge_marker): value
        for key, value in normalized_state_dict.items()
        if key.startswith(bridge_marker)
    }
    unexpected_checkpoint_keys = sorted(
        key
        for key in normalized_state_dict
        if not key.startswith((adapter_marker, bridge_marker))
    )
    if unexpected_checkpoint_keys:
        raise ValueError(
            "Unexpected tensors in trained C2R checkpoint: "
            f"{unexpected_checkpoint_keys[:20]}"
        )
    if not adapter_state_dict or not bridge_state_dict:
        raise ValueError(
            "Checkpoint must contain both dino_patch_adapter and dino_fusion_bridge tensors."
        )

    adapter_missing, adapter_unexpected = pipe.dit.dino_patch_adapter.load_state_dict(
        adapter_state_dict,
        strict=False,
    )
    bridge_missing, bridge_unexpected = pipe.dit.dino_fusion_bridge.load_state_dict(
        bridge_state_dict,
        strict=False,
    )
    if set(adapter_missing) != {"gate"} or adapter_unexpected:
        raise ValueError(
            "DINO adapter checkpoint mismatch: "
            f"missing={adapter_missing}, unexpected={adapter_unexpected}"
        )
    if set(bridge_missing) != {"gate"} or bridge_unexpected:
        raise ValueError(
            "DINO fusion bridge checkpoint mismatch: "
            f"missing={bridge_missing}, unexpected={bridge_unexpected}"
        )


def build_pipeline(cfg: InferenceConfig, local_rank: int = 0, rank: int = 0) -> WanVideoPipeline:
    dtype = dtype_from_name(cfg.dtype)
    dit_paths, text_encoder_path, vae_path, base_dir, dit_source = resolve_model_paths(cfg)
    tokenizer_config = resolve_tokenizer_config(cfg, base_dir)
    parallel_mode = cfg.normalized_parallel_mode()
    use_usp = parallel_mode == "usp"

    if use_usp:
        device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
    else:
        device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)

    configure_attention_backend(device=device, verbose=(rank == 0))
    model_configs = [
        ModelConfig(
            path=dit_paths if len(dit_paths) > 1 else dit_paths[0],
            model_name="wan_video_dit",
            model_resource="auto",
        ),
        ModelConfig(path=text_encoder_path, model_name="wan_video_text_encoder", model_resource="civitai"),
        ModelConfig(path=vae_path, model_name="wan_video_vae", model_resource="civitai"),
    ]
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=dtype,
        device=device,
        model_configs=model_configs,
        tokenizer_config=tokenizer_config,
        use_usp=use_usp,
        usp_attention_backend=cfg.normalized_usp_attention_backend(),
        dino_model_path=cfg.dino_model_path,
        verbose=(rank == 0),
    )
    pipe._prompt_embedding_cache_limit = cfg.prompt_embedding_cache_limit
    pipe._control_condition_cache_limit = cfg.control_condition_cache_limit
    if rank == 0:
        print(f"Using DiT weights from {dit_source}:")
        for dit_path in dit_paths:
            print(f"  - {dit_path}")
        if use_usp:
            print(f"USP attention backend: {pipe._usp_attention_backend}")
    if cfg.enable_vram_management:
        pipe.enable_vram_management(enabled=True, vram_buffer_gb=cfg.vram_buffer_gb)

    if cfg.dino_adapter_path:
        _load_trained_c2r_control(pipe, cfg.dino_adapter_path)
        if rank == 0:
            print(f"Loaded trained C2R adapter and fusion bridge: {cfg.dino_adapter_path}")
    return pipe


def read_prompts(path: str) -> List[str]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    prompts = []
    for line in lines:
        value = line.strip()
        if not value or value.startswith("#"):
            continue
        prompts.append(value)
    if not prompts:
        raise ValueError(f"No usable prompts found in {path}")
    return prompts


def read_control_videos(control_videos_dir: str | None) -> List[Path]:
    def list_videos(folder: Path) -> List[Path]:
        video_exts = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
        return sorted(path for path in folder.rglob("*") if path.is_file() and path.suffix.lower() in video_exts)

    if not control_videos_dir:
        raise ValueError("Set `control_videos_dir` in config to a folder containing control videos.")

    folder = Path(control_videos_dir)
    if not folder.is_dir():
        raise FileNotFoundError(f"Control videos dir not found: {control_videos_dir}")

    control_video_paths = list_videos(folder)
    if not control_video_paths:
        raise ValueError(f"No supported control videos found in `{control_videos_dir}`.")
    return control_video_paths


def prepare_control_videos(cfg: InferenceConfig, rank: int = 0) -> List[Path]:
    control_video_paths = read_control_videos(cfg.control_videos_dir)
    if not cfg.preprocess_control_videos:
        if rank == 0:
            print(
                f"Control video preprocessing disabled; using {len(control_video_paths)} clip(s) directly "
                f"from {cfg.control_videos_dir}."
            )
        return control_video_paths

    prep_config = ControlVideoPrepConfig(
        output_dir=Path(cfg.control_video_preprocess_dir),
        target_fps=cfg.control_video_target_fps,
        target_num_frames=cfg.num_frames,
        short_video_strategy=cfg.control_short_video_strategy,
        video_encoding_quality=cfg.video_encoding_quality,
    )
    return preprocess_control_videos(control_video_paths, config=prep_config, rank=rank)


def build_tasks(
    prompts: List[str],
    control_video_paths: List[Path],
    prompt_overrides: dict[tuple[int, int], str] | None = None,
    control_video_descriptions: dict[int, dict] | None = None,
) -> List[dict]:
    if not control_video_paths:
        raise ValueError("C2R inference requires at least one control video.")

    tasks = []
    task_id = 0
    for control_video_id, control_video_path in enumerate(control_video_paths):
        for prompt_id, prompt in enumerate(prompts):
            enhanced_prompt = prompt_overrides.get((prompt_id, control_video_id), prompt) if prompt_overrides else prompt
            tasks.append(
                {
                    "task_id": task_id,
                    "prompt_id": prompt_id,
                    "control_video_id": control_video_id,
                    "prompt": enhanced_prompt,
                    "raw_prompt": prompt,
                    "control_video_path": str(control_video_path),
                    "control_video_description": (
                        control_video_descriptions.get(control_video_id)
                        if control_video_descriptions
                        else None
                    ),
                }
            )
            task_id += 1
    return tasks


def maybe_init_dist(parallel_mode: str) -> tuple[int, int, int]:
    import torch.distributed as dist

    mode = parallel_mode.lower().strip()
    if mode not in {"single-gpu", "dp", "usp"}:
        raise ValueError(
            f"Unsupported parallel_mode='{parallel_mode}'. Use 'single-gpu', 'dp', or 'usp'."
        )

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    if mode in {"dp", "usp"} and torch.cuda.is_available():
        # Bind the CUDA context early so NCCL does not need to guess the device later.
        torch.cuda.set_device(local_rank)
    if mode == "single-gpu":
        if world_size > 1:
            raise ValueError(
                "parallel_mode='single-gpu' cannot be used with WORLD_SIZE > 1. "
                "Use 'dp' or 'usp' for multi-GPU launches."
            )
        if dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
        return rank, world_size, local_rank

    if mode == "dp" and world_size > 1 and not dist.is_initialized():
        init_kwargs = {
            "backend": "nccl",
            "init_method": "env://",
        }
        try:
            if "device_id" in inspect.signature(dist.init_process_group).parameters and torch.cuda.is_available():
                init_kwargs["device_id"] = torch.device(f"cuda:{local_rank}")
        except (TypeError, ValueError):
            pass
        dist.init_process_group(**init_kwargs)
    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    return rank, world_size, local_rank
