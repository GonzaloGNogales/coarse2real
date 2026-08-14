import os
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn

from ..utils.checkpoint import init_weights_on_device, load_state_dict
from .wan_video_dit import WanModel
from .wan_video_text_encoder import WanTextEncoder
from .wan_video_vae import WanVideoVAE


MODEL_CLASS_REGISTRY = {
    "wan_video_dit": WanModel,
    "wan_video_text_encoder": WanTextEncoder,
    "wan_video_vae": WanVideoVAE,
}


@torch.no_grad()
def init_dino_adapter(
    adapter: nn.Module,
    device="cuda",
    gate_init: float = 0.01,
    proj_zero: bool = False,
):
    adapter.to_empty(device=device)
    reset_parameters = getattr(adapter, "reset_parameters", None)
    if callable(reset_parameters):
        reset_parameters()
        return

    for name, p in adapter.named_parameters():
        if name.endswith("gate"):
            p.fill_(gate_init)

    temporal = getattr(adapter, "temporal_downsample", None)
    if temporal is not None and hasattr(temporal, "conv"):
        temporal.conv.weight.zero_()
        temporal.conv.weight[:, 0, :, 0, 0].fill_(1.0 / 9.0)

    for module in adapter.modules():
        if isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    proj = getattr(adapter, "proj", None)
    if isinstance(proj, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        if proj_zero:
            proj.weight.zero_()
        else:
            nn.init.xavier_normal_(proj.weight, gain=1e-2)
        if proj.bias is not None:
            proj.bias.zero_()

    mlp = getattr(adapter, "mlp", None)
    if mlp is not None and isinstance(mlp, nn.Sequential):
        if len(mlp) >= 1 and isinstance(mlp[0], nn.Linear):
            nn.init.kaiming_normal_(mlp[0].weight, nonlinearity="relu")
            if mlp[0].bias is not None:
                nn.init.zeros_(mlp[0].bias)
        if len(mlp) >= 1 and isinstance(mlp[-1], nn.Linear):
            nn.init.zeros_(mlp[-1].weight)
            if mlp[-1].bias is not None:
                nn.init.zeros_(mlp[-1].bias)


def _normalize_file_paths(file_path) -> list[str]:
    if isinstance(file_path, (str, os.PathLike)):
        paths = [str(file_path)]
    elif isinstance(file_path, Sequence):
        paths = [str(path) for path in file_path]
    else:
        raise TypeError(f"Unsupported model path container: {type(file_path)}")

    if not paths:
        raise ValueError("Empty model path list.")
    for path in paths:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Model file not found: {path}")
    return paths


def _merge_state_dict(paths: list[str]) -> dict:
    merged = {}
    for path in paths:
        merged.update(load_state_dict(path))
    return merged


def _dedup_preserve_order(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


def _resolve_model_resource_candidates(
    model_name: str,
    model_resource: str,
    paths: list[str],
    state_dict: dict,
) -> list[str]:
    if model_resource and model_resource != "auto":
        return [model_resource]

    file_names = [Path(path).name.lower() for path in paths]

    if model_name == "wan_video_dit":
        candidates: list[str] = []

        civitai_anchors = {
            "blocks.0.ffn.0.weight",
            "time_embedding.0.weight",
            "text_embedding.0.weight",
        }
        diffusers_anchors = {
            "blocks.0.ffn.net.0.proj.weight",
            "condition_embedder.time_embedder.linear_1.weight",
            "condition_embedder.text_embedder.linear_1.weight",
        }
        transformer_prefix_diffusers_anchors = {
            f"transformer.{key}" for key in diffusers_anchors
        }

        state_dict_keys = set(state_dict.keys())
        if state_dict_keys & civitai_anchors:
            candidates.append("civitai")
        if state_dict_keys & diffusers_anchors:
            candidates.append("diffusers")
        if state_dict_keys & transformer_prefix_diffusers_anchors:
            candidates.append("diffusers")

        if any("diffusion_pytorch_model" in name for name in file_names):
            candidates.append("diffusers")

        candidates.extend(["civitai", "diffusers"])
        return _dedup_preserve_order(candidates)

    if model_name in {"wan_video_text_encoder"} and any("diffusion_pytorch_model" in name for name in file_names):
        return ["diffusers", "civitai"]
    return ["civitai"]


def _convert_state_dict(model_class, model_resource: str, state_dict: dict):
    converter = model_class.state_dict_converter()
    if model_resource == "civitai":
        converted = converter.from_civitai(state_dict)
    elif model_resource == "diffusers":
        if not hasattr(converter, "from_diffusers"):
            raise ValueError(f"{model_class.__name__} does not support diffusers checkpoints.")
        converted = converter.from_diffusers(state_dict)
    else:
        raise ValueError(f"Unsupported model resource: {model_resource}")

    if isinstance(converted, tuple):
        return converted
    return converted, {}


def _instantiate_and_load_model(model_class, state_dict: dict, model_resource: str, torch_dtype, device, verbose: bool = True):
    model_state_dict, extra_kwargs = _convert_state_dict(model_class, model_resource, state_dict)
    effective_dtype = torch.float32 if extra_kwargs.get("upcast_to_float32", False) else torch_dtype

    with init_weights_on_device():
        model = model_class(**extra_kwargs)
    if hasattr(model, "eval"):
        model = model.eval()

    control_modules = (
        getattr(model, "dino_patch_adapter", None),
        getattr(model, "dino_fusion_bridge", None),
    )
    for control_module in control_modules:
        if control_module is None:
            continue
        init_dino_adapter(
            adapter=control_module,
            device=device,
            proj_zero=False,
        )
        control_module.to(dtype=effective_dtype)

    missing, unexpected = model.load_state_dict(model_state_dict, assign=True, strict=False)
    expected_control_only_missing = bool(missing) and all(
        name.startswith(("dino_patch_adapter.", "dino_fusion_bridge."))
        for name in missing
    )
    if verbose and missing and not expected_control_only_missing:
        print("missing:", missing[:5], "...")
    if verbose and unexpected:
        print("unexpected:", unexpected[:5], "...")
    return model.to(dtype=effective_dtype, device=device)


class ModelLoader:
    def __init__(
        self,
        torch_dtype=torch.float16,
        device="cuda",
        verbose: bool = True,
    ):
        self.torch_dtype = torch_dtype
        self.device = device
        self.verbose = verbose
        self.model = []
        self.model_path = []
        self.model_name = []

    def _log(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def load_model(
        self,
        file_path,
        model_name: str,
        model_resource: str = "auto",
        device=None,
        torch_dtype=None,
    ):
        if device is None:
            device = self.device
        if torch_dtype is None:
            torch_dtype = self.torch_dtype

        if model_name not in MODEL_CLASS_REGISTRY:
            raise ValueError(f"Unsupported model_name '{model_name}'. Expected one of: {sorted(MODEL_CLASS_REGISTRY)}")

        paths = _normalize_file_paths(file_path)
        state_dict = _merge_state_dict(paths)
        resolved_resources = _resolve_model_resource_candidates(model_name, model_resource, paths, state_dict)
        model_class = MODEL_CLASS_REGISTRY[model_name]

        errors: list[str] = []
        model = None
        resolved_resource = None
        for candidate_resource in resolved_resources:
            self._log(f"Loading {model_name} from: {paths} ({candidate_resource})")
            try:
                model = _instantiate_and_load_model(
                    model_class=model_class,
                    state_dict=state_dict,
                    model_resource=candidate_resource,
                    torch_dtype=torch_dtype,
                    device=device,
                    verbose=self.verbose,
                )
                resolved_resource = candidate_resource
                break
            except Exception as exc:
                errors.append(f"{candidate_resource}: {exc}")
                if len(resolved_resources) > 1:
                    self._log(f"    {candidate_resource} load failed, trying the next compatible layout...")

        if model is None or resolved_resource is None:
            details = " | ".join(errors) if errors else "Unknown model loading failure."
            raise ValueError(
                f"Failed to load {model_name} from {paths}. "
                f"Tried resources {resolved_resources}. Details: {details}"
            )

        self.model.append(model)
        self.model_path.append(paths if len(paths) > 1 else paths[0])
        self.model_name.append(model_name)
        self._log(f"    Loaded model: {model_name}.")

    def fetch_model(self, model_name, file_path=None):
        fetched_models = []
        fetched_model_paths = []
        for model, model_path, model_name_ in zip(self.model, self.model_path, self.model_name):
            if file_path is not None and file_path != model_path:
                continue
            if model_name == model_name_:
                fetched_models.append(model)
                fetched_model_paths.append(model_path)

        if len(fetched_models) == 0:
            self._log(f"No {model_name} models available.")
            return None

        if len(fetched_models) == 1:
            self._log(f"Using {model_name} from {fetched_model_paths[0]}.")
        else:
            self._log(f"More than one {model_name} model is loaded: {fetched_model_paths}. Using the first one.")

        return fetched_models[0]

    def to(self, device):
        for model in self.model:
            model.to(device)
