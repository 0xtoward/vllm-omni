"""Load-time W8A16 for MiniCPM-o 4.5's Stage0 Thinker.

The challenge launcher always points vLLM at the organizer's original BF16
checkpoint.  This module therefore does not ship replacement weights.  It
creates a tiny model-directory overlay whose files are symlinks to that
checkpoint and whose ``config.json`` adds an Ascend quantization description.
The selected Thinker linear weights are quantized once, after loading, and are
then consumed by ``npu_weight_quant_batchmatmul``.  Talker, Token2Wav, LM head,
embeddings, vision and audio modules remain in their original floating dtype.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path

import torch

_REGISTERED = False
_QUANT_RE = re.compile(
    r"^llm\.model\.layers\.\d+\."
    r"(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)\.weight$"
)


def build_quantization_description(model_dir: str | os.PathLike[str]) -> dict[str, str]:
    """Build the same complete description used by the validated W8A16 run."""
    root = Path(model_dir)
    index_path = root / "model.safetensors.index.json"
    weight_map = json.loads(index_path.read_text())["weight_map"]
    description: dict[str, str] = {}
    for name in weight_map:
        if name.endswith(".weight"):
            description[name] = "W8A16_RUNTIME" if _QUANT_RE.match(name) else "FLOAT"

    # vLLM asks for fused projection names and for stage-wrapper-prefixed names.
    # Emit both rather than relying on a missing-key fallback: ModelSlim treats
    # an absent description entry as an error.
    for name, scheme in list(description.items()):
        match = re.match(r"^(.*\.layers\.\d+\.)self_attn\.q_proj\.weight$", name)
        if match:
            description[match.group(1) + "self_attn.qkv_proj.weight"] = scheme
        match = re.match(r"^(.*\.layers\.\d+\.)mlp\.gate_proj\.weight$", name)
        if match:
            description[match.group(1) + "mlp.gate_up_proj.weight"] = scheme

    for name, scheme in list(description.items()):
        if name.startswith("llm."):
            description["thinker.llm." + name[len("llm.") :]] = scheme
        elif name.startswith("tts."):
            description["talker.tts_obj." + name[len("tts.") :]] = scheme
    return description


def prepare_model_overlay(model_dir: str | os.PathLike[str]) -> str:
    """Return a read-through model directory with the W8 description injected.

    Non-local paths, non-4.5 checkpoints, already-quantized checkpoints and an
    explicit opt-out are returned unchanged.
    """
    source = Path(model_dir).resolve()
    if os.environ.get("VLLM_OMNI_MINICPMO45_STAGE0_RUNTIME_W8A16", "0") == "0":
        return str(source)
    config_path = source / "config.json"
    index_path = source / "model.safetensors.index.json"
    if not config_path.is_file() or not index_path.is_file():
        return str(source)
    config = json.loads(config_path.read_text())
    if str(config.get("version", "")) != "4.5" or config.get("quantization_config"):
        return str(source)

    description = build_quantization_description(source)
    fingerprint = hashlib.sha256(
        (str(source) + "\0" + config_path.read_text()).encode()
    ).hexdigest()[:16]
    overlay = Path(tempfile.gettempdir()) / f"minicpmo45-w8a16-{fingerprint}"
    overlay.mkdir(mode=0o755, parents=True, exist_ok=True)
    for item in source.iterdir():
        if item.name == "config.json":
            continue
        target = overlay / item.name
        if not target.exists() and not target.is_symlink():
            target.symlink_to(item)
    config["quantization_config"] = description
    temporary = overlay / ".config.json.tmp"
    temporary.write_text(json.dumps(config, ensure_ascii=False, indent=1) + "\n")
    temporary.replace(overlay / "config.json")
    os.environ["_VLLM_OMNI_MINICPMO45_W8A16_OVERLAY"] = str(overlay)
    return str(overlay)


def register() -> None:
    """Register the runtime scheme in every stage process."""
    global _REGISTERED
    if _REGISTERED:
        return
    try:
        import torch_npu
        from vllm_ascend.quantization.methods.base import AscendLinearScheme
        from vllm_ascend.quantization.methods.registry import register_scheme
        from vllm_ascend.utils import maybe_trans_nz
    except Exception:
        return

    try:

        @register_scheme("W8A16_RUNTIME", "linear")
        class MiniCPMO45RuntimeW8A16LinearMethod(AscendLinearScheme):
            def get_weight(self, input_size, output_size, params_dtype=torch.bfloat16):
                return {
                    "weight": torch.empty(output_size, input_size, dtype=params_dtype)
                }

            def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
                weight_float = layer.weight.data.to(torch.float32)
                params_dtype = layer.weight.dtype
                scale = (weight_float.abs().amax(dim=1, keepdim=True) / 127.0).clamp(
                    min=1e-8
                )
                scale_store = scale.to(params_dtype)
                quantized = torch.round(
                    weight_float / scale_store.to(torch.float32)
                ).clamp(-127, 127)
                packed = quantized.to(torch.int8).transpose(0, 1).contiguous()
                layer.weight = torch.nn.Parameter(
                    maybe_trans_nz(packed), requires_grad=False
                )
                layer.weight_scale = torch.nn.Parameter(
                    scale_store.flatten(), requires_grad=False
                )
                layer.weight_offset = torch.nn.Parameter(
                    torch.zeros_like(scale_store).flatten(), requires_grad=False
                )

            def apply(self, layer, x, bias=None, tp_rank=0):
                if (
                    bias is not None
                    and x.dtype == torch.bfloat16
                    and bias.dtype != torch.float32
                ):
                    bias = bias.to(torch.float32)
                return torch_npu.npu_weight_quant_batchmatmul(
                    x=x,
                    weight=layer.weight,
                    antiquant_scale=layer.weight_scale,
                    antiquant_offset=layer.weight_offset,
                    bias=bias,
                )

        _REGISTERED = True
    except Exception:
        # Registration is process-global and another plugin import may have
        # installed the same scheme first.
        _REGISTERED = True
