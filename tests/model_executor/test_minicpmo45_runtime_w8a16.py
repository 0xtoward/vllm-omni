import json
import tempfile
from pathlib import Path

from vllm_omni.model_executor.layers.minicpmo45_runtime_w8a16 import (
    build_quantization_description,
    prepare_model_overlay,
)


def _fake_model(tmp_path: Path) -> Path:
    model = tmp_path / "MiniCPM-o-4_5"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"version": "4.5"}))
    weights = {
        "llm.model.layers.0.self_attn.q_proj.weight": "model.safetensors",
        "llm.model.layers.0.mlp.down_proj.weight": "model.safetensors",
        "tts.model.layers.0.self_attn.q_proj.weight": "model.safetensors",
        "llm.lm_head.weight": "model.safetensors",
    }
    (model / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weights})
    )
    (model / "model.safetensors").write_bytes(b"test")
    return model


def test_description_quantizes_only_thinker_linear_weights(tmp_path):
    model = _fake_model(tmp_path)
    desc = build_quantization_description(model)
    assert desc["llm.model.layers.0.self_attn.q_proj.weight"] == "W8A16_RUNTIME"
    assert desc["thinker.llm.model.layers.0.self_attn.qkv_proj.weight"] == "W8A16_RUNTIME"
    assert desc["talker.tts_obj.model.layers.0.self_attn.qkv_proj.weight"] == "FLOAT"
    assert desc["llm.lm_head.weight"] == "FLOAT"


def test_overlay_keeps_weights_external_and_injects_config(tmp_path, monkeypatch):
    model = _fake_model(tmp_path)
    overlay_root = tmp_path / "overlays"
    overlay_root.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(overlay_root))
    overlay = Path(prepare_model_overlay(model))
    assert overlay != model
    assert (overlay / "model.safetensors").is_symlink()
    config = json.loads((overlay / "config.json").read_text())
    assert config["quantization_config"][
        "llm.model.layers.0.mlp.down_proj.weight"
    ] == "W8A16_RUNTIME"
