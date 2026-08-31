from __future__ import annotations

import ast
from pathlib import Path

TARGET = (
    Path(__file__).parents[3]
    / "vllm_omni/model_executor/models/minicpmo_4_5/minicpmo_4_5_omni_tts.py"
)


def _method_source(method_name: str) -> str:
    source = TARGET.read_text()
    tree = ast.parse(source)
    model = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "MiniCPMO45OmniTTSForConditionalGeneration"
    )
    method = next(
        node
        for node in model.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )
    return ast.get_source_segment(source, method) or ""


def test_eos_batch_history_generation_is_initialized_and_advanced() -> None:
    preprocess = _method_source("preprocess")
    commit = _method_source("_commit_codec_prefix")

    assert '"history_version": 0' in preprocess
    assert (
        'state["history_version"] = int(state.get("history_version", 0)) + 1'
        in commit
    )
