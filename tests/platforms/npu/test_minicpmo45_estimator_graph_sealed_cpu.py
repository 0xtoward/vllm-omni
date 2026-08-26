# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import torch

from vllm_omni.model_executor.models.minicpmo_4_5.runtime_prompt_identity import (
    prompt_identity_from_f32_bytes,
)
from vllm_omni.model_executor.models.minicpmo_4_5.runtime_prompt_manifest import (
    RuntimePromptManifest,
)
from vllm_omni.platforms.npu.graph_tools import (
    CapturedDeviceGraph,
    EstimatorSemantic,
    GraphPhase,
    SealedNPUExactGraphRunner,
    tensor_signature,
)
from vllm_omni.platforms.npu.models import minicpmo_4_5_code2wav_graph as graph_impl

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _FunctionalReplay:
    def __init__(self, inputs, outputs, compute):
        self.inputs = inputs
        self.outputs = outputs
        self.compute = compute

    def replay(self):
        for target, value in zip(
            self.outputs,
            self.compute(*self.inputs),
            strict=True,
        ):
            target.copy_(value)


def _captured(inputs, compute, *, delta=0.0):
    static_inputs = tuple(value.detach().clone() for value in inputs)

    def captured_compute(*values):
        return tuple(value + delta for value in compute(*values))

    static_outputs = tuple(value.detach().clone() for value in captured_compute(*static_inputs))
    return CapturedDeviceGraph(
        graph=_FunctionalReplay(static_inputs, static_outputs, captured_compute),
        static_inputs=static_inputs,
        static_outputs=static_outputs,
    )


def _semantic(role="prompt_b1", *, cached=False):
    return EstimatorSemantic(
        role=role,
        request_batch=1,
        prompt_frames=302,
        codec_token_count=0 if role == "prompt_b1" else 28,
        last_chunk=role == "tail_b1",
        flush_encoder=False,
        cache_present=cached,
        attention_cache_length=302 if cached else 0,
        cfm_steps=3,
        codec_chunk_frames=25,
        left_context_frames=3,
    )


def _runner(*roles, mode="on", ephemeral_output_indices=()):
    runner = SealedNPUExactGraphRunner(
        mode=mode,
        expected_roles=roles,
        ephemeral_output_indices=ephemeral_output_indices,
    )
    runner._eligible = lambda inputs: True
    runner._capture = lambda inputs, compute: _captured(inputs, compute)
    return runner


def _run(runner, semantic, inputs, graph_compute, stock_compute=None):
    if stock_compute is None:

        def stock_compute():
            return graph_compute(*inputs)

    return runner.run(
        "unit",
        semantic,
        inputs,
        stock_compute=stock_compute,
        graph_compute=graph_compute,
    )


def test_tensor_signature_distinguishes_stride_offset_and_contiguity():
    base = torch.arange(24).reshape(4, 6)
    contiguous = base.clone()
    transposed = base.t()
    offset = base.reshape(-1)[1:13]

    assert tensor_signature(contiguous) != tensor_signature(transposed)
    assert tensor_signature(contiguous).contiguous
    assert not tensor_signature(transposed).contiguous
    assert tensor_signature(offset).storage_offset == 1


def test_graph_replay_accepts_same_contract_cache_views_at_different_offsets():
    runner = _runner("prompt_b1")
    backing = torch.arange(12.0)
    first = backing[1:5]
    second = backing[6:10]

    warm = _run(runner, _semantic(), (first,), lambda value: (value.square(),))
    runner.seal()
    replay = _run(runner, _semantic(), (second,), lambda value: (value.square(),))

    torch.testing.assert_close(warm[0], first.square())
    torch.testing.assert_close(replay[0], second.square())
    snapshot = runner.snapshot("offset-view")
    assert snapshot["resident_graphs"] == 1
    assert snapshot["sealed_misses"] == 0
    assert snapshot["replay_successes"] == 1


def test_runtime_only_is_eager_census_and_never_captures():
    runner = _runner("prompt_b1", mode="runtime_only")
    value = torch.tensor([1.0])
    output = _run(runner, _semantic(), (value,), lambda x: (x + 1,))

    torch.testing.assert_close(output[0], torch.tensor([2.0]))
    snapshot = runner.snapshot("ready")
    assert snapshot["runtime_only_eager_calls"] == 1
    assert snapshot["captures"] == 0
    assert snapshot["resident_graphs"] == 0
    assert snapshot["phase"] == GraphPhase.CENSUS.value


def test_warm_capture_shadows_stock_then_sealed_replays_changed_input_exactly():
    runner = _runner("prompt_b1")
    first = torch.tensor([1.0, 2.0])
    eager = _run(runner, _semantic(), (first,), lambda x: (x.square(),))
    runner.seal()
    second = torch.tensor([3.0, 4.0])
    replayed = _run(runner, _semantic(), (second,), lambda x: (x.square(),))

    torch.testing.assert_close(eager[0], first.square())
    torch.testing.assert_close(replayed[0], second.square())
    assert runner.snapshot("warm")["captures"] == 1
    assert runner.snapshot("warm")["shadow_replay_successes"] == 1
    assert runner.snapshot("warm")["replay_successes"] == 1


def test_every_warming_hit_returns_stock_and_uses_replay_only_as_shadow():
    runner = _runner("prompt_b1")
    stock_calls = []

    def graph_compute(value):
        return (value + 1,)

    def stock(value):
        def compute():
            stock_calls.append(value.item())
            return (value + 1,)

        return compute

    first = torch.tensor([2.0])
    second = torch.tensor([7.0])
    first_out = _run(
        runner,
        _semantic(),
        (first,),
        graph_compute,
        stock(first),
    )
    second_out = _run(
        runner,
        _semantic(),
        (second,),
        graph_compute,
        stock(second),
    )

    torch.testing.assert_close(first_out[0], torch.tensor([3.0]))
    torch.testing.assert_close(second_out[0], torch.tensor([8.0]))
    assert stock_calls == [2.0, 7.0]
    snapshot = runner.snapshot("warming")
    assert snapshot["eager_warm_calls"] == 2
    assert snapshot["shadow_replay_successes"] == 2
    assert snapshot["replay_successes"] == 0


def test_replay_outputs_are_request_owned_and_previous_output_is_stable():
    runner = _runner("prompt_b1")

    def compute(value):
        return (value + 7,)

    _run(runner, _semantic(), (torch.tensor([0.0]),), compute)
    runner.seal()
    first = _run(runner, _semantic(), (torch.tensor([1.0]),), compute)[0]
    first_copy = first.clone()
    second = _run(runner, _semantic(), (torch.tensor([9.0]),), compute)[0]

    assert first.data_ptr() != second.data_ptr()
    torch.testing.assert_close(first, first_copy)
    torch.testing.assert_close(second, torch.tensor([16.0]))


def test_certified_ephemeral_output_borrows_stride_but_cache_remains_owned():
    runner = _runner("prompt_b1", ephemeral_output_indices=(0,))

    def compute(value):
        return value.transpose(0, 1), value.square()

    first_input = torch.arange(6.0).reshape(2, 3)
    _run(runner, _semantic(), (first_input,), compute)
    runner.seal()
    first_estimate, first_cache = _run(
        runner,
        _semantic(),
        (first_input + 1,),
        compute,
    )
    first_cache_copy = first_cache.clone()
    second_estimate, second_cache = _run(
        runner,
        _semantic(),
        (first_input + 4,),
        compute,
    )

    assert first_estimate.data_ptr() == second_estimate.data_ptr()
    assert first_estimate.stride() == (1, 3)
    assert first_cache.data_ptr() != second_cache.data_ptr()
    torch.testing.assert_close(first_cache, first_cache_copy)
    torch.testing.assert_close(second_cache, (first_input + 4).square())
    assert runner.snapshot("sealed")["contract"]["ephemeral_output_indices"] == [0]


def test_biased_graph_compute_is_rejected_by_independent_stock_oracle():
    runner = _runner("prompt_b1")
    value = torch.tensor([1.0])

    with pytest.raises(RuntimeError, match="restart required"):
        _run(
            runner,
            _semantic(),
            (value,),
            lambda x: (x + 2,),
            stock_compute=lambda: (value + 1,),
        )
    snapshot = runner.snapshot("failed")
    assert snapshot["resident_graphs"] == 0
    assert snapshot["shadow_replay_failures"] == 1
    with pytest.raises(RuntimeError, match="poisoned"):
        next_value = torch.tensor([2.0])
        _run(runner, _semantic(), (next_value,), lambda x: (x + 1,))


def test_capture_failure_is_fatal_and_cannot_continue_eager():
    runner = _runner("prompt_b1")
    runner._capture = lambda inputs, compute: (_ for _ in ()).throw(RuntimeError("capture failed"))

    with pytest.raises(RuntimeError, match="restart required"):
        value = torch.tensor([1.0])
        _run(runner, _semantic(), (value,), lambda x: (x,))
    assert runner.snapshot("failed")["capture_failures"] == 1
    with pytest.raises(RuntimeError, match="poisoned"):
        next_value = torch.tensor([2.0])
        _run(runner, _semantic(), (next_value,), lambda x: (x,))


def test_seal_is_irreversible_and_unknown_signature_is_eager_miss():
    runner = _runner("prompt_b1")
    initial = torch.ones(2)
    _run(runner, _semantic(), (initial,), lambda x: (x + 1,))
    runner.seal()
    captures = runner.captures
    missed = torch.ones(3)
    graph_calls = []
    output = _run(
        runner,
        _semantic(),
        (missed,),
        lambda x: graph_calls.append(True) or (x - 100,),
        stock_compute=lambda: (missed + 2,),
    )

    torch.testing.assert_close(output[0], torch.full((3,), 3.0))
    assert runner.sealed_misses == 1
    assert runner.captures == captures
    assert len(runner._graphs) == 1
    assert graph_calls == []
    with pytest.raises(RuntimeError, match="only a warming runner"):
        runner.seal()


def test_tail_is_explicit_eager_not_sealed_miss():
    runner = _runner("prompt_b1")
    initial = torch.ones(2)
    _run(runner, _semantic(), (initial,), lambda x: (x,))
    runner.seal()
    value = torch.tensor([4.0])
    graph_calls = []
    output = _run(
        runner,
        _semantic("tail_b1", cached=True),
        (value,),
        lambda x: graph_calls.append(True) or (x - 100,),
        stock_compute=lambda: (value + 1,),
    )

    torch.testing.assert_close(output[0], torch.tensor([5.0]))
    assert runner.tail_eager_calls == 1
    assert runner.sealed_misses == 0
    assert graph_calls == []


def test_ineligible_batch_is_explicit_eager():
    runner = _runner("prompt_b1")
    value = torch.tensor([1.0])
    graph_calls = []
    output = _run(
        runner,
        _semantic("ineligible_batch"),
        (value,),
        lambda x: graph_calls.append(True) or (x - 100,),
        stock_compute=lambda: (value * 3,),
    )
    torch.testing.assert_close(output[0], torch.tensor([3.0]))
    assert runner.ineligible_eager_calls == 1
    assert runner.capture_requests == 0
    assert graph_calls == []


def test_second_signature_for_one_warm_role_fails_without_eviction():
    runner = _runner("prompt_b1")
    initial = torch.ones(2)
    _run(runner, _semantic(), (initial,), lambda x: (x,))
    with pytest.raises(RuntimeError, match="second signature"):
        changed = torch.ones(3)
        _run(runner, _semantic(), (changed,), lambda x: (x,))
    assert len(runner._graphs) == 1
    assert not hasattr(runner, "evict")


def test_seal_requires_exact_role_set():
    runner = _runner("prompt_b1", "steady_b1_p")
    value = torch.ones(2)
    _run(runner, _semantic(), (value,), lambda x: (x,))
    with pytest.raises(RuntimeError, match="cannot seal roles"):
        runner.seal()


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(None, "off"), (False, "off"), (True, "on"), ("runtime_only", "runtime_only")],
)
def test_mode_parser(raw, expected):
    extra = {} if raw is None else {graph_impl._MODE_KEY: raw}
    assert graph_impl._mode_from_extra(extra) == expected


def test_runtime_config_accepts_write_only_allow_internal_format(monkeypatch):
    class WriteOnlyConfig:
        def __init__(self):
            object.__setattr__(self, "writes", [])

        def __getattribute__(self, name):
            if name == "allow_internal_format":
                raise AttributeError(name)
            return object.__getattribute__(self, name)

        def __setattr__(self, name, value):
            if name == "allow_internal_format":
                self.writes.append(value)
                return
            object.__setattr__(self, name, value)

    config = WriteOnlyConfig()
    compile_modes = []
    fake_npu = SimpleNamespace(
        config=config,
        set_compile_mode=lambda **kwargs: compile_modes.append(kwargs),
    )
    monkeypatch.setattr(torch, "npu", fake_npu)

    graph_impl._configure_npu_graph_runtime()

    assert not hasattr(config, "allow_internal_format")
    assert config.writes == [False]
    assert compile_modes == [{"jit_compile": False}]


def test_runtime_prompt_roles_encode_prompt_and_attention_lengths():
    roles = graph_impl._roles_for_prompt_frames(417)
    assert roles == {
        "prompt_b1_pf417",
        "steady_b1_pf417_ac417",
        "steady_b1_pf417_ac467",
        "steady_b1_pf417_ac517",
    }
    assert graph_impl._steady_role_for_lengths(417, 468) == "ineligible_cache_pf417_ac468"


def test_runtime_prompt_manifest_is_hash_verified_and_model_default_only(tmp_path):
    metadata = tmp_path / "meta.lst"
    metadata.write_text("fixed rows\n")
    benchmark_sources = []
    for role in ("dataset_loader", "benchmark_patch", "benchmark_wrapper"):
        source_path = tmp_path / f"{role}.py"
        source_path.write_text(f"# {role}\n")
        benchmark_sources.append(
            {
                "role": role,
                "path": str(source_path),
                "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
            }
        )
    rows = []
    for index in range(1):
        path = tmp_path / f"ref-{index}.wav"
        path.write_bytes(b"RIFF" + bytes([index]) * 16)
        file_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        identity_sha = hashlib.sha256(f"identity-{index}".encode()).hexdigest()
        cache_key = hashlib.sha256(f"cache-{index}".encode()).hexdigest()
        rows.append(
            {
                "row_ordinal": index,
                "utterance_id": f"utt{index}",
                "ref_path": str(path),
                "ref_file_sha256": file_sha,
                "decoded_sample_rate": 24000,
                "decoded_samples": 16,
                "decoded_f32_sha256": hashlib.sha256(f"tensor-{index}".encode()).hexdigest(),
                "schema": "minicpmo45-runtime-prompt-identity-v1",
                "decoded_f32_payload_sha256": hashlib.sha256(f"payload-{index}".encode()).hexdigest(),
                "expected_cache_key": cache_key,
                "cache_key": cache_key,
                "runtime_prompt_identity_sha256": identity_sha,
                "canonical_wav_path": str(path),
                "canonical_wav_sha256": file_sha,
                "prompt_mel_sha256": hashlib.sha256(f"mel-{index}".encode()).hexdigest(),
                "prompt_frames": 302 + index,
            }
        )
        path.chmod(0o444)
    source = tmp_path / "source.json"
    source.write_text("{}")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "minicpmo45-runtime-prompt-manifest-v2",
                "sealed": True,
                "source_manifest_path": str(source),
                "source_manifest_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "dataset": {
                    "dataset_root": str(tmp_path),
                    "locale": "zh",
                    "metadata_path": str(metadata),
                    "metadata_sha256": hashlib.sha256(metadata.read_bytes()).hexdigest(),
                },
                "benchmark_sources": benchmark_sources,
                "rows": rows,
            }
        )
    )
    manifest.chmod(0o444)
    manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    specs = graph_impl._parse_prompt_wav_specs(
        {
            graph_impl._PROMPT_MANIFEST_MODE_KEY: "consumer",
            graph_impl._PROMPT_MANIFEST_KEY: str(manifest),
            graph_impl._PROMPT_MANIFEST_SHA_KEY: manifest_sha,
        }
    )
    assert len(specs) == 1
    manifest.chmod(0o644)
    with pytest.raises(RuntimeError, match="read-only"):
        graph_impl._parse_prompt_wav_specs(
            {
                graph_impl._PROMPT_MANIFEST_MODE_KEY: "consumer",
                graph_impl._PROMPT_MANIFEST_KEY: str(manifest),
                graph_impl._PROMPT_MANIFEST_SHA_KEY: manifest_sha,
            }
        )


def test_model_default_prompt_mode_uses_exact_model_asset(tmp_path):
    prompt = tmp_path / "HT_ref_audio.wav"
    prompt.write_bytes(b"RIFF" + b"\0" * 32)
    specs = graph_impl._parse_prompt_wav_specs(
        {graph_impl._PROMPT_MANIFEST_MODE_KEY: "model_default"},
        model_default_prompt=str(prompt),
    )
    assert len(specs) == 1
    assert specs[0].path == str(prompt.resolve())
    assert specs[0].sha256 == hashlib.sha256(prompt.read_bytes()).hexdigest()
    assert specs[0].manifest_row == {}


def test_file_prompt_path_seals_and_consumer_verifies_exact_features(tmp_path):
    metadata = tmp_path / "meta.lst"
    metadata.write_text("fixed rows\n")
    benchmark_sources = []
    for role in ("dataset_loader", "benchmark_patch", "benchmark_wrapper"):
        source_path = tmp_path / f"{role}.py"
        source_path.write_text(f"# {role}\n")
        benchmark_sources.append(
            {
                "role": role,
                "path": str(source_path),
                "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
            }
        )
    rows = []
    waves = []
    for index in range(1):
        path = tmp_path / f"live-{index}.wav"
        waveform = np.linspace(-0.2, 0.2, 32 + index, dtype=np.float32)
        sf.write(path, waveform, 24000, subtype="FLOAT")
        decoded, rate = sf.read(path, dtype="float32", always_2d=False)
        decoded = np.ascontiguousarray(decoded.reshape(-1), dtype=np.float32)
        identity = prompt_identity_from_f32_bytes(
            decoded.tobytes(),
            sample_rate=rate,
            samples=decoded.size,
        )
        rows.append(
            {
                "row_ordinal": index,
                "meta_line_index": index,
                "utterance_id": f"utt{index}",
                "locale": "zh",
                "target_text_sha256": hashlib.sha256(f"target-{index}".encode()).hexdigest(),
                "ref_path": str(path),
                "ref_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                **identity,
            }
        )
        waves.append(path)
    source = tmp_path / "source.json"
    source.write_text(
        json.dumps(
            {
                "schema": "minicpmo45-default-prompt-source-v1",
                "dataset_root": str(tmp_path),
                "locale": "zh",
                "metadata_path": str(metadata),
                "metadata_sha256": hashlib.sha256(metadata.read_bytes()).hexdigest(),
                "benchmark_sources": benchmark_sources,
                "rows": rows,
            }
        )
    )
    manifest = tmp_path / "runtime-prompts.json"
    producer = RuntimePromptManifest(
        mode="producer",
        manifest_path=manifest,
        source_manifest_path=source,
    )
    features = []
    for index, path in enumerate(waves):
        key = producer.observe_prompt_wav(path)
        producer.bind_request(f"request-{index}", key)
        value = SimpleNamespace(mels=torch.full((1, 300 + index, 80), index))
        features.append(value)
        producer.observe_features(f"request-{index}", value)
    assert manifest.is_file()
    assert not (manifest.stat().st_mode & 0o222)

    consumer = RuntimePromptManifest(
        mode="consumer",
        manifest_path=manifest,
        manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
    )
    for index, (path, value) in enumerate(zip(waves, features, strict=True)):
        key = consumer.observe_prompt_wav(path)
        consumer.bind_request(f"request-{index}", key)
        consumer.observe_features(f"request-{index}", value)


def test_prompt_census_deduplicates_only_by_real_prompt_length():
    bootstrap = graph_impl.Stage2EstimatorGraphBootstrap(
        SimpleNamespace(),
        mode="on",
        profile=graph_impl._CERTIFIED_PROFILE,
    )
    frames = {"/ref/0.wav": 302}
    bootstrap._prompt_specs = tuple(
        graph_impl._PromptWavSpec(
            path=f"/ref/{index}.wav",
            sha256=f"{index:064x}",
            manifest_row={
                "prompt_frames": frames[f"/ref/{index}.wav"],
                "prompt_mel_shape": [1, frames[f"/ref/{index}.wav"], 80],
                "prompt_mel_sha256": graph_impl.tensor_sha256(torch.zeros((1, frames[f"/ref/{index}.wav"], 80))),
            },
        )
        for index in range(1)
    )

    class _Backend:
        def prepare_prompt(self, cache_id, path):
            del cache_id
            return SimpleNamespace(mels=torch.zeros((1, frames[path], 80)))

        def evict_prompt(self, cache_id, path):
            del cache_id, path

    rows = bootstrap._build_prompt_census(_Backend())
    assert [row.prompt_frames for row, _ in rows] == [302]
    assert bootstrap._allowed_prompt_frames == {302}
    assert len(bootstrap._expected_roles) == 4


def test_prompt_census_accepts_model_default_without_external_manifest():
    bootstrap = graph_impl.Stage2EstimatorGraphBootstrap(
        SimpleNamespace(),
        mode="on",
        profile=graph_impl._CERTIFIED_PROFILE,
    )
    bootstrap._prompt_specs = (
        graph_impl._PromptWavSpec(
            path="/model/HT_ref_audio.wav",
            sha256="0" * 64,
            manifest_row={},
        ),
    )

    class _Backend:
        def prepare_prompt(self, cache_id, path):
            del cache_id, path
            return SimpleNamespace(mels=torch.zeros((1, 302, 80)))

        def evict_prompt(self, cache_id, path):
            del cache_id, path

    rows = bootstrap._build_prompt_census(_Backend())
    assert [row.prompt_frames for row, _ in rows] == [302]
    assert bootstrap._allowed_prompt_frames == {302}


def test_sealed_resident_fingerprint_ignores_replay_count_but_not_roles():
    runner = _runner("prompt_b1")
    initial = torch.ones(2)
    _run(runner, _semantic(), (initial,), lambda value: (value + 1,))
    runner.seal()
    warm = runner.snapshot("warm")
    _run(runner, _semantic(), (torch.ones(2) * 7,), lambda value: (value + 1,))
    final = runner.snapshot("final")
    assert warm["resident_fingerprint"] == final["resident_fingerprint"]
    assert warm["contract_fingerprint"] == final["contract_fingerprint"]
    assert warm["per_key"][0]["replay_successes"] == 0
    assert final["per_key"][0]["replay_successes"] == 1


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("token2wav_n_timesteps", 4),
        ("codec_chunk_frames", 50),
        ("codec_left_context_frames", 0),
    ],
)
def test_profile_mismatch_fails_before_capture(name, value):
    extra = {
        "token2wav_n_timesteps": 3,
        "codec_chunk_frames": 25,
        "codec_left_context_frames": 3,
    }
    extra[name] = value
    model = SimpleNamespace(_extra_config=lambda: extra)
    bootstrap = graph_impl.Stage2EstimatorGraphBootstrap(
        model,
        mode="on",
        profile=graph_impl._CERTIFIED_PROFILE,
    )
    backend = SimpleNamespace(
        n_timesteps=int(extra["token2wav_n_timesteps"]),
        flow=SimpleNamespace(training=False),
        _trt_stepper=None,
        _cfm_graph_wrapper=None,
    )
    with pytest.raises(RuntimeError, match="profile mismatch"):
        bootstrap._validate_runtime(backend)


def test_runtime_only_accepts_uncertified_profile_for_census():
    model = SimpleNamespace(
        _extra_config=lambda: {
            "token2wav_n_timesteps": 10,
            "codec_chunk_frames": 50,
            "codec_left_context_frames": 0,
        }
    )
    bootstrap = graph_impl.Stage2EstimatorGraphBootstrap(
        model,
        mode="runtime_only",
        profile="uncertified-census",
    )
    backend = SimpleNamespace(
        n_timesteps=10,
        flow=SimpleNamespace(training=False),
        _trt_stepper=None,
        _cfm_graph_wrapper=None,
    )
    bootstrap._validate_runtime(backend)


def test_runtime_only_bootstrap_calls_saved_original_cfm_solve(monkeypatch):
    bootstrap = graph_impl.Stage2EstimatorGraphBootstrap(
        SimpleNamespace(),
        mode="runtime_only",
        profile=graph_impl._CERTIFIED_PROFILE,
    )
    bootstrap._allowed_prompt_frames = frozenset({302})
    bootstrap.runner = _runner("prompt_b1_pf302", mode="runtime_only")
    bootstrap._cfm_steps = 3
    bootstrap._codec_chunk_frames = 25
    bootstrap._left_context_frames = 3
    original_calls = []

    def original(mu, speakers, cond, *, cnn_cache, att_cache):
        original_calls.append((speakers, cond, cnn_cache, att_cache))
        return (
            mu + 10,
            mu + 20,
            mu + 30,
        )

    bootstrap._original_decode_cfm = original

    def reject_graphable(*args, **kwargs):
        del args, kwargs
        raise AssertionError("runtime fallback called graphable CFM solve")

    monkeypatch.setattr(graph_impl, "_graphable_fixed_cfm3_solve", reject_graphable)
    mu = torch.tensor([[[1.0]]])
    zeros = torch.zeros_like(mu)
    with bootstrap._semantic_scope(
        role="prompt_b1",
        request_batch=1,
        prompt_frames=302,
        codec_token_count=0,
        last_chunk=False,
        flush_encoder=False,
    ):
        outputs = bootstrap._run_fixed_cfm3_solve(
            SimpleNamespace(),
            mu,
            speakers=torch.zeros((1, 1)),
            cond=zeros,
            cnn_cache=None,
            att_cache=None,
        )

    assert len(original_calls) == 1
    torch.testing.assert_close(outputs[0], mu + 10)
    torch.testing.assert_close(outputs[1], mu + 20)
    torch.testing.assert_close(outputs[2], mu + 30)


def test_on_bootstrap_four_warm_solves_use_saved_original(monkeypatch):
    bootstrap = graph_impl.Stage2EstimatorGraphBootstrap(
        SimpleNamespace(),
        mode="on",
        profile=graph_impl._CERTIFIED_PROFILE,
    )
    bootstrap._allowed_prompt_frames = frozenset({302})
    expected_roles = graph_impl._roles_for_prompt_frames(302)
    bootstrap.runner = _runner(*expected_roles)
    bootstrap._cfm_steps = 3
    bootstrap._codec_chunk_frames = 25
    bootstrap._left_context_frames = 3
    original_calls = []

    def original(mu, speakers, cond, *, cnn_cache, att_cache):
        del speakers, cond
        original_calls.append(
            bootstrap._current_semantic(
                cnn_cache=cnn_cache,
                att_cache=att_cache,
            ).role
        )
        return (
            mu + 10,
            mu.new_zeros((3, 1, 2, 1, 1)),
            mu.new_zeros((3, 1, 2, 1, 1, 1)),
        )

    graphable_calls = []

    def graphable(bound_backend, estimator, decoder, **kwargs):
        del bound_backend, estimator, decoder
        graphable_calls.append(True)
        mu = kwargs["mu_cfg"][:1]
        return (
            mu + 10,
            mu.new_zeros((3, 1, 2, 1, 1)),
            mu.new_zeros((3, 1, 2, 1, 1, 1)),
        )

    bootstrap._original_decode_cfm = original
    monkeypatch.setattr(graph_impl, "_graphable_fixed_cfm3_solve", graphable)
    estimator = SimpleNamespace(t_embedder=lambda value: value[:, None])
    decoder = SimpleNamespace(
        estimator=estimator,
        rand_noise=torch.zeros((1, 1, 1000)),
        inference_cfg_rate=1.0,
    )
    backend = SimpleNamespace(flow=SimpleNamespace(decoder=decoder))
    mu = torch.zeros((1, 1, 1))
    speakers = torch.zeros((1, 1))
    warm_roles = (
        ("prompt_b1", None),
        ("steady_b1", 302),
        ("steady_b1", 352),
        ("steady_b1", 402),
    )

    returned = []
    for role, cache_length in warm_roles:
        cnn_cache = None
        att_cache = None
        if cache_length is not None:
            cnn_cache = torch.zeros((3, 1, 2, 1, 1))
            att_cache = torch.zeros((3, 1, 2, 1, cache_length, 1))
        with bootstrap._semantic_scope(
            role=role,
            request_batch=1,
            prompt_frames=302,
            codec_token_count=0 if role == "prompt_b1" else 28,
            last_chunk=False,
            flush_encoder=False,
        ):
            outputs = bootstrap._run_fixed_cfm3_solve(
                backend,
                mu,
                speakers,
                mu,
                cnn_cache=cnn_cache,
                att_cache=att_cache,
            )
        returned.append(outputs[0])
        torch.testing.assert_close(outputs[0], mu + 10)

    assert original_calls == [
        "prompt_b1_pf302",
        "steady_b1_pf302_ac302",
        "steady_b1_pf302_ac352",
        "steady_b1_pf302_ac402",
    ]
    assert graphable_calls
    assert len(returned) == 4
    snapshot = bootstrap.runner.snapshot("pre_seal")
    assert snapshot["captures"] == 4
    assert snapshot["eager_warm_calls"] == 4
    assert snapshot["shadow_replay_successes"] == 4
    assert snapshot["replay_successes"] == 0
    assert snapshot["resident_graphs"] == 4
    bootstrap.runner.seal()
    assert bootstrap.runner.phase == GraphPhase.SEALED


def test_on_bootstrap_rejects_biased_fixed_solve_candidate(monkeypatch):
    bootstrap = graph_impl.Stage2EstimatorGraphBootstrap(
        SimpleNamespace(),
        mode="on",
        profile=graph_impl._CERTIFIED_PROFILE,
    )
    bootstrap._allowed_prompt_frames = frozenset({302})
    bootstrap.runner = _runner("prompt_b1_pf302")
    bootstrap._cfm_steps = 3
    bootstrap._codec_chunk_frames = 25
    bootstrap._left_context_frames = 3
    original_calls = []

    def original(mu, speakers, cond, *, cnn_cache, att_cache):
        del speakers, cond, cnn_cache, att_cache
        original_calls.append(True)
        return (mu + 1, mu + 2, mu + 3)

    def biased_graphable(bound_backend, estimator, decoder, **kwargs):
        del bound_backend, estimator, decoder
        mu = kwargs["mu_cfg"][:1]
        return (mu + 9, mu + 2, mu + 3)

    bootstrap._original_decode_cfm = original
    monkeypatch.setattr(
        graph_impl,
        "_graphable_fixed_cfm3_solve",
        biased_graphable,
    )
    estimator = SimpleNamespace(t_embedder=lambda value: value[:, None])
    decoder = SimpleNamespace(
        estimator=estimator,
        rand_noise=torch.zeros((1, 1, 1000)),
        inference_cfg_rate=1.0,
    )
    backend = SimpleNamespace(flow=SimpleNamespace(decoder=decoder))
    mu = torch.ones((1, 1, 1))
    with (
        bootstrap._semantic_scope(
            role="prompt_b1",
            request_batch=1,
            prompt_frames=302,
            codec_token_count=0,
            last_chunk=False,
            flush_encoder=False,
        ),
        pytest.raises(RuntimeError, match="restart required"),
    ):
        bootstrap._run_fixed_cfm3_solve(
            backend,
            mu,
            torch.zeros((1, 1)),
            torch.zeros_like(mu),
            cnn_cache=None,
            att_cache=None,
        )

    assert original_calls == [True]
    snapshot = bootstrap.runner.snapshot("failed")
    assert snapshot["resident_graphs"] == 0
    assert snapshot["shadow_replay_failures"] == 1


def test_instance_hook_does_not_modify_backend_class():
    class _Backend:
        def _decode_cfm(self):
            return "cfm"

        def setup_batch(self):
            return "setup"

        def decode_batch(self):
            return "decode"

    model = SimpleNamespace()
    bootstrap = graph_impl.Stage2EstimatorGraphBootstrap(
        model,
        mode="runtime_only",
        profile=graph_impl._CERTIFIED_PROFILE,
    )
    backend_a = _Backend()
    backend_b = _Backend()
    class_cfm = _Backend._decode_cfm
    bootstrap._install_instance_hooks(backend_a)

    assert _Backend._decode_cfm is class_cfm
    assert backend_b._decode_cfm.__func__ is class_cfm
    assert getattr(backend_a, graph_impl._BOOTSTRAP_ATTR) is bootstrap
    bootstrap._install_instance_hooks(backend_a)


def test_sdpa_context_does_not_mask_body_exception(monkeypatch):
    class _Context:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, traceback):
            return False

    from vllm_omni.platforms.npu.models import cosyvoice2_dit_attn

    monkeypatch.setattr(
        cosyvoice2_dit_attn,
        "npu_math_sdpa_context",
        lambda **kwargs: _Context(),
    )
    from vllm_omni.platforms.npu.models.step_audio2_token2wav import (
        npu_token2wav_sdpa_context,
    )

    with pytest.raises(ValueError, match="body failure"):
        with npu_token2wav_sdpa_context(require_math=True):
            raise ValueError("body failure")
