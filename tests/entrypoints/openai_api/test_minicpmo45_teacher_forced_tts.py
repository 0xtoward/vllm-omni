from types import SimpleNamespace

from vllm_omni.entrypoints.openai.serving_chat import OmniOpenAIServingChat


class _Tokenizer:
    _ids = {"<|tts_bos|>": 151703, "<|tts_eos|>": 151704}

    def convert_tokens_to_ids(self, token):
        return self._ids[token]

    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return [ord(char) for char in text]


def _serving():
    serving = OmniOpenAIServingChat.__new__(OmniOpenAIServingChat)
    serving.engine_client = SimpleNamespace(
        stage_configs=[
            SimpleNamespace(
                engine_args=SimpleNamespace(
                    model_arch="MiniCPMO45OmniForConditionalGeneration",
                    model_stage="llm",
                    max_model_len=32768,
                )
            ),
            SimpleNamespace(engine_args=SimpleNamespace(max_model_len=4096)),
        ]
    )
    return serving


def _request():
    return SimpleNamespace(
        chat_template_kwargs={"use_tts_template": True},
        modalities=["text", "audio"],
        tools=None,
        tool_choice=None,
        logprobs=False,
        prompt_logprobs=None,
        return_tokens_as_token_ids=False,
        n=1,
    )


def test_teacher_forced_tts_appends_declared_target(monkeypatch):
    monkeypatch.delenv("VLLM_OMNI_MINICPMO45_STAGE0_TEACHER_FORCED_TTS", raising=False)
    messages = [
        {
            "role": "system",
            "content": "The user message is the exact text you must speak.",
        },
        {"role": "user", "content": "你好"},
    ]
    prompt = {"prompt_token_ids": [10, 151703]}

    assert _serving()._maybe_apply_minicpmo45_teacher_forced_tts_prompt(
        _request(), messages, _Tokenizer(), prompt
    )
    target_ids = [ord("你"), ord("好")]
    assert prompt["prompt_token_ids"] == [10, 151703, *target_ids, 151704]
    assert prompt["additional_information"]["minicpmo45_teacher_forced_tts"] == {
        "target_start": 2,
        "target_end": 4,
        "target_token_ids": target_ids,
        "target_text": "你好",
    }


def test_teacher_forced_tts_accepts_official_multimodal_message_shape(monkeypatch):
    monkeypatch.setenv("VLLM_OMNI_MINICPMO45_STAGE0_TEACHER_FORCED_TTS", "1")
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "The user message is the exact text you must speak.",
                }
            ],
        },
        {"role": "user", "content": [{"type": "text", "text": "你好"}]},
    ]
    prompt = {"prompt_token_ids": [10, 151703]}

    assert _serving()._maybe_apply_minicpmo45_teacher_forced_tts_prompt(
        _request(), messages, _Tokenizer(), prompt
    )
    assert prompt["prompt_token_ids"] == [10, 151703, ord("你"), ord("好"), 151704]
    assert prompt["additional_information"]["minicpmo45_teacher_forced_tts"]["target_text"] == "你好"


def test_teacher_forced_tts_rejects_ordinary_chat(monkeypatch):
    monkeypatch.setenv("VLLM_OMNI_MINICPMO45_STAGE0_TEACHER_FORCED_TTS", "1")
    prompt = {"prompt_token_ids": [10, 151703]}
    messages = [
        {"role": "system", "content": "Answer the user's question."},
        {"role": "user", "content": "你好"},
    ]

    assert not _serving()._maybe_apply_minicpmo45_teacher_forced_tts_prompt(
        _request(), messages, _Tokenizer(), prompt
    )
    assert prompt == {"prompt_token_ids": [10, 151703]}


def test_teacher_forced_tts_explicit_opt_out(monkeypatch):
    monkeypatch.setenv("VLLM_OMNI_MINICPMO45_STAGE0_TEACHER_FORCED_TTS", "0")
    prompt = {"prompt_token_ids": [10, 151703]}
    messages = [
        {"role": "system", "content": "The user message is the exact text you must speak."},
        {"role": "user", "content": "你好"},
    ]

    assert not _serving()._maybe_apply_minicpmo45_teacher_forced_tts_prompt(
        _request(), messages, _Tokenizer(), prompt
    )
    assert prompt == {"prompt_token_ids": [10, 151703]}


def test_teacher_forced_tts_rejects_unsafe_request_contracts(monkeypatch):
    monkeypatch.setenv("VLLM_OMNI_MINICPMO45_STAGE0_TEACHER_FORCED_TTS", "1")
    messages = [
        {"role": "system", "content": "The user message is the exact text you must speak."},
        {"role": "user", "content": "你好"},
    ]
    for override in (
        {"_minicpmo45_native_duplex": True},
        {"logprobs": True},
        {"return_tokens_as_token_ids": True},
        {"n": 2},
        {"tool_choice": "auto"},
    ):
        request = _request()
        for key, value in override.items():
            setattr(request, key, value)
        prompt = {"prompt_token_ids": [10, 151703]}
        assert not _serving()._maybe_apply_minicpmo45_teacher_forced_tts_prompt(
            request, messages, _Tokenizer(), prompt
        )
        assert prompt == {"prompt_token_ids": [10, 151703]}


def test_teacher_forced_tts_rejects_multimodal_and_long_targets(monkeypatch):
    monkeypatch.setenv("VLLM_OMNI_MINICPMO45_STAGE0_TEACHER_FORCED_TTS", "1")
    system = {"role": "system", "content": "The user message is the exact text you must speak."}
    multimodal = [
        system,
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
                {"type": "text", "text": "你好"},
            ],
        },
    ]
    prompt = {"prompt_token_ids": [10, 151703]}
    assert not _serving()._maybe_apply_minicpmo45_teacher_forced_tts_prompt(
        _request(), multimodal, _Tokenizer(), prompt
    )

    serving = _serving()
    serving.engine_client.stage_configs[0].engine_args.max_model_len = 5
    prompt = {"prompt_token_ids": [10, 151703]}
    messages = [system, {"role": "user", "content": "你好"}]
    assert not serving._maybe_apply_minicpmo45_teacher_forced_tts_prompt(
        _request(), messages, _Tokenizer(), prompt
    )
