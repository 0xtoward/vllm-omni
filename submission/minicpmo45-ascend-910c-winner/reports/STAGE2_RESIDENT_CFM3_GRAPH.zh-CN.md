# Stage2 resident-state CFM3 NPUGraph 交付说明

## 范围

本变体在 conservative winner 之上额外包含 MiniCPM-o 4.5 Stage2 model-local NPUGraph：一次 replay 覆盖 3 次 estimator、CFG 与 Euler，并让 `302 -> 352 -> 402A/B` 的 estimator cache 留在图拥有的固定存储中。Flow encoder、HiFT、ISTFT、真实 tail 与未知 signature 保持 eager。

没有纳入 HiFT/ISTFT graph：其 primitive 虽快，但 matched product r1/r2 的 RTF 分别回退 `+0.692%/+1.813%`。

## 已有证据

- final mel、每步 CNN/attention cache、CFG/Euler 状态 exact；
- 启动期 shadow 通过后封图，请求期不 capture；unknown role/tail 计数后 eager fallback；
- trace 中 `LaunchKernelV2` 从 `55,527 / 239.717 ms` 降到 `20,860 / 89.210 ms`；
- 两对 matched `runtime_only/on`：mean RTF `0.267091 -> 0.253190`
  （-5.20%）、TTFP -3.46%、E2EL -4.99%。每个 on arm 31 次正式请求
  replay，四个 role 均命中，所有 miss/failure/external-load 为0；因此已进入
  最终默认 winner。

## 配置合同

最终 MiniCPM-o 4.5 NPU 缺省合同为：

```yaml
code2wav_npu_graph_mode: on
code2wav_npu_graph_profile: cfm3_ccf25_b1_model_default_prompt_v1
code2wav_npu_graph_prompt_manifest_mode: model_default
```

硬合同：CFM3、`codec_chunk_frames=25`、left context 3、B=1、模型默认
`HT_ref_audio.wav` 的 prompt 长度/内容 SHA。`runtime_only` 使用同一内部
格式/SDPA 路径但不 capture，可作为 matched control。官方 deploy 显式给出的
`off`、其他步数、profile 或 manifest 配置始终优先，不被缺省值覆盖。

需要运行期自定义参考音频时，仍可使用 producer/consumer sealed manifest：
producer 只观察 prompt，consumer 只读路径与 SHA。任何合同不匹配均 fail closed，
不会在正式请求期偷偷 capture。

## 最终 fresh-copy 验收

从独立 candidate source 复制并 `pip install -e . --no-build-isolation` 后，
使用逐字节未改的官方 deploy YAML，图自动完成 4 个 role capture、5/5 shadow
checks，并在正式请求 replay 31 次；最终源码（含真实 NUMA re-exec）RTF
`0.256868`、TTFP `653.568 ms`、TTFT `345.352 ms`，4/4 成功。Stage1
paged GE mentions=0。
