# Stage2 resident-state CFM3 NPUGraph 交付说明

## 范围

本变体在 conservative winner 之上额外包含 MiniCPM-o 4.5 Stage2 model-local NPUGraph：一次 replay 覆盖 3 次 estimator、CFG 与 Euler，并让 `302 -> 352 -> 402A/B` 的 estimator cache 留在图拥有的固定存储中。Flow encoder、HiFT、ISTFT、真实 tail 与未知 signature 保持 eager。

没有纳入 HiFT/ISTFT graph：其 primitive 虽快，但 matched product r1/r2 的 RTF 分别回退 `+0.692%/+1.813%`。

## 已有证据

- final mel、每步 CNN/attention cache、CFG/Euler 状态 exact；
- 启动期 shadow 通过后封图，请求期不 capture；unknown role/tail 计数后 eager fallback；
- trace 中 `LaunchKernelV2` 从 `55,527 / 239.717 ms` 降到 `20,860 / 89.210 ms`；
- E2E 是小幅正向候选，但低于 2% conservative-winner 集成门。因此本包提供完整实现，默认仍为 `off`。

## 配置合同

Stage2 `extra` 需要：

```yaml
code2wav_npu_graph_mode: on
code2wav_npu_graph_profile: cfm3_ccf25_b1_model_default_prompt_v1
code2wav_npu_graph_prompt_manifest_mode: consumer
code2wav_npu_graph_prompt_manifest: /absolute/path/to/sealed-runtime-prompts.json
code2wav_npu_graph_prompt_manifest_sha256: <64-hex-sha256>
```

硬合同：CFM3、`codec_chunk_frames=25`、left context 3、B=1、一个已封 runtime prompt 长度。`runtime_only` 使用同一内部格式/SDPA 路径但不 capture，可作为 matched control。

producer 必须用 `code2wav_npu_graph_mode: off` 和 `code2wav_npu_graph_prompt_manifest_mode: producer` 观察真实 prompt；consumer 只读 sealed manifest。路径和 SHA 不匹配时 fail closed，不会请求期偷偷扩图。

## 本轮限制

本轮按要求未启动服务、未占用 NPU。代码是此前已完成 NPU correctness/engagement/E2E 的原实现逐文件迁移；本轮只执行静态、AST、CPU 合同测试与打包检查。官方环境没有 portable 的绝对 prompt-manifest 路径，因此不把 graph 强制设为默认 ON。
