# 补丁台账（逐 ID）

本表是唯一权威状态表：每个补丁 ID 的实现与验证状态逐条维护，不用通配符
代替；能力在通过相应检查前不进入"已验证"列。

门控策略：megatron 补丁全部声明 `megatron-core` 门控；上界按
core_v0.17/0.18/0.19 分支逐目标校准（接口在边界处明显变化的拆分为独立
变体补丁，目标被移除/改名的封在移除线），策略见
`patches/megatron/__init__.py`。

验证记号：✅硬件 = 真实 MUSA 栈验证；✅单元 = CPU/桩单元测试通过；
⚠️ = 已知限制或缺口。

## 验证记录的适用范围

已有设备验证记录对应 torch / torch_musa 2.7.1、Transformer Engine
2.0.0+651a47b7、Megatron Core 0.16.1、transformers 5.16.1；这是指定栈的记录，
不是对版本门控内所有版本的硬件承诺。不同软件栈及改动后须按相关范围复验。
性能、硬件和训练结果必须附环境、命令及输出记录，不能由“已实现”或测试文件存在推断。

| 范围 | 本仓库验证入口 |
|---|---|
| 设备层与真实导入顺序 | [test_torch_cuda.py](../tests/test_torch_cuda.py) |
| RMSNorm / attention 前后向 | [RMSNorm](../tests/integration/test_rms_norm_musa.py)、[attention](../tests/integration/test_attention_musa.py) |
| GDN / RoPE / TE norm-linear | [SSM](../tests/test_ssm.py)、[RoPE](../tests/test_rope.py)、[TE norm](../tests/test_te_layer_norm.py)，按各用例条件启动硬件 worker |
| 控制通信 | [test_control_collectives.py](../tests/test_control_collectives.py)，包含两 rank 用例 |
| 注册表与逐 ID 覆盖 | [test_ledger.py](../tests/test_ledger.py) |

2026-09-22 静态审查修复及源码依据见 [审查记录](REVIEW_2026-09-22.md)。
本轮按用户要求未运行测试；下表相关 ID 已注明待复验，历史结果不覆盖本轮修改。

## 已有场景验证记录

| 场景 | 结果 |
|---|---|
| Core wheel 端到端（pretrain_gpt.py，2 卡，5 迭代，local impl） | ⚠️ 待重新验收：当前仓库无对应启动器与完整执行记录，不能据局部单元测试声明原始训练入口通过 |
| 无 Megatron 自动激活（transformers-only，Qwen3-VL RMSNorm） | ✅ 通过：bf16 前向/反向有限、数值对照 <2e-2、dtype 提升保留、ENABLED=0 硬退出、DISABLE 生效 |
| attention 硬件矩阵 | ✅ mudnn/te_unfused/GQA 前向+反向、数值 ≤2e-2、mate d=192 前向+反向与参考一致、packed THD 含 padded offsets/空序列零梯度、force 不降级、策略切换改实际路径 |
| fp8 norm-linear（1/2 卡，TP/SP，checkpoint 分片） | ✅ TE_PASS（16 项） |
| RoPE 融合 | ✅ ROPE_PASS（8 项，真实 apex 核） |
| GDN TileLang | ✅ SSM_PASS：分发位级一致、与 FLA 前向/反向一致、fla-only 降级位级一致、DISABLE 新进程恢复 |
| factory-shim torchscript | ✅ 子进程端到端（ms-swift zigzag_ring_attn 形态）；✅ [09-23 链式 shim 修复](REVIEW_2026-09-23.md)（torchada 0.1.86 在下、TE 在上，`import swift.megatron` 真实链） |

## 逐 ID 状态

### transformers / 设备层

| ID | 目标位置 | 状态 | 验证 |
|---|---|---|---|
| `transformers.qwen3-vl.text-rms-norm.fused-torch` | `patches/transformers/rms_norm.py` | 已实现（已验证） | ✅硬件；fp16/bf16 同 dtype 才融合 |
| `torch.cuda.compat-layer` | `patches/platform.py` + `backends/torch_cuda.py` | 已实现（已验证） | ✅ 边界测试（torch-first/megatron-first 两种顺序） |
| `torch.cuda.compat-layer.megatron-late-boundary` | 同上（parallel_state 边界） | 已实现（边界修正） | ✅ megatron-first 顺序接管 |
| `torch.cuda.compat-layer.transformers` | 同上（transformers 根边界） | 已实现（多框架） | ✅ transformers-only 独立激活 |
| `torch.cuda.compat-layer.transformers-late-boundary` | 同上（modeling_utils 边界） | 已实现（边界修正） | ✅ modeling-first 顺序接管；⚠️ 该顺序下 RMSNorm AttrPatch 错过边界（记录为兼容缺口） |
| `torch.cuda.device-capability.nvidia-scale` | `patches/megatron/device_arch.py` | 已实现 | ✅单元；⚠️ [09-22 修复待复验](REVIEW_2026-09-22.md) |

### megatron attention / norm / RoPE / SSM

| ID | 目标位置 | 状态 | 验证 |
|---|---|---|---|
| `megatron.te.attention.capability-dispatch` | `patches/megatron/attention.py` + `ops/attention.py` | 已实现（已验证） | ✅硬件全矩阵（见上）；框架契约保留；⚠️ [09-22 修复待复验](REVIEW_2026-09-22.md) |
| `transformer_engine.dot-product-attention.capability-dispatch` | — | 未实现 | ⚠️ TE 原生 DPA 包装（megatron-FSDP 直连 TE 场景），需要 megatron-FSDP suite 验证 |
| `megatron.te.norm.unfused-musa` | `patches/megatron/layer_norm.py` | 已实现 | ✅单元 + fp8 集成 |
| `megatron.te.layer-norm-linear.unfused` | 同上 | 已实现 | ✅单元 + fp8 集成 |
| `transformer_engine.layer-norm-linear.native-unfused` | 同上 | 已实现 | ✅单元 |
| `transformer_engine.layer-norm-mlp.native-unfused` | 同上 | 已实现 | ✅单元 |
| `megatron.te.fused-mlp.unfused` | 同上 | 已实现 | ✅单元 |
| `megatron.fusions.fused-layer-norm.pure-torch` | 同上 | 已实现 | ✅单元 |
| `megatron.fusions.fused-layer-norm.have-apex-flag` | 同上 | 已实现 | ✅单元 |
| `megatron.fusions.persist-layer-norm.disable` | 同上 | 已实现 | ✅单元 |
| `megatron.transformer-block.layer-norm.impl-local` | 同上 | 已实现 | ✅单元 |
| `megatron.embeddings.fused-rope.aten` | `patches/megatron/rope.py` | 新增（torch 融合算子优先，apex 回退） | ✅单元 + ✅MUSA 数值（fp32/bf16 × interleaved 双模式、直通拆分、梯度，子进程对照 unfused 参考）+ ✅性能实测（bf16 3.4-4.9x vs apex、7.8-11.1x vs unfused，MTT S5000）；⚠️ thd 未接入（算子无 cu_seqlens 形态）；⚠️ [09-22 修复待复验](REVIEW_2026-09-22.md) |
| `megatron.embeddings.fused-rope.apex` | `patches/megatron/rope.py` | 已实现（已验证） | ✅硬件 smoke；⚠️ [09-22 修复待复验](REVIEW_2026-09-22.md) |
| `megatron.embeddings.fused-rope-thd.apex` | 同上 | 已实现（已验证） | ✅硬件 smoke；⚠️ [09-22 修复待复验](REVIEW_2026-09-22.md) |
| `megatron.embeddings.rope-fusion.unfused-fallback` | 同上 | 已实现（已验证） | ✅硬件 smoke |
| `megatron.embeddings.fused-rope-thd.apex.core17` | 同上 | 新增（core≥0.17 变体：thd 调用点新增 interleaved=） | ✅单元（torch-free 桩）；⚠️ [09-22 修复待复验](REVIEW_2026-09-22.md) |
| `megatron.ssm.gated-delta-rule.tilelang` | `patches/megatron/ssm.py` + `ops/gated_delta_rule.py` | 已实现（已验证） | ✅硬件 smoke 全链路；⚠️ [09-22 修复待复验](REVIEW_2026-09-22.md) |
| `mcore_bridge.ssm.gated-delta-rule.tilelang` | `patches/mcore_bridge/ssm.py` | 已实现 | ✅单元；⚠️ mcore-bridge 未安装时真实桥接路径未跑（单元桩验证）；⚠️ [09-22 修复待复验](REVIEW_2026-09-22.md) |

### megatron MoE / GEMM / softmax

| ID | 目标位置 | 状态 | 验证 |
|---|---|---|---|
| `megatron.moe.router-gating.fp64-host` | `patches/megatron/moe.py` | 已实现 | ✅单元 |
| `megatron.moe.topk.fp64-reference` | 同上 | 已实现 | ✅单元（含 RNG 保持探测）；⚠️ [09-22 修复待复验](REVIEW_2026-09-22.md) |
| `megatron.moe.permutation.unfused-musa` | 同上 | 已实现 | ✅单元 |
| `megatron.moe.unpermutation.unfused-musa` | 同上 | 已实现 | ✅单元 |
| `megatron.moe.permutation.unfused-musa.core17` | 同上 | 新增（core≥0.17 变体：permute 新增 tokens_per_expert/align_size） | ⚠️ 单元已编写，待 torch 环境执行 |
| `megatron.moe.unpermutation.unfused-musa.core17` | 同上 | 新增（core≥0.17 变体：unpermute 新增 pad_offsets） | ⚠️ 单元已编写，待 torch 环境执行 |
| `megatron.moe.fused-router.topk-with-score-function` | 同上 | 已实现 | ✅单元 |
| `megatron.moe.fused-router.moe-aux-loss` | 同上 | 已实现 | ✅单元 |
| `megatron.moe.fused-router.score-for-aux-loss` | 同上 | 已实现 | ✅单元 |
| `megatron.moe.grouped-gemm.torch-ops` | `patches/megatron/grouped_gemm.py` | 已实现 | ✅单元 |
| `megatron.moe.grouped-gemm.available-flag` | 同上 | 已实现 | ✅单元 |
| `megatron.moe.grouped-gemm.assert-noop` | 同上 | 已实现 | ✅单元 |
| `transformer_engine.grouped-gemm.wgrad-reference` | 同上 | 已实现 | ✅单元 |
| `megatron.softmax.kernel-availability.musa` | `patches/megatron/softmax.py` | 已实现 | ✅单元：工厂仅探测一次；另有 pretrain smoke 间接覆盖 |

### megatron TE 适配

| ID | 目标位置 | 状态 | 验证 |
|---|---|---|---|
| `megatron.te.quantized-model-init.delayed-compat` | `patches/transformer_engine.py` | 已实现 | ✅单元 |
| `megatron.te.cpu-offload-context.signature-dispatch` | 同上 | 已实现 | ✅单元 |
| `megatron.te.grouped-linear.mem-monitor-compat` | 同上 | 已实现 | ✅单元 |
| `megatron.te.factory-shim.torchscript-compat` | 同上 | 已实现（已验证） | ✅子进程端到端；✅单元（[09-23 链式 shim 修复](REVIEW_2026-09-23.md)：torchada 层在 TE 之下时别名走查到真实 ATen 函数，真实 `import swift.megatron` 链验证） |
| `megatron.te.utils-module.safe-seed` | 同上 | 已实现 | ✅单元 |
| `megatron.te.make-weak-ref.graph-compat` | `patches/megatron/cuda_graphs.py` | 已实现 | ✅单元；⚠️ cuda graph 需 torch_musa>2.9，本栈 2.7.1 硬件验证延期 |
| `megatron.te.telinear.delayed-init` | `patches/megatron/delayed_wgrad.py` | 已实现 | ✅单元（含 requires 门控） |
| `megatron.te.telinear.delayed-forward` | `patches/megatron/delayed_wgrad.py` | 已实现 | ✅单元（含 requires 门控） |
| `megatron.te.telinear.delayed-backward` | `patches/megatron/delayed_wgrad.py` | 已实现 | ✅单元（含 requires 门控） |
| `megatron.te.tegroupedlinear.delayed-init` | `patches/megatron/delayed_wgrad.py` | 已实现 | ✅单元 |
| `megatron.te.tegroupedlinear.delayed-forward` | `patches/megatron/delayed_wgrad.py` | 已实现 | ✅单元 |
| `megatron.te.tegroupedlinear.delayed-backward` | `patches/megatron/delayed_wgrad.py` | 已实现 | ✅单元 |

### megatron 训练系统

| ID | 目标位置 | 状态 | 验证 |
|---|---|---|---|
| `megatron.dist-ckpt.musa-cpu-staging` | `patches/megatron/checkpointing.py` | 已实现 | ✅单元 |
| `megatron.dist-ckpt.no-fork-writer` | 同上 | 已实现 | ✅单元 |
| `megatron.serialization.signal-member-globals` | `patches/megatron/control_collectives.py` | 已实现 | ✅单元 |
| `megatron.training.checkpoint.host-barrier-context` | 同上 | 已实现 | ✅单元 |
| `megatron.training.checkpoint.host-barrier-proxy` | 同上 | 已实现 | ✅单元 |
| `megatron.training.get-device-arch-version.nvidia-scale` | `patches/megatron/device_arch.py` | 已实现 | ✅单元；⚠️ [09-22 修复待复验](REVIEW_2026-09-22.md) |
| `megatron.training.initialize.set-jit-fusion-options.noop` | `patches/megatron/training.py` | 已实现 | ✅单元 |
| `megatron.training.set-jit-fusion-options.noop` | 同上 | 已实现 | ✅单元 |
| `megatron.training.overlap-flags.noop` | 同上 | 已实现 | ✅单元 |
| `megatron.training.start-time.integer-microseconds` | `patches/megatron/control_collectives.py` | 已实现 | ✅单元 |
| `megatron.training.profile.pytorch` | `patches/megatron/training.py` | 已实现 | ✅单元 |
| `megatron.legacy.fused-kernels.load.noop` | 同上 | 已实现 | ✅单元 |
| `megatron.fsdp.premul-sum.device-prescale` | `patches/megatron/distributed.py` | 已实现 | ✅单元；⚠️ [09-22 修复待复验](REVIEW_2026-09-22.md) |
| `megatron.bridge-communicator.subgroups-backend` | 同上 | 已实现 | ✅单元 |
| `megatron.hyper-comm-grid.subgroups-backend` | 同上 | 已实现 | ✅单元 |
| `torch.distributed.clean-teardown` | 同上 | 已实现 | ✅单元 |
| `megatron.offloading.resident-parameters` | `patches/megatron/offloading.py` | 已实现 | ✅单元 |
| `transformer_engine.saved-tensors.offload-markers` | 同上 | 已实现 | ✅单元 |

### peft

| ID | 目标位置 | 状态 | 验证 |
|---|---|---|---|
| `peft.lora.torchao-probe.version-compat` | `patches/peft.py`（`peft.import_utils:is_torchao_available`） | 已实现 | ✅单元（含门控边界与异常透传）；✅LoRA SFT 训练入口（Qwen2.5-0.5B，1×MUSA，torchao 0.9.0 + peft 0.19.1 真实冲突链） |

门控证据（peft sdists 0.14.0/0.15.2/0.16.0/0.17.1/0.18.0/0.19.0/0.20.0/0.21.0，
证据副本在任务目录 `upstream-src/peft-*/`）：`is_torchao_available` 自 0.14.0 引入即对
低于最低版本的 torchao 抛 ImportError，但 0.14.0-0.18.x 的最低版本是 0.4.0；0.19.0 起
最低版本跳到 0.16.0，0.19.0-0.21.0 的消息与签名一致。本环境 torchao 0.9.0（社区构建）
能通过 0.14-0.18 的门、被 0.19+ 的门拒绝，而 `peft/tuners/lora/torchao.py:dispatch_torchao`
对每个 LoRA 目标先调用该探针再检查 weight 类型，因此 `get_peft_model` 全量崩溃。
补丁范围 `>=0.19,<0.22`：0.22 尚未发布，上界是保守核查边界，不是已证实的不兼容点。

### deepspeed

| ID | 目标位置 | 状态 | 验证 |
|---|---|---|---|
| `deepspeed.zero.grad-norm.fp32` | `patches/deepspeed.py`（`deepspeed.runtime.zero.utils:get_norm_dtype`） | 已实现 | ✅单元（CUDA 保持/MUSA 降级/门控边界）；✅ZeRO-3 LoRA SFT 训练入口（Qwen2.5-0.5B，2×MUSA，社区 deepspeed 0.19.7） |

门控证据（deepspeed sdists 0.17.2/0.18.9/0.19.7，证据副本在任务目录
`upstream-src/ds-*/`）：`get_norm_dtype` 0.19.0 引入（0.17.2/0.18.9 无此函数），
0.19.7 的 CUDA accelerator `is_fp64_supported()` 恒 True 而 torch_musa 无 fp64 norm
kernel，ZeRO-3 首个 optimizer step 在 `get_grad_norm_direct` 崩溃；MPS accelerator
`is_fp64_supported() -> False` 是上游对「无可用 fp64 设备」的既有先例。
补丁范围 `>=0.19,<0.20`：0.20 尚未发布，上界为保守核查边界。
deepspeed 采用社区版 0.19.7（用户指令：弃用 vendor fork `/home/DeepSpeed`）。

## 已知缺口（如实记录）

1. `transformer_engine.dot-product-attention.capability-dispatch`（TE 原生 DPA
   包装，megatron-FSDP 直连 te.TransformerLayer 场景）尚未实现——需要
   megatron-FSDP suite 验证后才能声明支持。
2. ms-swift / mcore-bridge 尚无完整原始训练入口的验收记录：其中 transformers RMSNorm、平台设备层和 Core GDN 有独立验证；
   mcore-bridge 的实际桥接路径仍只有桩测试，
   完整上层入口验证列为下一步。
3. 两 rank 控制通信测试已通过：实际启动归约片段、checkpoint host-barrier
   控制流与 SIGTERM 序列化。subgroups-backend、完整 checkpoint 恢复及
   故障退出仍需独立的跨 rank 验收，不能据此声明通信域全部验证。
4. `megatron.softmax.kernel-availability.musa` 已补工厂探测次数回归；
   数值路径仍主要通过 pretrain smoke 间接覆盖。
5. modeling-first 极端导入顺序下 transformers 建模模块自身的 AttrPatch 错过
   边界（设备层仍经 modeling_utils 就位）——记录为兼容缺口，见
   `patches/platform.py` 头注。
