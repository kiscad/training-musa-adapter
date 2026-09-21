# 旧 patch 迁移台账（Migration Ledger）

依据 `megatron-musa-patch`（基线 revision `a1090de`，2026-09-21 调研）向
`musa-adapter` 的逐项迁移记录。设计文档 §13.2 要求每个旧功能都有明确去向；
本文是唯一权威状态表，验收结论以实际运行为准。

状态图例：

- **已迁移（已验证）**：新架构中实现且在本仓库测试/硬件验证通过。
- **已迁移（待硬件验证）**：代码已迁移，能力窗口从旧项目移植，但尚未在
  本轮硬件矩阵中复跑。
- **P5 待迁移**：归入第 14 节 P5 阶段，迁移前旧项目仍是行为基线。
- **退役**：不再需要，理由附后。

## 聚合映射

| 旧模块 | 新去向 |
|---|---|
| `_engine.py` | `runtime/`（specs/registry/planner/imports/lifecycle/transactions） |
| `_compat.py` | `runtime/environment.py`（packaging 语义）+ 各 patch 声明式约束 |
| `_env.py` | `config/`（schema/loader/defaults.toml，冻结语义） |
| `activation.py` | `bootstrap.py` + `api.py`（框架无关激活） |
| `backends/torch_cuda.py` | `patches/platform/`（P5；不得成为引擎隐含依赖） |
| `patches/_attention.py` | `patches/megatron/attention.py` + `providers/attention/*` |
| `patches/_transformers.py` | `patches/transformers/qwen3_vl_rms_norm.py` + `providers/rms_norm/*` |
| 其余 `patches/_*.py` | P5 待迁移（见下表） |

## 逐项状态

| 旧 patch ID | 新 ID / 去向 | 状态 |
|---|---|---|
| `transformers.qwen3-vl.text-rms-norm.fused-torch` | `transformers.qwen3_vl.rms_norm.dispatch` + `rms_norm.torch_rms_norm`（加速）+ `rms_norm.upstream_chain`（参考） | 已迁移（已验证：MUSA 前向/反向、dtype 提升语义、prefer/force 切换） |
| `megatron.te.attention.capability-dispatch` | `megatron.core.attention.dispatch` + `attention.mudnn`/`attention.mate`/`attention.te_unfused` | 已迁移（已验证：mudnn 窗口/GQA/数值/force 不降级/策略切换；**mate 192 等维与混合 192/128 在标准 causal 构造下前向+反向+数值**（out ≤0.0156，梯度 bf16 量级）经 `TestMateHardwarePath` 通过；**packed THD 含 padded offsets/空序列**：输出与梯度有限、padded 行精确为零，旧 NaN 未复现。验证脚本存档 /tmp/attention_validation_b/） |
| `transformer_engine.dot-product-attention.capability-dispatch` | 需独立 TE 插件（trigger `transformer_engine.musa.pytorch.attention`），复用 attention providers | P5 待迁移 |
| `torch.cuda.compat-layer` | `patches/platform/`（torchada 之上的契约补偿） | P5 待迁移 |
| `torch.cuda.device-capability.nvidia-scale` | `patches/platform/` | P5 待迁移 |
| `torch.distributed.clean-teardown` | `patches/platform/` | P5 待迁移 |
| `megatron.te.norm.unfused-musa` | rms_norm operator family（norm 专项 binding） | P5 待迁移 |
| `megatron.te.layer-norm-linear.unfused` | norm family + TE binding | P5 待迁移 |
| `megatron.te.fused-mlp.unfused` | 保留原样迁移（compatibility patch） | P5 待迁移 |
| `megatron.te.factory-shim.torchscript-compat` | 保留（before_exec，trigger `transformer_engine`） | P5 待迁移 |
| `megatron.te.utils-module.safe-seed` | 保留（before_exec） | P5 待迁移 |
| `megatron.te.cpu-offload-context.signature-dispatch` | 保留 | P5 待迁移 |
| `megatron.te.make-weak-ref.graph-compat` | 保留 | P5 待迁移 |
| `megatron.te.grouped-linear.mem-monitor-compat` | 保留 | P5 待迁移 |
| `megatron.te.quantized-model-init.delayed-compat` | 保留 | P5 待迁移 |
| `megatron.fusions.fused-layer-norm.pure-torch` | norm family | P5 待迁移 |
| `megatron.fusions.fused-layer-norm.have-apex-flag` | norm family（capability 声明化） | P5 待迁移 |
| `megatron.fusions.persist-layer-norm.disable` | norm family | P5 待迁移 |
| `megatron.transformer-block.layer-norm.impl-local` | norm family | P5 待迁移 |
| `megatron.embeddings.fused-rope.apex` | 新 operator family `rope.v1` | P5 待迁移 |
| `megatron.embeddings.fused-rope-thd.apex` | `rope.v1` | P5 待迁移 |
| `megatron.embeddings.rope-fusion.unfused-fallback` | `rope.v1` | P5 待迁移 |
| `megatron.ssm.gated-delta-rule.tilelang` | 新 operator family `gated_delta_rule.v1` + provider（torch_kernels TileLang） | P5 待迁移 |
| `mcore_bridge.ssm.gated-delta-rule.tilelang` | 同一 `gated_delta_rule.v1` provider 的第二个 binding（`mcore_bridge.model.modules.gated_delta_net`） | P5 待迁移 |
| `megatron.moe.fused-router.*`（3 项） | MoE binding 组 | P5 待迁移 |
| `megatron.moe.permutation.unfused-musa` | MoE binding 组 | P5 待迁移 |
| `megatron.moe.unpermutation.unfused-musa` | MoE binding 组 | P5 待迁移 |
| `megatron.moe.router-gating.fp64-host` | MoE binding 组 | P5 待迁移 |
| `megatron.moe.topk.fp64-reference` | MoE binding 组 | P5 待迁移 |
| `megatron.moe.grouped-gemm.*`（3 项） | grouped_gemm binding 组 | P5 待迁移 |
| `megatron.softmax.kernel-availability.musa` | softmax family | P5 待迁移 |
| `megatron.dist-ckpt.musa-cpu-staging` | checkpoint 兼容组（compatibility patch） | P5 待迁移 |
| `megatron.dist-ckpt.no-fork-writer` | checkpoint 兼容组 | P5 待迁移 |
| `megatron.serialization.signal-member-globals` | checkpoint 兼容组 | P5 待迁移 |
| `megatron.training.checkpoint.host-barrier-context` | 控制面通信组（impact=collective） | P5 待迁移 |
| `megatron.training.checkpoint.host-barrier-proxy` | 控制面通信组 | P5 待迁移 |
| `megatron.training.get-device-arch-version.nvidia-scale` | platform 插件 | P5 待迁移 |
| `megatron.training.initialize.set-jit-fusion-options.noop` | 训练入口兼容组 | P5 待迁移 |
| `megatron.training.set-jit-fusion-options.noop` | 训练入口兼容组 | P5 待迁移 |
| `megatron.training.overlap-flags.noop` | 训练入口兼容组 | P5 待迁移 |
| `megatron.training.start-time.integer-microseconds` | 训练入口兼容组 | P5 待迁移 |
| `megatron.training.profile.pytorch` | 训练入口兼容组 | P5 待迁移 |
| `megatron.legacy.fused-kernels.load.noop` | 评估后决定（旧理由是 legacy 路径在 MUSA 崩溃） | P5 待迁移 |
| `megatron.hyper-comm-grid.subgroups-backend` | 通信组 | P5 待迁移 |
| `megatron.bridge-communicator.subgroups-backend` | 通信组 | P5 待迁移 |
| `megatron.fsdp.premul-sum.device-prescale` | 通信组 | P5 待迁移 |
| `megatron.offloading.resident-parameters` | offload 兼容组 | P5 待迁移 |
| `megatron.device-arch`（`megatron.training.get-device-arch-version.*`） | platform 插件 | P5 待迁移 |
| `transformer_engine.layer-norm-linear.native-unfused` | norm family 的 TE binding | P5 待迁移 |
| `transformer_engine.layer-norm-mlp.native-unfused` | norm family | P5 待迁移 |
| `transformer_engine.grouped-gemm.wgrad-reference` | grouped_gemm family | P5 待迁移 |
| `transformer_engine.saved-tensors.offload-markers` | offload 兼容组 | P5 待迁移 |

## 硬件验证中发现的已修复缺陷

1. **TE causal 窗口编码错配（已修复）**：TE 将 causal 初始化模块的
   `window_size` 规范为 `(-1, 0)`（显式 `(-1, -1)` 也会被改回）；binding
   最初原样透传，mate 拒绝 `(-1, 0)` → causal-LM 训练下 d=144/168..192 与
   混合 head_dim 永远到不了 mate，静默退到 O(S²) 参考。修复：binding 在
   describe 中把 `(-1,-1)/(-1,0)` 归一化为 None（mask_kind 承载因果语义）；
   回归测试 `tests/contracts/test_megatron_binding.py` +
   `TestMateHardwarePath`。

## 关键行为差异（相对旧项目，必须在迁移评审时知晓）

1. **ONLY/DISABLE 优先级反转**：旧项目 ONLY 优先；新项目
   `selected = (only or all) - disable`，DISABLE 恒为减法且优先。
2. **ATTN_BACKEND=mudnn 无等价迁移**：旧值只按 forward 放行（backward 可能
   失败）；新 `attention.mudnn` 的 supports 在 may_require_backward 时检查
   backward 窗口，force 也不放宽。显式迁移到 `force + attention.mudnn` 会
   得到更严格的语义。
3. **未知配置值不再静默回 auto**：ConfigError。
4. **Hook 不再在 find_spec 执行**：before_exec 在 loader 边界执行，纯
   find_spec 查询无副作用。
5. **应用失败不再破坏框架导入**：记录 failed + rollback 状态（旧引擎会抛出
   并中断导入）。
6. **reload 支持范围**：after_exec patch 随 reload 重建一次；before_exec 的
   目标模块自身 reload 不在 V1 承诺内。

## 验证证据索引

- transformers RMSNorm：`tests/integration/test_transformers_rms_norm.py`
  （MUSA 前向/反向、dtype 提升保留、策略切换、报告可解释）。
- attention 窗口：`tests/contracts/test_attention_selection.py`
  （描述符级窗口矩阵，源自旧项目实测）。
- 引擎机制：`tests/unit/runtime/test_lifecycle.py`（边界、事务、所有权、
  撤销、reload、失败路径）。
- 硬件矩阵复跑记录：见 `docs/compatibility/`（随 P4/P5 验收补充）。

## 性能基线（首轮测量）

- 选择器热路径（缓存决策命中，fake provider，Python 3.10）：**2.66 µs/调用**
  （纯 Python 调用基线 0.04 µs）。测量脚本：`tests/performance/bench_selector.py`。
  每层 attention 前向调用一次分发，占单步时间比例可忽略；预算上限待代表
  性训练基准建立后填写。默认候选顺序变更时必须复测。
