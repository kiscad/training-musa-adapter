# 迁移台账（v2.0，逐 ID）

按设计文档 §9.3 维护：每个旧 patch 与第一轮新增修复都有记录；
**本表是唯一权威状态表**，"保留/合并/退役/待迁移"与实际验证结果逐 ID 清点，
不用通配符代替。第一轮 v1.0 台账作为历史证据存于
`docs/reference/MIGRATION_LEDGER.v1.md`，不直接等同重构后的验证结果。

基线：

- 旧项目：`megatron-musa-patch` revision `a1090de`（行为与测试基线）。
- 第一轮：`musa-adapter` revision `70252e4`（本仓库 git 历史，计算修复与
  数值/硬件验证证据来源）。
- 本项目：`training-musa-adaptor`，版本独立演进。

状态图例：**已迁移（已验证）** / **已迁移（待验证）** / **待迁移** / **退役**。

## 逐 ID 状态

| 来源 ID | 目标文件/ID | 状态 | 保留契约与关键差异 | 相关测试 | 验证结果与缺口 |
|---|---|---|---|---|---|
| `megatron.te.attention.capability-dispatch` | `patches/megatron/attention.py`（同 ID）+ `ops/attention.py` | 已迁移（已验证） | 框架契约保留；实现选择下沉 ops 普通函数；TE (-1,0)/(-1,-1) 窗口归一化；packed THD 与空序列梯度边；force 不降级 | `tests/test_attention_ops.py`、`tests/test_attention.py`、`tests/integration/test_attention_musa.py` | MUSA：bf16/GQA/fp32 前向+反向、数值对比、force 拒绝、策略切换、mate 192、packed THD（随 S2c 测试落档更新） |
| `transformers.qwen3-vl.text-rms-norm.fused-torch` | `patches/transformers/rms_norm.py`（同 ID） | 已迁移（已验证） | dtype/device 完全匹配才走 torch.rms_norm；非 MUSA 输入保护；提升与 opt-in 保留 | `tests/test_transformers_rms_norm.py`、`tests/integration/test_rms_norm_musa.py` | MUSA：无 Megatron 自动激活、前向/反向、dtype 提升（随 S1e 落档更新） |
| `torch.cuda.compat-layer` | `patches/platform.py` + `backends/torch_cuda.py`（同 ID） | 已迁移（待验证） | 幂等 helper；namespace 根 trigger 改为具体边界；两种导入顺序 | `tests/test_torch_cuda.py` | 随 S1e 落档更新 |

## 待迁移（S3 批次）

| 来源 ID（megatron-musa-patch） | 目标位置 | 批次 |
|---|---|---|
| `megatron.te.norm.unfused-musa`、`megatron.te.layer-norm-linear.unfused`、`megatron.te.fused-mlp.unfused`、`megatron.fusions.fused-layer-norm.*`（2）、`megatron.fusions.persist-layer-norm.disable`、`megatron.transformer-block.layer-norm.impl-local` | `patches/megatron/layer_norm.py` | S3a |
| `megatron.te.factory-shim.torchscript-compat`、`megatron.te.utils-module.safe-seed`、`megatron.te.cpu-offload-context.signature-dispatch`、`megatron.te.make-weak-ref.graph-compat`、`megatron.te.grouped-linear.mem-monitor-compat`、`megatron.te.quantized-model-init.delayed-compat`、`transformer_engine.layer-norm-linear.native-unfused`、`transformer_engine.layer-norm-mlp.native-unfused`、`transformer_engine.grouped-gemm.wgrad-reference`、`transformer_engine.saved-tensors.offload-markers`、`transformer_engine.dot-product-attention.capability-dispatch` | `patches/transformer_engine.py` | S3b |
| `megatron.moe.*`（8）、`megatron.moe.grouped-gemm.*`（3）、`megatron.softmax.kernel-availability.musa` | `patches/megatron/moe.py`、`patches/megatron/softmax.py` | S3c |
| `megatron.embeddings.*`（3 RoPE）、`megatron.ssm.gated-delta-rule.tilelang`、`mcore_bridge.ssm.gated-delta-rule.tilelang` | `patches/megatron/rope.py`、`patches/megatron/ssm.py`、`patches/mcore_bridge.py`、`ops/gated_delta_rule.py` | S3d |
| `megatron.dist-ckpt.*`（2）、`megatron.serialization.signal-member-globals`、`megatron.training.*`（6）、`torch.distributed.clean-teardown`、`megatron.te.*graph*`、`megatron.offloading.resident-parameters`、`megatron.fsdp.premul-sum.device-prescale`、`megatron.training.checkpoint.host-barrier-*`（2）、`megatron.training.get-device-arch-version.nvidia-scale`、`torch.cuda.device-capability.nvidia-scale`、`megatron.legacy.fused-kernels.load.noop`、`megatron.hyper-comm-grid.subgroups-backend`、`megatron.bridge-communicator.subgroups-backend` | `patches/megatron/checkpointing.py`、`training.py`、`distributed.py`、`cuda_graphs.py`、`offloading.py`、`delayed_wgrad.py`、`control_collectives.py`、`device_arch.py` | S3e |

## 命名与配置迁移（S4）

| 来源 | 目标 | 差异 |
|---|---|---|
| `MEGATRON_MUSA_PATCH*` | `TRAINING_MUSA_ADAPTOR_*` | 见 §9.2 表；工具：`tools/migrate_legacy_config.py`（待改写） |
| 第一轮 `MUSA_ADAPTER_*` | 同上 | JSON 数组→逗号列表；bindings/operators 配置→attention/patch_options |
| 第一轮 patch ID（`*.dispatch`） | 旧 ID | 已按 §9.2 映射回旧 ID，台账标注 |
