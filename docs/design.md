# training-musa-adaptor 架构设计入口

**后续重构以 [软件架构与 Patch 设计 v2.0](../../megatron-musa-patch/docs/MUSA_ADAPTER_DESIGN_zh.md) 为准。** 更新日期：2026-09-21。

项目目标名称为 `training-musa-adaptor`，Python 包为 `training_musa_adaptor`，环境变量前缀为 `TRAINING_MUSA_ADAPTOR_`。当前目录、包和实现仍使用 `musa-adapter` / `musa_adapter`；本次仅修订设计，实际更名及代码重构尚未执行。

## 后续实现方向

- 沿用 `megatron-musa-patch` 的 `AttrPatch / HookPatch`、`replace(original)` 和显式 `PATCHES` 列表。
- 普通补丁在一个文件中说明目标、替换逻辑、适用条件和维护理由；引擎内部负责所有权、幂等及撤销。
- 只有确实共用的 attention、GDN 等逻辑才提取为 `ops/` 普通函数。
- 不再要求 PatchSpec / BindingSpec / ProviderSpec、Action / MutationPlan、通用 Planner 或统一算子协议。
- 保留第一轮的计算修复、支持范围、数值/梯度测试和诊断证据，重新验证重构后的行为。

完整目录、补丁示例、导入时序、配置、命名迁移和验收规则均在权威文档中维护，本文件不复制另一份协议。

## 与当前代码的关系

以下描述的是第一轮 v1.0 实现快照，**不是 v2.0 的完成清单或后续架构要求**：

| 当前实现 | v2.0 去向 |
|---|---|
| `runtime/specs.py`、`registry.py`、`planner.py` | 用旧式两种补丁记录和必要校验替代；不保留通用调度协议 |
| `runtime/imports.py`、`lifecycle.py`、`transactions.py` | 对照旧引擎回归收敛为 `_engine.py` / `_imports.py` 的内部机制 |
| `patches/*` 的 Action 与 Binding | 普通工厂、wrapper 和 PATCHES |
| `operators/`、`providers/` | 简单实现收回 patch，真实共享部分整理为 ops 函数 |
| `config/`、`diagnostics/` | 简化配置入口和小型报告 API/CLI |
| `tests/contracts/`、`tests/integration/` | 保留真实行为回归，更新只依赖旧协议形状的测试 |

当前 `AGENTS.md`、README、配置示例和 [ADR-0001](decisions/0001-naming-and-protocol-baseline.md) 描述第一轮实现。实际重构的 S0 阶段必须按 v2.0 同步更新；其中三种单元分离、强制 MutationPlan、旧配置层级等约定已被本次明确的设计修订取代，不应继续据此扩展旧架构。ADR-0001 作为历史决策保留。

## 已有工作与尚未完成事项

[迁移台账](MIGRATION_LEDGER.md) 保留第一轮的逐项记录、硬件验证和已知缺口。其 P4/P5 标签及旧章节号属于 v1.0 阶段划分；后续按 v2.0 的 S0–S4 迁移时更新，不把已有结果直接视为重构后的通过证据。

第一轮已具备 RMSNorm 和 Megatron attention 样例，并记录了 TE causal 窗口归一化、mate head dim、packed offsets/空序列等验证。其余旧补丁仍需逐项迁移，完整真实训练与性能矩阵以台账中的实际证据为准。

当前代码中“patch 应用失败只记录、不向框架导入传播”的处理不符合 v2.0：后续须清理已拥有的变更并报告原始失败，不能依赖未适配路径自行报错。普通 `import torch` 自动入口保留独立异常边界。

名称替换、代码收敛、配置转换和测试迁移按权威文档第 9–10 节实施；本次文档更新不表示这些工作已经完成。
