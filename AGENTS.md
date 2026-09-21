# training-musa-adaptor：Agent 开发操作手册

本文是本目录及子目录的 coding agent 工作约定。权威架构基线是
`megatron-musa-patch/docs/MUSA_ADAPTER_DESIGN_zh.md`（v2.0）；
`docs/design.md` 只做入口指向。适用的上级约定和用户明确指令优先。

## 1. 目标与完成标准

在明确验证过的框架、软件栈、硬件和负载组合内，让开源训练框架沿用
原始源码、入口和训练配置，在 MUSA 上正确、高效地训练。验收面：

| 验收面 | 必须保持的行为 |
|---|---|
| 原始训练入口 | 无需修改源码、插入适配 import、改设备字符串或关闭原本功能 |
| 上游单元测试 | 直接运行原始测试与断言；不新增 skip/xfail 掩盖失败 |
| 数值/梯度/优化器/RNG/分布式/checkpoint | 语义保持；数值回退要有参考实现与容差依据 |
| 多实现选择 | 用户只改配置即可在已验证实现间切换；能查明选中/拒绝/回退原因 |
| 可维护性 | 普通补丁的目标、工厂、条件与维护说明在一个文件内可读 |

"先跑起来"不是缩小目标。未验证能力不得写成已支持；
`docs/MIGRATION_LEDGER.md` 是唯一逐 ID 状态表。

## 2. 不得破坏的约束

1. **改动收敛在本仓库**；不修改上游安装源码，不复制整份上游模块，
   不修改上游测试断言。不通过依赖/import 旧包（megatron-musa-patch、
   第一版 musa-adapter）运行；检测到已知旧引擎活跃必须拒绝。
2. **补丁写法**：普通补丁只用 `AttrPatch`/`HookPatch` + 普通函数 +
   `PATCHES`；不需要 PatchSpec/BindingSpec/ProviderSpec、Action、
   MutationPlan、通用 Planner 或统一算子协议。工厂只构造对象，不自行
   写目标属性；`replace(original)` 返回 `None` 表示主动放弃，不能吞异常。
3. **顶层只读**：`patches/` 被导入时顶层只用标准库和本包轻量模块；
   torch/框架/加速库在工厂或安全初始化函数内部加载。
4. **引擎框架无关**：`_engine.py`/`_imports.py` 不导入 torch、框架或
   kernel，不增加框架名称分支。Hook 只在真实 exec 边界运行，find_spec
   只查找和包装；namespace 根包不是默认可执行边界。
5. **共享逻辑**：确实两个调用方才提取到 `ops/` 或 `backends/` 的普通
   幂等函数；单调用点的转换逻辑留在补丁文件内。
6. **生命周期**：重复 install/apply 不叠加 wrapper；uninstall 只撤销
   本包仍拥有的绑定，不覆盖第三方后写对象；清理失败禁止直接重装。
7. **配置**：逗号分隔环境变量、显式 TOML、第一次相关模块真实执行前
   冻结；未知字段/ID/实现名/枚举报错，不静默回 auto；总开关硬退出。
8. **依赖规则**：`requires` 只引用同模块不同属性的 patch ID；循环、
   同目标和跨模块 requires 拒绝；消费者 skipped 记录原因，不自动启用前置。

## 3. 从哪里读、在哪里改

| 位置 | 职责 |
|---|---|
| `src/training_musa_adaptor/_engine.py` | AttrPatch/HookPatch、注册、绑定所有权、撤销、报告；仅通用机制 |
| `src/training_musa_adaptor/_imports.py` | finder/loader（exec 边界）；不做模型判断 |
| `src/training_musa_adaptor/_config.py` | 开关、显式 TOML、冻结；环境变量集中说明 |
| `src/training_musa_adaptor/_compat.py` | 目标解析、version_gates（延迟 packaging）、来源检查 |
| `src/training_musa_adaptor/activation.py` | 自动入口、install/apply/uninstall/report |
| `src/training_musa_adaptor/backends/` | 共享设备检查与 torchada 后的契约补偿（幂等 helper） |
| `src/training_musa_adaptor/patches/` | 具体补丁；`patches/__init__.py` 显式 MODULES 顺序 |
| `src/training_musa_adaptor/ops/` | 确有两处调用方的实现选择普通函数（attention、GDN） |
| `tests/` | 引擎/配置/补丁契约与集成回归；`integration/` 为子进程与硬件 |

## 4. 开工与验证

```bash
cd /path/to/training-musa-adaptor
git status --short && git log --oneline -3
python3 -m pip install --no-deps -e .
python3 -m pytest tests -q
python3 -m training_musa_adaptor list
```

- 新补丁：在对应 `patches/` 模块添加记录，填齐 rationale/strategy/
  upstream/remove_when；加入 `patches/__init__.py` 的 MODULES；同步
  `docs/MIGRATION_LEDGER.md`。
- 引擎/导入机制改动：必须重跑引擎生命周期测试（幂等、同目标链、
  有限 requires、descriptor/继承属性、别名、reload、失败回滚、第三方
  覆盖、不可逆项）。
- 硬件测试用 `musa` marker；报告列出实际收集与跳过原因，全 skip 不算通过。
- 纯文档改动只检查链接、示例及设计一致性，不要求启动训练。

## 5. 交付记录模板

```text
改动及原因：
保留的契约：
验证环境/命令/结果（收集/通过/失败/跳过与退出码）：
性能与限制：
台账位置：
```
