# training-musa-adaptor

[English](README.en.md) · [贡献指南](CONTRIBUTING.md) · [Agent 开发规范](AGENTS.md) · [支持与验收台账](docs/PATCH_LEDGER.md)

`training-musa-adaptor` 为摩尔线程 MUSA 训练环境提供运行时适配与算子选择。
目标是在**经过验证的软件栈和训练场景内**，让 transformers、deepspeed、megatron 及其上层训练框架
沿用原始源码、入口和训练配置，在 MUSA 上正确运行，并使用适用的计算实现。

安装后，本库通过 PyTorch 的自动加载入口注册补丁，在相关框架模块加载时生效。
补丁集中维护在本库，不需要把修改散落到上游源码或 `site-packages` 中。
本库不提供 MUSA 驱动、完整训练框架或模型训练配置，也不自动安装、升级厂商计算栈。

当前处于 **pre-alpha**。补丁实现、版本门控命中、单元测试通过和完整训练验证是不同状态；
实际支持范围以[逐 ID 台账](docs/PATCH_LEDGER.md)为准。

## 能做什么

| 功能 | 用途与边界 |
|---|---|
| 设备接口适配 | 在框架导入边界接入 torchada，并补偿已知设备接口差异 |
| 局部框架兼容 | 针对 norm、RoPE、MoE、GEMM、训练初始化等具体调用点安装小型补丁 |
| Attention 实现选择 | 通过配置选择 MuDNN、mate、TE unfused 等已接入路径；按输入能力决定是否可用 |
| GDN 接入 | 为 Megatron 和 mcore-bridge 的独立调用点复用 GDN 分发逻辑 |
| 训练系统适配 | 包含 checkpoint、控制通信、offload 等局部适配；各场景单独验收 |
| 诊断与开关 | 按 patch ID 启停，查看有效配置、应用状态、跳过或失败原因 |

单个补丁通常就是目标路径、替换函数和维护说明；需要新增适配时，见
[CONTRIBUTING.md](CONTRIBUTING.md)。

## 快速上手

### 1. 准备环境并安装

需要 Python **3.10+**，以及已安装并相互匹配的 MUSA 驱动、`torch`、`torch_musa`、
`torchada` 和目标训练框架。TE、mate、torch-kernels 等由所选功能决定，不要求全部安装。
确认各 worker 使用相同的软件环境和适配配置。

以下在本仓库根目录执行，使用训练任务所用的 Python：

```bash
# --no-deps 使用已有环境，不让安装命令解析或替换依赖。
# 请预先准备 pyproject.toml 声明的 packaging，以及 Python 3.10 所需的 tomli。
python3 -m pip install --no-deps .

python3 -m training_musa_adaptor list
python3 -m training_musa_adaptor config
```

开发时将安装命令换成 `python3 -m pip install --no-deps -e .`。
只设置 `PYTHONPATH` 不能替代安装：自动加载依赖安装后的 `torch.backends` entry point。
不要在同一进程同时激活改写相同目标的其他适配引擎。

### 2. 做一次小规模检查

已安装 transformers 的 MUSA 环境可以执行下面的例子。它只构造一个 RMSNorm，
不下载模型，也不需要 Megatron；用于确认自动激活和一次前向/反向，而非完整训练验收。
请在新进程中执行，且不要设置关闭自动加载的开关。

```python
import torch
import torch_musa
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextRMSNorm

import training_musa_adaptor as tma

assert torch.musa.is_available(), "请先确认 MUSA 环境和设备可见性"
patch_id = "transformers.qwen3-vl.text-rms-norm.fused-torch"
assert tma.is_applied(patch_id), tma.report()

norm = Qwen3VLTextRMSNorm(128).to(device="musa", dtype=torch.bfloat16)
x = torch.randn(2, 8, 128, device="musa", dtype=torch.bfloat16, requires_grad=True)
y = norm(x)
y.float().square().mean().backward()
assert torch.isfinite(y).all().item()
assert torch.isfinite(x.grad).all().item()
print("MUSA RMSNorm forward/backward OK")
```

例子使用 torch-first 导入顺序。若版本缺少该模型类，先按台账选择已有验证的组合，
不要据此升级整个计算栈。更完整的数值对照见
[硬件 RMSNorm 测试](tests/integration/test_rms_norm_musa.py)。

### 3. 启动原训练任务

通常无需增加适配 import。安装本库后，在同一环境运行已有训练脚本，
PyTorch 自动入口会在每个 worker 内分别注册补丁。以下是命令结构，路径、卡数和
省略的训练参数必须替换成你已有的配置；它不是一份完整训练配方：

```bash
# 默认 attention 为 auto；也可以在启动前指定偏好。
export TRAINING_MUSA_ADAPTOR_ATTN_POLICY=prefer
export TRAINING_MUSA_ADAPTOR_ATTN_IMPLS=mate,mudnn
# torchrun --nproc_per_node=2 /path/to/Megatron-LM/pretrain_gpt.py <原有训练参数>
```

默认使用环境内已验证的路径，不要求先设置 attention 选项。`prefer` 不是强制成功；
若要检验某个实现是否覆盖你的输入，使用下面的 `force` 策略。

## 配置

优先级为 **内置默认值 < 显式 TOML < 环境变量**。同一来源内，patch 专项配置覆盖
通用 attention 项；更高来源仍优先。列表整体替换，不合并。

自动激活在第一个相关模块执行边界解析并冻结配置；显式 `install()` 立即冻结。
请在启动 worker 前完成配置，变更配置后使用新进程。DEBUG 日志级别也随配置冻结。
没有当前目录或 HOME 下的隐式配置搜索。
完整带注释的配置样例见
[`examples/configs/training-musa-adaptor.toml`](examples/configs/training-musa-adaptor.toml)：
包含全部字段、策略约束与常见取值，复制后按需修改。

### 环境变量

| 变量 | 默认值 | 用途 |
|---|---|---|
| `TRAINING_MUSA_ADAPTOR_ENABLED` | `1` | 总开关；`0` 同时关闭自动与显式激活 |
| `TRAINING_MUSA_ADAPTOR_AUTOLOAD` | `1` | 自动通道开关；`0` 时显式 API 仍可使用 |
| `TRAINING_MUSA_ADAPTOR_CONFIG` | 未设置 | 指定一个 TOML 文件 |
| `TRAINING_MUSA_ADAPTOR_ONLY` | 空 | 逗号分隔的 patch ID 白名单 |
| `TRAINING_MUSA_ADAPTOR_DISABLE` | 空 | 逗号分隔的 patch ID 禁用列表 |
| `TRAINING_MUSA_ADAPTOR_ATTN_POLICY` | `auto` | `auto / prefer / force / upstream` |
| `TRAINING_MUSA_ADAPTOR_ATTN_IMPLS` | 空 | 有序实现名，如 `mate,mudnn` |
| `TRAINING_MUSA_ADAPTOR_ATTN_FALLBACK` | `reference` | `reference / upstream / error` |
| `TRAINING_MUSA_ADAPTOR_DEBUG` | `0` | 将补丁状态日志提升到 INFO；需应用的日志配置输出该级别 |

布尔值接受 `0/1/true/false`，忽略大小写。列表使用逗号分隔，不使用 JSON 数组；
未知 ID、实现名、配置字段、重复项和中间空项会报错。列表条目也可以是**套件名**，
展开为该套件下全部补丁 ID：`megatron`（patches/megatron/ 全部适配，含其中
TE/torch 目标补丁）、`transformer_engine`、`transformers`、`mcore_bridge`、
`platform`。例如把 Megatron 适配整体交给另一套实现时：

```bash
export TRAINING_MUSA_ADAPTOR_DISABLE=megatron
```

**ONLY 非空时忽略 DISABLE**，但两者中的 ID 都会校验；ONLY 为空时使用默认集合减去 DISABLE。
ONLY 也会过滤设备 Hook 和前置补丁，不会自动补齐依赖。仅排除一个补丁时优先用 DISABLE：

```bash
export TRAINING_MUSA_ADAPTOR_DISABLE=transformers.qwen3-vl.text-rms-norm.fused-torch
```

### Attention 策略

| 策略 | implementations | 行为 |
|---|---|---|
| `auto` | 空 | 按当前内置候选顺序选择，全部拒绝后按 fallback |
| `prefer` | 非空有序列表 | 先尝试指定项，再尝试其余默认候选，最后按 fallback |
| `force` | 恰好一个名字 | MUSA 输入必须由该实现执行，不适用就报错，不降级 |
| `upstream` | 空 | 原参数交回捕获的原函数，不保证该路径可在 MUSA 上工作 |

已声明实现名为 `mudnn`、`mate`、`te_unfused`、`flash_attn`、`torch_sdpa_math`。
名字有效不代表依赖已安装或任意输入可用。当前默认加速候选顺序是 `mudnn → mate`，
这是当前实现的选择顺序，不是面向所有设备的性能排名。

`reference` 只使用调用点明确接入的参考路径；`fallback=upstream` 仍需确认原路径能力；
`error` 表示无候选就失败。fallback 只用于 auto/prefer，force/upstream 会归一化为 `error`。
CPU/CUDA 输入保留原函数行为。kernel 开始执行后的异常直接传播，不换实现重跑。

```bash
export TRAINING_MUSA_ADAPTOR_ATTN_POLICY=force
export TRAINING_MUSA_ADAPTOR_ATTN_IMPLS=mate
```

`upstream` 仅控制这个 attention 调用点，不撤销设备层或其他补丁。
需要比较完全关闭本库的行为时，在新进程设置 `TRAINING_MUSA_ADAPTOR_ENABLED=0`。

### TOML 文件

保存为 `training-musa-adaptor.toml`：

```toml
[patches]
only = []
disable = []

[attention]
policy = "prefer"
implementations = ["mate", "mudnn"]
fallback = "reference"

# 只有需要覆盖通用 attention 配置时才添加此段。
[patch_options."megatron.te.attention.capability-dispatch"]
policy = "force"
implementations = ["mate"]
```

```bash
export TRAINING_MUSA_ADAPTOR_CONFIG=/absolute/path/training-musa-adaptor.toml
python3 -m training_musa_adaptor config
```

`patch_options` 目前仅支持已声明的 attention 字段，不是任意属性配置。
ENABLED/AUTOLOAD 等启动开关使用环境变量；TOML 没有 `[runtime]` 表。

## 确认补丁是否生效

```bash
training-musa-adaptor list           # 声明的 ID 与目标
training-musa-adaptor config         # 此进程的有效配置、来源及归一化说明
training-musa-adaptor report --json  # 此 CLI 进程的状态
```

CLI 不连接正在训练的 worker。查看训练状态，应在该 worker 的调试入口调用：

```python
import json
import training_musa_adaptor as tma

print(json.dumps(tma.report(), indent=2, default=str))
```

| 状态 | 含义 |
|---|---|
| `pending` | 已登记，相关路径尚未触发 |
| `applied` | 属性已替换或 Hook 已完成；不等于某次输入一定使用加速 kernel |
| `skipped` | 被配置、版本、依赖或工厂条件排除；看 `detail` |
| `failed` | 导入、应用或清理失败；看异常及 `detail` |
| `reverted` | 本包拥有的可逆绑定已撤销 |

`last_attention_dispatch` 若存在，记录各调用点最后一次成功 dispatcher 调用的实现、
入口和拒绝原因；直接 passthrough 不更新，不是每个 batch 的执行轨迹。
`bootstrap_errors`、`cleanup_pending` 和 `restart_required` 分别帮助识别初始化错误、
未完成的清理和不可逆 Hook 效果。report 本身不触发适配、设备初始化或通信。

### 显式 API

自定义接入和诊断可以在导入目标框架前显式安装：

```python
import training_musa_adaptor as tma

tma.install()  # 可传 config_path="/absolute/path/config.toml"
# 随后导入并使用目标框架。
# 诊断时可主动导入指定目标；apply() 不接受省略 ID 的全量导入。
# tma.apply(patch_ids=("transformers.qwen3-vl.text-rms-norm.fused-torch",))
# 仅在训练结束或隔离测试边界清理：
# tma.uninstall()
```

`install(config_path=...)` 指定的文件优先于 `TRAINING_MUSA_ADAPTOR_CONFIG`，
但文件内容仍会被环境变量覆盖。单独 `import training_musa_adaptor` 不安装补丁。
显式 install 会处理允许晚应用的已加载属性，
但不能修复所有已有实例、缓存或错过时机的 Hook。uninstall 只恢复本包仍拥有的绑定，
不会覆盖第三方后写对象，也不能完全撤销 torchada、扩展注册或编译缓存。切换实验请用新进程。

## 常见问题与已知边界

| 现象 | 检查方向 |
|---|---|
| 只有源码，没有自动生效 | 确认安装到了训练解释器；检查 ENABLED、AUTOLOAD 和 PyTorch 的 `TORCH_DEVICE_BACKEND_AUTOLOAD` |
| `pending` 或 `phase_missed` | 检查实际导入路径和顺序；已错过导入前 Hook 的进程应重新启动 |
| `version gate` 跳过 | 核对发行包版本与实际厂商源码；不要把放宽 gate 当作兼容修复 |
| `force` 报不支持 | 查看实现拒绝原因，核对 dtype、布局、mask、head dim、反向和依赖能力 |
| 清理失败 | 保留异常，排除原因后重试 uninstall；未完成前不重装 |
| 引擎冲突 | 新进程中只激活一套引擎，检查遗留 editable 安装与启动变量 |

RoPE 的可选 apex/扩展缺失可以保留上游路径；已安装扩展的 ABI 或内部依赖损坏会报错，
不会按“未安装”处理。

当前台账仍记录 TE 原生 DPA 未实现、部分 modeling-first 顺序错过属性补丁、
完整 ms-swift/mcore-bridge 入口与恢复语义待验收，以及当前验证栈上的 CUDA graph 限制。
版本门控中更宽的源码核对范围不等于硬件验证范围。不能据单算子或 smoke 结果
推断所有模型的兼容性与吞吐。

## 参与开发

[CONTRIBUTING.md](CONTRIBUTING.md) 解释生效原理、开发示例、测试与提交检查；
[AGENTS.md](AGENTS.md) 约束 Coding agent 的工作流程；
[设计入口](docs/design.md) 说明本项目的架构约束与文档分工。许可证见 [LICENSE](LICENSE)。
