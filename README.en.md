# training-musa-adaptor

[中文（默认）](README.md) · [Contributor guide](CONTRIBUTING.md) · [Agent rules](AGENTS.md) · [Patch ledger](docs/PATCH_LEDGER.md)

`training-musa-adaptor` provides runtime compatibility patches and operator selection for
training on Moore Threads MUSA GPUs. Its goal is to preserve upstream source code, entry
points and training configuration **within explicitly validated stacks and workloads**.
It targets selected transformers, deepspeed, megatron and related training-framework call sites.

The package registers through PyTorch's `torch.backends` entry point and applies patches
when relevant framework modules load. It does not ship drivers, a training framework or
model recipes, and does not automatically install or upgrade the vendor stack.

Status: **pre-alpha**. Implemented patches, matching version gates, unit coverage and
end-to-end training validation are different milestones. The
[patch ledger](docs/PATCH_LEDGER.md) is the authoritative per-patch status record.

## Features

- Device-interface compatibility through torchada and targeted contract fixes.
- Local patches for norm, RoPE, MoE, GEMM and selected training-system interfaces.
- Configurable attention selection across integrated implementations such as MuDNN,
  mate and TE unfused, with input-specific capability checks.
- Shared GDN dispatch at independent Megatron and mcore-bridge call sites.
- Per-patch switches, effective-configuration inspection and in-process diagnostics.

Checkpoint, communication, offload and graph behavior require their own validation;
the existence of a patch does not establish support for an entire training framework.

## Quick start

Use Python **3.10+** with matching MUSA drivers, `torch`, `torch_musa`, `torchada` and
the target framework already installed. Optional acceleration packages depend on the
selected path. All workers need consistent software and configuration.

From this repository, using the training interpreter:

```bash
# Preinstall packaging and, on Python 3.10, tomli as declared in pyproject.toml.
python3 -m pip install --no-deps .
python3 -m training_musa_adaptor list
python3 -m training_musa_adaptor config
```

Use `--no-deps -e .` for development. `PYTHONPATH` alone does not install the autoload
entry point. Do not activate another adaptor engine that modifies the same targets.

With transformers installed, this small check uses no model download or Megatron:

```python
import torch
import torch_musa
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextRMSNorm

import training_musa_adaptor as tma

assert torch.musa.is_available(), "Check the MUSA runtime and visible devices"
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

Run in a fresh process with autoload enabled. This checks activation and a forward/backward
pass, not numerical equivalence or complete training. Use a validated transformers version
that contains this model class. See the [RMSNorm tests](tests/integration/test_rms_norm_musa.py)
for broader checks.

Then run your existing training command in that environment. Normal use requires no adaptor
import; each worker activates independently. The following shows optional configuration and
the command structure, not a complete training recipe:

```bash
export TRAINING_MUSA_ADAPTOR_ATTN_POLICY=prefer
export TRAINING_MUSA_ADAPTOR_ATTN_IMPLS=mate,mudnn
# torchrun --nproc_per_node=2 /path/to/Megatron-LM/pretrain_gpt.py <existing arguments>
```

## Configuration

Priority: **defaults < explicit TOML < environment**. Within one source, per-patch options
win over generic attention options; lists replace rather than merge. Automatic activation
freezes configuration at the first relevant execution boundary; explicit `install()` freezes
it immediately, including the DEBUG log level. Configure before starting workers and use a
fresh process for changes.
There is no implicit config-file search. A fully commented sample covering every TOML
field lives at
[`examples/configs/training-musa-adaptor.toml`](examples/configs/training-musa-adaptor.toml);
copy it and edit as needed.

| Variable | Default | Purpose |
|---|---|---|
| `TRAINING_MUSA_ADAPTOR_ENABLED` | `1` | Master switch; `0` blocks both activation channels |
| `TRAINING_MUSA_ADAPTOR_AUTOLOAD` | `1` | Automatic channel only |
| `TRAINING_MUSA_ADAPTOR_CONFIG` | unset | Explicit TOML path |
| `TRAINING_MUSA_ADAPTOR_ONLY` | empty | Comma-separated patch whitelist |
| `TRAINING_MUSA_ADAPTOR_DISABLE` | empty | Comma-separated patch denylist |
| `TRAINING_MUSA_ADAPTOR_ATTN_POLICY` | `auto` | Selection policy |
| `TRAINING_MUSA_ADAPTOR_ATTN_IMPLS` | empty | Ordered implementation names |
| `TRAINING_MUSA_ADAPTOR_ATTN_FALLBACK` | `reference` | `reference / upstream / error` |
| `TRAINING_MUSA_ADAPTOR_DEBUG` | `0` | Emit patch-state logs at INFO; configure the application's logger to display them |

Booleans accept `0/1/true/false`. Lists use commas, not JSON arrays. Unknown values,
duplicate items and empty middle items are rejected. **Nonempty ONLY overrides DISABLE**;
both lists are validated. ONLY also filters device hooks and prerequisites without enabling
dependencies automatically. Use DISABLE when excluding a single patch. An entry may also be a
**suite name**, expanding to every patch id of that suite: `megatron` (the whole
patches/megatron/ adaptation, including its TE/torch-target patches), `transformer_engine`,
`transformers`, `mcore_bridge`, `platform`. For example, to hand the Megatron adaptation
to another implementation wholesale:

```bash
export TRAINING_MUSA_ADAPTOR_DISABLE=megatron
```

| Policy | Implementations | Behavior |
|---|---|---|
| `auto` | empty | Default candidates, then fallback |
| `prefer` | nonempty ordered list | Listed candidates, remaining defaults, then fallback |
| `force` | exactly one | Unsupported MUSA inputs fail; no fallback |
| `upstream` | empty | Original arguments go directly to the captured original function |

Declared names: `mudnn`, `mate`, `te_unfused`, `flash_attn`, `torch_sdpa_math`. A valid name
does not guarantee an installed backend or compatible input. Current default accelerated
order is `mudnn → mate`, not a performance ranking for every device.

Fallback applies only to auto/prefer: `reference` uses an integrated reference path;
`upstream` still needs the original path to be supported; `error` fails immediately.
Force/upstream normalize fallback to `error`. CPU/CUDA inputs retain the original behavior.
Errors after kernel execution starts propagate without retrying another implementation.

```toml
[patches]
only = []
disable = []

[attention]
policy = "prefer"
implementations = ["mate", "mudnn"]
fallback = "reference"

# Optional call-site override:
[patch_options."megatron.te.attention.capability-dispatch"]
policy = "force"
implementations = ["mate"]
```

Set `TRAINING_MUSA_ADAPTOR_CONFIG` to the file's absolute path and inspect `config`.
Only declared attention options are accepted in `patch_options`. Bootstrap switches remain
environment variables; there is no `[runtime]` table.

## Diagnostics and explicit activation

```bash
training-musa-adaptor list
training-musa-adaptor config
training-musa-adaptor report --json
```

The CLI reports its own process, not a running worker. Inspect the actual worker using:

```python
import json
import training_musa_adaptor as tma

print(json.dumps(tma.report(), indent=2, default=str))
```

States are `pending`, `applied`, `skipped`, `failed`, `reverted`; inspect `detail` for reasons.
Applied means a binding/hook was installed, not that every input used an accelerated kernel.
`last_attention_dispatch`, when present, describes the last successful dispatcher call per
site. Direct passthrough does not update it. `bootstrap_errors`, `cleanup_pending` and
`restart_required` expose initialization, incomplete cleanup and irreversible-hook effects.

For custom integration, call `tma.install()` or `tma.install(config_path="/absolute/path/config.toml")`
before importing target frameworks. An explicit config_path takes precedence over the
CONFIG environment variable when choosing a file; environment field overrides still apply.
Importing the adaptor alone does not install patches.
Diagnostic `tma.apply(patch_ids=(... ,))` requires explicit IDs. `tma.uninstall()` restores only
bindings still owned by this package and must run outside active training. It cannot reset
all imported vendor libraries, existing instances or compilation caches. Retry failed cleanup
before reinstalling; use a new process for configuration experiments.

## Limitations and troubleshooting

Check installation in the actual interpreter, master/autoload switches and PyTorch's
`TORCH_DEVICE_BACKEND_AUTOLOAD` if activation is missing. Inspect import order for pending
patches or `phase_missed` hooks. Review version-gate evidence instead of blindly widening it.
An explicit upstream attention policy does not undo device compatibility or other patches;
use `TRAINING_MUSA_ADAPTOR_ENABLED=0` in a fresh process for a fully disabled comparison.

RoPE may retain the upstream path when optional apex packages/extensions are absent; ABI or
internal-dependency failures in installed extensions propagate as errors.

Current recorded gaps include native TE DPA support, some modeling-first import orders,
full ms-swift/mcore-bridge training and recovery validation, and graph support on the recorded
vendor stack. See the [ledger](docs/PATCH_LEDGER.md). Historical test results do not prove
support for other stacks.

Development instructions and executable patch examples are maintained in the Chinese
[CONTRIBUTING guide](CONTRIBUTING.md); Coding agents must also read [AGENTS.md](AGENTS.md).
Architecture context: [docs/design.md](docs/design.md). License: [LICENSE](LICENSE).
