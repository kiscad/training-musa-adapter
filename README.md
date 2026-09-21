# training-musa-adaptor

Minimal, reversible runtime patches that let upstream training frameworks
(Megatron-Core, transformers, ms-swift, mcore-bridge, ...) train on Moore
Threads MUSA GPUs **with their original source code, entry points and
training configs**.

> Status: pre-alpha.  The architecture baseline is
> `megatron-musa-patch/docs/MUSA_ADAPTER_DESIGN_zh.md` (v2.0); this repo is
> the refactored successor of both `megatron-musa-patch` and the first-round
> `musa-adapter` prototype.  Everything not listed as migrated+verified in
> `docs/MIGRATION_LEDGER.md` is **not** yet supported.

## Quick start

```bash
pip install --no-deps .            # vendor stack (torch/torch_musa/...) stays untouched
```

No source changes, no adaptor imports, no device-string edits:

```bash
export TRAINING_MUSA_ADAPTOR_ATTN_POLICY=prefer
export TRAINING_MUSA_ADAPTOR_ATTN_IMPLS=mate
torchrun --nproc_per_node=8 pretrain_gpt.py ...   # original entry point
```

Switches (design doc §7.1):

| Variable | Meaning |
|---|---|
| `TRAINING_MUSA_ADAPTOR_ENABLED=0/1` | master switch (hard exit) |
| `TRAINING_MUSA_ADAPTOR_AUTOLOAD=0/1` | only disables the automatic channel |
| `TRAINING_MUSA_ADAPTOR_CONFIG=path` | one explicit TOML file (no implicit search) |
| `TRAINING_MUSA_ADAPTOR_ONLY=a,b` | comma-separated patch ID whitelist (wins over DISABLE) |
| `TRAINING_MUSA_ADAPTOR_DISABLE=a,b` | comma-separated patch ID denylist |
| `TRAINING_MUSA_ADAPTOR_ATTN_POLICY` | `auto / prefer / force / upstream` |
| `TRAINING_MUSA_ADAPTOR_ATTN_IMPLS` | comma-separated implementation names (e.g. `mate,mudnn`) |
| `TRAINING_MUSA_ADAPTOR_ATTN_FALLBACK` | `reference / upstream / error` |
| `TRAINING_MUSA_ADAPTOR_DEBUG=0/1` | verbose per-patch logging |

Diagnostics (read-only; the CLI process is never the training process):

```bash
training-musa-adaptor list
training-musa-adaptor config
training-musa-adaptor report --json
```

Python API (explicit channel; `import training_musa_adaptor` never patches):

```python
import training_musa_adaptor as tma
tma.install()                       # install the import watcher
records = tma.report()              # read-only process state
tma.apply(patch_ids=("transformers.qwen3-vl.text-rms-norm.fused-torch",))
tma.uninstall()                     # undo owned changes
```

## Writing a patch

A patch is a small declarative record in one file; the engine handles
ownership, idempotence and undo (design doc §4):

```python
# patches/transformers/rms_norm.py
from functools import wraps
from ..._engine import AttrPatch


def replace_rms_norm(original):
    import torch  # factories run only when the target module is ready

    @wraps(original)
    def forward(self, hidden_states, *args, **kwargs):
        if (args or kwargs
                or type(hidden_states) is not torch.Tensor
                or hidden_states.device.type != "musa"
                or self.weight.device != hidden_states.device
                or hidden_states.dtype != self.weight.dtype):
            return original(self, hidden_states, *args, **kwargs)
        return torch.rms_norm(
            hidden_states, (hidden_states.shape[-1],),
            self.weight, self.variance_epsilon,
        )

    return forward


PATCHES = (
    AttrPatch(
        id="transformers.qwen3-vl.text-rms-norm.fused-torch",
        target="transformers.models.qwen3_vl.modeling_qwen3_vl:Qwen3VLTextRMSNorm.forward",
        replace=replace_rms_norm,
        version_gates=("transformers >=4.57",),
        rationale="upstream RMSNorm small-op chain costs several kernel launches",
        strategy="dtype-matched MUSA path uses the fused torch.rms_norm; contracts kept",
        upstream="transformers/models/qwen3_vl/modeling_qwen3_vl.py:Qwen3VLTextRMSNorm.forward",
        remove_when="upstream fuses the op and the regressions pass with the patch disabled",
    ),
)
```

See `AGENTS.md` for development conventions and `docs/MIGRATION_LEDGER.md`
for the per-patch migration/verification status.
