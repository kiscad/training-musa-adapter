"""Integration: transformers RMSNorm end-to-end on MUSA (S1 exit condition).

The transformers-only activation contract: the automatic channel must
apply the patch at the import boundary without any adaptor import in user
code, without Megatron installed/loaded, and with exact delegation semantics
for everything outside the fully matched fp16/bf16 MUSA path.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

torch = pytest.importorskip("torch")

musa_available = False
try:
    import torch_musa  # noqa: F401

    musa_available = torch.musa.is_available() and torch.musa.device_count() > 0
except Exception:
    musa_available = False

pytestmark = pytest.mark.musa


def _run(code: str, env_extra: dict[str, str] | None = None):
    env = {
        k: v for k, v in os.environ.items() if not k.startswith("TRAINING_MUSA_ADAPTOR")
    }
    env["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "1"  # the real automatic channel
    env.update(env_extra or {})
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        env=env,
        timeout=600,
        cwd="/tmp",
    )


@pytest.mark.skipif(not musa_available, reason="no live MUSA stack")
class TestRMSNormMusa:
    def test_automatic_channel_applies_without_megatron(self):
        result = _run("""
            import sys
            import torch  # the entry point installs the watcher here
            import torch_musa
            import transformers.models.qwen3_vl.modeling_qwen3_vl as m
            import training_musa_adaptor as tma
            rep = tma.report()
            by_id = {p["id"]: p["status"] for p in rep["patches"]}
            print("STATUS:", by_id["transformers.qwen3-vl.text-rms-norm.fused-torch"])
            print("MEGATRON-LOADED:", any(n == "megatron" for n in sys.modules))
            print("ATTN-PATCH:", by_id["megatron.te.attention.capability-dispatch"])
            """)
        assert result.returncode == 0, result.stderr
        assert "STATUS: applied" in result.stdout
        assert "MEGATRON-LOADED: False" in result.stdout
        assert "ATTN-PATCH: pending" in result.stdout

    def test_forward_backward_and_delegation(self):
        result = _run("""
            import torch
            import torch_musa
            import transformers.models.qwen3_vl.modeling_qwen3_vl as m
            import training_musa_adaptor as tma

            norm = m.Qwen3VLTextRMSNorm(1024, eps=1e-6).to("musa").to(torch.bfloat16)
            x = torch.randn(4, 64, 1024, device="musa", dtype=torch.bfloat16, requires_grad=True)
            out = norm(x)
            assert torch.isfinite(out).all()
            out.float().pow(2).mean().backward()
            assert x.grad is not None and torch.isfinite(x.grad).all()

            # fused vs upstream chain on the same input (bf16 tolerance)
            w = norm.weight
            hidden = x.detach().to(torch.float32)
            variance = hidden.pow(2).mean(-1, keepdim=True)
            reference = w * (hidden * torch.rsqrt(variance + 1e-6)).to(x.dtype)
            diff = (out.detach().float() - reference.float()).abs().max().item()
            assert diff < 2e-2, diff

            # dtype mismatch (fp32 weight): upstream promotion is preserved
            norm32 = m.Qwen3VLTextRMSNorm(1024, eps=1e-6).to("musa").to(torch.float32)
            out32 = norm32(x.detach())
            assert out32.dtype == torch.float32
            assert torch.isfinite(out32).all()

            # fp32 activation: delegates (documented restriction)
            out_fp32 = norm(x.detach().float())
            assert out_fp32.dtype == torch.float32
            print("OK", diff)
            """)
        assert result.returncode == 0, result.stderr
        assert "OK" in result.stdout

    def test_master_switch_disables_everything(self):
        result = _run(
            """
            import torch
            import torch_musa
            import transformers.models.qwen3_vl.modeling_qwen3_vl as m
            import training_musa_adaptor as tma
            rep = tma.report()
            by_id = {p["id"]: p["status"] for p in rep["patches"]}
            print("STATUS:", by_id.get("transformers.qwen3-vl.text-rms-norm.fused-torch", "absent"))
            """,
            env_extra={"TRAINING_MUSA_ADAPTOR_ENABLED": "0"},
        )
        assert result.returncode == 0, result.stderr
        assert "STATUS: applied" not in result.stdout

    def test_disable_by_id_keeps_upstream(self):
        result = _run(
            """
            import torch
            import torch_musa
            import transformers.models.qwen3_vl.modeling_qwen3_vl as m
            import training_musa_adaptor as tma
            rep = tma.report()
            by_id = {p["id"]: p["status"] for p in rep["patches"]}
            print("STATUS:", by_id["transformers.qwen3-vl.text-rms-norm.fused-torch"])
            """,
            env_extra={
                "TRAINING_MUSA_ADAPTOR_DISABLE": "transformers.qwen3-vl.text-rms-norm.fused-torch"
            },
        )
        assert result.returncode == 0, result.stderr
        assert "STATUS: skipped" in result.stdout
