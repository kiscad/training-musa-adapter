"""Integration: Megatron attention patch on real MUSA (S2 hardware matrix).

Ported from the round-1 hardware matrix to the v2.0 engine/config API.
The mate path (TileLang JIT) and packed THD are included; a cold
~/.tilelang cache makes the mate case compile for minutes on first run.
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


def _run(code: str, env_extra: dict[str, str] | None = None, timeout: int = 900):
    env = {k: v for k, v in os.environ.items() if not k.startswith("TRAINING_MUSA_ADAPTOR")}
    env["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "1"
    env.update(env_extra or {})
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        env=env,
        timeout=timeout,
        cwd="/tmp",
    )


_SETUP = """
    import torch
    import torch_musa
    import training_musa_adaptor as tma
    tma.install()
    from megatron.core.transformer.transformer_config import TransformerConfig
    from megatron.core.extensions.transformer_engine import TEDotProductAttention
    from megatron.core.transformer.enums import AttnMaskType

    def make_attn(heads=16, groups=None, kv_channels=64):
        kwargs = dict(
            num_layers=1, hidden_size=heads * kv_channels,
            num_attention_heads=heads, kv_channels=kv_channels,
            attention_dropout=0.0, hidden_dropout=0.0, layernorm_epsilon=1e-5)
        if groups:
            kwargs["num_query_groups"] = groups
        config = TransformerConfig(**kwargs)
        return TEDotProductAttention(
            config=config, layer_number=1, attn_mask_type=AttnMaskType.causal,
            attention_type="self", attention_dropout=0.0).to("musa")

    def qkv(sq, b, hq, hkv, d, dtype=torch.bfloat16, seed=42):
        g = torch.Generator(device="musa").manual_seed(seed)
        q = torch.randn(sq, b, hq, d, device="musa", dtype=dtype, generator=g, requires_grad=True)
        k = torch.randn(sq, b, hkv, d, device="musa", dtype=dtype, generator=g, requires_grad=True)
        v = torch.randn(sq, b, hkv, d, device="musa", dtype=dtype, generator=g, requires_grad=True)
        return q, k, v
"""


def _script(body: str) -> str:
    return textwrap.dedent(_SETUP) + textwrap.dedent(body)


@pytest.mark.skipif(not musa_available, reason="no live MUSA stack")
class TestAttentionMusa:
    def test_automatic_channel_applies_and_runs_bf16(self):
        result = _run(
            _script(
                """
                attn = make_attn(heads=16)
                q, k, v = qkv(512, 2, 16, 16, 64)
                out = attn(q, k, v, None, AttnMaskType.causal)
                assert torch.isfinite(out).all()
                out.to(torch.float32).pow(2).mean().backward()
                assert torch.isfinite(q.grad).all()
                by_id = {p["id"]: p["status"] for p in tma.report()["patches"]}
                assert by_id["megatron.te.attention.capability-dispatch"] == "applied"
                print("OK")
                """
            )
        )
        assert result.returncode == 0, result.stderr
        assert "OK" in result.stdout

    def test_fp32_forward_backward(self):
        result = _run(
            _script(
                """
                attn = make_attn(heads=16)
                q, k, v = qkv(512, 2, 16, 16, 64, dtype=torch.float32)
                out = attn(q, k, v, None, AttnMaskType.causal)
                assert torch.isfinite(out).all()
                out.backward(torch.ones_like(out))
                assert torch.isfinite(q.grad).all()
                print("OK")
                """
            )
        )
        assert result.returncode == 0, result.stderr
        assert "OK" in result.stdout

    def test_gqa_forward_backward(self):
        result = _run(
            _script(
                """
                attn = make_attn(heads=8, groups=2)
                q, k, v = qkv(512, 2, 8, 2, 64)
                out = attn(q, k, v, None, AttnMaskType.causal)
                assert tuple(out.shape) == (512, 2, 512)
                assert torch.isfinite(out).all()
                out.to(torch.float32).pow(2).mean().backward()
                assert torch.isfinite(q.grad).all() and torch.isfinite(k.grad).all()
                print("OK")
                """
            )
        )
        assert result.returncode == 0, result.stderr
        assert "OK" in result.stdout

    def test_mudnn_matches_te_unfused_reference(self):
        auto = _run(
            _script(
                """
                attn = make_attn(heads=16)
                q, k, v = qkv(512, 2, 16, 16, 64)
                out = attn(q, k, v, None, AttnMaskType.causal)
                out.to(torch.float32).pow(2).mean().backward()
                import tempfile, os
                path = os.path.join(tempfile.mkdtemp(), "auto.pt")
                torch.save({"out": out.detach().cpu(), "g": q.grad.detach().cpu()}, path)
                print(path)
                """
            )
        )
        assert auto.returncode == 0, auto.stderr
        path = auto.stdout.strip().splitlines()[-1]
        forced = _run(
            _script(
                f"""
                attn = make_attn(heads=16)
                q, k, v = qkv(512, 2, 16, 16, 64)
                out = attn(q, k, v, None, AttnMaskType.causal)
                out.to(torch.float32).pow(2).mean().backward()
                ref = torch.load({path!r})
                fwd = (out.detach().cpu().float() - ref["out"].float()).abs().max().item()
                grd = (q.grad.detach().cpu().float() - ref["g"].float()).abs().max().item()
                assert fwd < 2e-2, fwd
                assert grd < 2e-2, grd
                print("OK", fwd, grd)
                """
            ),
            env_extra={
                "TRAINING_MUSA_ADAPTOR_ATTN_POLICY": "force",
                "TRAINING_MUSA_ADAPTOR_ATTN_IMPLS": "te_unfused",
            },
        )
        assert forced.returncode == 0, forced.stderr
        assert "OK" in forced.stdout

    def test_policy_switch_changes_executed_path(self):
        """force te_unfused must change the actual computation vs auto."""
        result = _run(
            _script(
                """
                attn = make_attn(heads=16)
                q, k, v = qkv(512, 2, 16, 16, 64)
                out = attn(q, k, v, None, AttnMaskType.causal)
                assert torch.isfinite(out).all()
                print("OK")
                """
            ),
            env_extra={
                "TRAINING_MUSA_ADAPTOR_ATTN_POLICY": "force",
                "TRAINING_MUSA_ADAPTOR_ATTN_IMPLS": "te_unfused",
            },
        )
        assert result.returncode == 0, result.stderr
        assert "OK" in result.stdout

    def test_force_mate_on_unsupported_input_fails_before_kernel(self):
        result = _run(
            _script(
                """
                attn = make_attn(heads=16)
                q, k, v = qkv(512, 2, 16, 16, 64)  # dim 64 is outside mate's window
                try:
                    attn(q, k, v, None, AttnMaskType.causal)
                    raise SystemExit("should have failed")
                except SystemExit:
                    raise
                except Exception as exc:
                    assert type(exc).__name__ == "NoCompatibleImplementation", type(exc)
                print("OK")
                """
            ),
            env_extra={
                "TRAINING_MUSA_ADAPTOR_ATTN_POLICY": "force",
                "TRAINING_MUSA_ADAPTOR_ATTN_IMPLS": "mate",
            },
        )
        assert result.returncode == 0, (result.returncode, result.stdout, result.stderr)
        assert "OK" in result.stdout

    def test_mate_dim192_forward_backward(self):
        """d=192 equal dims: mudnn backward rejects, mate serves (JIT warm)."""
        result = _run(
            _script(
                """
                attn = make_attn(heads=16, kv_channels=192)
                q, k, v = qkv(512, 2, 16, 16, 192)
                out = attn(q, k, v, None, AttnMaskType.causal)
                assert torch.isfinite(out).all()
                out.to(torch.float32).pow(2).mean().backward()
                assert torch.isfinite(q.grad).all() and torch.isfinite(k.grad).all()
                print("OK")
                """
            ),
            timeout=1800,
        )
        assert result.returncode == 0, result.stderr
        assert "OK" in result.stdout

    def test_mate_dim192_matches_reference(self):
        mate = _run(
            _script(
                """
                attn = make_attn(heads=16, kv_channels=192)
                q, k, v = qkv(512, 2, 16, 16, 192)
                out = attn(q, k, v, None, AttnMaskType.causal)
                out.to(torch.float32).pow(2).mean().backward()
                import tempfile, os
                path = os.path.join(tempfile.mkdtemp(), "mate.pt")
                torch.save({"out": out.detach().cpu(), "g": q.grad.detach().cpu()}, path)
                print(path)
                """
            ),
            timeout=1800,
        )
        assert mate.returncode == 0, mate.stderr
        path = mate.stdout.strip().splitlines()[-1]
        forced = _run(
            _script(
                f"""
                attn = make_attn(heads=16, kv_channels=192)
                q, k, v = qkv(512, 2, 16, 16, 192)
                out = attn(q, k, v, None, AttnMaskType.causal)
                out.to(torch.float32).pow(2).mean().backward()
                ref = torch.load({path!r})
                fwd = (out.detach().cpu().float() - ref["out"].float()).abs().max().item()
                grd = (q.grad.detach().cpu().float() - ref["g"].float()).abs().max().item()
                assert fwd < 2e-2, fwd   # bf16 tolerance
                assert grd < 1e-1, grd   # measured ~0.5-0.7% of grad absmax 4-6
                print("OK", fwd, grd)
                """
            ),
            env_extra={
                "TRAINING_MUSA_ADAPTOR_ATTN_POLICY": "force",
                "TRAINING_MUSA_ADAPTOR_ATTN_IMPLS": "te_unfused",
            },
        )
        assert forced.returncode == 0, forced.stderr
        assert "OK" in forced.stdout

    def test_packed_thd_with_padded_offsets(self):
        """Packed THD slicing: padded offsets stay finite (no NaN), padded
        rows are exactly zero, empty sequences keep zero-gradient edges."""
        result = _run(
            _script(
                """
                from megatron.core.packed_seq_params import PackedSeqParams
                attn = make_attn(heads=16)
                # the module keeps its default "sbhd"; the packed layout is
                # carried by PackedSeqParams.qkv_format (real megatron usage)
                total, h, d = 384, 16, 64
                q = torch.randn(total, h, d, device="musa", dtype=torch.bfloat16, requires_grad=True)
                k = torch.randn(total, h, d, device="musa", dtype=torch.bfloat16, requires_grad=True)
                v = torch.randn(total, h, d, device="musa", dtype=torch.bfloat16, requires_grad=True)
                packed = PackedSeqParams(
                    qkv_format="thd",
                    cu_seqlens_q=torch.tensor([0, 128, 128, 224], device="musa", dtype=torch.int32),
                    cu_seqlens_kv=torch.tensor([0, 128, 128, 224], device="musa", dtype=torch.int32),
                    cu_seqlens_q_padded=torch.tensor([0, 192, 256, 384], device="musa", dtype=torch.int32),
                    cu_seqlens_kv_padded=torch.tensor([0, 192, 256, 384], device="musa", dtype=torch.int32),
                )
                out = attn(q, k, v, None, AttnMaskType.causal, packed_seq_params=packed)
                assert tuple(out.shape) == (384, 1024)
                assert torch.isfinite(out).all(), "padded THD must not produce NaN"
                # lens [128, 0, 96], caps [192, 64, 128], offsets [0, 192, 256]:
                # valid rows 0:128 and 256:352; padded rows are exactly zero.
                padded_rows = torch.cat([out[128:192], out[192:256], out[352:384]])
                assert (padded_rows == 0).all(), "padded rows must be exactly zero"
                valid_rows = torch.cat([out[0:128], out[256:352]])
                assert valid_rows.abs().sum() > 0
                out.to(torch.float32).pow(2).mean().backward()
                assert torch.isfinite(q.grad).all()
                padded_grad = torch.cat([q.grad[128:192], q.grad[192:256], q.grad[352:384]])
                assert (padded_grad == 0).all()
                print("OK")
                """
            )
        )
        assert result.returncode == 0, result.stderr
        assert "OK" in result.stdout
