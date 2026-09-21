"""Megatron attention patch tests: domain eligibility, meta construction,
packed-THD span parsing (CPU-safe, SimpleNamespace stubs)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from training_musa_adaptor.patches.megatron import attention as attn


def _module(**overrides):
    fields = dict(
        training=False,
        attention_dropout=0.0,
        qkv_format="sbhd",
        window_size=None,
        config=SimpleNamespace(
            fp8_dot_product_attention=False,
            fp8_multi_head_attention=False,
            qk_clip=False,
            log_max_attention_logit=False,
            apply_query_key_layer_scaling=False,
            softmax_type="vanilla",
            context_parallel_size=1,
        ),
        cp_group=None,
        num_splits=None,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _qkv(head_dim=64, v_dim=None, dtype=torch.bfloat16):
    v_dim = v_dim or head_dim
    q = torch.randn(512, 2, 16, head_dim, dtype=dtype)
    k = torch.randn(512, 2, 16, head_dim, dtype=dtype)
    v = torch.randn(512, 2, 16, v_dim, dtype=dtype)
    return q, k, v


class TestEligibility:
    def test_plain_call_eligible(self):
        q, k, v = _qkv()
        assert attn._eligible(_module(), q, k, v, None, None, "causal", None)

    @pytest.mark.parametrize(
        "flag",
        ["fp8_dot_product_attention", "fp8_multi_head_attention", "qk_clip",
         "log_max_attention_logit", "apply_query_key_layer_scaling"],
    )
    def test_config_protocol_flags_rejected(self, flag):
        module = _module()
        setattr(module.config, flag, True)
        q, k, v = _qkv()
        assert not attn._eligible(module, q, k, v, None, None, "causal", None)

    def test_non_vanilla_softmax_rejected(self):
        module = _module()
        module.config.softmax_type = "learned"
        q, k, v = _qkv()
        assert not attn._eligible(module, q, k, v, None, None, "causal", None)

    def test_num_splits_rejected(self):
        q, k, v = _qkv()
        assert not attn._eligible(_module(), q, k, v, None, 2, "causal", None)
        assert not attn._eligible(_module(num_splits=2), q, k, v, None, None, "causal", None)

    def test_window_restrictions(self):
        q, k, v = _qkv()
        assert attn._eligible(_module(window_size=(-1, -1)), q, k, v, None, None, "causal", None)
        assert attn._eligible(_module(window_size=(-1, 0)), q, k, v, None, None, "causal", None)
        assert not attn._eligible(_module(window_size=(16, 0)), q, k, v, None, None, "causal", None)
        # (-1, 0) with a non-causal mask is contradictory -> upstream
        assert not attn._eligible(
            _module(window_size=(-1, 0)), q, k, v, None, None, "no_mask", None
        )

    def test_cp_rejected(self):
        group = SimpleNamespace(size=lambda: 2)
        q, k, v = _qkv()
        assert not attn._eligible(_module(cp_group=group), q, k, v, None, None, "causal", None)
        module = _module()
        module.config.context_parallel_size = 2
        assert not attn._eligible(module, q, k, v, None, None, "causal", None)

    def test_mask_domain(self):
        q, k, v = _qkv()
        assert not attn._eligible(_module(), q, k, v, None, None, "bogus_mask", None)
        # dense: padding/arbitrary without a mask tensor -> upstream
        assert not attn._eligible(_module(), q, k, v, None, None, "padding", None)
        assert not attn._eligible(_module(), q, k, v, None, None, "arbitrary", None)
        # packed THD: padding qualifier is allowed without a mask tensor
        # (spans drop it per sequence -- the restored old-code guard)
        packed = SimpleNamespace(cp_group=None, local_cp_size=None)
        assert attn._eligible(_module(), q, k, v, packed, None, "padding_causal", None)


class TestBuildMeta:
    def test_te_causal_window_normalized(self):
        """Regression (round 1): TE normalizes causal modules to (-1, 0);
        the encoding must not reach implementations."""
        q, k, v = _qkv()
        meta = attn._build_meta(
            _module(window_size=(-1, 0)), q, k, v, None, None, None, "causal", "sbhd"
        )
        assert meta.sliding_window is None
        assert meta.mask_kind == "causal"

    def test_no_window_normalized(self):
        q, k, v = _qkv()
        meta = attn._build_meta(
            _module(window_size=(-1, -1)), q, k, v, None, None, None, "causal", "sbhd"
        )
        assert meta.sliding_window is None

    def test_may_require_backward_from_training(self):
        q, k, v = _qkv()
        with torch.no_grad():
            meta = attn._build_meta(_module(training=True), q, k, v, None, None, None, "causal", "sbhd")
            assert meta.may_require_backward is True
            meta = attn._build_meta(_module(training=False), q, k, v, None, None, None, "causal", "sbhd")
            assert meta.may_require_backward is False

    def test_may_require_backward_from_grad_inputs(self):
        q, k, v = _qkv()
        q.requires_grad_(True)
        meta = attn._build_meta(_module(training=False), q, k, v, None, None, None, "causal", "sbhd")
        assert meta.may_require_backward is True

    def test_effective_dropout(self):
        assert attn._effective_dropout(_module(training=True, attention_dropout=0.2)) == pytest.approx(0.2)
        assert attn._effective_dropout(_module(training=False, attention_dropout=0.2)) == 0.0

    def test_shape_fields(self):
        q, k, v = _qkv(head_dim=192, v_dim=128)
        meta = attn._build_meta(_module(), q, k, v, None, None, None, "causal", "sbhd")
        assert meta.heads_q == 16 and meta.heads_kv == 16
        assert meta.head_dim_qk == 192 and meta.head_dim_v == 128
        assert meta.seq_q == 512 and meta.seq_k == 512 and meta.batch == 2


class TestPackedSpans:
    def _cu(self, values):
        return torch.tensor(values, dtype=torch.int32)

    def test_valid_spans(self):
        spans = attn._packed_spans(self._cu([0, 128, 128, 224]), self._cu([0, 192, 256, 384]), 384)
        assert spans == [(0, 128, 192), (192, 0, 64), (256, 96, 128)]

    def test_valid_without_padding(self):
        spans = attn._packed_spans(self._cu([0, 128, 224]), None, 224)
        assert spans == [(0, 128, 128), (128, 96, 96)]

    @pytest.mark.parametrize(
        "lengths",
        [[], [0], [128, 224], [0, 128, 999], [0, 96, 64], [0, 224, 128]],
    )
    def test_invalid_spans(self, lengths):
        with pytest.raises(ValueError):
            attn._packed_spans(self._cu(lengths), None, 224)


class TestFactoryDecline:
    def test_factory_declines_without_flash_marker(self, monkeypatch):
        monkeypatch.setattr(
            attn, "module_source_contains", lambda module, needle: False
        )
        assert attn._tedpa_forward(lambda self, *a, **k: None) is None

    def test_factory_wraps_with_marker(self, monkeypatch):
        monkeypatch.setattr(
            attn, "module_source_contains", lambda module, needle: True
        )
        def original(self, *a, **k):
            return "original"

        wrapper = attn._tedpa_forward(original)
        assert wrapper is not None
        module = _module()
        # CPU tensor: non-MUSA calls keep the original path
        q, k, v = _qkv()
        assert wrapper(module, q, k, v, None, SimpleNamespace(name="causal")) == "original"
