"""Attention ops tests: capability windows, candidates, selection (CPU-safe)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from training_musa_adaptor.ops import attention as ops

BF16 = "torch.bfloat16"
FP32 = "torch.float32"


def meta(**overrides) -> ops.AttentionMeta:
    fields = dict(
        dtype=BF16,
        dtype_mixed=False,
        device_type="musa",
        plain_tensors=True,
        heads_q=32,
        heads_kv=32,
        head_dim_qk=128,
        head_dim_v=128,
        layout="sbhd",
        batch=2,
        seq_q=512,
        seq_k=512,
        packed=False,
        mask_kind="causal",
        has_attention_mask=False,
        has_attention_bias=False,
        sliding_window=None,
        softmax_scale=None,
        has_alibi=False,
        training=False,
        dropout_p=0.0,
        may_require_backward=False,
        deterministic=False,
        cp_size=1,
        fp8=False,
        extra_outputs=False,
    )
    fields.update(overrides)
    return ops.AttentionMeta(**fields)


def supports(name: str, m: ops.AttentionMeta) -> tuple[bool, str]:
    return ops.IMPLEMENTATIONS[name].supports(m)


class TestMudnnWindow:
    def test_bf16_supported_dim(self):
        ok, _ = supports("mudnn", meta())
        assert ok

    @pytest.mark.parametrize("dtype", [FP32, "torch.float64"])
    def test_wider_dtypes_rejected(self, dtype):
        ok, reason = supports("mudnn", meta(dtype=dtype))
        assert not ok and "dtype" in reason

    def test_mixed_dtypes_rejected(self):
        ok, _ = supports("mudnn", meta(dtype_mixed=True))
        assert not ok

    @pytest.mark.parametrize("head_dim", [32, 56, 256])
    def test_head_dim_outside_forward_window(self, head_dim):
        ok, reason = supports("mudnn", meta(head_dim_qk=head_dim, head_dim_v=head_dim))
        assert not ok and "forward window" in reason

    @pytest.mark.parametrize("head_dim", [144, 168, 176, 184, 192])
    def test_backward_window_rejects_broken_dims_in_training(self, head_dim):
        ok, reason = supports(
            "mudnn",
            meta(head_dim_qk=head_dim, head_dim_v=head_dim, may_require_backward=True),
        )
        assert not ok and "backward" in reason
        # inference-only keeps the fast forward for the full window
        ok2, _ = supports("mudnn", meta(head_dim_qk=head_dim, head_dim_v=head_dim))
        assert ok2

    @pytest.mark.parametrize("safe_dim", [64, 80, 96, 112, 128, 160])
    def test_backward_safe_dims_supported_in_training(self, safe_dim):
        ok, _ = supports(
            "mudnn",
            meta(head_dim_qk=safe_dim, head_dim_v=safe_dim, may_require_backward=True),
        )
        assert ok

    def test_backward_window_rejects_mixed_dims(self):
        ok, reason = supports(
            "mudnn", meta(head_dim_qk=192, head_dim_v=128, may_require_backward=True)
        )
        assert not ok and "backward" in reason

    def test_dropout_rejected(self):
        ok, _ = supports("mudnn", meta(dropout_p=0.1))
        assert not ok

    def test_packed_rejected(self):
        ok, _ = supports("mudnn", meta(packed=True))
        assert not ok

    def test_cpu_rejected(self):
        ok, reason = supports("mudnn", meta(device_type="cpu"))
        assert not ok and "musa" in reason


class TestMateWindow:
    @pytest.mark.parametrize("head_dim", [144, 168, 176, 184, 192])
    def test_equal_broken_dims_supported(self, head_dim):
        ok, _ = supports(
            "mate",
            meta(head_dim_qk=head_dim, head_dim_v=head_dim, may_require_backward=True),
        )
        assert ok

    @pytest.mark.parametrize("dqk,dv", [(192, 128), (160, 128)])
    def test_verified_mixed_pairs_supported(self, dqk, dv):
        ok, _ = supports(
            "mate", meta(head_dim_qk=dqk, head_dim_v=dv, may_require_backward=True)
        )
        assert ok

    @pytest.mark.parametrize("dqk,dv", [(128, 64), (256, 128), (192, 160)])
    def test_unverified_mixed_pairs_rejected(self, dqk, dv):
        ok, _ = supports("mate", meta(head_dim_qk=dqk, head_dim_v=dv))
        assert not ok

    def test_inside_mudnn_safe_window_not_claimed(self):
        ok, _ = supports("mate", meta(head_dim_qk=128, head_dim_v=128))
        assert not ok

    def test_dropout_bias_mask_rejected(self):
        assert not supports("mate", meta(head_dim_qk=192, dropout_p=0.1))[0]
        assert not supports("mate", meta(head_dim_qk=192, has_attention_bias=True))[0]
        assert not supports("mate", meta(head_dim_qk=192, mask_kind="padding"))[0]
        assert not supports("mate", meta(head_dim_qk=192, mask_kind="arbitrary"))[0]

    def test_bottom_right_requires_equal_lengths(self):
        assert supports(
            "mate",
            meta(
                head_dim_qk=192, mask_kind="causal_bottom_right", seq_q=128, seq_k=128
            ),
        )[0]
        assert not supports(
            "mate",
            meta(
                head_dim_qk=192, mask_kind="causal_bottom_right", seq_q=128, seq_k=256
            ),
        )[0]

    def test_gqa_indivisibility_rejected(self):
        assert not supports("mate", meta(head_dim_qk=192, heads_q=7, heads_kv=3))[0]

    def test_unnormalized_window_encoding_rejected(self):
        """TE's (-1, 0) causal encoding must be normalized by the patch;
        the raw encoding must not reach the implementations."""
        ok, reason = supports("mate", meta(head_dim_qk=192, sliding_window=(-1, 0)))
        assert not ok and "window" in reason

    def test_packed_rejected(self):
        assert not supports("mate", meta(head_dim_qk=192, packed=True))[0]


class TestTEUnfusedWindow:
    def test_plain_dtypes_supported(self):
        assert supports("te_unfused", meta())[0]
        assert supports("te_unfused", meta(dtype=FP32))[0]

    def test_packed_rejected(self):
        assert not supports("te_unfused", meta(packed=True))[0]

    def test_cp_rejected(self):
        assert not supports("te_unfused", meta(cp_size=2))[0]

    def test_padding_without_mask_rejected(self):
        ok, _ = supports(
            "te_unfused", meta(mask_kind="padding", has_attention_mask=False)
        )
        assert not ok


class TestClosedSlots:
    def test_flash_attn_closed(self):
        ok, reason = supports("flash_attn", meta())
        assert not ok and "backward" in reason

    def test_sdpa_math_closed(self):
        ok, reason = supports("torch_sdpa_math", meta())
        assert not ok and "math" in reason


class TestResolveCandidates:
    def test_auto_default_order_with_reference_fallback(self):
        candidates, pre = ops.resolve_candidates("auto", (), "reference")
        assert [c.name for c in candidates if c.kind == "implementation"] == [
            "mudnn",
            "mate",
            "te_unfused",
        ]
        assert pre == []

    def test_auto_with_upstream_fallback_appends_original(self):
        candidates, _ = ops.resolve_candidates("auto", (), "upstream")
        assert candidates[-1].kind == "original"

    def test_auto_with_error_fallback_no_tail(self):
        candidates, _ = ops.resolve_candidates("auto", (), "error")
        assert [c.name for c in candidates] == ["mudnn", "mate"]

    def test_prefer_listed_first_then_default(self):
        candidates, pre = ops.resolve_candidates("prefer", ("mate",), "error")
        assert [c.name for c in candidates] == ["mate", "mudnn"]

    def test_prefer_unknown_recorded_and_continues(self):
        candidates, pre = ops.resolve_candidates("prefer", ("nope", "mate"), "error")
        assert ("nope", "unknown implementation") in pre
        assert [c.name for c in candidates] == ["mate", "mudnn"]

    def test_force_single(self):
        candidates, _ = ops.resolve_candidates("force", ("te_unfused",), "error")
        assert [(c.kind, c.name) for c in candidates] == [
            ("implementation", "te_unfused")
        ]

    def test_force_unknown_raises(self):
        with pytest.raises(ops.NoCompatibleImplementation):
            ops.resolve_candidates("force", ("nope",), "error")

    def test_upstream_single_original(self):
        candidates, _ = ops.resolve_candidates("upstream", (), "error")
        assert [(c.kind,) for c in candidates] == [("original",)]


class TestSelectAndRun:
    def _call(self, name):
        return SimpleNamespace(module=None, meta=meta())

    def _original(self, call):
        return "original-ran"

    def _original_supports(self, call):
        return True, ""

    def test_supports_rejected_means_run_not_called(self):
        ran = []
        impl = ops.IMPLEMENTATIONS["mate"]
        original_run = impl.run_fn
        impl.run_fn = lambda payload, call: ran.append("mate")
        try:
            candidates, pre = ops.resolve_candidates(
                "prefer", ("mate", "mudnn"), "error"
            )
            # mate cannot serve dim 128; mudnn can (fake its run too)
            mudnn = ops.IMPLEMENTATIONS["mudnn"]
            mudnn_run = mudnn.run_fn
            mudnn.run_fn = lambda payload, call: "mudnn-ran"
            try:
                result = ops.select_and_run(
                    "test",
                    candidates,
                    pre,
                    meta(),
                    self._call("x"),
                    self._original_supports,
                    self._original,
                )
            finally:
                mudnn.run_fn = mudnn_run
            assert result == "mudnn-ran"
            assert ran == []
        finally:
            impl.run_fn = original_run

    def test_execution_exception_never_retries(self):
        boom = RuntimeError("mid-kernel failure")
        mudnn = ops.IMPLEMENTATIONS["mudnn"]
        original_run = mudnn.run_fn
        mudnn.run_fn = lambda payload, call: (_ for _ in ()).throw(boom)
        try:
            candidates, _ = ops.resolve_candidates("auto", (), "reference")
            with pytest.raises(RuntimeError, match="mid-kernel"):
                ops.select_and_run(
                    "test",
                    candidates,
                    [],
                    meta(),
                    self._call("x"),
                    self._original_supports,
                    self._original,
                )
        finally:
            mudnn.run_fn = original_run

    def test_fallback_upstream_calls_original(self):
        # dim 32: nothing supports it, original does
        candidates, pre = ops.resolve_candidates("auto", (), "upstream")
        result = ops.select_and_run(
            "test",
            candidates,
            pre,
            meta(head_dim_qk=32, head_dim_v=32),
            self._call("x"),
            self._original_supports,
            self._original,
        )
        assert result == "original-ran"

    def test_nothing_applies_raises_with_reasons(self):
        candidates, pre = ops.resolve_candidates("auto", (), "error")
        with pytest.raises(ops.NoCompatibleImplementation) as excinfo:
            ops.select_and_run(
                "test",
                candidates,
                pre,
                meta(head_dim_qk=32, head_dim_v=32),
                self._call("x"),
                self._original_supports,
                self._original,
            )
        assert "mudnn" in str(excinfo.value) and "mate" in str(excinfo.value)

    def test_original_candidate_rejected_when_unsupported(self):
        candidates, _ = ops.resolve_candidates("auto", (), "upstream")
        with pytest.raises(ops.NoCompatibleImplementation) as excinfo:
            ops.select_and_run(
                "test",
                candidates,
                [],
                meta(head_dim_qk=32, head_dim_v=32),
                self._call("x"),
                lambda call: (False, "original not ok"),
                self._original,
            )
        assert "original: original not ok" in str(excinfo.value)
