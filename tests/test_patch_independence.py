# 要点: 用同构的合成补丁把独立性语义钉在引擎上——norm 三件套(类替换+两个跟随
# 旗标)、ONLY/DISABLE 选择不隐式启用伴随、缺失伴随按原因 skipped、单目标多
# wrapper 任意子集/顺序可交换、dispatch 运行时读活状态且用 vars() 有界探测;
# 环境开关为 TRAINING_MUSA_ADAPTOR_ONLY/DISABLE.
"""Selective activation must not rely on incidental registration order."""

from __future__ import annotations

from itertools import permutations
from types import SimpleNamespace

import pytest

from training_musa_adaptor._engine import AttrPatch


def _wrap(label):
    def replace(old):
        def wrapper():
            return old() + label

        return wrapper

    return replace


def _norm_trio(module_name):
    """Synthetic mirror of the old fused-layer-norm trio: a class replacement
    plus two flag patches that must only follow the class (requires)."""
    return [
        AttrPatch(
            "test.norm.class",
            f"{module_name}:FusedLayerNorm",
            lambda old: type("FallbackNorm", (old,), {"_test_fallback": True}),
        ),
        AttrPatch(
            "test.norm.have-flag",
            f"{module_name}:HAVE_FUSED_LAYER_NORM",
            lambda old: True,
            requires=("test.norm.class",),
        ),
        AttrPatch(
            "test.norm.persist-flag",
            f"{module_name}:HAVE_PERSIST_LAYER_NORM",
            lambda old: False,
            requires=("test.norm.class",),
        ),
    ]


def _norm_module(stub_module, name):
    original = type("UpstreamNorm", (), {})
    return original, stub_module(
        name,
        FusedLayerNorm=original,
        HAVE_FUSED_LAYER_NORM=False,
        HAVE_PERSIST_LAYER_NORM=True,
    )


@pytest.mark.parametrize("order", list(permutations(range(3))))
def test_norm_flags_follow_class_in_any_registration_order(engine, stub_module, order):
    original, module = _norm_module(stub_module, "test_norm_mod")
    patches = _norm_trio(module.__name__)
    engine.register([patches[i] for i in order])
    engine.install()
    assert module.FusedLayerNorm._test_fallback
    assert module.HAVE_FUSED_LAYER_NORM is True
    assert module.HAVE_PERSIST_LAYER_NORM is False
    engine.uninstall()
    assert module.FusedLayerNorm is original
    assert module.HAVE_FUSED_LAYER_NORM is False
    assert module.HAVE_PERSIST_LAYER_NORM is True


@pytest.mark.parametrize("switch", ["ONLY", "DISABLE"])
def test_flag_selection_does_not_enable_class_implicitly(
    engine, stub_module, monkeypatch, switch
):
    original, module = _norm_module(stub_module, "test_norm_select")
    patches = _norm_trio(module.__name__)
    selected = [
        p.id
        for p in patches
        if (p.attr_name == "FusedLayerNorm") == (switch == "DISABLE")
    ]
    monkeypatch.setenv("TRAINING_MUSA_ADAPTOR_" + switch, ",".join(selected))
    engine.register(patches)
    engine.install()
    assert module.FusedLayerNorm is original
    assert module.HAVE_FUSED_LAYER_NORM is False
    assert module.HAVE_PERSIST_LAYER_NORM is True
    for record in engine.report()["patches"]:
        assert record["status"] == "skipped"
        if record["requires"]:
            assert "requires applied companion" in record["detail"]


def test_independent_target_applies_without_sibling(engine, stub_module):
    """Mirror of the old block-norm test's independence half: a patch on one
    module applies cleanly without the sibling module's class replacement
    (no implicit cross-patch enabling, no torch numerics here -- those return
    with the actual patch's contract tests)."""
    upstream = type("BrokenApexNorm", (), {})
    local = stub_module("test_block_local", FusedLayerNorm=upstream)
    block = stub_module("test_block_impl", LayerNormImpl=lambda: "upstream")
    engine.register(
        [AttrPatch("test.block.impl", "test_block_impl:LayerNormImpl", _wrap("fb"))]
    )
    engine.install()
    assert block.LayerNormImpl() == "upstreamfb"
    assert local.FusedLayerNorm is upstream


def test_consumer_alone_reports_missing_companion(engine, stub_module):
    """Mirror of the old checkpoint-context test: a consumer whose requires
    companion is absent stays skipped and names the companion."""
    original = lambda: "saved"
    module = stub_module("test_ckpt_mod", save_checkpoint=original)
    engine.register(
        [
            AttrPatch(
                "test.ckpt.wrapper",
                "test_ckpt_mod:save_checkpoint",
                _wrap("x"),
                requires=("test.ckpt.host-barrier-proxy",),
            )
        ]
    )
    engine.install()
    assert module.save_checkpoint is original
    assert engine.report()["patches"][0]["status"] == "skipped"
    assert "host-barrier-proxy" in engine.report()["patches"][0]["detail"]


@pytest.mark.parametrize("order", [(0,), (1,), (0, 1), (1, 0)])
def test_wrappers_are_selectable_and_commute(engine, stub_module, order):
    """Mirror of the old training validate_args pair: two independent
    wrappers on one target compose in registration order; any subset works."""

    def flip_profiler(old):
        def wrapper(args):
            args.use_pytorch_profiler = True
            return old(args)

        return wrapper

    def drop_overlap(old):
        def wrapper(args):
            args.overlap_grad_reduce = False
            return old(args)

        return wrapper

    patches = [
        AttrPatch("test.args.profile", "test_args_mod:validate_args", flip_profiler),
        AttrPatch("test.args.overlap", "test_args_mod:validate_args", drop_overlap),
    ]
    module = stub_module("test_args_mod", validate_args=lambda args: args)
    engine.register([patches[i] for i in order])
    engine.install()
    args = SimpleNamespace(
        profile=True, use_pytorch_profiler=False, overlap_grad_reduce=True, rank=1
    )
    assert module.validate_args(args) is args
    assert args.use_pytorch_profiler is (0 in order)
    assert args.overlap_grad_reduce is (1 not in order)


@pytest.mark.parametrize("order", [(0, 1), (1, 0)])
def test_dispatch_observes_live_module_state(engine, stub_module, order):
    """Mirror of the old rope dispatch test: the wrapper reads the module's
    live attributes at call time, so a third party installing kernels after
    the wrapper was built is still observed; registration order of an
    unrelated sibling patch does not matter."""
    module = stub_module(
        "test_rope_mod",
        fused_kernel=None,
        apply_op=lambda *args: "reference",
        FLAG=True,
    )

    def replace(old):
        def dispatch(*args, **kwargs):
            fused = vars(module).get("fused_kernel")
            return fused(*args) if fused is not None else old(*args, **kwargs)

        return dispatch

    patches = [
        AttrPatch("test.rope.dispatch", "test_rope_mod:apply_op", replace),
        AttrPatch("test.rope.flag", "test_rope_mod:FLAG", lambda old: False),
    ]
    engine.register([patches[i] for i in order])
    engine.install()
    assert module.apply_op(1) == "reference"
    module.fused_kernel = lambda *args: "kernel"  # third-party late install
    assert module.apply_op(1) == "kernel"
    assert module.FLAG is False


def test_dispatch_uses_bounded_vars_lookup(engine, stub_module):
    """Mirror of the old no-apex-probe test: dispatch probes the module's own
    __dict__ only -- a missing optional attribute must not wake any module
    __getattr__ (bounded probe discipline)."""
    module = stub_module("test_rope_safe", apply_op=lambda *args: "reference")

    def forbidden(name):
        raise AssertionError(f"dispatch probed {name} via __getattr__")

    module.__getattr__ = forbidden

    def replace(old):
        def dispatch(*args, **kwargs):
            fused = vars(module).get("fused_kernel")
            return fused(*args) if fused is not None else old(*args, **kwargs)

        return dispatch

    engine.register([AttrPatch("test.rope.safe", "test_rope_safe:apply_op", replace)])
    engine.install()
    assert module.apply_op(1) == "reference"
