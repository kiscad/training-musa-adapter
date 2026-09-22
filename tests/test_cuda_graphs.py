"""The TE graph import gate: make_weak_ref port, probes and hook lifecycle.

Unit tests are CPU-safe and use synthetic modules; the tests marked ``musa``
require a live MUSA device and exercise the alias on real accelerator memory.
``tests/cuda_graphs_smoke.py`` is the full hardware worker.
"""

from __future__ import annotations

import sys
import types

import pytest
from packaging.version import Version

from training_musa_adaptor.patches import PATCHES
from training_musa_adaptor.patches.megatron import cuda_graphs as gate

torch = pytest.importorskip("torch")
torch_musa = pytest.importorskip("torch_musa")

# Gate only the hardware cases; CPU contracts still run on older stacks.
musa_device = pytest.mark.skipif(
    Version(torch_musa.__version__) < Version("2.9.0") or not torch_musa.is_available(),
    reason=(
        "cuda-graph requires torch_musa >= 2.9.0 and a live MUSA device "
        f"(installed: {torch_musa.__version__})"
    ),
)


@pytest.fixture(autouse=True)
def _clean_owned_state():
    gate._uninstall_make_weak_ref()
    yield
    gate._uninstall_make_weak_ref()


@pytest.fixture
def make_weak_ref():
    return gate._build_make_weak_ref()


@pytest.fixture
def te_utils(monkeypatch):
    module = types.ModuleType("transformer_engine.pytorch.utils")
    monkeypatch.setitem(sys.modules, "transformer_engine.pytorch.utils", module)
    return module


@pytest.fixture
def capable(monkeypatch):
    """Pretend every prerequisite holds and the port self-check passes."""
    monkeypatch.setattr(
        torch, "musa", types.SimpleNamespace(is_available=lambda: True), raising=False
    )
    monkeypatch.setattr(gate, "_musa_fork_installed", lambda: True)
    monkeypatch.setattr(gate, "_missing_graph_prerequisites", lambda: [])
    monkeypatch.setattr(gate, "_weak_ref_self_check", lambda impl, device=None: None)


# ---------------------------------------------------------------- the ledger


def test_patch_is_registered_with_megatron_trigger():
    matches = [p for p in PATCHES if p.id == "megatron.te.make-weak-ref.graph-compat"]
    assert len(matches) == 1
    patch = matches[0]
    # namespace roots are not hook boundaries; the concrete boundary
    # is megatron.core.parallel_state (reachable in both import orders)
    assert patch.trigger == "megatron.core.parallel_state"
    assert callable(patch.run) and callable(patch.undo)


# ------------------------------------------------------- make_weak_ref port


def test_passes_cpu_tensors_through_unchanged(make_weak_ref):
    tensor = torch.zeros(3)
    assert make_weak_ref(tensor) is tensor


def test_recurses_containers_and_passes_scalars(make_weak_ref):
    tensor = torch.zeros(2)
    out = make_weak_ref((tensor, [tensor], {"k": tensor}, None, 1, 2.5, True))
    assert isinstance(out, tuple)
    assert out[0] is tensor
    assert out[1][0] is tensor
    assert out[2]["k"] is tensor
    assert out[3] is None and out[4] == 1 and out[5] == 2.5 and out[6] is True


def test_rejects_unsupported_types(make_weak_ref):
    with pytest.raises(TypeError, match="Invalid type"):
        make_weak_ref(object())


@musa_device
def test_alias_is_zero_copy_and_isolates_autograd(make_weak_ref):
    device = torch.device("musa", 0)
    original = torch.arange(4, device=device)
    ref = make_weak_ref(original)
    assert torch.is_tensor(ref)
    assert ref.data_ptr() == original.data_ptr()
    assert tuple(ref.shape) == tuple(original.shape)
    assert ref.dtype == original.dtype
    assert ref.grad_fn is None
    ref.can_skip_replay_copy = True
    ref.requires_grad = True
    with torch.no_grad():
        ref.copy_(torch.ones_like(original))
        assert bool(torch.all(original == 1))
        assert torch.equal(ref.clone(), original)


@musa_device
def test_alias_preserves_non_contiguous_layout(make_weak_ref):
    device = torch.device("musa", 0)
    base = torch.arange(12, device=device).view(3, 4)
    view = base.t()
    ref = make_weak_ref(view)
    assert ref.data_ptr() == view.data_ptr()
    assert tuple(ref.stride()) == tuple(view.stride())
    with torch.no_grad():
        ref.copy_(torch.full((4, 3), 7.0, device=device))
    assert torch.equal(view, torch.full((4, 3), 7.0, device=device))


@musa_device
def test_self_check_passes_on_live_musa():
    impl = gate._build_make_weak_ref()
    assert gate._weak_ref_self_check(impl, device=torch.device("musa", 0)) is None


# ------------------------------------------------------------ prerequisites


def _fake_torch(
    *,
    graph_ctx="stock",
    musa_graph=True,
    generator_trio=True,
    register_generator_state=True,
):
    """A torch module shaped like an adapted (or unadapted) MUSA build."""
    fake = types.ModuleType("torch")

    class Generator:
        pass

    if generator_trio:
        for name in ("graphsafe_set_state", "graphsafe_get_state", "clone_state"):
            setattr(Generator, name, lambda *a, **k: None)
    fake.Generator = Generator

    class CUDAGraph:
        pass

    if register_generator_state:
        CUDAGraph.register_generator_state = lambda *a, **k: None

    cuda = types.ModuleType("torch.cuda")
    cuda.is_available = lambda: True
    cuda.CUDAGraph = CUDAGraph
    for name in (
        "graph_pool_handle",
        "synchronize",
        "current_stream",
        "default_stream",
        "set_stream",
        "Stream",
    ):
        setattr(cuda, name, lambda *a, **k: None)
    graphs = types.ModuleType("torch.cuda.graphs")
    stock_ctx = lambda *a, **k: None  # noqa: E731 - stock CUDA-spelled wrapper
    graphs.graph = stock_ctx
    cuda.graphs = graphs
    if graph_ctx == "stock":
        cuda.graph = stock_ctx
    else:  # routed away from the stock wrapper by some owner
        cuda.graph = lambda *a, **k: None

    musa = types.ModuleType("torch.musa")
    musa.is_available = lambda: True
    if musa_graph:
        musa.graph = lambda *a, **k: None
    fake.cuda, fake.musa = cuda, musa
    return fake


def test_probe_accepts_a_fully_adapted_stack(monkeypatch):
    fake = _fake_torch(graph_ctx="routed")
    monkeypatch.setitem(sys.modules, "torch", fake)
    monkeypatch.setattr(gate, "_missing_te_graph_apis", lambda: [])
    assert gate._missing_graph_prerequisites() == []


@pytest.mark.parametrize(
    "kwargs, fragment",
    [
        ({"musa_graph": False}, "stock CUDA wrapper"),
        ({"graph_ctx": "stock"}, "stock CUDA wrapper"),
        ({"register_generator_state": False}, "register generator states"),
        ({"generator_trio": False}, "torch.Generator.graphsafe_set_state"),
    ],
)
def test_probe_declines_unadapted_surfaces(monkeypatch, kwargs, fragment):
    fake = _fake_torch(**kwargs)
    monkeypatch.setitem(sys.modules, "torch", fake)
    missing = gate._missing_torch_graph_apis(fake)
    assert any(fragment in reason for reason in missing), missing


def test_probe_requires_a_live_device(monkeypatch):
    fake = _fake_torch(graph_ctx="routed")
    fake.musa.is_available = lambda: False
    missing = gate._missing_torch_graph_apis(fake)
    assert any("MUSA runtime" in reason for reason in missing), missing


def test_probe_reports_gaps_without_device_adaptation():
    if torch.cuda.is_available():
        pytest.skip("device adaptation is active in this process")
    missing = gate._missing_torch_graph_apis(torch)
    assert missing and all(isinstance(reason, str) and reason for reason in missing)


# ------------------------------------------------------------ hook lifecycle


def test_installs_only_owned_symbol_and_uninstall_removes_it(capable, te_utils):
    assert "make_weak_ref" not in vars(te_utils)
    assert gate._install_make_weak_ref() is True
    installed = te_utils.make_weak_ref
    assert callable(installed)
    assert gate._install_make_weak_ref() is False  # idempotent
    gate._uninstall_make_weak_ref()
    assert "make_weak_ref" not in vars(te_utils)
    assert gate._install_make_weak_ref() is True  # reinstallable
    assert callable(te_utils.make_weak_ref)
    gate._uninstall_make_weak_ref()
    assert "make_weak_ref" not in vars(te_utils)


def test_undo_leaves_third_party_replacement_alone(capable, te_utils):
    gate._install_make_weak_ref()
    ours = te_utils.make_weak_ref

    def vendor(tensor):
        return tensor

    te_utils.make_weak_ref = vendor
    gate._uninstall_make_weak_ref()
    assert te_utils.make_weak_ref is vendor
    assert ours is not vendor


def test_declines_when_vendor_provides_the_symbol(capable, te_utils):
    def vendor(tensor):
        return tensor

    te_utils.make_weak_ref = vendor
    assert gate._install_make_weak_ref() is False
    assert te_utils.make_weak_ref is vendor


def test_declines_when_prerequisites_are_missing(te_utils, monkeypatch):
    monkeypatch.setattr(
        torch, "musa", types.SimpleNamespace(is_available=lambda: True), raising=False
    )
    monkeypatch.setattr(gate, "_musa_fork_installed", lambda: True)
    monkeypatch.setattr(
        gate,
        "_missing_graph_prerequisites",
        lambda: [
            "torch.cuda.CUDAGraph cannot register generator states (graph-safe RNG)"
        ],
    )
    assert gate._install_make_weak_ref() is False
    assert "make_weak_ref" not in vars(te_utils)


def test_declines_when_self_check_fails(capable, te_utils, monkeypatch):
    monkeypatch.setattr(gate, "_weak_ref_self_check", lambda impl, device=None: "boom")
    assert gate._install_make_weak_ref() is False
    assert "make_weak_ref" not in vars(te_utils)


def test_declines_without_musa_runtime(te_utils, monkeypatch):
    monkeypatch.setattr(gate, "_musa_fork_installed", lambda: True)
    monkeypatch.delattr(torch, "musa", raising=False)
    assert gate._install_make_weak_ref() is False
    assert "make_weak_ref" not in vars(te_utils)


def test_declines_without_the_musa_te_fork(te_utils, monkeypatch):
    monkeypatch.setattr(gate, "_musa_fork_installed", lambda: False)
    assert gate._install_make_weak_ref() is False
    assert "make_weak_ref" not in vars(te_utils)
