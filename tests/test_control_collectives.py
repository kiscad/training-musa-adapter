"""Regression contracts for source changes migrated out of Megatron-LM."""

from io import BytesIO
from signal import SIGTERM, Signals
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from tests.conftest import integration_env

from training_musa_adaptor.patches.megatron import control_collectives as control

torch = pytest.importorskip("torch")


def make_torch(backend="mccl"):
    dist = SimpleNamespace(
        is_initialized=lambda: True,
        get_backend=lambda: backend,
        group=SimpleNamespace(WORLD=object()),
        new_group=Mock(side_effect=[object(), object()]),
        all_reduce=Mock(),
        barrier=Mock(),
        ReduceOp=torch.distributed.ReduceOp,
    )
    return (
        SimpleNamespace(
            distributed=dist, float64=torch.float64, int64=torch.int64, tensor=torch.tensor
        ),
        dist,
    )


def define(name, source, **globals):
    namespace = dict(__name__=name, **globals)
    exec(source, namespace)
    return namespace


def test_startup_integer_microseconds_preserves_upstream_globals():
    base, dist = make_torch()
    earliest = 1789452300.123456

    def reduce(value, **kwargs):
        assert value.dtype == torch.int64
        value.fill_(int(earliest * 1_000_000))

    dist.all_reduce.side_effect = reduce
    proxy = control._startup_torch(base)
    namespace = define(
        "megatron.training.training",
        """
def pretrain(timestamp):
    global _LEGACY_TRAIN_START_TIME
    value = torch.tensor([timestamp], dtype=torch.float64)
    torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.MIN)
    _LEGACY_TRAIN_START_TIME = value.item()
    return value.item()
""",
        torch=proxy,
    )
    assert abs(namespace["pretrain"](earliest + 10) - earliest) < 2e-6
    assert namespace["_LEGACY_TRAIN_START_TIME"] == namespace["pretrain"](earliest)
    assert proxy.tensor is base.tensor


@pytest.mark.parametrize(
    "backend,name,dtype,op,async_op",
    [
        ("gloo", "pretrain", torch.float64, "MIN", False),
        ("nccl", "pretrain", torch.float64, "MIN", False),
        ("mccl", "train_step", torch.float64, "MIN", False),
        ("mccl", "pretrain", torch.float32, "MIN", False),
        ("mccl", "pretrain", torch.float64, "SUM", False),
        ("mccl", "pretrain", torch.float64, "MIN", True),
    ],
)
def test_unrelated_reductions_pass_through(backend, name, dtype, op, async_op):
    base, dist = make_torch(backend)
    proxy = control._startup_torch(base)
    namespace = define(
        "megatron.training.training",
        f"""
def {name}(value):
    return torch.distributed.all_reduce(value, op=op, async_op=asynchronous)
""",
        torch=proxy,
        op=getattr(dist.ReduceOp, op),
        asynchronous=async_op,
    )
    value = torch.tensor([12.5], dtype=dtype)
    assert namespace[name](value) is dist.all_reduce.return_value
    assert dist.all_reduce.call_args.args[0] is value


def test_checkpoint_group_cache_and_reinitialization():
    base, dist = make_torch()
    proxy = control._checkpoint_torch(base).distributed
    group = proxy.checkpoint_group(60)
    assert proxy.checkpoint_group(60) is group
    assert dist.new_group.call_count == 1
    assert dist.new_group.call_args.kwargs["backend"] == "gloo"
    assert dist.new_group.call_args.kwargs["timeout"].total_seconds() == 3600
    dist.group.WORLD = object()
    assert proxy.checkpoint_group(60) is not group
    assert dist.new_group.call_count == 2


@pytest.mark.parametrize("initialized,backend", [(False, "mccl"), (True, "nccl"), (True, "gloo")])
def test_other_backends_keep_default_group(initialized, backend):
    base, dist = make_torch(backend)
    dist.is_initialized = lambda: initialized
    assert control._checkpoint_torch(base).distributed.checkpoint_group(60) is None
    dist.new_group.assert_not_called()


@pytest.mark.parametrize("fail", [False, True])
def test_checkpoint_group_created_before_io_and_context_reset(stub_module, fail):
    base, dist = make_torch()
    proxy = control._checkpoint_torch(base)
    subgroup = object()

    def disk_io():
        assert dist.new_group.call_count == 1

    namespace = define(
        "megatron.training.checkpointing",
        """
def other_function():
    torch.distributed.barrier()
def save_checkpoint():
    disk_io()
    torch.distributed.barrier()
    other_function()
    torch.distributed.barrier(group=subgroup)
    torch.distributed.barrier()
    if fail:
        raise ValueError('disk failure')
    return 42
""",
        torch=proxy,
        subgroup=subgroup,
        disk_io=disk_io,
        fail=fail,
    )
    module = stub_module(
        "megatron.training.checkpointing",
        torch=proxy,
        get_args=lambda: SimpleNamespace(distributed_timeout_minutes=60),
    )
    stub_module("megatron.training", checkpointing=module)
    wrapped = control._checkpoint_save(namespace["save_checkpoint"])
    if fail:
        with pytest.raises(ValueError, match="disk failure"):
            wrapped()
    else:
        assert wrapped() == 42
    calls = dist.barrier.call_args_list
    assert calls[0].kwargs["group"] is proxy.distributed._group
    assert calls[1].kwargs == {}
    assert calls[2].kwargs["group"] is subgroup
    assert calls[3].kwargs["group"] is proxy.distributed._group
    assert proxy.distributed._active_group.get() is control._UNSET


def test_proxy_only_does_not_redirect_barriers():
    base, dist = make_torch()
    namespace = define(
        "megatron.training.checkpointing",
        """
def save_checkpoint():
    torch.distributed.barrier()
""",
        torch=control._checkpoint_torch(base),
    )
    namespace["save_checkpoint"]()
    dist.barrier.assert_called_once_with()


@pytest.mark.parametrize("order", [(0,), (1,), (0, 1), (1, 0)])
def test_checkpoint_companions_are_ordered_and_selectable(engine, stub_module, order):
    base, dist = make_torch()
    module = stub_module(
        "megatron.training.checkpointing",
        torch=base,
        get_args=lambda: SimpleNamespace(distributed_timeout_minutes=10),
    )
    stub_module("megatron.training", checkpointing=module)
    exec("def save_checkpoint():\n    torch.distributed.barrier()\n", module.__dict__)
    original = module.save_checkpoint
    patches = [p for p in control.PATCHES if "host-barrier" in p.id]
    engine.register([patches[i] for i in order])
    engine.install()
    module.save_checkpoint()
    if len(order) == 2:
        assert dist.new_group.call_count == 1
        dist.barrier.assert_called_once_with(group=module.torch.distributed._group)
    else:
        dist.new_group.assert_not_called()
        dist.barrier.assert_called_once_with()
    engine.uninstall()
    assert module.torch is base
    assert module.save_checkpoint is original


def test_signal_member_list_is_nonmutating_idempotent_and_loadable():
    original = [Signals]
    expanded = control._signal_globals(original)
    assert original == [Signals]
    assert control._signal_globals(expanded) == expanded
    with torch.serialization.safe_globals(expanded):
        stream = BytesIO()
        torch.save({"exit_signal": SIGTERM}, stream)
        stream.seek(0)
        assert torch.load(stream, weights_only=True)["exit_signal"] == SIGTERM


@pytest.mark.parametrize(
    "iteration,elapsed,saves,exits",
    [
        (1, 34, 0, False),
        (1000, 100, 1, False),
        (1, 236 * 60, 1, True),
        (1000, 236 * 60, 1, True),
    ],
)
def test_checkpoint_exit_policy(iteration, elapsed, saves, exits):
    import ast
    import os
    from pathlib import Path

    checkout = os.environ.get("MEGATRON_LM_PATH")
    if not checkout:
        pytest.skip("set MEGATRON_LM_PATH for upstream exit-policy regression")
    TRAINING = Path(checkout) / "megatron" / "training"
    # Execute the real policy without importing the entire training/model stack.
    tree = ast.parse((TRAINING / "training.py").read_text())
    try:
        function = next(
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "checkpoint_and_decide_exit"
        )
    except StopIteration:
        pytest.skip("checkpoint_and_decide_exit not found in upstream training.py")
    args = SimpleNamespace(
        exit_signal_handler=False,
        save="/tmp/checkpoint",
        save_interval=1000,
        non_persistent_save_interval=None,
        exit_duration_in_mins=235,
        exit_interval=None,
        phase_transition_iterations=None,
    )
    save = Mock()
    namespace = dict(
        get_args=lambda: args,
        get_timers=lambda: None,
        time=SimpleNamespace(time=lambda: 1789452300 + elapsed),
        _TRAIN_START_TIME=1789452300,
        save_checkpoint_and_time=save,
        print_datetime=Mock(),
        torch=SimpleNamespace(
            int=torch.int,
            tensor=lambda data, **kw: torch.tensor(data, dtype=kw["dtype"]),
            distributed=SimpleNamespace(all_reduce=Mock(), ReduceOp=torch.distributed.ReduceOp),
        ),
    )
    exec(
        compile(
            ast.Module(body=[function], type_ignores=[]), str(TRAINING / "training.py"), "exec"
        ),
        namespace,
    )
    assert (
        namespace["checkpoint_and_decide_exit"](None, None, None, iteration, 0, None, None) is exits
    )
    assert save.call_count == saves


@pytest.mark.integration
def test_two_rank_control_collectives():
    import os
    import subprocess
    import sys
    from pathlib import Path

    if os.environ.get("TMA_RUN_INTEGRATION") != "1":
        pytest.skip("set TMA_RUN_INTEGRATION=1 for two-rank MUSA control collectives")
    env = integration_env({"CUDA_DEVICE_MAX_CONNECTIONS": "1"})
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=2",
            str(Path(__file__).with_name("control_collectives_smoke.py")),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count("CONTROL_PASS") == 2
