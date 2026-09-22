"""Two-rank validation of control-collective patches against pristine upstream source.
Hardware worker for test_control_collectives.py; run via torch.distributed.run
(see test_two_rank_control_collectives)."""

import ast
import inspect
import os
from io import BytesIO
from signal import SIGTERM
from types import SimpleNamespace

import torch
from megatron.core import safe_globals
from megatron.training import checkpointing, training

import training_musa_adaptor as tma

tma.install()  # idempotent; the automatic channel already fired
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
torch.distributed.init_process_group("nccl")
try:
    rank = torch.distributed.get_rank()
    timestamp = 1789452300.123456
    # Execute the actual two startup reductions from upstream pretrain. The
    # original function's global name is retained for the module-local proxy.
    tree = ast.parse(inspect.getsource(training.pretrain))
    body = tree.body[0].body
    start = next(
        i
        for i, node in enumerate(body)
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "program_start_global"
    )
    end = next(
        i
        for i in range(start, len(body))
        if isinstance(body[i], ast.Assign)
        and isinstance(body[i].targets[0], ast.Name)
        and body[i].targets[0].id == "_LEGACY_TRAIN_START_TIME"
    )
    function = ast.parse("def pretrain():\n    pass").body[0]
    function.body = body[start : end + 1]
    namespace = dict(
        __name__=training.__name__,
        torch=training.torch,
        _TRAIN_START_TIME=timestamp + rank,
        _LEGACY_TRAIN_START_TIME=timestamp + rank,
        _STARTUP_TIMESTAMPS={"program_start": timestamp + rank},
    )
    namespace["set_startup_timestamps"] = lambda **kw: namespace[
        "_STARTUP_TIMESTAMPS"
    ].update(kw)
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
            "<upstream startup reductions>",
            "exec",
        ),
        namespace,
    )
    namespace["pretrain"]()
    assert abs(namespace["_LEGACY_TRAIN_START_TIME"] - timestamp) < 2e-6
    assert abs(namespace["_STARTUP_TIMESTAMPS"]["program_start"] - timestamp) < 2e-6

    proxy = checkpointing.torch
    checkpointing.get_args = lambda: SimpleNamespace(distributed_timeout_minutes=1)

    def disk_io():
        group = proxy.distributed._active_group.get()
        assert torch.distributed.get_backend(group) == "gloo"

    scope = dict(__name__=checkpointing.__name__, torch=proxy, disk_io=disk_io)
    exec(
        "def save_checkpoint():\n    disk_io()\n    torch.distributed.barrier()\n    torch.distributed.barrier()",
        scope,
    )
    from training_musa_adaptor.patches.megatron.control_collectives import (
        _checkpoint_save,
    )

    _checkpoint_save(scope["save_checkpoint"])()

    with torch.serialization.safe_globals([]):
        safe_globals.register_safe_globals()
        stream = BytesIO()
        torch.save({"exit_signal": SIGTERM}, stream)
        stream.seek(0)
        assert torch.load(stream, weights_only=True)["exit_signal"] == SIGTERM
    tma.uninstall()
    assert training.torch is torch
    assert checkpointing.torch is torch
    print("CONTROL_PASS", rank, flush=True)
finally:
    torch.distributed.destroy_process_group()
