"""Bucket-writer protocol tests without importing the vendor Megatron/TE stack.

The stand-in mirrors filesystem_async.py's CPU worker: write one (index,
list-or-Exception) message, then consume a count token and call task_done.
Hardware save/load and external async process safety require separate tests.
"""

from __future__ import annotations

import pytest

from training_musa_adaptor.patches.megatron import checkpointing as _checkpointing


class _Sink:
    def __init__(self) -> None:
        self.payloads = []

    def put(self, item) -> None:
        self.payloads.append(item)

    @property
    def payload(self):
        assert len(self.payloads) == 1, "writer must publish exactly one final payload"
        return self.payloads[0]


@pytest.fixture()
def writer(stub_module, monkeypatch):

    class Writer:
        @staticmethod
        def write_preloaded_data_multiproc(*args, **kwargs):
            pytest.fail("forked launcher must not run")

        @staticmethod
        def write_preloaded_data(
            transforms, idx, bucket, results_queue, count_queue, **kwargs
        ):
            results_queue.put((idx, [f"result-{idx}"]))
            count_queue.get()
            count_queue.task_done()

    stub_module(
        "megatron.core.dist_checkpointing.strategies.filesystem_async",
        FileSystemWriterAsync=Writer,
    )
    replacement = _checkpointing._serial_writer(Writer.write_preloaded_data_multiproc)
    assert isinstance(replacement, staticmethod)
    Writer.write_preloaded_data_multiproc = replacement
    return Writer


BUCKETS = [("f0", "k0", ()), ("f1", "k1", ())]


def test_successful_buckets_are_collected_without_binding_self(writer):
    sink = _Sink()
    writer().write_preloaded_data_multiproc([], False, 0, BUCKETS, sink)
    assert sink.payload == {0: ["result-0"], 1: ["result-1"]}


def test_empty_buckets_publish_empty_dict(writer):
    sink = _Sink()
    writer.write_preloaded_data_multiproc([], False, 0, [], sink)
    assert sink.payload == {}


@pytest.mark.parametrize("use_msc", [False, True])
def test_worker_arguments_preserved(writer, monkeypatch, use_msc):
    transforms = [object()]
    seen = []

    def worker(transform_list, idx, bucket, results_queue, count_queue, **kwargs):
        assert transform_list is transforms
        seen.append((idx, bucket, kwargs))
        results_queue.put((idx, []))
        count_queue.get()
        count_queue.task_done()

    monkeypatch.setattr(writer, "write_preloaded_data", worker)
    sink = _Sink()
    writer.write_preloaded_data_multiproc(transforms, use_msc, 7, BUCKETS, sink)
    assert seen == [
        (idx, bucket, {"use_fsync": True, "use_msc": use_msc})
        for idx, bucket in enumerate(BUCKETS)
    ]
    assert sink.payload == {0: [], 1: []}


@pytest.mark.parametrize("failing_index", [0, 1])
def test_failed_bucket_replaces_whole_payload_and_stops(
    writer, monkeypatch, failing_index
):
    error = RuntimeError(f"bucket {failing_index} failed")
    seen = []

    def worker(transforms, idx, bucket, results_queue, count_queue, **kwargs):
        seen.append(idx)
        results_queue.put((idx, error if idx == failing_index else ["success"]))
        count_queue.get()
        count_queue.task_done()

    monkeypatch.setattr(writer, "write_preloaded_data", worker)
    sink = _Sink()
    writer.write_preloaded_data_multiproc([], False, 0, BUCKETS, sink)
    assert sink.payload is error
    assert seen == list(range(failing_index + 1))


def test_escaping_exception_replaces_payload(writer, monkeypatch):
    error = ValueError("escaped")

    def worker(*args, **kwargs):
        raise error

    monkeypatch.setattr(writer, "write_preloaded_data", worker)
    sink = _Sink()
    writer.write_preloaded_data_multiproc([], False, 0, BUCKETS, sink)
    assert sink.payload is error


def test_missing_result_publishes_failure_instead_of_escaping(writer, monkeypatch):
    monkeypatch.setattr(writer, "write_preloaded_data", lambda *args, **kwargs: None)
    sink = _Sink()
    writer.write_preloaded_data_multiproc([], False, 0, BUCKETS, sink)
    assert isinstance(sink.payload, RuntimeError)
    assert "result queue is empty" in str(sink.payload)


@pytest.mark.parametrize(
    "message,error_type,match",
    [
        ((5, []), RuntimeError, "unexpected index"),
        ((0, "not a list"), TypeError, "not list"),
        ((0,), ValueError, "not enough values"),
    ],
)
def test_malformed_result_publishes_failure(
    writer, monkeypatch, message, error_type, match
):
    def worker(transforms, idx, bucket, results_queue, count_queue, **kwargs):
        results_queue.put(message)
        count_queue.get()
        count_queue.task_done()

    monkeypatch.setattr(writer, "write_preloaded_data", worker)
    sink = _Sink()
    writer.write_preloaded_data_multiproc([], False, 0, BUCKETS, sink)
    assert isinstance(sink.payload, error_type)
    assert match in str(sink.payload)


def test_results_queue_is_fifo_and_never_blocks_when_empty():
    queue = _checkpointing._ImmediateQueue()
    queue.put("first")
    queue.put("second")
    assert queue.get() == "first"
    assert queue.get() == "second"
    with pytest.raises(RuntimeError, match="result queue is empty"):
        queue.get()


def test_counter_detects_underflow():
    counter = _checkpointing._Counter()
    counter.put(0)
    counter.get()
    counter.task_done()
    with pytest.raises(RuntimeError, match="count queue underflow"):
        counter.get()


@pytest.fixture
def dcp_selector(stub_module):
    from types import SimpleNamespace

    _checkpointing._uninstall_dcp_device()
    device = SimpleNamespace(type="musa")
    cuda = SimpleNamespace(current_stream=lambda: SimpleNamespace(device=device))
    stub_module("torch", musa=SimpleNamespace(is_available=lambda: True), cuda=cuda)
    original = lambda: "cuda"
    module = stub_module(
        "torch.distributed.checkpoint.filesystem", _get_available_device_type=original
    )
    yield module, original, device
    _checkpointing._uninstall_dcp_device()


def test_dcp_selector_lifecycle_and_real_cuda(dcp_selector):
    import inspect

    module, original, device = dcp_selector
    assert _checkpointing._install_dcp_device()
    replacement = module._get_available_device_type
    assert inspect.signature(replacement) == inspect.signature(original)
    assert replacement() == "musa"
    assert not _checkpointing._install_dcp_device()
    assert module._get_available_device_type is replacement
    device.type = "cuda"
    assert replacement() == "cuda"
    _checkpointing._uninstall_dcp_device()
    assert module._get_available_device_type is original
    assert _checkpointing._install_dcp_device()
    third_party = lambda: "other"
    module._get_available_device_type = third_party
    _checkpointing._uninstall_dcp_device()
    assert module._get_available_device_type is third_party


@pytest.mark.parametrize("selected", ["cpu", "musa", "xpu", None])
def test_dcp_selector_preserves_other_devices(dcp_selector, selected):
    module, _, _ = dcp_selector
    module._get_available_device_type = lambda: selected
    assert _checkpointing._install_dcp_device()
    assert module._get_available_device_type() == selected


def test_dcp_selector_without_musa(dcp_selector, monkeypatch):
    import sys

    module, original, _ = dcp_selector
    monkeypatch.setattr(sys.modules["torch"].musa, "is_available", lambda: False)
    assert not _checkpointing._install_dcp_device()
    assert module._get_available_device_type is original
