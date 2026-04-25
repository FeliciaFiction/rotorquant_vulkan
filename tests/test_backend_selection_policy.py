import pytest

import turboquant as tq


@pytest.mark.parametrize(
    "vulkan_ok,cuda_ok,request_vulkan,expected_backend",
    [
        (True, True, True, "vulkan"),
        (False, True, True, "cuda"),
        (False, False, True, "pytorch"),
        (True, True, False, "cuda"),
        (True, False, False, "pytorch"),
        (False, True, False, "cuda"),
        (False, False, False, "pytorch"),
    ],
)
def test_select_backend_auto_truth_table(
    monkeypatch, vulkan_ok, cuda_ok, request_vulkan, expected_backend
):
    monkeypatch.setattr(tq, "is_vulkan_available", lambda: vulkan_ok)
    monkeypatch.setattr(tq, "is_cuda_available", lambda: cuda_ok)

    backend, reason = tq.select_backend(request_vulkan=request_vulkan)
    assert backend == expected_backend
    assert isinstance(reason, str) and reason


def test_select_backend_explicit_vulkan(monkeypatch):
    monkeypatch.setattr(tq, "is_vulkan_available", lambda: True)
    monkeypatch.setattr(tq, "is_cuda_available", lambda: True)
    backend, reason = tq.select_backend(backend="vulkan")
    assert backend == "vulkan"
    assert "explicit override" in reason.lower()


def test_select_backend_explicit_cuda(monkeypatch):
    monkeypatch.setattr(tq, "is_vulkan_available", lambda: True)
    monkeypatch.setattr(tq, "is_cuda_available", lambda: True)
    backend, reason = tq.select_backend(backend="cuda")
    assert backend == "cuda"
    assert "explicit override" in reason.lower()


def test_select_backend_explicit_pytorch(monkeypatch):
    monkeypatch.setattr(tq, "is_vulkan_available", lambda: False)
    monkeypatch.setattr(tq, "is_cuda_available", lambda: False)
    backend, reason = tq.select_backend(backend="pytorch")
    assert backend == "pytorch"
    assert "explicit override" in reason.lower()


def test_select_backend_explicit_unavailable_raises(monkeypatch):
    monkeypatch.setattr(tq, "is_vulkan_available", lambda: False)
    monkeypatch.setattr(tq, "is_cuda_available", lambda: False)

    with pytest.raises(RuntimeError, match="Vulkan"):
        tq.select_backend(backend="vulkan")
    with pytest.raises(RuntimeError, match="CUDA"):
        tq.select_backend(backend="cuda")


def test_select_backend_unknown_override_raises():
    with pytest.raises(ValueError, match="Unknown backend override"):
        tq.select_backend(backend="metal")
