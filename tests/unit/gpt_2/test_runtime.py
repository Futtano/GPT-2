import pytest
import torch

from gpt_2.runtime import resolve_device, seed_random_sources


def _always_true():
    return True


def _always_false():
    return False


@pytest.mark.parametrize("name", ["cuda", "mps"])
def test_resolve_device_correctly_resolves_supported_devices(name, monkeypatch):
    module_to_patch = torch.cuda
    if name == "mps":
        module_to_patch = torch.backends.mps
    monkeypatch.setattr(module_to_patch, "is_available", _always_true)

    assert torch.device(name) == resolve_device(name)


@pytest.mark.parametrize(
    "patches, desired_result",
    [
        ([_always_true, _always_true], torch.device("cuda")),
        ([_always_true, _always_false], torch.device("cuda")),
        ([_always_false, _always_true], torch.device("mps")),
        ([_always_false, _always_false], torch.device("cpu")),
    ],
)
def test_resolve_device_correctly_handles_auto(patches, desired_result, monkeypatch):
    modules_to_patch = [torch.cuda, torch.backends.mps]
    for module, patch in zip(modules_to_patch, patches, strict=False):
        monkeypatch.setattr(module, "is_available", patch)

    assert resolve_device("auto") == desired_result


def test_resolve_device_correctly_raises_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", _always_false)
    with pytest.raises(RuntimeError, match="cuda"):
        resolve_device("cuda")


def test_resolve_device_correctly_raises_mps(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", _always_false)
    with pytest.raises(RuntimeError, match="mps"):
        resolve_device("mps")


def test_resolve_device_resolves_cpu():
    assert resolve_device("cpu") == torch.device("cpu")


def test_auto_prefers_cuda_without_checking_mps(mocker):
    cuda_available = mocker.patch(
        "gpt_2.runtime.torch.cuda.is_available",
        return_value=True,
    )
    mps_available = mocker.patch(
        "gpt_2.runtime.torch.backends.mps.is_available",
    )

    assert resolve_device("auto") == torch.device("cuda")
    cuda_available.assert_called_once_with()
    mps_available.assert_not_called()


def test_cpu_does_not_check_accelerator_availability(mocker):
    cuda_available = mocker.patch(
        "gpt_2.runtime.torch.cuda.is_available",
    )
    mps_available = mocker.patch(
        "gpt_2.runtime.torch.backends.mps.is_available",
    )

    assert resolve_device("cpu") == torch.device("cpu")
    cuda_available.assert_not_called()
    mps_available.assert_not_called()


@pytest.fixture
def seed():
    return 42


def test_random_seed_all_sources_exactly_once(seed, mocker):
    mock_random = mocker.patch("gpt_2.runtime.random.seed")
    mock_numpy = mocker.patch("gpt_2.runtime.np.random.seed")
    mock_torch = mocker.patch("gpt_2.runtime.torch.manual_seed")

    seed_random_sources(seed)

    mock_random.assert_called_once_with(seed)
    mock_numpy.assert_called_once_with(seed)
    mock_torch.assert_called_once_with(seed)
