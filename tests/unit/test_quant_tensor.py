import torch

from frontierguard.quant.tensor import fake_quantize


def test_fake_quant_identity_at_16_bits():
    tensor = torch.randn(3, 7)
    assert fake_quantize(tensor, 16) is tensor


def test_fake_quant_preserves_shape_dtype_and_finiteness():
    tensor = torch.tensor([[0.0, -2.0, 0.5, 4.0, 1.0]], dtype=torch.float32)
    result = fake_quantize(tensor, 4, group_size=4, symmetric=True)
    assert result.shape == tensor.shape
    assert result.dtype == tensor.dtype
    assert torch.isfinite(result).all()
    assert not torch.equal(result, tensor)


def test_asymmetric_constant_group_is_stable():
    tensor = torch.ones(2, 8, dtype=torch.float16)
    result = fake_quantize(tensor, 4, group_size=4, symmetric=False)
    assert torch.isfinite(result).all()
    assert torch.allclose(result, tensor)


def test_fake_quant_supports_non_last_axis():
    tensor = torch.randn(2, 5, 3)
    result = fake_quantize(tensor, 3, group_size=2, axis=1)
    assert result.shape == tensor.shape
