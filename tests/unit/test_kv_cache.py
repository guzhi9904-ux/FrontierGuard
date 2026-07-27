import torch

from frontierguard.quant.kv_cache import fake_quantize_kv_cache


class TinyCache:
    def __init__(self):
        self.key_cache = [torch.randn(1, 2, 3, 5)]
        self.value_cache = [torch.randn(1, 2, 3, 5)]


def test_legacy_kv_quantization():
    cache = ((torch.randn(1, 2, 3, 5), torch.randn(1, 2, 3, 5)),)
    result = fake_quantize_kv_cache(cache, 4, group_size=4)
    assert result[0][0].shape == cache[0][0].shape
    assert not torch.equal(result[0][0], cache[0][0])


def test_dynamic_cache_is_updated():
    cache = TinyCache()
    original = cache.key_cache[0].clone()
    returned = fake_quantize_kv_cache(cache, 4, group_size=4)
    assert returned is cache
    assert not torch.equal(cache.key_cache[0], original)
