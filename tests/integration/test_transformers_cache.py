import torch
from transformers.cache_utils import DynamicCache

from frontierguard.quant.kv_cache import fake_quantize_kv_cache


def _first_key(cache):
    if hasattr(cache, "key_cache"):
        return cache.key_cache[0]
    return cache.layers[0].keys


def test_installed_transformers_dynamic_cache_is_supported():
    key = torch.randn(1, 2, 3, 8)
    value = torch.randn(1, 2, 3, 8)
    cache = DynamicCache()
    cache.update(key, value, 0)
    original = _first_key(cache).clone()

    returned = fake_quantize_kv_cache(cache, 4, group_size=4)

    assert returned is cache
    assert _first_key(cache).shape == original.shape
    assert not torch.equal(_first_key(cache), original)
