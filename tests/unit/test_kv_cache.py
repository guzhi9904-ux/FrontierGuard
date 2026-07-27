import torch

from frontierguard.quant.kv_cache import fake_quantize_kv_cache


class TinyCache:
    def __init__(self):
        self.key_cache = [torch.randn(1, 2, 3, 5)]
        self.value_cache = [torch.randn(1, 2, 3, 5)]


class TinyCacheLayer:
    def __init__(self):
        self.keys = torch.randn(1, 2, 3, 5)
        self.values = torch.randn(1, 2, 3, 5)
        self.cumulative_length = 3


class NewStyleTinyCache:
    def __init__(self):
        self.layers = [TinyCacheLayer()]


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


def test_new_dynamic_cache_layers_are_updated_without_losing_metadata():
    cache = NewStyleTinyCache()
    original = cache.layers[0].keys.clone()
    returned = fake_quantize_kv_cache(cache, 4, group_size=4)
    assert returned is cache
    assert not torch.equal(cache.layers[0].keys, original)
    assert cache.layers[0].cumulative_length == 3


def test_new_dynamic_cache_skips_uninitialized_layers():
    cache = NewStyleTinyCache()
    cache.layers[0].keys = None
    cache.layers[0].values = None
    assert fake_quantize_kv_cache(cache, 4) is cache


def test_incremental_quantization_only_changes_appended_suffix():
    cache = NewStyleTinyCache()
    original_key = cache.layers[0].keys.clone()
    original_value = cache.layers[0].values.clone()

    returned = fake_quantize_kv_cache(
        cache,
        4,
        group_size=4,
        new_tokens=1,
    )

    assert returned is cache
    assert torch.equal(cache.layers[0].keys[..., :-1, :], original_key[..., :-1, :])
    assert torch.equal(
        cache.layers[0].values[..., :-1, :],
        original_value[..., :-1, :],
    )
    assert not torch.equal(cache.layers[0].keys[..., -1:, :], original_key[..., -1:, :])
    assert not torch.equal(
        cache.layers[0].values[..., -1:, :],
        original_value[..., -1:, :],
    )


def test_incremental_quantization_requires_positive_token_count():
    cache = NewStyleTinyCache()
    try:
        fake_quantize_kv_cache(cache, 4, new_tokens=0)
    except ValueError as error:
        assert "new_tokens must be positive" in str(error)
    else:
        raise AssertionError("expected a positive new_tokens validation error")
