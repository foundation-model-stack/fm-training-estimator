# Local
from .parser import parse
from .utils import is_fsdp


def test_fsdp_empty():
    config = {}
    _, ta, _, _, _, _ = parse(config)

    assert is_fsdp(ta) is False


def test_fsdp_enabled():
    config = {"fsdp": "full_shard"}
    _, ta, _, _, _, _ = parse(config)

    assert is_fsdp(ta) is True

    config = {"fsdp": ["hybrid_shard", "offload"]}
    _, ta, _, _, _, _ = parse(config)

    assert is_fsdp(ta) is True
