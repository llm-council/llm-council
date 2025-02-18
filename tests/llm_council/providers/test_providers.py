import os
import pytest

from llm_council.providers.base_provider import PROVIDER_REGISTRY

print(PROVIDER_REGISTRY)


@pytest.mark.parametrize("service", PROVIDER_REGISTRY)
def test_api_keys_exist(service):
    assert PROVIDER_REGISTRY[service]["enabled"]
