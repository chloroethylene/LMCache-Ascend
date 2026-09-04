# SPDX-License-Identifier: Apache-2.0
# First Party
from lmcache_ascend import _build_info

if _build_info.__framework_name__ == "pytorch":
    # First Party
    import lmcache_ascend  # noqa: F401
elif _build_info.__framework_name__ == "mindspore":
    # First Party
    import lmcache_ascend.mindspore  # noqa: F401
else:
    raise ValueError("Unsupported Framework")

# Third Party
from lmcache.integration.vllm.lmcache_connector_v1 import LMCacheConnectorV1Dynamic


class LMCacheAscendConnectorV1Dynamic(LMCacheConnectorV1Dynamic):
    pass
