REGISTRY = {}

from .basic_controller import BasicMAC
from .maac_controller import MAACMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["maac_mac"] = MAACMAC