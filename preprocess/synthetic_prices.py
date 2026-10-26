"""Synthetic price generator for ShopTalk.

ABO listings have no price field. These prices are SIMULATED so the price-filter
logic ("under $50") can be demonstrated. They are NOT real prices.
Deterministic: the same item_id always gets the same price.
"""
import hashlib
import numpy as np

SALT = "shoptalk-price-v1"

# Exact product-type overrides: (median USD, sigma). Checked first.
EXACT = {
    "HOME_BED_AND_BATH": (60, 0.55),
    "SUITCASE": (70, 0.50),
    "PORTABLE_AUDIO": (60, 0.60),
    "PORTABLE_TOOL_BOX": (35, 0.50),
    "SPORT_TABLE_GAME": (120, 0.60),
    "GROCERY": (12, 0.50),
}

# Whole-word matches on the product type split by "_". First match wins.
WORDS = [
    ({"SOFA", "COUCH"}, 650, 0.45),
    ({"BED", "MATTRESS"}, 400, 0.50),
    ({"DESK"}, 200, 0.50),
    ({"TABLE"}, 180, 0.55),
    ({"CHAIR"}, 110, 0.50),
    ({"RUG"}, 90, 0.60),
    ({"BOOT", "BOOTS"}, 75, 0.45),
    ({"SHOES", "SHOE"}, 55, 0.45),
    ({"SANDAL"}, 30, 0.45),
    ({"SHIRT"}, 25, 0.40),
    ({"LAMP"}, 45, 0.50),
    ({"CASE"}, 14, 0.45),
]
DEFAULT_PROFILE = (40, 0.60)


def _profile_for(product_type):
    t = (product_type or "").upper()
    if t in EXACT:
        return EXACT[t]
    words = set(t.split("_"))
    for keys, median, sigma in WORDS:
        if words & keys:
            return median, sigma
    return DEFAULT_PROFILE


def synthetic_price(item_id, product_type):
    """Return a deterministic, price-like number ending in .99."""
    digest = hashlib.sha256(f"{SALT}:{item_id}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    median, sigma = _profile_for(product_type)
    raw = rng.lognormal(mean=np.log(median), sigma=sigma)
    return round(max(raw, 2.0)) - 0.01
