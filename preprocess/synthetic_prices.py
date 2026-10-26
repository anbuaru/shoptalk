
"""Synthetic price generator for ShopTalk.

ABO listings have no price field. These prices are SIMULATED, so the price-filter
logic ("under $50") can be demonstrated. They are NOT real prices.
Deterministic: the same item_id always gets the same price.
"""
import hashlib
import numpy as np

SALT = "shoptalk-price-v1"

# (substring in product type, median price in USD, log-normal sigma)
# Edit these after you look at your top product types.
PROFILES = [
    ("SOFA", 650, 0.45),
    ("BED", 400, 0.50),
    ("TABLE", 180, 0.55),
    ("DESK", 200, 0.50),
    ("CHAIR", 110, 0.50),
    ("LAMP", 45, 0.50),
    ("SHOE", 55, 0.45),
    ("BOOT", 75, 0.45),
    ("SHIRT", 25, 0.40),
    ("CASE", 14, 0.45),
    ("RUG", 90, 0.60),
]
DEFAULT_PROFILE = (40, 0.60)


def _profile_for(product_type):
    t = (product_type or "").upper()
    for key, median, sigma in PROFILES:
        if key in t:
            return median, sigma
    return DEFAULT_PROFILE


def synthetic_price(item_id, product_type):
    """Return a deterministic, price-like number ending in .99."""
    digest = hashlib.sha256(f"{SALT}:{item_id}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    median, sigma = _profile_for(product_type)
    raw = rng.lognormal(mean=np.log(median), sigma=sigma)
    return round(max(raw, 2.0)) - 0.01
