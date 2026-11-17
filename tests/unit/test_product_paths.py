"""The product is named twice (scripts/product.sh for jobs, decklight.search.product for Python): they must agree."""

import re
from pathlib import Path

from decklight.search import product

ROOT = Path(__file__).resolve().parents[2]


def test_product_sh_and_python_name_the_same_files():
    sh = (ROOT / "scripts/product.sh").read_text()
    ckpt = re.search(r"^PRODUCT_CKPT=(\S+)", sh, re.M).group(1)
    value = re.search(r"^PRODUCT_VALUE=(\S+)", sh, re.M).group(1)
    assert product.PRODUCT_CKPT == ROOT / ckpt
    assert product.PRODUCT_VALUE == ROOT / value
