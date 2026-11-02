"""``python -m orkexa.dynamic.docker --self-test``: see :mod:`orkexa.dynamic.docker._cli`."""

from __future__ import annotations

import sys

from ._cli import main

if __name__ == "__main__":
    sys.exit(main())
