"""Write ``nodestro_catalog.json``: Nodestro's statement catalog rendered with ``nodestro_`` (MUP-ONA
T-PAR-2).

Usage (standard library, psycopg and orkexa; run from the repository root)::

    git -C ../nodestro-runtime archive 76f9cb2 src/nodestro_runtime/pg_generic | tar -x -C
    /tmp/nodestro-76f9cb2 python tests/fixtures/postgres/make_nodestro_catalog.py /tmp/nodestro-76f9cb2
    76f9cb2

The generic part of nodestro-runtime (``src/nodestro_runtime/pg_generic``) is loaded from that tree as a
package of its own (it imports nothing of Nodestro), and every entry of its ``CATALOG`` is written with its
rendered text (``render(name, "nodestro_").as_string(None)``), parameters, kind and flags. The file is test
data: never edit it, regenerate it from a named commit and update ``nodestro_catalog.json.sha256``.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


def load(tree: Path) -> object:
    package = tree / "src" / "nodestro_runtime" / "pg_generic"
    spec = importlib.util.spec_from_file_location(
        "nodestro_pg_generic", package / "__init__.py", submodule_search_locations=[str(package)]
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return sys.modules["nodestro_pg_generic._sql"]


def main(tree: str, commit: str) -> None:
    sql_module = load(Path(tree))
    catalog = sql_module.CATALOG  # type: ignore[attr-defined]
    render = sql_module.render  # type: ignore[attr-defined]
    data = {
        "origin": f"nodestro-runtime {commit}: src/nodestro_runtime/pg_generic/_sql.py (MUP-ONA B4)",
        "prefix": "nodestro_",
        "statements": {
            name: {
                "text": render(name, "nodestro_").as_string(None),
                "params": list(entry.params),
                "kind": entry.kind,
                "partitioned": entry.partitioned,
                "rows": entry.rows,
            }
            for name, entry in catalog.items()
        },
    }
    out = Path(__file__).resolve().parent / "nodestro_catalog.json"
    out.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
