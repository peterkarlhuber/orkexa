"""Storage backends of Orkexa, one subpackage each (:mod:`orkexa.backends.sqlite`).

A backend implements the storage contracts of :mod:`orkexa.retrieval`, :mod:`orkexa.files` and
:mod:`orkexa.memory`; it imports those contract modules, never the other way round.
"""
