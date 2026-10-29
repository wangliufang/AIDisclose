"""Disclosure sources.

==========  ===============================================  ======================
``edgar``   SEC EDGAR Form 10-K                              United States
``xbrl``    filings.xbrl.org (ESEF, UKSEF reports; annual status requires review)    EU/EEA, United Kingdom
``local``   any folder of HTML/XHTML/PDF/TXT reports         any jurisdiction
==========  ===============================================  ======================
"""

from __future__ import annotations

from typing import Any

from .base import HttpClient, Source, fiscal_year_from_period_end
from .edgar import EdgarSource
from .local import LocalSource
from .xbrl_filings import XbrlFilingsSource

REGISTRY = {"edgar": EdgarSource, "xbrl": XbrlFilingsSource, "esef": XbrlFilingsSource, "local": LocalSource}


def get_source(name: str, **options: Any) -> Source:
    try:
        cls = REGISTRY[name.lower()]
    except KeyError:
        raise ValueError(f"unknown source {name!r}; available: {sorted(REGISTRY)}") from None
    return cls(**options)


BUILTIN_SOURCES = frozenset(REGISTRY)


def register_source(name: str, cls: type[Source]) -> None:
    """Register a custom adapter under *name*.

    Registered adapters can be used with :func:`get_source` and, when the
    registering module is loaded with ``aidisclose --plugin <module>`` or
    exposed through the ``aidisclose.sources`` entry-point group, from the CLI.
    """
    if not isinstance(cls, type) or not issubclass(cls, Source):
        raise TypeError("adapters must subclass aidisclose.sources.Source")
    REGISTRY[name.lower()] = cls


def load_entry_point_sources() -> list[str]:
    """Register adapters published by installed packages.

    A package exposes an adapter with, in its ``pyproject.toml``::

        [project.entry-points."aidisclose.sources"]
        myregulator = "my_package.adapters:MyRegulatorSource"
    """
    from importlib.metadata import entry_points

    loaded = []
    for ep in entry_points(group="aidisclose.sources"):
        try:
            register_source(ep.name, ep.load())
            loaded.append(ep.name)
        except Exception as exc:  # a broken plugin must not break the CLI
            import logging

            logging.getLogger(__name__).warning("could not load source plugin %s: %s", ep.name, exc)
    return loaded


__all__ = [
    "HttpClient",
    "Source",
    "EdgarSource",
    "XbrlFilingsSource",
    "LocalSource",
    "get_source",
    "register_source",
    "load_entry_point_sources",
    "BUILTIN_SOURCES",
    "REGISTRY",
    "fiscal_year_from_period_end",
]
