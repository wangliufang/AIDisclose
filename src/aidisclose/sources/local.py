"""Adapter for local files: any jurisdiction, any document type.

Point it at a folder of HTML, XHTML, PDF or plain-text reports. Optional
metadata can be supplied as a CSV file with one row per file::

    file,firm_id,firm_name,fiscal_year,country,industry_code,industry_scheme,form,period_end
    acme_2023.pdf,ACME,ACME plc,2023,GB,J,NACE,annual-report,2023-12-31

Without a metadata file the firm identifier is the file stem with any year
removed, and the fiscal year is the first four-digit year in the file name.
Writing an adapter for a new regulator usually means subclassing
:class:`~aidisclose.sources.base.Source` and implementing two methods; see
``docs/sources.md``.
"""

from __future__ import annotations

import hashlib
import hmac
import re
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import url2pathname

import pandas as pd

from ..industry import sic_to_nace_section
from ..models import Document, Filing
from .base import Source

SUPPORTED = (".htm", ".html", ".xhtml", ".xml", ".txt", ".pdf")
_YEAR = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")


_FORM_IN_NAME = re.compile(r"(?<![a-z0-9])(10-?k|20-?f|40-?f)(?![a-z0-9])", re.I)


def _form_from_name(name: str) -> str:
    m = _FORM_IN_NAME.search(name)
    if not m:
        return ""
    code = m.group(1).upper().replace("-", "")
    return f"{code[:2]}-{code[2:]}"


def _path_from_uri(uri: str) -> str:
    """``file:///C:/x%20y.pdf`` -> ``C:\\x y.pdf`` on Windows, ``/x y.pdf`` elsewhere."""
    parsed = urlparse(uri)
    if parsed.scheme != "file":
        return uri
    path = url2pathname(parsed.path)  # also decodes %-escapes
    if parsed.netloc:  # UNC path file://server/share/x
        path = f"//{parsed.netloc}{path}"
    return path


class LocalSource(Source):
    """Read reports from a local folder.

    10-K and 20-F documents (``form`` column, or "10-K"/"20-F" in the file
    name) are split into Items; other documents into annual-report parts
    (``AR:management``, ``AR:risk``, ...), see :mod:`aidisclose.text.sections`.
    """

    name = "local"

    def __init__(self, directory: str | Path, metadata: str | Path | None = None, recursive: bool = True):
        self.directory = Path(directory).resolve()
        if not self.directory.is_dir():
            raise NotADirectoryError(self.directory)
        self.metadata_path = Path(metadata) if metadata else None
        self.recursive = recursive

    def config(self) -> dict:
        return {"name": self.name, "recursive": self.recursive, "metadata_supplied": self.metadata_path is not None}

    def _files(self) -> list[Path]:
        pattern = "**/*" if self.recursive else "*"
        return sorted(p for p in self.directory.glob(pattern) if p.is_file() and p.suffix.lower() in SUPPORTED)

    def list_filings(self, **_ignored) -> list[Filing]:
        meta = None
        if self.metadata_path is not None:
            meta = pd.read_csv(self.metadata_path, dtype=str, keep_default_na=False)
            if "file" not in meta.columns:
                raise ValueError("metadata CSV needs a 'file' column")
            if meta["file"].str.strip().eq("").any() or meta["file"].duplicated().any():
                raise ValueError("metadata file paths must be nonempty and unique")
            meta = meta.set_index("file")
        paths = self._files()
        names: dict[str, list[Path]] = {}
        for path in paths:
            names.setdefault(path.name, []).append(path)
        filings = []
        matched_metadata: set[str] = set()
        inferred_id_locations: dict[str, list[tuple[Path, bool]]] = {}
        for path in paths:
            rel = path.relative_to(self.directory).as_posix()
            row = {}
            if meta is not None:
                if rel in meta.index:
                    row = meta.loc[rel].to_dict()
                    matched_metadata.add(rel)
                elif path.name in meta.index:
                    if len(names[path.name]) > 1:
                        raise ValueError(f"ambiguous metadata basename {path.name!r}; use relative paths for each file")
                    row = meta.loc[path.name].to_dict()
                    matched_metadata.add(path.name)
                else:
                    continue  # files not listed in the metadata are ignored
            year_match = _YEAR.search(path.stem)
            default_id = _YEAR.sub("", path.stem).strip("_- .") or path.stem
            fy = row.get("fiscal_year") or (year_match.group(1) if year_match else "")
            declared_id = row.get("firm_id", "")
            firm_id = declared_id if str(declared_id).strip() else default_id
            inferred_id_locations.setdefault(firm_id, []).append((path.parent, not bool(str(declared_id).strip())))
            form = row.get("form") or _form_from_name(path.name)
            scheme = row.get("industry_scheme") or None
            code = row.get("industry_code") or None
            extra = {"path": str(path)}
            if scheme == "SIC":
                extra["nace_section"] = sic_to_nace_section(code)
            elif scheme == "NACE" and code:
                extra["nace_section"] = code[:1].upper()
            filings.append(
                Filing(
                    source=self.name,
                    filing_id=f"local:{rel}",
                    firm_id=firm_id,
                    firm_name=row.get("firm_name") or default_id,
                    fiscal_year=fy or None,
                    period_end=row.get("period_end") or None,
                    country=(row.get("country") or "").upper() or None,
                    industry_code=code,
                    industry_scheme=scheme,
                    form=form or None,
                    url=path.as_uri(),
                    language=row.get("language") or None,
                    extra=extra,
                )
            )
        if meta is not None:
            unmatched = sorted(set(meta.index) - matched_metadata)
            if unmatched:
                examples = ", ".join(repr(x) for x in unmatched[:5])
                raise ValueError(f"metadata references {len(unmatched)} unmatched file(s): {examples}")
        for firm_id, locations in inferred_id_locations.items():
            if len({parent for parent, _ in locations}) > 1 and any(inferred for _, inferred in locations):
                raise ValueError(
                    f"inferred firm_id {firm_id!r} occurs in multiple directories; provide explicit firm_id metadata"
                )
        return filings

    def input_content_path(self, filing: Filing) -> Path:
        """Resolve the same file path used for reading and checkpoint hashing."""
        return Path(filing.extra.get("path") or _path_from_uri(filing.url or ""))

    def can_verify_input_content(self) -> bool:
        """Do not bypass a subclass's custom fetch_document implementation."""
        return type(self).fetch_document is LocalSource.fetch_document

    def fetch_document_verified(self, filing: Filing, expected_sha256: str | None) -> Document:
        """Check the exact bytes analysed against the run-start manifest."""
        path = self.input_content_path(filing)
        if expected_sha256 is None:
            raise FileNotFoundError(f"local input was unavailable when the run manifest was created: {path}")
        data = path.read_bytes()
        if not hmac.compare_digest(hashlib.sha256(data).hexdigest(), expected_sha256):
            raise ValueError(f"local input changed after the run manifest was created: {path}")
        return self._document_from_bytes(filing, path, data)

    def _document_from_bytes(self, filing: Filing, path: Path, data: bytes) -> Document:
        doc = self.make_document(filing, data, filename=path.name)
        # Local metadata is an explicit declaration, unlike filename hints
        # supplied by some remote adapters. Retain scores for inspection.
        if filing.language and str(filing.language).strip():
            doc.language = str(filing.language).strip().lower()
        return doc

    def fetch_document(self, filing: Filing) -> Document:
        path = self.input_content_path(filing)
        return self._document_from_bytes(filing, path, path.read_bytes())
