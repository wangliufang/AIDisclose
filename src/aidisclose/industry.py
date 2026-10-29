"""Approximate concordance from US SIC codes to NACE Rev. 2 sections.

Cross-country comparison requires a common industry scheme. EDGAR reports a
four-digit SIC code per registrant, while European statistics (e.g. Eurostat's
survey on AI use in enterprises) use NACE Rev. 2. This module maps SIC code
ranges to the 21 NACE sections (A-U). The mapping is deliberately coarse: it is
exact at the level of most SIC major groups but some groups straddle NACE
sections (e.g. SIC 73 business services spans J, M and N), which are split at
the four-digit level where it matters for AI-intensive industries. Users who
need finer or official concordances can pass their own mapping to
:func:`aidisclose.aggregate.firm_year_panel` via the ``industry_map`` argument.
"""

from __future__ import annotations

NACE_SECTIONS = {
    "A": "Agriculture, forestry and fishing",
    "B": "Mining and quarrying",
    "C": "Manufacturing",
    "D": "Electricity, gas, steam and air conditioning supply",
    "E": "Water supply; sewerage, waste management and remediation",
    "F": "Construction",
    "G": "Wholesale and retail trade; repair of motor vehicles",
    "H": "Transportation and storage",
    "I": "Accommodation and food service activities",
    "J": "Information and communication",
    "K": "Financial and insurance activities",
    "L": "Real estate activities",
    "M": "Professional, scientific and technical activities",
    "N": "Administrative and support service activities",
    "O": "Public administration and defence; compulsory social security",
    "P": "Education",
    "Q": "Human health and social work activities",
    "R": "Arts, entertainment and recreation",
    "S": "Other service activities",
    "T": "Activities of households as employers",
    "U": "Activities of extraterritorial organisations",
}

# (first SIC, last SIC, NACE section); more specific ranges are listed first.
_SIC_RANGES: list[tuple[int, int, str]] = [
    # Manufacturing exceptions: publishing belongs to J (58)
    (2710, 2741, "J"),
    # Transportation, communications, utilities
    (4724, 4729, "N"),  # travel agencies and arrangement
    (4800, 4899, "J"),  # communications
    (4900, 4939, "D"),  # electric and gas services
    (4940, 4959, "E"),  # water supply, sanitary services
    (4960, 4969, "D"),  # steam and air-conditioning supply
    (4970, 4999, "E"),
    (4000, 4799, "H"),
    # Trade
    (5812, 5813, "I"),  # eating and drinking places
    (5000, 5999, "G"),
    # Finance, insurance, real estate
    (6500, 6553, "L"),  # real estate
    (6798, 6798, "L"),  # real estate investment trusts
    (6000, 6799, "K"),
    # Services
    (7000, 7099, "I"),  # hotels and lodging
    (7200, 7299, "S"),  # personal services
    (7310, 7319, "M"),  # advertising
    (7320, 7339, "N"),  # credit reporting, mailing, reproduction
    (7340, 7349, "N"),  # services to buildings
    (7350, 7359, "N"),  # equipment rental and leasing
    (7360, 7369, "N"),  # personnel supply
    (7370, 7379, "J"),  # computer programming, data processing
    (7380, 7389, "N"),  # miscellaneous business services
    (7510, 7519, "N"),  # automotive rental and leasing
    (7500, 7599, "G"),  # automotive repair
    (7600, 7699, "S"),  # miscellaneous repair
    (7800, 7899, "J"),  # motion pictures
    (7900, 7999, "R"),  # amusement and recreation
    (8000, 8099, "Q"),  # health services
    (8100, 8199, "M"),  # legal services
    (8200, 8299, "P"),  # educational services
    (8300, 8399, "Q"),  # social services
    (8400, 8499, "R"),  # museums
    (8600, 8699, "S"),  # membership organisations
    (8700, 8799, "M"),  # engineering, accounting, research, management
    (8800, 8899, "T"),  # private households
    (8900, 8999, "M"),
    # Divisions
    (100, 999, "A"),
    (1000, 1499, "B"),
    (1500, 1799, "F"),
    (2000, 3999, "C"),
    (9100, 9729, "O"),
]


def sic_to_nace_section(sic: str | int | None) -> str | None:
    """Return the NACE Rev. 2 section letter for a SIC code (``None`` if unknown)."""
    if sic is None:
        return None
    try:
        code = int(str(sic).strip()[:4])
    except ValueError:
        return None
    if code < 100 and len(str(sic).strip()) <= 2:  # two-digit major group
        code *= 100
    for lo, hi, section in _SIC_RANGES:
        if lo <= code <= hi:
            return section
    return None
