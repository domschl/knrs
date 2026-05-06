"""
knrs.timelines.indra_time — Parser for the IndraTime deep-time format.

Handles:
- Standard dates (AD/CE): YYYY, YYYY-MM, YYYY-MM-DD
- BC dates: Y BC, Y-M BC, Y-M-D BC
- BP (Before Present, 1950): N BP
- kya (Thousand years ago): N kya
- Ma (Million years ago): N Ma
- Ga (Billion years ago): N Ga

Internal representation is a float representing the year (Astronomical convention):
- 1 AD = 1.0
- 1 BC = 0.0
- 2 BC = -1.0
- 100 BC = -99.0
"""

from __future__ import annotations

import re

# Regex for points
_AD_RE = re.compile(r'^(\d+)(?:-(\d{1,2}))?(?:-(\d{1,2}))?(?:\s+AD)?$', re.IGNORECASE)
_BC_RE = re.compile(r'^(\d+)(?:-(\d{1,2}))?(?:-(\d{1,2}))?\s+BC$', re.IGNORECASE)
_BP_RE = re.compile(r'^([\d.]+)\s+BP$', re.IGNORECASE)
_KYA_RE = re.compile(r'^([\d.]+)\s+(?:kya|ka|kyr)(?:\s+BP)?$', re.IGNORECASE)
_MA_RE = re.compile(r'^([\d.]+)\s+(?:Ma|mya)(?:\s+BP)?$', re.IGNORECASE)
_GA_RE = re.compile(r'^([\d.]+)\s+(?:Ga|bya)(?:\s+BP)?$', re.IGNORECASE)

def parse_point(s: str) -> float:
    """Parse a time point string into a year float."""
    s = s.strip()
    
    # AD / Standard
    m = _AD_RE.match(s)
    if m:
        year = int(m.group(1))
        month = int(m.group(2)) if m.group(2) else 1
        day = int(m.group(3)) if m.group(3) else 1
        return year + (month - 1) / 12 + (day - 1) / 365
    
    # BC
    m = _BC_RE.match(s)
    if m:
        year = int(m.group(1))
        month = int(m.group(2)) if m.group(2) else 1
        day = int(m.group(3)) if m.group(3) else 1
        # 1 BC = 0.0, 2 BC = -1.0
        astronomical_year = 1.0 - year
        return astronomical_year + (month - 1) / 12 + (day - 1) / 365

    # BP (Present = 1950)
    m = _BP_RE.match(s)
    if m:
        val = float(m.group(1))
        return 1950.0 - val
    
    # kya (1000 BP)
    m = _KYA_RE.match(s)
    if m:
        val = float(m.group(1))
        return 1950.0 - (val * 1_000)
    
    # Ma (1,000,000 BP)
    m = _MA_RE.match(s)
    if m:
        val = float(m.group(1))
        return 1950.0 - (val * 1_000_000)
    
    # Ga (1,000,000,000 BP)
    m = _GA_RE.match(s)
    if m:
        val = float(m.group(1))
        return 1950.0 - (val * 1_000_000_000)

    raise ValueError(f"Unknown IndraTime format: {s}")

def parse_interval(s: str) -> tuple[float, float]:
    """Parse an interval 'Start - End'."""
    parts = s.split(" - ")
    if len(parts) != 2:
        p = parse_point(s)
        return (p, p)
    return (parse_point(parts[0]), parse_point(parts[1]))

def format_point(year: float) -> str:
    """Format a year float back to its canonical IndraTime string."""
    if year >= 1.0:
        y = int(year)
        rem = year - y
        if rem < 0.0001:
            return f"{y:04d}"
        m = int(rem * 12) + 1
        d_rem = (rem * 12) - (m - 1)
        d = int(d_rem * 30.44) + 1
        if d == 1:
            if m == 1: return f"{y:04d}"
            return f"{y:04d}-{m:02d}"
        return f"{y:04d}-{m:02d}-{d:02d}"
    
    # BC
    if year > -11050.0: # ~13000 years ago from present
        # 0.0 -> 1 BC, -1.0 -> 2 BC
        val = 1.0 - year
        return f"{int(val)} BC"
    
    age_from_1950 = 1950.0 - year
    
    if age_from_1950 <= 100_000:
        return f"{int(age_from_1950)} BP"
    
    if age_from_1950 <= 100_000_000:
        val = age_from_1950 / 1_000
        return f"{val:.2f} kya BP"
    
    if age_from_1950 <= 10_000_000_000:
        val = age_from_1950 / 1_000_000
        return f"{val:.2f} Ma BP"
    
    val = age_from_1950 / 1_000_000_000
    return f"{val:.3f} Ga BP"
