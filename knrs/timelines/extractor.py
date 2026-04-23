"""
knrs.timelines.extractor — Extract timelines from Wiki/Notes Markdown tables.

Scans all .md files in Wiki/Notes for tables containing 'Date' and 'Event'
columns. Parsed events are sorted and saved to KnrsData/timelines.json.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, asdict
from pathlib import Path

from knrs.timelines.indra_time import parse_interval, format_point

logger = logging.getLogger(__name__)

@dataclass
class TimelineEvent:
    start_year: float
    end_year: float
    source_file: str
    data: dict[str, str]

    def to_dict(self) -> dict:
        """Return a flattened dictionary for JSON serialization."""
        d = {
            "start_year": self.start_year,
            "end_year": self.end_year,
            "source_file": self.source_file,
        }
        d.update(self.data)
        return d

def _parse_row(line: str) -> list[str]:
    """Parse a Markdown table row into a list of cell contents."""
    parts = [p.strip() for p in line.split('|')]
    if parts and not parts[0]:
        parts.pop(0)
    if parts and not parts[-1]:
        parts.pop(-1)
    return parts

def extract_from_file(path: Path, notes_root: Path) -> list[TimelineEvent]:
    """Extract timeline events from a single Markdown file."""
    try:
        content = path.read_text(encoding="utf-8")
    except Exception as e:
        logger.error("Failed to read %s: %s", path, e)
        return []

    events = []
    rel_path = str(path.relative_to(notes_root))
    
    # Simple table extractor: find rows starting with |
    # We look for a header row containing 'Date' and 'Event'
    lines = content.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith('|') and 'Date' in line:
            header = _parse_row(line)
            if not header or header[0] != 'Date':
                i += 1
                continue
            
            # Skip separator line if it exists (e.g. |---|---|)
            i += 1
            if i < len(lines) and (lines[i].strip().startswith('|') and '-' in lines[i]):
                i += 1
            
            # Process data rows
            while i < len(lines) and lines[i].strip().startswith('|'):
                row = _parse_row(lines[i])
                # A valid row must have the same column count as the header
                # and the first column must not be 'Date' (to skip potential repeated headers)
                if len(row) == len(header) and row[0] != 'Date':
                    date_val = row[0]
                    if date_val:
                        try:
                            start, end = parse_interval(date_val)
                            # Map columns to their header names
                            event_data = {header[j]: row[j] for j in range(len(header))}
                            
                            events.append(TimelineEvent(
                                start_year=start,
                                end_year=end,
                                source_file=rel_path,
                                data=event_data
                            ))
                        except ValueError as e:
                            logger.error("Invalid date format '%s' in %s: %s", date_val, path.name, e)
                i += 1
        else:
            i += 1
            
    return events

def run_extraction(notes_path: Path, output_file: Path) -> None:
    """Scan notes_path for timelines and save to output_file."""
    all_events = []
    logger.info("Scanning %s for timelines...", notes_path)
    
    for md_path in notes_path.rglob("*.md"):
        events = extract_from_file(md_path, notes_path)
        all_events.extend(events)
        
    if not all_events:
        logger.info("No timeline events found.")
        # Ensure output file exists but empty list
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with output_file.open('w', encoding='utf-8') as f:
            json.dump([], f)
        return

    # Sort by start_year, then end_year
    all_events.sort(key=lambda x: (x.start_year, x.end_year))
    
    logger.info("Extracted %d events. Saving to %s", len(all_events), output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open('w', encoding='utf-8') as f:
        json.dump([e.to_dict() for e in all_events], f, indent=2)
