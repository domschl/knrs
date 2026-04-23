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
    date_str: str
    event: str
    description: str
    source_file: str
    start_year: float
    end_year: float

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
        if line.startswith('|') and 'Date' in line and 'Event' in line:
            # Found a potential header
            header = [c.strip() for c in line.split('|') if c.strip()]
            try:
                date_idx = header.index('Date')
                event_idx = header.index('Event')
                desc_idx = header.index('Description') if 'Description' in header else -1
            except ValueError:
                i += 1
                continue
            
            # Skip separator line if it exists
            i += 1
            if i < len(lines) and re.match(r'^|[ \-:]+|', lines[i].strip()):
                i += 1
            
            # Process data rows
            while i < len(lines) and lines[i].strip().startswith('|'):
                row = [c.strip() for c in lines[i].split('|')]
                # Split creates empty strings at start/end if line is | a | b |
                if row[0] == '': row = row[1:]
                if row and row[-1] == '': row = row[:-1]
                
                if len(row) > max(date_idx, event_idx):
                    date_val = row[date_idx]
                    event_val = row[event_idx]
                    desc_val = row[desc_idx] if desc_idx != -1 and len(row) > desc_idx else ""
                    
                    if date_val and event_val and date_val != 'Date':
                        try:
                            start, end = parse_interval(date_val)
                            events.append(TimelineEvent(
                                date_str=date_val,
                                event=event_val,
                                description=desc_val,
                                source_file=rel_path,
                                start_year=start,
                                end_year=end
                            ))
                        except ValueError as e:
                            logger.warning("Skipping invalid date '%s' in %s: %s", date_val, path.name, e)
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
        json.dump([asdict(e) for e in all_events], f, indent=2)
