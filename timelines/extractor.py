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
from typing import Any

from timelines.indra_time import parse_interval, format_point

logger = logging.getLogger(__name__)

@dataclass
class TimelineEvent:
    start_year: float
    end_year: float
    source_file: str
    context: str
    data: dict[str, str]

    def to_dict(self) -> dict:
        """Return a flattened dictionary for JSON serialization."""
        d = {
            "start_year": self.start_year,
            "end_year": self.end_year,
            "source_file": self.source_file,
            "context": self.context,
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
    
    # Extract context from frontmatter
    context = ""
    lines = content.splitlines()
    if lines and lines[0].strip() == "---":
        for j in range(1, min(len(lines), 20)):
            if lines[j].strip() == "---":
                break
            if ":" in lines[j]:
                k, v = lines[j].split(":", 1)
                if k.strip() == "context":
                    context = v.strip().strip("'\"")
                    break
    
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
            last_start = None
            prev_date_val = None
            while i < len(lines) and lines[i].strip().startswith('|'):
                row = _parse_row(lines[i])
                # A valid row must have the same column count as the header
                # and the first column must not be 'Date' (to skip potential repeated headers)
                if len(row) == len(header) and row[0] != 'Date':
                    date_val = row[0]
                    if date_val:
                        try:
                            start, end = parse_interval(date_val)
                            
                            # Sanity Check 2: Interval [a, b], b < a
                            if end < start:
                                logger.warning(
                                    "Inverted interval '%s' in %s", 
                                    date_val, path.name
                                )

                            # Sanity Check 1: Rows not in ascending order
                            if last_start is not None and start < last_start:
                                logger.warning(
                                    "Timeline row out of order: '%s' follows '%s' in %s",
                                    date_val, prev_date_val, path.name
                                )
                            
                            last_start = start
                            prev_date_val = date_val

                            # Map columns to their header names
                            event_data = {header[j]: row[j] for j in range(len(header))}
                            
                            events.append(TimelineEvent(
                                start_year=start,
                                end_year=end,
                                source_file=rel_path,
                                context=context,
                                data=event_data
                            ))
                        except ValueError as e:
                            logger.error("Invalid date format '%s' in %s: %s", date_val, path.name, e)
                i += 1
        else:
            i += 1
            
    return events

def run_extraction(notes_path: Path, output_file: Path) -> dict[str, Any]:
    """Scan notes_path for timelines and save to output_file."""
    all_events = []
    logger.info("Scanning %s for timelines...", notes_path)
    files_scanned = 0
    errors: list[str] = []
    
    for md_path in notes_path.rglob("*.md"):
        files_scanned += 1
        try:
            events = extract_from_file(md_path, notes_path)
            all_events.extend(events)
        except Exception as e:
            logger.error("Error extracting from %s: %s", md_path, e)
            errors.append(f"{md_path.name}: {e}")
        
    if not all_events:
        logger.info("No timeline events found.")
        # Ensure output file exists but empty list
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with output_file.open('w', encoding='utf-8') as f:
            json.dump([], f)
        return {
            "files_scanned": files_scanned,
            "events_extracted": 0,
            "failure_count": len(errors),
            "failed_items": errors,
        }

    # Sort by start_year, then end_year
    all_events.sort(key=lambda x: (x.start_year, x.end_year))
    
    logger.info("Extracted %d events. Saving to %s", len(all_events), output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open('w', encoding='utf-8') as f:
        json.dump([e.to_dict() for e in all_events], f, indent=2)

    return {
        "files_scanned": files_scanned,
        "events_extracted": len(all_events),
        "failure_count": len(errors),
        "failed_items": errors,
    }

def show_timeline(
    output_file: Path,
    start_year: float | None = None,
    end_year: float | None = None,
    context_filters: list[str] | None = None,
    keywords: list[str] | None = None,
    raw: bool = False
) -> None:
    """Load, filter, and display timeline events."""
    from utils.search import SearchTools
    from rich.table import Table
    from rich.console import Console
    
    console = Console()
    
    results = query_timeline_data(output_file, start_year, end_year, context_filters, keywords)
    
    if not results:
        console.print("[yellow]No timeline events matched the filter.[/yellow]")
        return
        
    _render_events(results, raw, console)

def query_timeline_data(
    output_file: Path,
    start_year: float | None = None,
    end_year: float | None = None,
    context_filters: list[str] | None = None,
    keywords: list[str] | None = None
) -> list[dict]:
    """Load and filter timeline events, returning the raw data list."""
    from utils.search import SearchTools
    if not output_file.exists():
        return []
        
    with output_file.open("r", encoding="utf-8") as f:
        events = json.load(f)
        
    filtered = []
    for ev in events:
        # Context filter
        if context_filters:
            if not SearchTools.match(ev.get("context", ""), context_filters, any_match=True):
                continue
                
        # Keyword filter
        if keywords:
            data_text = " ".join(str(v) for v in ev.values())
            if not SearchTools.match(data_text, keywords, any_match=True):
                continue
        
        # Date filter logic
        ev_start = ev["start_year"]
        ev_end = ev["end_year"]
        
        # Base overlap check
        if start_year is not None and ev_end < start_year:
            continue
        if end_year is not None and ev_start > end_year:
            continue
            
        filtered.append(ev)
        
    # Categorize and filter strictly within range
    results = []
    if start_year is not None and end_year is not None and start_year < end_year:
        for ev in filtered:
            if ev["start_year"] >= start_year and ev["end_year"] <= end_year:
                results.append(ev)
    else:
        results = filtered
        
    return results

def format_timeline_as_markdown_table(events: list[dict]) -> str:
    """Format a list of timeline events as a Markdown table."""
    if not events:
        return ""
        
    exclude = {"start_year", "end_year", "source_file", "context", "Date"}
    
    headers = ["Date", "Description", "Context"]
    lines = []
    lines.append(f"| {' | '.join(headers)} |")
    lines.append(f"| {' | '.join(['---']*len(headers))} |")
    for ev in events:
        date_str = ev.get("Date", f"{ev['start_year']}")
        ctx_str = ev.get("context", "")
        # Collect all other fields as description
        desc = " | ".join(str(v) for k, v in ev.items() if k not in exclude)
        lines.append(f"| {date_str} | {desc} | {ctx_str} |")
        
    return "\n".join(lines)


def _render_events(events: list[dict], raw: bool, console) -> None:
    from rich.table import Table
    # Metadata keys to exclude from the description column
    exclude = {"start_year", "end_year", "source_file", "context", "Date"}
    
    if raw:
        headers = ["Date", "Description", "Context"]
        console.print(f"| {' | '.join(headers)} |")
        console.print(f"| {' | '.join(['---']*len(headers))} |")
        for ev in events:
            date_str = ev.get("Date", f"{ev['start_year']}")
            ctx_str = ev.get("context", "")
            # Collect all other fields as description
            desc = " | ".join(str(v) for k, v in ev.items() if k not in exclude)
            console.print(f"| {date_str} | {desc} | {ctx_str} |")
    else:
        table = Table(box=None)
        table.add_column("Date", style="cyan", width=30)
        table.add_column("Description", style="white")
        table.add_column("Context", style="magenta")
        
        for ev in events:
            date_str = ev.get("Date", f"{ev['start_year']}")
            ctx_str = ev.get("context", "")
            # Collect all other fields as description
            desc = " | ".join(str(v) for k, v in ev.items() if k not in exclude)
            table.add_row(date_str, desc, ctx_str)
            
        console.print(table)
