from __future__ import annotations

import json
import logging
import os
import re
import shutil
from pathlib import Path
from typing import Any

from config import KnrsConfig
from agent.engine import AgentSession
from agent.tools import CACHE_DIR_NAMES

logger = logging.getLogger("wiki_organizer")

# Files that live under the Cache/ subtree are managed by agent tools
# and must never be re-arranged by the organizer.
_CACHE_SUBDIR = "Cache"

def gather_research_files(research_root: Path) -> list[dict[str, Any]]:
    """Scan AINotes/Research/ and collect files metadata and body snippets."""
    files_data: list[dict[str, Any]] = []
    
    # We walk recursively through the research directory
    for md_path in research_root.rglob("*.md"):
        if md_path.name.startswith(".") or ".sessions" in md_path.parts:
            continue
        # Skip files inside the Cache/ subtree
        if _CACHE_SUBDIR in md_path.relative_to(research_root).parts:
            continue
        try:
            content = md_path.read_text(encoding="utf-8")
        except Exception as e:
            logger.warning("Could not read %s: %s", md_path, e)
            continue
            
        from calibre.converter import _split_frontmatter
        import yaml
        
        fm_raw, body = _split_frontmatter(content)
        try:
            meta = yaml.safe_load(fm_raw) if fm_raw else {}
        except Exception:
            meta = {}
            
        if not isinstance(meta, dict):
            meta = {}
            
        title = meta.get("title") or md_path.stem
        tags = meta.get("tags") or []
        
        # Grab first 200 characters of the body as a semantic snippet
        # (shorter = smaller prompt = more room for the large JSON output)
        snippet = body[:200].strip()
        
        rel_path = md_path.relative_to(research_root)
        files_data.append({
            "path": str(rel_path).replace("\\", "/"),
            "title": title,
            "tags": tags,
            "snippet": snippet
        })
        
    return files_data

def classify_research_files(config: KnrsConfig, files_data: list[dict[str, Any]]) -> dict[str, str]:
    """Call the active LLM to get proposed relative paths for all files."""
    if not files_data:
        return {}

    prompt = f"""\
You are a library cataloger. Below is a list of research documents in our wiki, including their current relative paths, titles, tags, and small text snippets.

Your goal is to organize these files into a logical, hierarchical directory structure (taxonomy) within the research folder. 
Establish broad categories like "Philosophy", "History", "Mathematics", "Science", etc., and subcategories where appropriate (e.g., "Philosophy/Philosophy of Science", "History/Tudor England").
Do not create too many nested levels (maximum 2-3 folders deep).

Files to organize:
{json.dumps(files_data, indent=2)}

Please output a JSON object mapping each current path to its proposed new relative path under the research root.

CRITICAL REQUIREMENTS:
1. You MUST include every single file path from the input list as a key in the output JSON. Do NOT omit any files.
2. Do NOT use "..." or placeholder entries in the JSON. The output MUST be a single, complete, fully-expanded, valid JSON object mapping all input files.
3. Respond ONLY with the raw JSON object inside a ```json ... ``` code block. Do not include any preambles, follow-up explanations, or conversational text outside the code block.

Example output format:
```json
{{
  "Roman Law.md": "History/Roman Law.md",
  "Mathematics/Integral_1_over_2_plus_tanh_x.md": "Mathematics/Calculus/Integral_1_over_2_plus_tanh_x.md"
}}
```
"""

    messages = [
        {
            "role": "system", 
            "content": "You are a precise database organizer. You MUST output a complete, valid JSON object mapping EVERY input file to its new path. Do NOT write explanation. Do NOT use '...' or placeholders under any circumstances. Every single file must be fully mapped."
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    logger.info("Querying LLM for categorization (%d files)...", len(files_data))
    with AgentSession(config) as session:
        # Use a generous token budget so the full mapping JSON is never truncated.
        response = session.generate(messages, max_tokens=32000)

    res_text = response.strip()
    
    # Strip <think>...</think> block if present
    res_text = re.sub(r"<think>.*?</think>", "", res_text, flags=re.DOTALL).strip()
    if "</think>" in res_text:
        parts = res_text.split("</think>")
        res_text = parts[-1].strip()

    # Clean up surrounding text to focus on JSON code blocks or curly braces
    candidate = res_text
    
    # Try markdown json block
    if "```" in candidate:
        blocks = re.findall(r"```(?:json)?\s*(.*?)\s*```", candidate, re.DOTALL)
        for b in blocks:
            b_clean = b.strip()
            if b_clean.startswith("{") and b_clean.endswith("}"):
                candidate = b_clean
                break
        else:
            if blocks:
                candidate = blocks[-1].strip()
            
    # Look for first '{' and last '}'
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start != -1 and end == -1:
        # Opening brace found but no closing brace — the response was cut off.
        logger.error(
            "LLM response was truncated (no closing '}' found). "
            "Consider reducing the number of files or increasing max_tokens. "
            "Partial response:\n%s", response[-500:]
        )
        raise RuntimeError(
            "LLM response was cut off before the JSON was complete. "
            "Try again or reduce the number of files being organised."
        )
    if start != -1 and end >= start:
        candidate = candidate[start:end+1]
        
    # Check if the output has ellipses/placeholders
    if "..." in candidate or "…" in candidate:
        logger.error("LLM response was abbreviated or incomplete. Full raw response was:\n%s", response)
        raise RuntimeError(
            "LLM response was abbreviated or incomplete (contained '...'). Please try again."
        )

    # Load JSON
    try:
        mapping = json.loads(candidate)
    except json.JSONDecodeError as e:
        # Try a quick regex cleanup of common trailing comma errors in json
        cleaned = re.sub(r',\s*([\]}])', r'\1', candidate)
        try:
            mapping = json.loads(cleaned)
        except json.JSONDecodeError as e_inner:
            logger.error("Failed to parse LLM response as JSON: %s. Response was: %s", e_inner, response)
            raise RuntimeError(f"LLM response was not a parseable JSON object mapping: {e_inner}")
            
    if not isinstance(mapping, dict):
        raise RuntimeError("LLM response JSON root was not an object/dictionary.")
        
    return {str(k): str(v) for k, v in mapping.items()}

def move_associated_assets(src_file: Path, dst_file: Path, dry_run: bool = False) -> list[str]:
    """Identify and relocate any relative assets (Images/ or Resources/) referenced by the file."""
    moved_assets = []
    try:
        content = src_file.read_text(encoding="utf-8")
    except Exception:
        return moved_assets

    # Extract relative links like Images/something.png or Resources/data.csv
    refs = re.findall(r"\]\(((?:Images|Resources)/[^)]+)\)", content)
    refs += re.findall(r"src=\"((?:Images|Resources)/[^\"]+)\"", content)

    for ref in sorted(set(refs)):
        ref_path = Path(ref)
        asset_src = src_file.parent / ref_path
        if asset_src.exists() and asset_src.is_file():
            asset_dst = dst_file.parent / ref_path
            moved_assets.append(f"  Asset: {ref} -> {asset_dst}")
            if not dry_run:
                asset_dst.parent.mkdir(parents=True, exist_ok=True)
                # Overwrite destination asset if it already exists
                if asset_dst.exists():
                    asset_dst.unlink()
                shutil.move(str(asset_src), str(asset_dst))
                
    return moved_assets

def cleanup_empty_dirs(root: Path) -> None:
    """Recursively delete empty folders under root (ignoring hidden files like .DS_Store)."""
    for dirpath, dirnames, filenames in os.walk(root, topdown=False):
        p = Path(dirpath)
        if p == root:
            continue
        try:
            # Items ignoring hidden files
            items = [item for item in p.iterdir() if not item.name.startswith(".")]
            if not items:
                # Clean up any leftover hidden files (like .DS_Store)
                for hidden in p.iterdir():
                    hidden.unlink()
                p.rmdir()
                logger.info("Cleaned up empty directory: %s", p)
        except Exception as e:
            logger.debug("Failed cleaning empty directory %s: %s", p, e)

def organize_research_directory(config: KnrsConfig, dry_run: bool = False) -> list[str]:
    """Orchestrate the LLM-guided directory reorganization."""
    research_root = config.wiki_path / "AINotes" / "Research"
    if not research_root.exists():
        return ["Research directory does not exist."]
        
    files = gather_research_files(research_root)
    if not files:
        return ["No research documents found to organize."]
        
    try:
        proposed_mapping = classify_research_files(config, files)
    except Exception as e:
        return [f"LLM Classification failed: {e}"]
        
    log_lines: list[str] = []
    
    # Track destination paths to prevent conflicts / duplicate assignments
    dest_tracker: set[str] = set()
    
    for f in files:
        current_rel = f["path"]
        proposed_rel = proposed_mapping.get(current_rel)
        if not proposed_rel:
            continue

        # Never touch files already inside Cache/
        if _CACHE_SUBDIR in Path(current_rel).parts:
            continue
            
        # Enforce that the folder structure changes, but the filename stem remains the same
        proposed_rel = proposed_rel.replace("\\", "/").strip().strip("/")
        proposed_parent = Path(proposed_rel).parent
        filename = Path(current_rel).name
        
        # Clean proposed path
        clean_rel = str(proposed_parent / filename).replace("\\", "/").strip().strip("/")
        if clean_rel == "." or not clean_rel:
            clean_rel = filename
            
        if clean_rel == current_rel:
            # File is already in the correct proposed folder
            continue
            
        src_path = research_root / current_rel
        dst_path = research_root / clean_rel
        
        if dst_path.exists():
            log_lines.append(f"SKIP: Destination already exists for {current_rel} -> {clean_rel}")
            continue
            
        if clean_rel in dest_tracker:
            log_lines.append(f"SKIP: Collision in proposed mapping for {current_rel} -> {clean_rel}")
            continue
            
        dest_tracker.add(clean_rel)
        
        log_lines.append(f"MOVE: {current_rel} -> {clean_rel}")
        
        # Scan and list asset moves
        assets = move_associated_assets(src_path, dst_path, dry_run=True)
        log_lines.extend(assets)
        
        if not dry_run:
            try:
                from wiki.checker import ensure_minimal_frontmatter
                from calibre.converter import _split_frontmatter, atomic_write
                import yaml
                
                content = src_path.read_text(encoding="utf-8")
                fm_raw, body = _split_frontmatter(content)
                try:
                    meta = yaml.safe_load(fm_raw) if fm_raw else {}
                except Exception:
                    meta = {}
                if not isinstance(meta, dict):
                    meta = {}
                    
                # Setup correct parent directory for context update
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Update context metadata in frontmatter
                ensure_minimal_frontmatter(dst_path, config.wiki_path, meta)
                
                new_fm = yaml.dump(meta, default_flow_style=False, allow_unicode=True, indent=2)
                new_content = f"---\n{new_fm}---\n{body}"
                
                # Write to new path
                atomic_write(dst_path, new_content)
                
                # Move assets for real
                move_associated_assets(src_path, dst_path, dry_run=False)
                
                # Delete source
                src_path.unlink()
                
            except Exception as e:
                logger.error("Failed moving %s: %s", current_rel, e)
                log_lines.append(f"ERROR moving {current_rel}: {e}")
                
    if not dry_run:
        cleanup_empty_dirs(research_root)
        
    return log_lines

def main() -> None:
    import argparse
    from config import load_config
    
    parser = argparse.ArgumentParser(description="Organize research directory hierarchically.")
    parser.add_argument("--dry-run", action="store_true", help="Simulate proposed changes without writing them")
    args = parser.parse_args()
    
    cfg = load_config()
    dry_run = args.dry_run
    
    print("Starting LLM-Guided Research Organizer...")
    if dry_run:
        print("DRY-RUN MODE (No changes will be written)\n")
    else:
        print("APPLYING CHANGES...\n")
        
    log_lines = organize_research_directory(cfg, dry_run=dry_run)
    for line in log_lines:
        print(line)
        
    print("\nOrganizer Finished.")

if __name__ == "__main__":
    main()
