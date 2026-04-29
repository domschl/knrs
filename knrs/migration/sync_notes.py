import logging
import os
import shutil
import unicodedata
import re
from pathlib import Path
import yaml

from knrs.calibre.library import scan_existing_markdowns

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("sync_notes")

class OldStuff:
    """Helper to map old-style book links to new filenames using UUIDs."""
    
    def __init__(self, src_folder: Path, dst_folder: Path):
        self.src_folder = src_folder
        self.dst_folder = dst_folder
        self.old_title_to_uuid: dict[str, str] = {}
        self.uuid_to_new_title: dict[str, str] = {}
        self.converted_links_count = 0

        # 1. Scan old books in source to map old-sanitized-filename -> UUID
        books_dir = self.src_folder / "Books"
        if books_dir.exists():
            logger.info("Scanning old books in %s...", books_dir)
            for root, _dirs, files in os.walk(books_dir):
                for file in files:
                    if file.endswith(".md"):
                        src_file = Path(root) / file
                        try:
                            # Quick frontmatter parse
                            text = src_file.read_text(encoding="utf-8")
                            if text.startswith("---"):
                                end = text.find("\n---\n", 4)
                                if end != -1:
                                    meta = yaml.safe_load(text[4:end])
                                    if isinstance(meta, dict) and meta.get("uuid"):
                                        uuid = meta["uuid"]
                                        title_sort = meta.get("title_sort")
                                        if title_sort:
                                            cleaned = self._clean_filename(str(title_sort))
                                            self.old_title_to_uuid[cleaned] = uuid
                                        
                                        # Also map the raw filename without .md
                                        base_name = file[:-3]
                                        self.old_title_to_uuid[base_name] = uuid
                        except Exception as e:
                            logger.error("Failed to parse old book frontmatter in %s: %s", src_file, e)

        # 2. Scan new books in configured AINotes directory to map UUID -> new-filename-stem
        try:
            from knrs.config import load_config
            cfg = load_config()
            new_books_dir = cfg.ai_notes_books
            if new_books_dir.exists():
                logger.info("Scanning new books in %s...", new_books_dir)
                new_index = scan_existing_markdowns(new_books_dir)
                for uuid, info in new_index.items():
                    # We want the filename without .md for the wiki link
                    self.uuid_to_new_title[uuid] = Path(info['filename']).stem
        except Exception as e:
            logger.error("Failed to scan new books: %s", e)

        logger.info("Mapped %d old titles and %d new UUIDs.", 
                    len(self.old_title_to_uuid), len(self.uuid_to_new_title))

    def _clean_filename(self, s: str) -> str:
        """Old-style filename sanitizer used in previous project."""
        if not s:
            return ""
        bad_chars = ["<", ">", ":", '"', "/", "\\", "|", "?", "*"]
        for c in bad_chars:
            s = s.replace(c, ",")
        s = s.replace("__", "_")
        s = s.replace(" _ ", ", ")
        s = s.replace("_ ", ", ")
        # Collapse multiple spaces (emulating the old project's repeated calls)
        for _ in range(3):
            s = s.replace("  ", " ")
        s = s.replace(",,", ",")
        s = s.replace(" ,", " ")
        for _ in range(2):
            s = s.replace("  ", " ")
        s = s.strip()
        s = unicodedata.normalize("NFC", s)
        return s
    
    def convert_links(self, src: Path, dst: Path) -> None:
        """Read src note, replace book links, and write to dst."""
        try:
            full_text = src.read_text(encoding="utf-8")
        except OSError as e:
            logger.error("Failed to read note %s: %s", src, e)
            return
        except Exception as e:
            logger.error("Failed to read note %s: %s", src, e)
            exit(-1)

        def replace_link(match):
            link_inner = match.group(1)
            parts = link_inner.split("|", 1)
            link_target = parts[0]
            display_text = parts[1] if len(parts) > 1 else None

            # Check if this link corresponds to an old book
            uuid = self.old_title_to_uuid.get(link_target)
            if uuid:
                new_title = self.uuid_to_new_title.get(uuid)
                if new_title:
                    self.converted_links_count += 1
                    print(f"Found link: {link_target} -> {new_title}")
                    if display_text:
                        return f"[[{new_title}|{display_text}]]"
                    else:
                        return f"[[{new_title}]]"
            return match.group(0)

        # Regex for standard wiki links: [[Link Name]]
        new_text = re.sub(r'\[\[(.*?)\]\]', replace_link, full_text)
        
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(new_text, encoding="utf-8")
        except OSError as e:
            logger.error("Failed to write converted note to %s: %s", dst, e)

def sync_notes(src: str, dst: str, exclude_dirs: list[str] | None = None, dry_run: bool = True) -> bool:
    """Synchronize notes from src to dst, converting book links along the way."""
    src_path = Path(src)
    dst_path = Path(dst)
    
    # Initialize link converter
    old_stuff = OldStuff(src_path, dst_path)
    
    dst_cur: list[str] = []
    cnt_new: int = 0
    cnt_exist: int = 0
    cnt_debris: int = 0
    cnt_src_garbage: int = 0
    cnt_all: int = 0

    # Build index of existing files in destination
    for root, _dirs, files in os.walk(dst):
        path_parts = Path(root).parts
        if any(p.startswith('.') for p in path_parts):
            continue
        for file in files:
            dst_cur.append(os.path.join(root, file))

    # Scan source and sync to destination
    for root, _dirs, files in os.walk(src):
        path_parts = Path(root).parts
        if any(p.startswith('.') for p in path_parts):
            continue
            
        rel_path = os.path.relpath(root, src)
        if rel_path == ".":
            rel_path = ""
            
        # Check exclusion
        if exclude_dirs:
            if any(ex in rel_path for ex in exclude_dirs):
                continue

        for file in files:
            cnt_all += 1
            src_file = Path(root) / file
            if file.endswith("~") or file.startswith("."):
                cnt_src_garbage += 1
                continue

            dst_file = dst_path / rel_path / file
            
            if dst_file.exists():
                # Remove from debris list
                try:
                    dst_cur.remove(str(dst_file))
                except ValueError:
                    pass
                
                # Update links in existing markdown files
                if not dry_run and src_file.suffix.lower() == ".md":
                    old_stuff.convert_links(src_file, dst_file)
                    
                cnt_exist += 1
                continue

            if dry_run:
                print(f"[dry-run] NEW: {src_file} -> {dst_file}")
            else:
                # print(f"NEW: {src_file} -> {dst_file}")
                if src_file.suffix.lower() == ".md":
                    old_stuff.convert_links(src_file, dst_file)
                else:
                    dst_file.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src_file, dst_file)
            cnt_new += 1

    # Cleanup debris
    for dst_file in dst_cur:
        if dry_run:
            print(f"[dry-run] DEBRIS: {dst_file}")
        else:
            print(f"DEBRIS: {dst_file}")
            # os.remove(dst_file) # Disabled for safety until explicit requirement
        cnt_debris += 1

    print(f"\nSummary:")
    print(f"Existing notes:  {cnt_exist}")
    print(f"New notes:       {cnt_new}")
    print(f"Debris notes:    {cnt_debris}")
    print(f"Source garbage:  {cnt_src_garbage}")
    print(f"All source notes: {cnt_all}")
    print(f"Total links converted: {old_stuff.converted_links_count}")
    return True

def main():
    import sys
    src = os.path.expanduser("~/Notes")
    dst = os.path.expanduser("~/Wiki/Notes")
    
    dry_run = "--execute" not in sys.argv
    if dry_run:
        print("Dry run active. Use --execute to actually perform the sync and conversion.\n")
        
    sync_notes(src, dst, exclude_dirs=["Books"], dry_run=dry_run)
    
if __name__ == "__main__":
    main()
