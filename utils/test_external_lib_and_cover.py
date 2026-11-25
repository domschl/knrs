from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from PIL import Image

from calibre.cover import generate_cover_icon, icon_is_current
from calibre.library import CalibreBook
from calibre.sync import plan_sync, _execute_action
from config import KnrsConfig
from external_lib.sync import (
    CATALOG_FILENAME,
    filename_md5,
    partial_koreader_md5,
    run_external_sync,
)


def _make_image(path: Path, width: int, height: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (width, height), color=(40, 80, 120))
    img.save(path, format="JPEG")


class TestCoverAndExternalCatalog(unittest.TestCase):
    def test_cover_resize_512_no_upscale_and_current_check(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            icons_dir = root / "BookCoverIcons"

            # Large cover -> resized to 512px height
            big_src = root / "big.jpg"
            _make_image(big_src, 600, 1000)
            uuid_big = "11111111-1111-1111-1111-111111111111"

            # Simulate legacy 256px icon
            icons_dir.mkdir(parents=True, exist_ok=True)
            _make_image(icons_dir / f"{uuid_big}.jpg", 154, 256)
            self.assertFalse(icon_is_current(icons_dir, uuid_big, big_src))

            out_big = generate_cover_icon(big_src, icons_dir, uuid_big)
            self.assertIsNotNone(out_big)
            with Image.open(icons_dir / f"{uuid_big}.jpg") as im:
                self.assertEqual(im.height, 512)
                self.assertEqual(im.width, 307)
            self.assertTrue(icon_is_current(icons_dir, uuid_big, big_src))

            # Small cover -> never upscaled
            small_src = root / "small.jpg"
            _make_image(small_src, 140, 200)
            uuid_small = "22222222-2222-2222-2222-222222222222"
            generate_cover_icon(small_src, icons_dir, uuid_small)
            with Image.open(icons_dir / f"{uuid_small}.jpg") as im:
                self.assertEqual((im.width, im.height), (140, 200))
            self.assertTrue(icon_is_current(icons_dir, uuid_small, small_src))

    def test_sync_regenerates_stale_icon_without_rewriting_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            md_root = root / "MarkdownBooks"
            icon_root = root / "BookCoverIcons"
            book_dir = root / "Calibre" / "Author" / "Book"
            book_dir.mkdir(parents=True)
            md_root.joinpath("Science Fiction").mkdir(parents=True)
            icon_root.mkdir(parents=True)

            epub_path = book_dir / "Book.epub"
            epub_path.write_bytes(b"EPUB" * 50)
            cover_path = book_dir / "cover.jpg"
            _make_image(cover_path, 600, 900)
            opf_path = book_dir / "metadata.opf"
            opf_path.write_text("<package/>", encoding="utf-8")

            uuid = "33333333-3333-3333-3333-333333333333"
            book = CalibreBook(
                uuid=uuid,
                calibre_id="1",
                title="Dune",
                title_sort="Dune",
                authors=["Frank Herbert"],
                series="Science Fiction",
                tags=["scifi"],
                languages=["eng"],
                identifiers=["calibre_id/1"],
                publisher="Chilton",
                publication_date="1965-01-01T00:00:00+00:00",
                creation_date="2026-01-01T00:00:00+00:00",
                description="Desert planet",
                book_dir=book_dir,
                opf_path=opf_path,
                cover_path=cover_path,
                source_file=epub_path,
                source_format="epub",
                source_hash="hash123",
            )

            md_file = md_root / "Science Fiction" / book.expected_filename
            original_md = "---\nuuid: 33333333-3333-3333-3333-333333333333\ntitle: Dune\n---\nBody\n"
            md_file.write_text(original_md, encoding="utf-8")

            # Legacy 256px icon
            _make_image(icon_root / f"{uuid}.jpg", 170, 256)

            markdown_index = {
                uuid: {
                    "path": md_file,
                    "filename": book.expected_filename,
                    "series": "Science Fiction",
                    "title": book.title,
                    "authors": book.authors,
                    "source_hash": "hash123",
                    "metadata": {
                        "uuid": uuid,
                        "title": book.title,
                        "title_sort": book.title_sort,
                        "authors": book.authors,
                        "series": book.series,
                        "tags": book.tags,
                        "publisher": book.publisher,
                        "publication_date": book.publication_date,
                        "description": book.description,
                        "context": "AINotes/Books/Science Fiction",
                    },
                }
            }

            actions = plan_sync({uuid: book}, markdown_index, md_root, icon_root)
            self.assertEqual(len(actions), 1)
            self.assertEqual(actions[0].action, "UPDATE_METADATA")
            self.assertFalse(actions[0].needs_frontmatter_update)

            cfg = MagicMock(spec=KnrsConfig)
            cfg.book_cover_icons = icon_root
            ok = _execute_action(actions[0], cfg, dry_run=False, idx=1, total=1)
            self.assertTrue(ok)
            # Markdown untouched, icon upgraded to 512px
            self.assertEqual(md_file.read_text(encoding="utf-8"), original_md)
            with Image.open(icon_root / f"{uuid}.jpg") as im:
                self.assertEqual(im.height, 512)

    def test_external_sync_writes_and_preserves_catalog(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            calibre_dir = root / "Calibre"
            ext_dir = root / "MetaLibrary"
            ext_dir.mkdir()

            book_dir = calibre_dir / "Philip K. Dick" / "The Man in the High Castle (1)"
            book_dir.mkdir(parents=True)
            epub_path = book_dir / "castle.epub"
            epub_path.write_bytes(b"Sample EPUB content for testing" * 200)

            uuid = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
            opf = book_dir / "metadata.opf"
            opf.write_text(
                f"""<?xml version='1.0' encoding='utf-8'?>
<package xmlns="http://www.idpf.org/2007/opf" unique-identifier="uuid_id" version="2.0">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:opf="http://www.idpf.org/2007/opf">
    <dc:identifier opf:scheme="calibre" id="calibre_id">42</dc:identifier>
    <dc:identifier opf:scheme="uuid" id="uuid_id">{uuid}</dc:identifier>
    <dc:title>The Man in the High Castle</dc:title>
    <dc:creator opf:role="aut">Philip K. Dick</dc:creator>
    <dc:subject>SciFi</dc:subject>
    <opf:meta name="calibre:series" content="Science Fiction"/>
  </metadata>
</package>""",
                encoding="utf-8",
            )

            cfg = MagicMock(spec=KnrsConfig)
            cfg.calibre_path = calibre_dir
            cfg.external_library = ext_dir
            cfg.target_series = []

            # Dry run must not create catalog
            dry_res = run_external_sync(cfg, dry_run=True)
            self.assertEqual(dry_res["catalog_books"], 1)
            catalog_path = ext_dir / CATALOG_FILENAME
            self.assertFalse(catalog_path.exists())

            # Real run writes catalog and copies file
            res = run_external_sync(cfg, dry_run=False)
            self.assertEqual(res["copied"], 1)
            self.assertEqual(res["catalog_books"], 1)
            self.assertTrue(catalog_path.exists())

            data = json.loads(catalog_path.read_text(encoding="utf-8"))
            self.assertEqual(data["version"], 1)
            self.assertIn(uuid, data["books"])
            entry = data["books"][uuid]
            self.assertEqual(entry["title"], "The Man in the High Castle")
            self.assertEqual(entry["authors"], ["Philip K. Dick"])
            self.assertEqual(entry["series"], "Science Fiction")
            self.assertEqual(len(entry["files"]), 1)
            finfo = entry["files"][0]
            self.assertEqual(finfo["rel_path"], "Science Fiction/Philip K. Dick - The Man in the High Castle.epub")
            self.assertEqual(finfo["format"], "epub")
            self.assertEqual(finfo["filename_md5"], filename_md5("Philip K. Dick - The Man in the High Castle.epub"))
            self.assertEqual(finfo["koreader_partial_md5"], partial_koreader_md5(epub_path))

            # Second run must not remove .knrs-catalog.json and must not rewrite if unchanged
            mtime_before = catalog_path.stat().st_mtime_ns
            time.sleep(0.02)
            res2 = run_external_sync(cfg, dry_run=False)
            self.assertEqual(res2["removed_files"], 0)
            self.assertTrue(catalog_path.exists())
            self.assertEqual(catalog_path.stat().st_mtime_ns, mtime_before)


if __name__ == "__main__":
    unittest.main()
