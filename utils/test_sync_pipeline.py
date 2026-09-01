from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path
from typing import Any

from config import KnrsConfig
from repl.commands import (
    cmd_sync,
    cmd_sync_calibre,
    cmd_sync_summaries,
    cmd_sync_wiki,
    cmd_sync_external_lib,
    cmd_wiki_check,
    cmd_timeline,
    cmd_index,
    cmd_sync_git,
    cmd_unload,
    GIT_STATE,
)


class TestSyncPipeline(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = MagicMock(spec=KnrsConfig)
        self.cfg.auto_git_sync = False
        self.cfg.summarizer_name = "summarizer_api"
        self.cfg.knrs_data = Path("/mock/data")
        self.cfg.wiki_path = Path("/mock/wiki")
        self.cfg.notes_path = Path("/mock/wiki/Notes")
        self.cfg.timelines = Path("/mock/data/timelines")
        self.cfg.markdown_books = Path("/mock/data/MarkdownBooks")
        self.cfg.book_summaries = Path("/mock/data/BookSummaries")
        self.cfg.external_library = Path("/mock/external")

        # Set git safety to true for testing
        GIT_STATE["knrs_data_safe_remote"] = True
        GIT_STATE["wiki_path_safe_remote"] = True
        GIT_STATE["knrs_data_safe_local"] = True
        GIT_STATE["wiki_path_safe_local"] = True

    @patch("repl.commands.cmd_sync_calibre")
    @patch("repl.commands.cmd_sync_summaries")
    @patch("repl.commands.cmd_sync_wiki")
    @patch("repl.commands.cmd_wiki_check")
    @patch("repl.commands.cmd_timeline")
    @patch("repl.commands.cmd_index")
    @patch("repl.commands.cmd_sync_external_lib")
    def test_cmd_sync_all_success(
        self,
        mock_ext: MagicMock,
        mock_idx: MagicMock,
        mock_tl: MagicMock,
        mock_chk: MagicMock,
        mock_wiki: MagicMock,
        mock_sum: MagicMock,
        mock_cal: MagicMock,
    ) -> None:
        mock_cal.return_value = {"success_count": 2, "failure_count": 0, "total": 2, "actions": {"ADD": 2}}
        mock_sum.return_value = {"success_count": 2, "failure_count": 0, "total": 2, "actions": {"ADD": 2}}
        mock_wiki.return_value = {"success_count": 2, "failure_count": 0, "total": 2, "actions": {"ADD": 2}, "frontmatter_updated": 0}
        mock_chk.return_value = {"checked": 10, "updated": 0, "duplicates": 0, "broken_links": 0, "fixed_links": 0, "malformed_links": 0, "errors": 0}
        mock_tl.return_value = {"files_scanned": 5, "events_extracted": 20, "failure_count": 0, "failed_items": []}
        mock_idx.return_value = {"files_indexed": 2, "files_failed": 0, "chunks_indexed": 10, "total_chunks": 10, "total_files": 2, "failed_items": []}
        mock_ext.return_value = {"success_count": 2, "failure_count": 0, "total": 2, "copied": 2, "updated": 0, "removed_files": 0, "failed_items": []}

        res = cmd_sync([], self.cfg)
        self.assertIsNotNone(res)
        assert res is not None
        self.assertEqual(res["total_faults"], 0)
        self.assertEqual(res["total_warnings"], 0)
        self.assertEqual(len(res["steps"]), 7)

    @patch("repl.commands.cmd_sync_calibre")
    @patch("repl.commands.cmd_sync_summaries")
    @patch("repl.commands.cmd_sync_wiki")
    @patch("repl.commands.cmd_wiki_check")
    @patch("repl.commands.cmd_timeline")
    @patch("repl.commands.cmd_index")
    @patch("repl.commands.cmd_sync_external_lib")
    def test_cmd_sync_handles_faults_and_exceptions(
        self,
        mock_ext: MagicMock,
        mock_idx: MagicMock,
        mock_tl: MagicMock,
        mock_chk: MagicMock,
        mock_wiki: MagicMock,
        mock_sum: MagicMock,
        mock_cal: MagicMock,
    ) -> None:
        # Step 1: 1 failure
        mock_cal.return_value = {"success_count": 1, "failure_count": 1, "total": 2, "actions": {"ADD": 2}, "failed_items": ["ADD: Book Failed"]}
        # Step 2: raises unexpected exception
        mock_sum.side_effect = RuntimeError("Subprocess crash")
        # Step 3: success
        mock_wiki.return_value = {"success_count": 1, "failure_count": 0, "total": 1, "actions": {"ADD": 1}}
        # Step 4: warning (broken links)
        mock_chk.return_value = {"checked": 10, "updated": 0, "duplicates": 1, "broken_links": 2, "fixed_links": 0, "malformed_links": 0, "errors": 0}
        # Step 5-7: success
        mock_tl.return_value = {"files_scanned": 5, "events_extracted": 20, "failure_count": 0, "failed_items": []}
        mock_idx.return_value = {"files_indexed": 0, "files_failed": 0, "chunks_indexed": 0, "total_chunks": 10, "total_files": 2, "failed_items": []}
        mock_ext.return_value = {"success_count": 0, "failure_count": 0, "total": 0, "copied": 0, "updated": 0, "removed_files": 0, "failed_items": []}

        res = cmd_sync([], self.cfg)
        self.assertIsNotNone(res)
        assert res is not None
        # Calibre failed (1) + Summaries exception (1) = 2 faults
        self.assertEqual(res["total_faults"], 2)
        # Wiki check warning (1)
        self.assertEqual(res["total_warnings"], 1)

    @patch("summarizer.engine.unload_model")
    def test_cmd_unload(self, mock_unload_model: MagicMock) -> None:
        mock_unload_model.return_value = True
        cmd_unload([], self.cfg)
        mock_unload_model.assert_called_once_with("summarizer_api", force=True)


if __name__ == "__main__":
    unittest.main()
