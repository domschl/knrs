from __future__ import annotations

import sys
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path
from typing import Any

# Add project root and subprocesses to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(root_dir / "subprocesses" / "summarizer_api"))
sys.path.insert(0, str(root_dir / "subprocesses" / "summarizer_core"))

from subprocesses.summarizer_api.summarizer_api import (
    DEFAULT_LOCAL_CONFIG,
    summarize_file,
    answer_query,
    main,
)


class TestSummarizerApiConfig(unittest.TestCase):
    def test_default_config_has_auto_unload_model(self) -> None:
        self.assertIn("auto_unload_model", DEFAULT_LOCAL_CONFIG)
        self.assertIs(DEFAULT_LOCAL_CONFIG["auto_unload_model"], False)

    @patch("subprocesses.summarizer_api.summarizer_api.parse_markdown")
    @patch("subprocesses.summarizer_api.summarizer_api.chunked_summarize")
    @patch("subprocesses.summarizer_api.summarizer_api.assemble_markdown")
    @patch("subprocesses.summarizer_api.summarizer_api.WorkCache")
    @patch("subprocesses.summarizer_api.summarizer_api.ApiEngine")
    def test_summarize_file_does_not_unload_per_doc(
        self,
        mock_api_engine: MagicMock,
        mock_cache: MagicMock,
        mock_assemble: MagicMock,
        mock_chunked: MagicMock,
        mock_parse: MagicMock,
    ) -> None:
        mock_engine_instance = MagicMock()
        mock_api_engine.return_value = mock_engine_instance
        mock_parse.return_value = ({}, "markdown content")
        mock_chunked.return_value = "summary"
        mock_assemble.return_value = "full summary"

        source_file = str(root_dir / "pyproject.toml")
        dest_file = str(root_dir / "scratch" / "test_out.md")

        config: dict[str, Any] = {
            "chunk_size": 200000,
            "model_name": "test-model",
            "summary_max_tokens": 2500,
            "auto_unload_model": True,
        }
        server_config: dict[str, Any] = {"url": "http://localhost:8180"}

        with patch("sys.exit") as mock_exit:
            summarize_file(source_file, dest_file, config, server_config, 2500)
            mock_exit.assert_called_once_with(0)

        # summarize_file should never unload per-document
        mock_engine_instance.unload.assert_not_called()

    @patch("subprocesses.summarizer_api.summarizer_api.get_llm_server_config")
    @patch("subprocesses.summarizer_api.summarizer_api.get_platform_config")
    @patch("subprocesses.summarizer_api.summarizer_api.ApiEngine")
    def test_main_unload_auto_true(
        self,
        mock_api_engine: MagicMock,
        mock_get_config: MagicMock,
        mock_server_config: MagicMock,
    ) -> None:
        mock_engine_instance = MagicMock()
        mock_engine_instance.unload.return_value = True
        mock_api_engine.return_value = mock_engine_instance
        mock_get_config.return_value = {"auto_unload_model": True, "model_name": "test-model"}
        mock_server_config.return_value = {"url": "http://localhost:8180"}

        with patch("sys.argv", ["summarizer_api.py", "--unload"]):
            with patch("sys.exit") as mock_exit:
                main()
                mock_exit.assert_called_once_with(0)

        mock_engine_instance.unload.assert_called_once()

    @patch("subprocesses.summarizer_api.summarizer_api.get_llm_server_config")
    @patch("subprocesses.summarizer_api.summarizer_api.get_platform_config")
    @patch("subprocesses.summarizer_api.summarizer_api.ApiEngine")
    def test_main_unload_auto_false(
        self,
        mock_api_engine: MagicMock,
        mock_get_config: MagicMock,
        mock_server_config: MagicMock,
    ) -> None:
        mock_engine_instance = MagicMock()
        mock_api_engine.return_value = mock_engine_instance
        mock_get_config.return_value = {"auto_unload_model": False, "model_name": "test-model"}
        mock_server_config.return_value = {"url": "http://localhost:8180"}

        with patch("sys.argv", ["summarizer_api.py", "--unload"]):
            with patch("sys.exit") as mock_exit:
                main()
                mock_exit.assert_called_once_with(0)

        mock_engine_instance.unload.assert_not_called()

    @patch("subprocesses.summarizer_api.summarizer_api.get_llm_server_config")
    @patch("subprocesses.summarizer_api.summarizer_api.get_platform_config")
    @patch("subprocesses.summarizer_api.summarizer_api.ApiEngine")
    def test_main_unload_force(
        self,
        mock_api_engine: MagicMock,
        mock_get_config: MagicMock,
        mock_server_config: MagicMock,
    ) -> None:
        mock_engine_instance = MagicMock()
        mock_engine_instance.unload.return_value = True
        mock_api_engine.return_value = mock_engine_instance
        mock_get_config.return_value = {"auto_unload_model": False, "model_name": "test-model"}
        mock_server_config.return_value = {"url": "http://localhost:8180"}

        with patch("sys.argv", ["summarizer_api.py", "--unload", "--force"]):
            with patch("sys.exit") as mock_exit:
                main()
                mock_exit.assert_called_once_with(0)

        mock_engine_instance.unload.assert_called_once()


if __name__ == "__main__":
    unittest.main()
