import sys
from pathlib import Path
from config import KnrsConfig
from agent.tools import AgentTools

def test_file_read_limit(tmp_path: Path) -> None:
    # Setup folders
    wiki_path = tmp_path / "wiki"
    wiki_path.mkdir(parents=True)
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)
    
    books_dir = data_dir / "MarkdownBooks"
    books_dir.mkdir(parents=True)
    summaries_dir = data_dir / "BookSummaries"
    summaries_dir.mkdir(parents=True)

    # Initialize KnrsConfig
    cfg = KnrsConfig(
        calibre_path=tmp_path / "calibre",
        notes_path=tmp_path / "notes",
        knrs_data=data_dir,
        wiki_path=wiki_path,
        vector_db_path=tmp_path / "vectordb",
    )

    # Setup AgentTools
    tools = AgentTools(cfg)

    # Write small file (10 lines)
    small_file = books_dir / "small.md"
    small_lines = [f"Line {i}" for i in range(1, 11)]
    small_file.write_text("\n".join(small_lines), encoding="utf-8")

    # Test reading small file completely
    result_small = tools.file_read("books:small.md", 1, -1)
    assert result_small == "\n".join(small_lines)
    assert "[TRUNCATED" not in result_small

    # Write large file (1200 lines)
    large_file = books_dir / "large.md"
    large_lines = [f"Line {i}" for i in range(1, 1201)]
    large_file.write_text("\n".join(large_lines), encoding="utf-8")

    # Test reading large file completely (should be truncated to 1000 lines)
    result_large_truncated = tools.file_read("books:large.md", 1, -1)
    expected_lines_first_chunk = large_lines[:1000]
    expected_first_chunk_text = "\n".join(expected_lines_first_chunk)
    assert result_large_truncated.startswith(expected_first_chunk_text)
    assert "[TRUNCATED: Read limit of 1000 lines reached. The file has 1200 lines total. Use file_read with start_line=1001" in result_large_truncated

    # Test reading next chunk of large file (lines 1001-1200)
    result_large_second_chunk = tools.file_read("books:large.md", 1001, -1)
    expected_lines_second_chunk = large_lines[1000:1200]
    assert result_large_second_chunk == "\n".join(expected_lines_second_chunk)
    assert "[TRUNCATED" not in result_large_second_chunk

    # Test reading exact slice within limit
    result_slice = tools.file_read("books:large.md", 50, 150)
    expected_slice = large_lines[49:150]
    assert result_slice == "\n".join(expected_slice)
    assert "[TRUNCATED" not in result_slice

    # Test input validation errors
    # start_line > total_lines
    err_start = tools.file_read("books:small.md", 15, -1)
    assert "Error: start_line (15) is greater than total lines in file (10)" in err_start

    # end_line < start_line
    err_end = tools.file_read("books:small.md", 5, 3)
    assert "Error: end_line (3) is less than start_line (5)" in err_end
