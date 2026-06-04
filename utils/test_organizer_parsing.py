import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import load_config
from wiki.organizer import classify_research_files

def test_organizer_parsing():
    print("Loading config...")
    cfg = load_config()

    # 1. Test case: perfect JSON inside code blocks
    response_valid = """
Some thinking here...
```json
{
  "file1.md": "History/file1.md",
  "file2.md": "Science/file2.md"
}
```
More text.
"""
    # 2. Test case: trailing comma in JSON (common LLM typo)
    response_trailing_comma = """
```json
{
  "file1.md": "History/file1.md",
}
```
"""
    # 3. Test case: response containing ellipsis placeholder
    response_ellipsis = """
```json
{
  "file1.md": "History/file1.md",
  ...
}
```
"""
    # 4. Test case: invalid JSON altogether
    response_invalid = "This is not json at all."

    # 5. Test case: JSON block preceded by a thinking block with quoted code blocks
    response_with_quoted_blocks = """
<think>
Checking constraints: "Do NOT use '...' or placeholders under any circumstances." "Respond ONLY with the raw JSON object inside a ```json ... ``` code block."
</think>

```json
{
  "file1.md": "History/file1.md"
}
```
"""

    files_dummy = [{"path": "file1.md", "title": "File 1", "tags": [], "snippet": ""}]

    # Mock the AgentSession
    mock_session = MagicMock()
    mock_session.__enter__.return_value = mock_session
    
    with patch("wiki.organizer.AgentSession", return_value=mock_session):
        # Test 1: Valid JSON parsing
        mock_session.generate.return_value = response_valid
        print("Testing valid JSON parsing...")
        result = classify_research_files(cfg, files_dummy)
        assert result == {"file1.md": "History/file1.md", "file2.md": "Science/file2.md"}
        print("Passed valid JSON parsing.")

        # Test 2: Trailing comma fixing
        mock_session.generate.return_value = response_trailing_comma
        print("Testing trailing comma parsing...")
        result = classify_research_files(cfg, files_dummy)
        assert result == {"file1.md": "History/file1.md"}
        print("Passed trailing comma parsing.")

        # Test 3: Ellipsis detection
        mock_session.generate.return_value = response_ellipsis
        print("Testing ellipsis detection...")
        try:
            classify_research_files(cfg, files_dummy)
            raise AssertionError("Should have raised RuntimeError for ellipsis!")
        except RuntimeError as e:
            print(f"Caught expected error: {e}")
            assert "abbreviated or incomplete" in str(e)
        print("Passed ellipsis detection.")

        # Test 4: Invalid JSON
        mock_session.generate.return_value = response_invalid
        print("Testing invalid JSON handling...")
        try:
            classify_research_files(cfg, files_dummy)
            raise AssertionError("Should have raised RuntimeError for invalid JSON!")
        except RuntimeError as e:
            print(f"Caught expected error: {e}")
            assert "not a parseable JSON object mapping" in str(e)
        print("Passed invalid JSON handling.")

        # Test 5: Quoted code blocks inside thinking process
        mock_session.generate.return_value = response_with_quoted_blocks
        print("Testing thinking block with quoted backticks...")
        result = classify_research_files(cfg, files_dummy)
        assert result == {"file1.md": "History/file1.md"}
        print("Passed thinking block with quoted backticks.")

        # Test 6: Cut-off response (no closing brace — max_tokens hit mid-JSON)
        response_cutoff = """```json
{
  "file1.md": "History/file1.md",
  "file2.md": "Science/file2.md
"""
        mock_session.generate.return_value = response_cutoff
        print("Testing cut-off response detection...")
        try:
            classify_research_files(cfg, files_dummy)
            raise AssertionError("Should have raised RuntimeError for cut-off response!")
        except RuntimeError as e:
            print(f"Caught expected error: {e}")
            assert "cut off" in str(e).lower() or "not a parseable" in str(e).lower()
        print("Passed cut-off response detection.")

    print("\nAll organizer parsing tests PASSED!")

if __name__ == "__main__":
    test_organizer_parsing()
