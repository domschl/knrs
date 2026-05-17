import os
import sys
import shutil
from pathlib import Path

# Add project root to sys.path so we can import modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import load_config
from agent.tools import AgentTools

def main():
    print("Loading knrs configuration...")
    try:
        config = load_config()
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)

    # Override paths to writeable workspace subdirectory for the sandbox
    test_temp_dir = Path(__file__).resolve().parent.parent / "test_temp_sandbox"
    if test_temp_dir.exists():
        shutil.rmtree(test_temp_dir)
    test_temp_dir.mkdir(parents=True, exist_ok=True)

    # Set paths on config
    config.wiki_path = test_temp_dir / "Wiki"
    config.knrs_data = test_temp_dir / "KnrsData"

    print(f"wiki_path (sandbox-overridden): {config.wiki_path}")
    print(f"markdown_books (sandbox-overridden): {config.markdown_books}")

    # Set up temporary directories for testing
    test_note_dir = config.wiki_path / "Notes" / "TestingAgent"
    test_note_dir.mkdir(parents=True, exist_ok=True)
    test_book_dir = config.markdown_books / "TestingAgent"
    test_book_dir.mkdir(parents=True, exist_ok=True)

    # Create dummy files
    dummy_note = test_note_dir / "Dummy Note Page.md"
    dummy_note.write_text("Hello dummy note", encoding="utf-8")
    
    dummy_book = test_book_dir / "Dummy Book Page.md"
    dummy_book.write_text("Hello dummy book", encoding="utf-8")

    # Instantiate AgentTools
    tools = AgentTools(config)

    # List of test cases: (input_string, expected_resolved_path)
    test_cases = [
        # 1. Exact path
        (f"wiki:Notes/TestingAgent/Dummy Note Page.md", dummy_note),
        # 2. Missing extension with prefix
        (f"wiki:Notes/TestingAgent/Dummy Note Page", dummy_note),
        # 3. Missing extension, no prefix (direct relative check)
        (f"Notes/TestingAgent/Dummy Note Page", dummy_note),
        # 4. Bare stem (recursive search fallback)
        ("Dummy Note Page", dummy_note),
        # 5. Bracketed stem (recursive search fallback with brackets cleaning)
        ("[[Dummy Note Page]]", dummy_note),
        # 6. Bracketed exact prefix link
        ("[[wiki:Notes/TestingAgent/Dummy Note Page.md]]", dummy_note),
        # 7. Prefix missing extension in bracketed link
        ("[[wiki:Notes/TestingAgent/Dummy Note Page]]", dummy_note),
        # 8. Books prefixed stem search
        ("books:Dummy Book Page", dummy_book),
        # 9. Books bare stem search
        ("Dummy Book Page", dummy_book),
        # 10. Books bracketed stem search
        ("[[Dummy Book Page]]", dummy_book),
    ]

    failed = False
    print("\nRunning test cases...")
    for idx, (input_str, expected) in enumerate(test_cases, 1):
        print(f"Test {idx}: Input: '{input_str}' -> ", end="")
        resolved = tools._resolve_read_path(input_str)
        if resolved is None:
            print("FAILED (returned None)")
            failed = True
        elif resolved.resolve() != expected.resolve():
            print(f"FAILED (expected {expected}, got {resolved})")
            failed = True
        else:
            print(f"PASSED (resolved to {resolved})")

    # 11. Test Wikipedia Local Cache Hit (Exact match)
    print("\nTest 11: Wikipedia Local Cache Hit (Exact Case) -> ", end="")
    wiki_wiki_dir = config.wiki_path / "AINotes" / "Research" / "Wikipedia"
    wiki_wiki_dir.mkdir(parents=True, exist_ok=True)
    cached_art = wiki_wiki_dir / "Dummy Wiki Article.md"
    cached_art.write_text("---\ntitle: \"Dummy Wiki Article\"\nsource: \"Wikipedia\"\n---\n\n# Dummy Wiki Article\n\nDummy Article Content", encoding="utf-8")
    
    result = tools.wikipedia_fetch("Dummy Wiki Article")
    if "Successfully loaded cached article" in result:
        print("PASSED")
    else:
        print(f"FAILED (got result: {result})")
        failed = True

    # 12. Test Wikipedia Local Cache Hit (Case-insensitive match)
    print("Test 12: Wikipedia Local Cache Hit (Case-insensitive) -> ", end="")
    result_lower = tools.wikipedia_fetch("dummy wiki article")
    if "Successfully loaded cached article" in result_lower:
        print("PASSED")
    else:
        print(f"FAILED (got result: {result_lower})")
        failed = True

    # Cleanup
    print("\nCleaning up temporary sandbox directory...")
    if test_temp_dir.exists():
        shutil.rmtree(test_temp_dir)

    if failed:
        print("\nSome tests FAILED.")
        sys.exit(1)
    else:
        print("\nAll tests PASSED successfully!")

if __name__ == "__main__":
    main()
