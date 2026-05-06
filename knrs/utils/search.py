import re
import logging

logger = logging.getLogger(__name__)

class SearchTools:
    """
    Search and matching utilities ported and improved from the legacy project.
    """

    @staticmethod
    def match(text: str, keys: list[str]) -> bool:
        """
        Checks if the text matches the search keys.
        Keys support:
        - 'word': matches if 'word' is in text (case-insensitive)
        - '!word': matches if 'word' is NOT in text
        - 'w1|w2': matches if 'w1' OR 'w2' is in text
        - 'word*': matches if text contains a word starting with 'word'
        - '*word': matches if text contains a word ending with 'word'
        - '*word*': matches if 'word' is a substring
        
        All top-level keys in the list must match (AND logic).
        """
        if not keys:
            return True
            
        s_text = text.lower()
        
        for key in keys:
            if key.startswith("!"):
                # Negative match
                neg_key = key[1:].lower()
                if SearchTools._check_single_key(s_text, neg_key):
                    return False
                continue
            
            # Positive match (OR logic within the key)
            or_keys = key.split("|")
            or_found = False
            for or_key in or_keys:
                if SearchTools._check_single_key(s_text, or_key.lower()):
                    or_found = True
                    break
            
            if not or_found:
                return False
                
        return True

    @staticmethod
    def _check_single_key(text: str, key: str) -> bool:
        """Helper to check a single key (already lowered) against text."""
        if not key:
            return True
            
        # Wildcard handling
        if '*' in key:
            # Escape regex special chars but keep * as .*
            pattern = re.escape(key).replace(r'\*', '.*')
            try:
                return re.search(pattern, text, re.IGNORECASE) is not None
            except re.error:
                logger.error("Invalid regex pattern generated from key: %s", key)
                return key.lower() in text.lower()
        else:
            # Substring match (matching legacy behavior)
            return key.lower() in text.lower()
