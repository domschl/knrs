import re
import logging

logger = logging.getLogger(__name__)

class SearchTools:
    """
    Search and matching utilities ported and improved from the legacy project.
    """

    @staticmethod
    def match(text: str, keys: list[str], any_match: bool = False) -> bool:
        """
        Checks if the text matches the search keys.
        Keys support:
        - 'word': matches if 'word' is in text (case-insensitive)
        - '!word': matches if 'word' is NOT in text
        - 'w1|w2': matches if 'w1' OR 'w2' is in text
        - 'word*': matches if text contains a word starting with 'word'
        - '*word': matches if text contains a word ending with 'word'
        - '*word*': matches if 'word' is a substring
        
        By default, all top-level keys in the list must match (AND logic).
        If any_match=True, only one positive key must match (OR logic).
        Negative keys (!word) are always absolute exclusions.
        """
        if not keys:
            return True
            
        s_text = text.lower()
        
        positive_keys = [k for k in keys if not k.startswith("!")]
        negative_keys = [k[1:] for k in keys if k.startswith("!")]
        
        # Absolute exclusions
        for neg_key in negative_keys:
            if SearchTools._check_single_key(s_text, neg_key.lower()):
                return False
                
        if not positive_keys:
            return True
            
        matches = []
        for key in positive_keys:
            or_keys = key.split("|")
            or_found = any(SearchTools._check_single_key(s_text, ok.lower()) for ok in or_keys)
            matches.append(or_found)
            
        if any_match:
            return any(matches)
        else:
            return all(matches)

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
