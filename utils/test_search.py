from utils.search import SearchTools

def test_search():
    text = "Notes/Humanities/History/Europe"
    
    print(f"Testing text: '{text}'")
    
    # Test cases: (keys, expected, any_match=False)
    cases = [
        (["History"], True, False),
        (["history"], True, False),
        (["Europe"], True, False),
        (["Humanities"], True, False),
        (["Notes"], True, False),
        (["!Asia"], True, False),
        (["!Europe"], False, False),
        (["Hist*"], True, False),
        (["*tory"], True, False),
        (["*story*"], True, False),
        (["Hist|Geo"], True, False),
        (["History", "Europe"], True, False),
        (["History", "Asia"], False, False),
        (["History", "Asia"], True, True),
        (["History", "!Europe"], False, False),
        (["History", "!Asia"], True, False),
        (["History", "!Europe"], False, True), # !Europe is absolute
    ]
    
    for keys, expected, any_match in cases:
        result = SearchTools.match(text, keys, any_match=any_match)
        keys_str = str(keys)
        any_str = f"any={any_match}"
        print(f"Keys: {keys_str:20} | {any_str:10} | Expected: {str(expected):5} | Result: {str(result):5} | {'PASS' if result == expected else 'FAIL'}")

if __name__ == "__main__":
    test_search()
