from knrs.utils.search import SearchTools

def test_search():
    text = "Notes/Humanities/History/Europe"
    
    print(f"Testing text: '{text}'")
    
    # Test cases
    cases = [
        (["History"], True),
        (["history"], True),
        (["Europe"], True),
        (["Humanities"], True),
        (["Notes"], True),
        (["!Asia"], True),
        (["!Europe"], False),
        (["Hist*"], True),
        (["*tory"], True),
        (["*story*"], True),
        (["Hist|Geo"], True),
    ]
    
    for keys, expected in cases:
        result = SearchTools.match(text, keys)
        keys_str = str(keys)
        print(f"Keys: {keys_str:20} | Expected: {str(expected):5} | Result: {str(result):5} | {'PASS' if result == expected else 'FAIL'}")

if __name__ == "__main__":
    test_search()
