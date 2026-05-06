from knrs.utils.search import SearchTools

def test_boundaries():
    text = "1932.0 2016.0 Science/Geology"
    print(f"Text: '{text}'")
    print(f"Match '1945': {SearchTools.match(text, ['1945'])}")
    print(f"Match '1932': {SearchTools.match(text, ['1932'])}")
    print(f"Match 'Geology': {SearchTools.match(text, ['Geology'])}")
    print(f"Match 'Geo': {SearchTools.match(text, ['Geo'])}")

if __name__ == "__main__":
    test_boundaries()
