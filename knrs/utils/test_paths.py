from knrs.utils.search import SearchTools

def test_paths():
    text = "Notes/Humanities/History/Europe"
    print(f"Text: {text}")
    print(f"Match 'History': {SearchTools.match(text, ['History'])}")
    print(f"Match 'Hist': {SearchTools.match(text, ['Hist'])}")
    print(f"Match 'Hist*': {SearchTools.match(text, ['Hist*'])}")
    print(f"Match 'Europe': {SearchTools.match(text, ['Europe'])}")

if __name__ == "__main__":
    test_paths()
