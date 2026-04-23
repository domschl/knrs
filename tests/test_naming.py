import unittest
from knrs.naming import title_to_filename

class TestNaming(unittest.TestCase):
    def test_basic_naming(self):
        self.assertEqual(
            title_to_filename("The Structure of Scientific Revolutions", "Thomas S. Kuhn"),
            "The Structure of Scientific Revolutions - Thomas S. Kuhn"
        )

    def test_truncation(self):
        # Long title that should be truncated
        long_title = "A Very Very Long Book Title That Goes On And On And On And On And On And On And On And On"
        filename = title_to_filename(long_title, "Some Author", max_length=50)
        self.assertLessEqual(len(filename), 50)
        self.assertTrue(filename.endswith(" - Some Author"))

    def test_numbering_preservation(self):
        title = "History of the World, Vol. 3"
        author = "John Doe"
        # Truncate to a small length to force body truncation but preserve suffix
        filename = title_to_filename(title, author, max_length=40)
        self.assertTrue(filename.endswith("Vol. 3 - John Doe"))
        self.assertLessEqual(len(filename), 40)

    def test_roman_numbering(self):
        title = "Decline and Fall of the Roman Empire Volume IV"
        author = "Edward Gibbon"
        filename = title_to_filename(title, author, max_length=50)
        self.assertTrue(filename.endswith("Volume IV - Edward Gibbon"))

    def test_sanitization(self):
        self.assertEqual(
            title_to_filename("Title: With Subtitle", "Author/Name"),
            "Title — With Subtitle - Author-Name"
        )

if __name__ == "__main__":
    unittest.main()
