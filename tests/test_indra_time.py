import unittest
from knrs.timelines.indra_time import parse_point, format_point, parse_interval

class TestIndraTime(unittest.TestCase):
    def test_ad_dates(self):
        self.assertEqual(parse_point("2024"), 2024.0)
        self.assertEqual(parse_point("1066"), 1066.0)
        # Month/Day handling
        self.assertAlmostEqual(parse_point("2024-02"), 2024.0833, places=4)
        
    def test_bc_dates(self):
        self.assertEqual(parse_point("1 BC"), 0.0)
        self.assertEqual(parse_point("44 BC"), -43.0)
        
    def test_bp_and_deep_time(self):
        # 1950 is base for BP, kya, Ma, Ga
        self.assertEqual(parse_point("1000 BP"), 950.0)
        self.assertEqual(parse_point("10 kya"), 1950.0 - 10_000)
        self.assertEqual(parse_point("65 Ma"), 1950.0 - 65_000_000)
        self.assertEqual(parse_point("4.5 Ga"), 1950.0 - 4_500_000_000)

    def test_intervals(self):
        start, end = parse_interval("100 BC - 50 BC")
        self.assertEqual(start, -99.0)
        self.assertEqual(end, -49.0)
        
    def test_formatting_roundtrip(self):
        points = ["2024", "1066", "44 BC", "1000 BP", "10.00 kya BP", "65.00 Ma BP"]
        for p in points:
            with self.subTest(point=p):
                val = parse_point(p)
                formatted = format_point(val)
                # We expect canonical form which might differ slightly (e.g. 10 kya BP vs 10 kya)
                # but should parse back to same value
                self.assertAlmostEqual(parse_point(formatted), val, places=2)

if __name__ == "__main__":
    unittest.main()
