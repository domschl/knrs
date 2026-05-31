import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.agent import parse_gemma_args

def test_parse_gemma_args():
    # Test cases: (input_string, expected_output_dict)
    cases = [
        ('{query:<|"|>Vladimir Terziski<|"|>}', {'query': 'Vladimir Terziski'}),
        ('{path:<|"|>books:History Of Rome.md<|"|>,start_line:1,end_line:-1}', {
            'path': 'books:History Of Rome.md',
            'start_line': 1,
            'end_line': -1
        }),
        ('{context_filters:[<|"|>filter1<|"|>,<|"|>filter2<|"|>],start_year:-100}', {
            'context_filters': ['filter1', 'filter2'],
            'start_year': -100
        }),
        ('{}', {}),
        ('{start_line: 1, end_line: 10}', {
            'start_line': 1,
            'end_line': 10
        }),
        # Test case with comma inside a string
        ('{path:<|"|>History, Law and Rome.md<|"|>,start_line:1}', {
            'path': 'History, Law and Rome.md',
            'start_line': 1
        }),
        # Test case with multiple colons
        ('{query:<|"|>Vladimir Terziski: The Legend<|"|>}', {
            'query': 'Vladimir Terziski: The Legend'
        }),
    ]

    failed = False
    for idx, (input_str, expected) in enumerate(cases, 1):
        try:
            result = parse_gemma_args(input_str)
            if result != expected:
                print(f"Test {idx} FAILED: Expected {expected}, got {result}")
                failed = True
            else:
                print(f"Test {idx} PASSED")
        except Exception as e:
            print(f"Test {idx} EXCEPTION: {e}")
            failed = True

    if failed:
        sys.exit(1)
    else:
        print("All Gemma parsing tests PASSED!")

if __name__ == "__main__":
    test_parse_gemma_args()
