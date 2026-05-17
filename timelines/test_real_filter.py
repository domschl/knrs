import json
from pathlib import Path
from timelines.indra_time import parse_point

def test_filter():
    json_path = Path.home() / "KnrsData/Timelines/timelines.json"
    if not json_path.exists():
        print("File not found")
        return
        
    with open(json_path, "r") as f:
        events = json.load(f)
        
    start_year = parse_point("1945-05-08")
    end_year = parse_point("1945-09-02")
    
    print(f"Filter: {start_year} to {end_year}")
    
    count = 0
    for ev in events:
        ev_start = ev["start_year"]
        ev_end = ev["end_year"]
        
        if ev_end < start_year:
            continue
        if ev_start > end_year:
            continue
            
        count += 1
        if count <= 5:
            print(f"Match: {ev_start} - {ev_end} | {ev.get('context')}")

    print(f"Total matches: {count}")

if __name__ == "__main__":
    test_filter()
