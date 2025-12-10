import subprocess
import sys
from pathlib import Path

steps = [
    "00_clean_address.py",
    "01_filter_geodistance.py",
    "02_split_by_usercount.py",
    "03_poi_rule_based_cleaning.py",
    "04_create_canonical_poi_ids.py",
]

BASE_DIR = Path(__file__).resolve().parent

for step in steps:
    print(f"Running: {step}")
    script_path = BASE_DIR / step
    subprocess.run([sys.executable, str(script_path)], check=True)

print("=== POI preprocessing finished successfully ===")
