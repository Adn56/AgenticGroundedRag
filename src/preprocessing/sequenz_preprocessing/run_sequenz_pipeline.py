import subprocess
import sys
from pathlib import Path
import os

# =====================================================
# Add src/ to PYTHONPATH for all subprocess scripts
# =====================================================
SRC_DIR = Path(__file__).resolve().parents[2]   # .../src
env = os.environ.copy()
env["PYTHONPATH"] = str(SRC_DIR)

steps = [
    "00_filter_media_2018_2019.py",
    "01_generate_user_sequences.py",
    "02_filter_sequences_by_allowed_pois.py",
    "03_add_canonical_ids_to_sequences.py",
    "04_sequence_time_duplicat_cleaning.py",
    "05_speed_based_cleaning.py",
    "06_remove_length1_sequences.py"
]

BASE_DIR = Path(__file__).resolve().parent

for step in steps:
    print(f"Running: {step}")
    script_path = BASE_DIR / step
    subprocess.run([sys.executable, str(script_path)], check=True, env=env)

print("=== Sequence preprocessing finished successfully ===")
