import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent

print("=== Running POI preprocessing ===")
subprocess.run([sys.executable, str(BASE / "poi_preprocessing" / "run_poi_pipeline.py")], check=True)

print("=== Running Sequence preprocessing ===")
subprocess.run([sys.executable, str(BASE / "sequenz_preprocessing" / "run_sequenz_pipeline.py")], check=True)

print("=== All preprocessing finished ===")
